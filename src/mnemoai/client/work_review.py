"""Artifact-focused strategy and change checkpoints in the original user turn."""

import copy
import json
import re
import threading
import time
from contextlib import contextmanager

from langchain_core.messages import HumanMessage, SystemMessage

from mnemoai.client import review
from mnemoai.client.agent import mid_turn, supervised_turn
from mnemoai.client.agent.reasoning_utils import extract_visible_text
from mnemoai.client.review_evidence import Capture, text
from mnemoai.utils.config import config
from mnemoai.utils.review_protocol import DATA_MARKER, FEEDBACK_PREFIX, ReviewStopped

_SOCIAL = re.compile(
    r"^(?:hi|hello|hey|thanks|thank you|bye|goodbye|ciao|salve|grazie|buongiorno|"
    r"buonasera|arrivederci)[\s!.?]*$", re.IGNORECASE,
)


def obvious_conversation(prompt):
    return not prompt.strip() or bool(_SOCIAL.fullmatch(prompt.strip()))


def parse_strategy(response):
    if response is None or getattr(response, "tool_calls", None):
        raise ValueError("Strategy preparation must not execute tools")
    raw = extract_visible_text(response.content)
    if len(raw) > 12_000:
        raise ValueError("Strategy exceeded its output limit")
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw.partition("\n")[2].rsplit("```", 1)[0]
    result = json.loads(raw)
    if not isinstance(result, dict) or set(result) != {"kind", "strategy"}:
        raise ValueError("Invalid strategy response")
    if result["kind"] not in {"code", "document", "none"}:
        raise ValueError("Invalid review task kind")
    plan = result["strategy"]
    if not isinstance(plan, str) or len(plan) > 8_000:
        raise ValueError("Invalid strategy text")
    if result["kind"] != "none" and not plan.strip():
        raise ValueError("Artifact strategy is empty")
    if any(not c.isprintable() and not c.isspace() for c in plan):
        raise ValueError("Invalid strategy control characters")
    return result


class WorkReview:
    """One foreground task; reviewing never grants permissions or runs tools."""

    def __init__(self, client, task, context):
        self.client, self.agent, self.reviewer = client, client.agent, client.reviewer
        self.task, self.context = task, context
        self.previous_task = self.reviewer.artifact_task
        self.reviewer.reset(clear_task=False)
        self.generation = self.reviewer._generation
        self.capture = None
        self.kind = None
        self.strategy = ""
        self.disabled = False
        self.checkpoints = []
        self.wave_notes = []
        self.actor_steps = 0
        self.change_reviews = 0
        self.remaining = None
        self._writes = self._shells = 0
        self._git_revision = None
        self.owner = threading.get_ident()
        self.finished = False

    def _current(self):
        return self.reviewer.enabled and self.reviewer._generation == self.generation

    def _cancel_check(self):
        if self.agent._cancelled():
            raise KeyboardInterrupt
        if not self._current():
            raise ReviewStopped("Review mode or task changed")

    @contextmanager
    def _charged(self):
        """Count supervision overhead, not normal implementation time between checks."""
        started = time.monotonic()
        try:
            self._cancel_check()
            if self.remaining is None or self.remaining <= 0:
                raise ReviewStopped("Review time budget exhausted")
            yield
        finally:
            if self.remaining is not None:
                self.remaining = max(0, self.remaining - (time.monotonic() - started))

    def _unavailable(self, reason, checkpoint):
        self.disabled = True
        self.agent._completion_supervisor = None
        self.agent._review_capture = None
        if not self._current():
            return
        report = self.reviewer.incomplete(reason)
        report.update(unavailable=True, checkpoint=checkpoint)
        self._remember(report)
        self.client._publish_review("complete", self.reviewer.last)

    def _remember(self, report):
        if report.get("checkpoint") == "strategy":
            report = {**report, "strategy": self.strategy}
        self.checkpoints.append(copy.deepcopy(report))
        # UI and audit retain the checkpoint sequence, but not recursive copies.
        self.reviewer.last = {
            **report, "checkpoints": copy.deepcopy(self.checkpoints),
            "strategy": self.strategy, "task_kind": self.kind,
        }

    def _feedback(self, report, checkpoint):
        cited = {ref for finding in report.get("findings", []) for ref in finding["evidence_ids"]}
        data = {
            "checkpoint": checkpoint, "strategy": self.strategy,
            "summary": report["summary"], "verdict": report["verdict"],
            "findings": report.get("findings", []),
            "reviewer": report.get("reviewer", ""),
            "evidence": [item for item in self.reviewer._last_evidence if item["id"] in cited],
        }
        return HumanMessage(
            content=FEEDBACK_PREFIX + "\n" + config.require_prompt("WORK_REVIEW_FEEDBACK_PROMPT")
            + DATA_MARKER + json.dumps(data, ensure_ascii=True),
            name="reviewer",
        )

    def _strategy_call(self, feedback=None):
        """Tool-free actor call. Raw strategy JSON is never printed or executed."""
        context = []
        for item in self.context:
            value, clipped = text(item.get("text", ""), 1200)
            context.append({"kind": item.get("kind"), "text": value, "truncated": clipped})
        task, clipped = text(self.task, 8000)
        previous, previous_clipped = text(self.previous_task, 4000)
        payload = {"request": task, "request_truncated": clipped,
                   "context": context, "previous_artifact_task": previous,
                   "previous_task_truncated": previous_clipped}
        if feedback:
            payload.update(strategy=self.strategy, reviewer_feedback=feedback)
        instruction = config.require_prompt("WORK_STRATEGY_PROMPT")
        serialized = json.dumps(payload, ensure_ascii=True)
        while context and review.estimated_input_tokens(instruction, serialized) > self.options["MAX_INPUT_TOKENS"]:
            context.pop()
            payload["context_truncated"] = True
            serialized = json.dumps(payload, ensure_ascii=True)
        if review.estimated_input_tokens(instruction, serialized) > self.options["MAX_INPUT_TOKENS"]:
            payload.update(previous_artifact_task="", previous_task_truncated=bool(previous))
            serialized = json.dumps(payload, ensure_ascii=True)
        if review.estimated_input_tokens(instruction, serialized) > self.options["MAX_INPUT_TOKENS"]:
            raise ValueError("Strategy input exceeds the review input budget")
        messages = [
            SystemMessage(content=instruction),
            HumanMessage(content=serialized),
        ]
        model = self.agent._callback_free_model()
        if model is self.agent.model and getattr(model, "callbacks", None):
            raise RuntimeError("Cannot isolate strategy model callbacks")
        budget = review.LoopBudget(
            min(self.options["TIMEOUT"], self.remaining), 1,
            cancel=self.agent._cancelled, current=self._current,
        )
        response = None
        with self._charged(), supervised_turn.budget_scope(self.agent, budget):
            self.actor_steps += 1
            try:
                response, _ = self.agent._stream_response(messages, {}, model=model, quiet=True)
            finally:
                self.agent._record_usage(response)
        return parse_strategy(response)

    def _check(self, checkpoint, answer):
        self.client._publish_review("reviewing_" + checkpoint, announce=False)
        self.agent._start_spinner(f"Reviewing {checkpoint}")
        try:
            with self._charged():
                report = self.reviewer.finish(
                    self.capture, answer, checkpoint=checkpoint,
                    time_limit=self.remaining,
                    **self.client._review_options(self.context, checkpoint),
                )
        finally:
            self.agent._stop_spinner()
        self._cancel_check()
        if report.get("unavailable"):
            self._unavailable(report["summary"], checkpoint)
            return None
        self._remember(report)
        self.client._publish_review("feedback", self.reviewer.last)
        return report

    def prepare(self, state, step_limit):
        """Select artifact work and discuss its strategy before the graph can edit."""
        try:
            self.options = review.settings()
            self.remaining = self.options["TOTAL_TIMEOUT"]
            if obvious_conversation(self.task):
                self.agent._completion_supervisor = None
                return 0
            if step_limit <= 1:
                raise ReviewStopped("No spare agent step for strategy preparation")
            self.agent._start_spinner("Preparing response")
            plan = self._strategy_call()
            self._cancel_check()
            if plan["kind"] == "none":
                self.reviewer.artifact_task = ""
                self.reviewer.view = None
                self.agent._completion_supervisor = None
                return self.actor_steps
            self.kind, self.strategy = plan["kind"], plan["strategy"]
            self.client._publish_review("strategy", announce=False)
            strategy_context = {"id": "strategy", "kind": "actor_strategy", "text": self.strategy}
            self.context.append(strategy_context)
            self.reviewer.artifact_task = self.task
            self.capture = Capture(self.task)
            self.agent._review_capture = self.capture
            state["messages"].extend(mid_turn.drain(self.agent))
            self._git_revision = (self.capture.before or {}).get("revision")
            for revision in range(self.options["MAX_STRATEGY_ROUNDS"] + 1):
                report = self._check("strategy", self.strategy)
                if report is None:
                    # Retain the actor's plan even when the judge is down, without
                    # turning an unavailable verdict into an approval or warning loop.
                    state["messages"].append(self._feedback({
                        "verdict": "inconclusive", "summary": "Chat-model strategy for this task.",
                        "findings": [], "reviewer": "unavailable",
                    }, "strategy"))
                    break
                state["messages"].append(self._feedback(report, "strategy"))
                if (
                    report["verdict"] == "pass" or not report.get("valid_verdict")
                    or not report.get("findings") or report.get("stale")
                    or revision == self.options["MAX_STRATEGY_ROUNDS"]
                    or self.actor_steps >= step_limit - 1
                ):
                    break
                plan = self._strategy_call({
                    "summary": report["summary"], "findings": report["findings"],
                })
                if plan["kind"] not in {"code", "document"}:
                    break
                self.strategy = plan["strategy"]
                strategy_context["text"] = self.strategy
                self.capture.reopen()
            if not self.disabled:
                self.capture.reopen()
                self.agent._completion_supervisor = self.complete
            state["messages"].extend(mid_turn.drain(self.agent))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self._unavailable(f"Strategy review unavailable ({type(exc).__name__}); task continues.", "strategy")
        finally:
            self.agent._stop_spinner()
        return self.actor_steps

    def after_batch(self, messages, *, wave=False):
        """Review completed edit batches, never in a worker or during corrections."""
        if (
            self.disabled or self.capture is None or not self._current()
            or threading.get_ident() != self.owner or self.agent._cancelled()
            or supervised_turn.budget_for(self.agent) is not None
            or self.change_reviews >= self.options["MAX_CHANGE_REVIEWS"]
        ):
            return []
        writes, shells = self.capture.write_serial, self.capture.shell_serial
        if writes == self._writes and shells == self._shells:
            return []
        try:
            changed = writes != self._writes
            self._writes, self._shells = writes, shells
            if not changed:
                now = self.capture._git()
                changed = (now or {}).get("revision") != self._git_revision
            if not changed:
                return []
            self.change_reviews += 1
            answer = self.agent._last_visible_from(messages) or "An implementation batch has completed."
            report = self._check("changes", answer)
            self._git_revision = (self.capture.after or {}).get("revision")
            if report is None:
                return []
            self.capture.reopen()
            if report.get("valid_verdict") and not report.get("stale") and report.get("findings"):
                note = self._feedback(report, "changes")
                if wave:
                    self.wave_notes.append(note)
                return [note]
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self._unavailable(f"Change review unavailable ({type(exc).__name__}); task continues.", "changes")
        return []

    def complete(self, state, steps):
        if self.disabled or self.capture is None or not self._current():
            return
        try:
            self.finished = True
            with self._charged():
                self.client._supervise_turn(
                    self.capture, state, self.context, steps, time_limit=self.remaining,
                )
            if self.reviewer.last is not None:
                report = dict(self.reviewer.last)
                report.pop("checkpoints", None)
                self._remember(report)
                self.client._publish_review("complete", self.reviewer.last, announce=False)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self._unavailable(f"Completion review unavailable ({type(exc).__name__}); task continues.", "completion")

    def finalize(self):
        """A strategy pass must never masquerade as final artifact verification."""
        if not self.finished and not self.disabled and self.capture is not None and self._current():
            report = self.reviewer.incomplete(
                "The actor stopped before final artifact review; completed work is retained."
            )
            self._remember(report)
            self.client._publish_review("complete", self.reviewer.last)
