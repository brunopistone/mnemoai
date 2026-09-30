"""A bounded supervisor dialogue; only the chat model can execute corrections."""

import copy
import json
import math
import queue
import threading
import time
import uuid

import tiktoken
from langchain_core.messages import HumanMessage, SystemMessage

from mnemoai.client.agent.reasoning_utils import extract_visible_text
from mnemoai.client.memory.reflection import redact
from mnemoai.client.review_evidence import Capture, digest, text
from mnemoai.utils.config import config
from mnemoai.utils.review_protocol import DATA_MARKER, FEEDBACK_PREFIX, ReviewStopped

SETTINGS = {
    "TIMEOUT": ("Review wait (seconds, 1–120)", 45, "float", 1, 120),
    "MAX_INPUT_TOKENS": ("Estimated review input tokens (1000–32000)", 6000, "int", 1000, 32000),
    "MAX_ROUNDS": ("Chat-model correction rounds (0–4)", 2, "int", 0, 4),
    "TOTAL_TIMEOUT": ("Shared supervision budget (seconds, 1–1800)", 180, "float", 1, 1800),
}
DEFAULT_OUTPUT_TOKENS = 2048


class LoopBudget:
    """One deadline and remaining actor steps, never reset by a revision or retry."""

    def __init__(self, seconds, steps, cancel=None, current=None):
        self.deadline = time.monotonic() + seconds
        self.steps = steps
        self.cancel = cancel
        self.current = current
        self.closed = False

    def remaining(self):
        return max(0, self.deadline - time.monotonic())

    def check(self):
        if self.cancel and self.cancel():
            raise KeyboardInterrupt("Supervisor exchange cancelled")
        if self.closed or self.remaining() <= 0:
            raise ReviewStopped("Shared supervision time budget exhausted")
        if self.current is not None and not self.current():
            raise ReviewStopped("Review mode or task requirements changed")

    def take_step(self):
        self.check()
        if self.steps <= 0:
            raise ReviewStopped("The original agent step budget is exhausted")
        self.steps -= 1


def settings():
    raw = config.get("REVIEW", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("REVIEW must be a mapping")
    result = {}
    for key, (_, default, kind, minimum, maximum) in SETTINGS.items():
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"REVIEW.{key} must be numeric")
        if not math.isfinite(value) or not minimum <= value <= maximum or (kind == "int" and int(value) != value):
            raise ValueError(f"REVIEW.{key} is outside its allowed range")
        result[key] = value
    return result


def parse_verdict(response, ids):
    if getattr(response, "tool_calls", None):
        raise ValueError("Reviewer returned tool calls; no tools are available")
    raw = extract_visible_text(getattr(response, "content", ""))
    if len(raw) > 16_000:
        raise ValueError("Reviewer output exceeded its limit")
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw.partition("\n")[2].rsplit("```", 1)[0]
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != {"verdict", "summary", "findings"}:
        raise ValueError("Invalid review response shape")
    if data["verdict"] not in ("pass", "revise", "inconclusive"):
        raise ValueError("Invalid review verdict")

    def prose(value, limit):
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError("Invalid review text")
        if any(not c.isprintable() and not c.isspace() for c in value) or redact(value) != value:
            raise ValueError("Unsafe review text")
        return " ".join(value.split())

    data["summary"] = prose(data["summary"], 1000)
    if not isinstance(data["findings"], list) or len(data["findings"]) > 6:
        raise ValueError("Invalid findings list")
    for finding in data["findings"]:
        if not isinstance(finding, dict) or set(finding) != {"issue", "evidence_ids", "verification"}:
            raise ValueError("Invalid finding shape")
        finding["issue"] = prose(finding["issue"], 700)
        finding["verification"] = prose(finding["verification"], 700)
        refs = finding["evidence_ids"]
        if not isinstance(refs, list) or not refs or any(not isinstance(i, str) or i not in ids for i in refs):
            raise ValueError("Finding cites evidence the reviewer did not receive")
    if (data["verdict"] == "pass" and data["findings"]) or (data["verdict"] == "revise" and not data["findings"]):
        raise ValueError("Review verdict contradicts its findings")
    return data


def packet(items, gaps, limit):
    prompt = config.prompt("REVIEWER_SYSTEM_PROMPT")
    encoder = tiktoken.get_encoding("o200k_base")
    kept = []
    for item in items:
        cleaned = copy.deepcopy(item)
        for field in ("text", "content"):
            if field in cleaned:
                cleaned[field], clipped = text(cleaned[field], 6000)
                if clipped:
                    gaps.add(f"Evidence {item['id']} was truncated.")
        kept.append(cleaned)

    def build():
        payload = json.dumps({"evidence": kept, "coverage_gaps": sorted(gaps)}, ensure_ascii=True)
        estimate = math.ceil(len(encoder.encode(prompt + payload, disallowed_special=())) * 1.5) + 256
        return payload, estimate

    payload, estimated = build()
    while estimated > limit and len(kept) > 2:
        kept.pop()
        gaps.add("Some evidence was omitted to fit the input budget.")
        payload, estimated = build()
    if estimated > limit:
        raise ValueError("Task, answer and review instructions exceed the input budget")
    return [SystemMessage(content=prompt), HumanMessage(content=payload)], kept, estimated


class Reviewer:
    """Session-local coordinator; a timed-out call cannot launch another or publish late."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.last = None
        self._capture = None
        self._in_flight = threading.Lock()
        self._generation = 0
        self._last_evidence = []

    def reset(self):
        self._generation += 1
        self.last = None
        self._capture = None
        self._last_evidence = []

    def begin(self, task):
        self.reset()
        return Capture(task)

    def incomplete(self, reason, preserve=False):
        if preserve and self.last is not None:
            self.last = {**self.last, "verdict": "inconclusive", "summary": reason}
            return self.last
        self.last = {
            "id": "review-" + uuid.uuid4().hex, "checkpoint": "completion",
            "reviewer": "not invoked", "verdict": "inconclusive", "summary": reason,
            "findings": [], "coverage_gaps": [], "evidence": [],
            "input_tokens_estimate": 0, "elapsed_seconds": 0,
        }
        self._capture = None
        return self.last

    def finish(self, capture, answer, *, model_factory, model_label, context=(), cancel=None, usage=None,
               time_limit=None):
        """Returns an advisory report, never changes the actor's answer or artifacts."""
        generation = self._generation
        started = time.monotonic()
        report = {
            "id": "review-" + uuid.uuid4().hex, "checkpoint": "completion",
            "reviewer": model_label, "verdict": "inconclusive", "summary": "",
            "findings": [], "coverage_gaps": [], "evidence": [], "input_tokens_estimate": 0,
            "valid_verdict": False, "stale": False,
        }
        try:
            options = settings()
            if not self.enabled or (cancel and cancel()):
                raise InterruptedError("Review cancelled; the actor's answer is preserved")
            if not self._in_flight.acquire(blocking=False):
                raise RuntimeError("A prior reviewer call is still running; no new call started")
            inbox = queue.Queue(maxsize=1)
            accounted = threading.Event()
            accounting_lock = threading.Lock()

            def account(response):
                with accounting_lock:
                    if not accounted.is_set():
                        accounted.set()
                        if usage is not None:
                            try:
                                usage(response)
                            except Exception:
                                pass  # accounting cannot alter the verdict

            abandoned = threading.Event()
            sent = threading.Event()

            def invoke():
                try:
                    items = capture.finish(answer, context() if callable(context) else context)
                    messages, kept, estimate = packet(items, capture.gaps, options["MAX_INPUT_TOKENS"])
                    if abandoned.is_set() or (cancel and cancel()) or generation != self._generation:
                        return
                    model = model_factory()
                    if abandoned.is_set() or (cancel and cancel()) or generation != self._generation:
                        return
                    if model is None:
                        raise RuntimeError("Reviewer model unavailable; no fallback was substituted")
                    sent.set()
                    response = model.invoke(messages, config={"callbacks": []})
                    account(response)
                    verdict = parse_verdict(response, {i["id"] for i in kept})
                    stale = capture.stale()
                    if stale:
                        verdict.update(verdict="inconclusive", summary="Artifacts changed during review; findings are historical.")
                    elif verdict["verdict"] == "pass" and capture.gaps:
                        verdict.update(verdict="inconclusive", summary="No blocking finding in the supplied evidence, but coverage is incomplete.")
                    verdict.update(
                        valid_verdict=True, stale=stale,
                        coverage_gaps=sorted(capture.gaps), input_tokens_estimate=estimate,
                        evidence=[{"id": i["id"], "kind": i["kind"], "revision": i.get("revision", digest(i))}
                                  for i in kept],
                    )
                    inbox.put(((verdict, kept), None))
                except BaseException as exc:
                    if sent.is_set():
                        account(None)
                    inbox.put((None, exc))
                finally:
                    self._in_flight.release()

            try:
                threading.Thread(target=invoke, daemon=True, name="mnemoai-reviewer").start()
            except BaseException:
                self._in_flight.release()
                raise
            try:
                timeout = options["TIMEOUT"] if time_limit is None else min(options["TIMEOUT"], time_limit)
                deadline = started + timeout
                while True:
                    if (cancel and cancel()) or generation != self._generation or not self.enabled:
                        raise InterruptedError("Review cancelled; the actor's answer is preserved")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Review timed out; no verdict assumed")
                    try:
                        response, error = inbox.get(timeout=min(0.1, remaining))
                        break
                    except queue.Empty:
                        continue
                if error is not None:
                    raise error
                verdict, evidence = response
                report.update(verdict)
                if generation == self._generation:
                    self._last_evidence = evidence
            finally:
                abandoned.set()
                if sent.is_set():
                    account(None)  # timeout/cancel: a call happened, but usage is unknown
        except (Exception, KeyboardInterrupt) as exc:
            report["summary"] = {
                TimeoutError: "Review timed out; no verdict assumed.",
                InterruptedError: "Review cancelled; the actor's answer is preserved.",
                KeyboardInterrupt: "Review cancelled; the actor's answer is preserved.",
            }.get(type(exc), f"Review unavailable or invalid ({type(exc).__name__}); no verdict assumed.")
            report["coverage_gaps"] = ["Review did not complete; evidence coverage was not established."]
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if generation == self._generation:
            self.last, self._capture = report, capture
        return report

    def supervise(self, capture, answer, *, actor, steps, model_factory, model_label,
                  context=(), cancel=None, usage=None, progress=None):
        """Ask, respond, and recheck within one user task; return the latest actor answer."""
        generation = self._generation
        rounds, revisions = [], 0
        started = time.monotonic()
        budget = None
        latest = answer
        try:
            options = settings()
            budget = LoopBudget(
                options["TOTAL_TIMEOUT"], steps, cancel,
                current=lambda: self.enabled and generation == self._generation,
            )
            while True:
                budget.check()
                dialogue = [
                    {"id": f"exchange-{i + 1}", "kind": "prior_review_exchange",
                     "text": json.dumps({"reviewer": r["summary"], "findings": r["findings"],
                                        "actor_response": r.get("actor_response", "")}, ensure_ascii=True)}
                    for i, r in enumerate(rounds)
                ]
                report = self.finish(
                    capture, latest, model_factory=model_factory, model_label=model_label,
                    context=[*(context() if callable(context) else context), *dialogue], cancel=cancel, usage=usage,
                    time_limit=budget.remaining(),
                )
                rounds.append(copy.deepcopy(report))
                if progress is not None:
                    progress("review", report, revisions)
                budget.check()
                if report["verdict"] == "pass":
                    break
                if not report.get("valid_verdict") or report.get("stale") or not report["findings"]:
                    break
                if revisions >= options["MAX_ROUNDS"]:
                    report["summary"] += " Correction round limit reached; findings remain unresolved."
                    break
                if capture.stale():
                    self.incomplete("Artifacts changed before the correction; feedback was not applied.", preserve=True)
                    self.last["stale"] = True
                    break
                ids = {i for finding in report["findings"] for i in finding["evidence_ids"]}
                data = {
                    "round": revisions + 1, "reviewer": model_label,
                    "verdict": report["verdict"], "summary": report["summary"],
                    "findings": report["findings"],
                    "evidence": [i for i in self._last_evidence if i["id"] in ids],
                }
                feedback = HumanMessage(
                    content=FEEDBACK_PREFIX + "\n" + config.prompt("REVIEW_ACTOR_PROMPT")
                    + DATA_MARKER + json.dumps(data, ensure_ascii=True),
                    name="reviewer",
                )
                capture.reopen()
                revisions += 1
                if progress is not None:
                    progress("actor", report, revisions)
                updated = actor(feedback, budget)
                if not isinstance(updated, str) or not updated.strip():
                    raise ReviewStopped("The chat model returned no correction or counterevidence")
                latest = updated
                rounds[-1]["actor_response"], clipped = text(latest, 1600)
                rounds[-1]["actor_response_truncated"] = clipped
        except KeyboardInterrupt:
            self.incomplete("Supervisor exchange cancelled; completed work is retained.", preserve=True)
            raise
        except ReviewStopped as exc:
            self.incomplete(str(exc) + "; completed work is retained.", preserve=True)
        except Exception as exc:
            self.incomplete(f"Supervisor exchange stopped ({type(exc).__name__}); completed work is retained.",
                            preserve=True)
        finally:
            if budget is not None:
                budget.closed = True
            if self.last is not None and generation == self._generation:
                self.last = {**self.last, "rounds": rounds, "revisions": revisions,
                             "elapsed_seconds": round(time.monotonic() - started, 3)}
        return latest

    def current_report(self):
        report = copy.deepcopy(self.last)
        if report and report["verdict"] != "inconclusive" and self._capture is not None and self._capture.stale():
            report.update(verdict="inconclusive", summary="Artifacts changed since review; this report is historical.")
        return report


def render(report, history=True):
    def plain(value):
        return " ".join("".join(c for c in str(value) if c.isprintable() or c.isspace()).split())
    lines = [
        f"Peer review · {report['verdict']} · {plain(report['reviewer'])}",
        report["summary"],
    ]
    if history and report.get("rounds"):
        for i, item in enumerate(report["rounds"]):
            lines.append(f"  Round {i + 1} · reviewer: {item['verdict']} — {item['summary']}")
            for finding in item["findings"]:
                lines.append(f"    {finding['issue']} [{', '.join(finding['evidence_ids'])}]")
            if item.get("actor_response"):
                lines.append(f"    Chat model: {plain(item['actor_response'])}")
    for finding in report["findings"]:
        lines.extend([
            f"  • {finding['issue']} [{', '.join(finding['evidence_ids'])}]",
            f"    Suggested verification (not executed): {finding['verification']}",
        ])
    lines.extend("  Coverage: " + gap for gap in report["coverage_gaps"])
    lines.append(
        "Reviewer advice grants no permission; only the chat model executes changes. "
        "Not proof of correctness, permission, or completed verification."
    )
    return "\n".join(lines)


def command(client, arguments):
    """User-only controls; never dispatch a slash command through a model."""
    reviewer = getattr(client, "reviewer", None)
    if reviewer is None:
        return "Peer review is unavailable (client not initialized)."
    action = arguments.strip().lower()
    if action in {"on", "off"}:
        reviewer.reset()
        reviewer.enabled = action == "on"
        return (
            f"Peer review {action} for this session. "
            + ("The reviewer and chat model may exchange bounded correction rounds. " if reviewer.enabled else "")
            + "Permissions unchanged. /model → Reviewer; /config review; /features for the startup default."
        )
    if action not in {"", "last"}:
        return "Usage: /review [on|off|last]"
    report = reviewer.current_report()
    return (
        f"Peer review: {'on' if reviewer.enabled else 'off'} (supervised).\n"
        + (render(report) if report else "No completion review in this session yet.")
    )
