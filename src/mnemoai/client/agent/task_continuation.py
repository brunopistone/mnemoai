"""Per-request instructions and local progress checks for long-running work."""

import hashlib
import json
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mnemoai.utils.request_intent import obvious_conversation
from mnemoai.utils.tool_results import is_error_result, is_unchanged_result


class IncompleteResult(str):
    """A worker report that must not be promoted to a successful completion."""

    incomplete = True


def compact_worker(compactor, messages, *, force, cancelled, timeout):
    """Allow a stopped worker to abandon a summary without mutating any history."""
    if compactor is None:
        return None
    result = queue.Queue()
    abandoned = threading.Event()

    def stopped():
        return abandoned.is_set() or cancelled()

    def run():
        try:
            if stopped():
                return
            result.put((compactor(list(messages), force=force, cancelled=stopped), None))
        except BaseException as error:
            result.put((None, error))
    threading.Thread(target=run, daemon=True).start()
    deadline = time.monotonic() + timeout
    try:
        while True:
            if cancelled():
                raise KeyboardInterrupt("worker stopped during compaction")
            if time.monotonic() >= deadline:
                return None
            try:
                value, error = result.get(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if error is not None:
                raise error
            if cancelled():
                raise KeyboardInterrupt("worker stopped during compaction")
            return value
    finally:
        abandoned.set()


@dataclass
class TaskContext:
    """Keep the request verbatim while conversation summaries change around it."""

    request: str
    guidance: tuple[str, ...] = ()
    revision: int = 0
    had_tools: bool = False
    needs_follow_through: bool = False
    context: str = ""
    explicit_request: bool = False

    def add_guidance(self, texts):
        self.guidance += tuple(texts)
        self.revision += 1
        self.needs_follow_through = self.had_tools

    def tool_round(self):
        self.had_tools = True
        self.needs_follow_through = False

    def follow_through(self):
        self.needs_follow_through = False
        # A new input event is required after an assistant status reply by
        # providers that forbid assistant prefilling. This is control traffic,
        # never a new user goal and never stored as a user message.
        return HumanMessage(content=(
            "<task-continuation>\n"
            "You have addressed a message sent during ongoing work. Now finish the "
            "active task under the latest user instructions; a status answer alone "
            "does not complete it. Do not repeat the status answer or completed "
            "actions. If the user cancelled/replaced the task, honor that instead. "
            "If the work is already complete or genuinely blocked, conclude with "
            "that result; otherwise continue executing it.\n"
            "</task-continuation>"
        ), additional_kwargs={"mnemoai_control": True})

    def reminder(self, messages):
        if obvious_conversation(self.request):
            return ""
        humans = [
            m.content for m in messages if isinstance(m, HumanMessage)
            and isinstance(m.content, str) and not m.additional_kwargs.get("mnemoai_control")
        ]
        missing = []
        if not any(self.request in text for text in humans):
            missing.append("Original request:\n" + self.request)
        for index, text in enumerate(self.guidance, 1):
            if not any(text in content for content in humans):
                missing.append(f"User follow-up {index}:\n{text}")
        if not missing:
            return ""
        return (
            "<active-task>\n"
            "Continue the active user task through completion. Compaction and execution "
            "checkpoints do not complete or replace it. Treat follow-ups as guidance for "
            "this task unless the user explicitly cancels or replaces it. Answer status "
            "questions briefly, then continue the authorized work. Respect newer "
            "constraints, cancellations and permission gates. Use retained results; do "
            "not repeat completed actions. Stop when the requested work is complete or "
            "you cannot make progress, explaining any unresolved blocker.\n"
            + "\n\n".join(missing)
            + "\n</active-task>"
        )

    def apply(self, messages):
        reminder = self.reminder(messages)
        additions = [text for text in (self.context, reminder) if text]
        if not additions:
            return messages
        result = list(messages)
        index = next((i for i, m in enumerate(result) if isinstance(m, SystemMessage)), None)
        content = result[index].content if index is not None else ""
        existing = content if isinstance(content, str) else "\n".join(
            block if isinstance(block, str) else str(block.get("text", "")) for block in content
        )
        additions = [text for text in additions if text not in existing]
        if not additions:
            return messages
        extra = "\n\n".join(additions)
        if index is None:
            result.insert(0, SystemMessage(content=extra))
        else:
            updated = content + "\n\n" + extra if isinstance(content, str) else [
                *content, {"type": "text", "text": extra},
            ]
            result[index] = result[index].model_copy(update={"content": updated})
        return result


@dataclass
class ProgressLedger:
    """Detect unchanged tool activity without a model call or an unbounded log.

    New successful call/result pairs permit another execution segment. This is
    a repetition guard, not a claim that a tool's success proves task completion.
    Failed/refused calls and prose alone never renew the execution budget.
    """

    seen: OrderedDict = field(default_factory=OrderedDict)
    advanced: bool = False

    def observe(self, messages):
        calls = {}
        for message in messages:
            if isinstance(message, AIMessage):
                calls.update({c["id"]: c for c in message.tool_calls})
            elif isinstance(message, ToolMessage):
                call = calls.get(message.tool_call_id)
                if call is None:
                    continue
                outcome = message.additional_kwargs.get("execution_outcome", "completed")
                if (outcome != "completed" or message.status == "error"
                        or is_error_result(message.content) or is_unchanged_result(message.content)):
                    continue
                name, args = call["name"], call["args"]
                # Fresh job IDs alone must not make repeated spawning look new.
                result = None if name in {"spawn_agent", "resume_agent", "start_background_task"} else message.content
                if isinstance(result, str):
                    try:
                        decoded = json.loads(result)
                        if isinstance(decoded, dict):
                            result = {k: v for k, v in decoded.items() if k != "progress_notice"}
                    except (TypeError, ValueError):
                        pass
                encoded = json.dumps([name, args, result], sort_keys=True, default=str).encode()
                key = hashlib.sha256(encoded).digest()
                if key not in self.seen:
                    self.advanced = True
                self.seen[key] = None
                self.seen.move_to_end(key)
                if len(self.seen) > 4096:
                    self.seen.popitem(last=False)

    def take_progress(self):
        advanced, self.advanced = self.advanced, False
        return advanced


def task_messages(agent, messages):
    task = getattr(agent, "_task_context", None)
    return task.apply(messages) if isinstance(task, TaskContext) else messages
