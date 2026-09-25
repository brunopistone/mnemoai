"""One bounded model extraction from observed tool evidence, without tool access."""

import hashlib
import json
import queue
import re
import threading
import time

from langchain_core.messages import HumanMessage, SystemMessage

from mnemoai.client.agent.reasoning_utils import extract_visible_text
from mnemoai.client.memory.playbook_records import PlaybookEntry
from mnemoai.utils.config import config

_SENSITIVE = re.compile(r"api.?key|password|secret|token|authorization", re.I)
_TOKEN = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|pypi-[A-Za-z0-9_-]{16,}|AKIA[A-Z0-9]{16})\b")


def redact(value):
    if isinstance(value, dict):
        return {str(k): "[redacted]" if _SENSITIVE.search(str(k)) else redact(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = _TOKEN.sub("[redacted]", value)
        return re.sub(
            r"""(?i)(["']?(?:password|secret|api[_-]?key|authorization|access_token)"""
            r"""["']?\s*[:=]\s*)(?:"[^"]*"|'[^']*'|(?:Bearer|Basic)\s+[^\s,;]+|[^\s,;]+)""",
            r'\1"[redacted]"', value,
        )
    return value


def evidence_item(name, args, result, call_id, outcome, source):
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    try:
        preview = json.dumps(redact(json.loads(text)), ensure_ascii=True)
    except (ValueError, TypeError):
        preview = redact(text)
    return {
        "tool": name, "args": json.dumps(redact(args), ensure_ascii=True)[:600],
        "result": preview[:1200], "truncated": len(preview) > 1200,
        "outcome": outcome,
        "ref": {
            **source, "tool_call_id": call_id, "tool": name, "outcome": outcome,
            "result_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "args_sha256": hashlib.sha256(
                json.dumps(args, sort_keys=True, default=str).encode()
            ).hexdigest(),
        },
    }


def extract(evidence, task, model, *, scope, timeout=30, cancel=None, record_usage=None,
            in_flight=None):
    """Errors/timeouts create no entries; late responses never write to the store."""
    evidence = evidence[-12:]
    payload = {
        "task": redact(task)[:1500],
        "evidence": [{"id": i, **{k: v for k, v in item.items() if k != "ref"}}
                     for i, item in enumerate(evidence)],
    }
    prompt = config.prompt("REFLECTOR_SYSTEM_PROMPT")
    if not prompt:
        raise ValueError("REFLECTOR_SYSTEM_PROMPT is missing")
    if cancel is not None and cancel():
        raise InterruptedError("Reflection cancelled")
    inbox = queue.Queue(maxsize=1)

    def invoke():
        try:
            response = model.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=json.dumps(payload, ensure_ascii=True)),
            ], config={"callbacks": []})
            if record_usage is not None:
                record_usage(response)
            inbox.put((response, None))
        except Exception as exc:
            inbox.put((None, exc))
        finally:
            if in_flight is not None:
                in_flight.clear()

    if in_flight is not None:
        in_flight.set()
    threading.Thread(target=invoke, daemon=True, name="mnemoai-reflector").start()
    deadline = time.monotonic() + timeout
    while True:
        if cancel is not None and cancel():
            raise InterruptedError("Reflection cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Reflection timed out; no lesson stored")
        try:
            response, error = inbox.get(timeout=min(0.1, remaining))
            break
        except queue.Empty:
            pass
    if error is not None:
        raise error
    text = extract_visible_text(getattr(response, "content", ""))
    if len(text) > 12_000:
        raise ValueError("Reflection response exceeds its size limit")
    if text.startswith("```") and text.endswith("```"):
        text = text.partition("\n")[2].rsplit("```", 1)[0]
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("lessons"), list):
        raise ValueError("Reflection must return a lessons list")
    if len(data["lessons"]) > 3:
        raise ValueError("Reflection returned too many lessons")
    entries = []
    for lesson in data["lessons"]:
        if not isinstance(lesson, dict):
            raise ValueError("Invalid lesson")
        context, strategy = lesson.get("context"), lesson.get("strategy")
        ids = lesson.get("evidence_ids")
        for value, limit in ((context, 200), (strategy, 800)):
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise ValueError("Invalid lesson text")
            if any(not c.isprintable() and not c.isspace() for c in value):
                raise ValueError("Invalid lesson control characters")
            if redact(value) != value:
                raise ValueError("Possible credential in lesson")
        if not isinstance(ids, list) or not ids or any(
            type(i) is not int or i < 0 or i >= len(evidence) for i in ids
        ):
            raise ValueError("Lesson references unobserved evidence")
        selected = [evidence[i] for i in sorted(set(ids))]
        entries.append(PlaybookEntry(
            context=" ".join(context.split()), strategy=" ".join(strategy.split()),
            source="Model inference from recorded tool evidence", confidence=0.5,
            outcome="failure" if any(e["outcome"] == "failure" for e in selected) else "success",
            tools=sorted({e["tool"] for e in selected}),
            source_refs=[e["ref"] for e in selected], scope=scope, provenance="model",
        ))
    return entries
