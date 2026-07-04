"""Pure helpers for rendering, normalizing, and error-mapping tool calls.

Extracted from ``agent.py`` (no agent state); the agent keeps thin delegating
methods so its historical surface — and the unit tests — stay unchanged.
"""

import re
from typing import Any

# A key that is really a ``field="value"`` expression — a small-model malformation
# where Python-call syntax is emitted as a single JSON arg key.
_ARG_KEY_EXPR = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*$", re.DOTALL)

# pydantic's "Field required" line; captures the missing field name above it.
_MISSING_FIELD_RE = re.compile(r"^\s*(\w[\w.]*)\s*\n\s*Field required", re.MULTILINE)


def elide_middle(text: str, limit: int = 72) -> str:
    """Shorten ``text`` to ``limit`` chars keeping both ends (``head…tail``).

    A long command/path is informative at both ends, so middle elision beats a
    plain head cut that hides the tail.
    """
    if len(text) <= limit:
        return text
    keep = limit - 1  # room for the ellipsis
    head = (keep + 1) // 2
    tail = keep // 2
    return f"{text[:head]}…{text[-tail:]}"


def truncate_tool_result(text: str, max_chars: int) -> str:
    """Cap a tool result to ``max_chars`` (0 disables), keeping head+tail with a
    middle note, so one runaway result can't overflow the context window.

    Both ends are kept — the head (counts, first matches) and tail (summary
    lines) of grep/read output are the useful parts; the note tells the model
    output was trimmed so it can narrow the call rather than assume it saw all.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    note = f"\n\n… [truncated {dropped} chars / ~{dropped // 4} tokens] …\n\n"
    keep = max_chars - len(note)
    if keep <= 0:
        return text[:max_chars]
    head = (keep + 1) // 2
    tail = keep // 2
    return f"{text[:head]}{note}{text[-tail:]}"


def format_tool_call(tool_call: dict) -> str:
    """Compact one-line ``name(arg=value, …)`` rendering; values middle-elided."""
    name = tool_call.get("name", "tool")
    args = tool_call.get("args") or {}
    parts = []
    for key, value in args.items():
        text = str(value).replace("\n", " ")
        parts.append(f"{key}={elide_middle(text)}")
    return f"{name}({', '.join(parts)})"


def normalize_tool_args(args: Any) -> Any:
    """Repair a malformed single-arg shape from smaller models.

    Rebuilds ``{'query="USPTO fees"': ''}`` (a ``field=value`` packed into the
    dict key with an empty value) into ``{'query': 'USPTO fees'}``. Well-formed
    args are returned unchanged.
    """
    if not isinstance(args, dict) or len(args) != 1:
        return args
    ((key, value),) = args.items()
    # Empty value is the tell-tale sign; a real single-arg call has it populated.
    if value not in ("", None):
        return args
    m = _ARG_KEY_EXPR.match(str(key))
    if not m:
        return args
    field, raw = m.group(1), m.group(2)
    if (raw.startswith('"') and raw.endswith('"')) or (
        raw.startswith("'") and raw.endswith("'")
    ):
        raw = raw[1:-1]
    return {field: raw}


def tool_error_message(tool_name: str, exc: Exception) -> str:
    """Turn a tool exception into model-facing guidance.

    Translates a pydantic "Field required" failure (opaque text the model tends
    to retry verbatim) into a plain instruction to supply the missing args.
    """
    text = str(exc)
    missing = _MISSING_FIELD_RE.findall(text)
    if missing:
        fields = ", ".join(sorted(set(missing)))
        return (
            f"Error: the call to `{tool_name}` is missing required "
            f"argument(s): {fields}. Provide every required argument and try "
            f"again — do not omit them. (For file_edit, `new_string` is "
            f'required; pass "" to delete text.)'
        )
    return f"Error: {text}"
