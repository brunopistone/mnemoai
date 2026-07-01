"""Pure helpers for rendering, capturing, normalizing, and error-mapping tool calls.

Extracted from ``agent.py``: none of these touch agent state. The agent keeps thin
delegating methods so its historical surface (and the unit tests that call
``LangGraphAgent._format_tool_call`` etc.) is unchanged.
"""

import re
from typing import Any

# Matches a key that is really a ``field="value"`` (or field='value', or bare
# field=value) expression — a common small-model malformation where the model
# emits Python-call syntax as a single JSON arg key.
_ARG_KEY_EXPR = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*$", re.DOTALL)

# Matches pydantic's "Field required" line, capturing the missing field name (the
# line above the "Field required" message is the field path).
_MISSING_FIELD_RE = re.compile(r"^\s*(\w[\w.]*)\s*\n\s*Field required", re.MULTILINE)


def elide_middle(text: str, limit: int = 72) -> str:
    """Shorten ``text`` to ``limit`` chars keeping BOTH ends.

    A long value (e.g. a shell command or a path) is most informative at its head
    AND tail — the program/path at the front and the meaningful subcommand/flags
    at the end. Middle elision (``head…tail``) preserves both, unlike a plain head
    cut that hides the tail.
    """
    if len(text) <= limit:
        return text
    keep = limit - 1  # room for the ellipsis
    head = (keep + 1) // 2
    tail = keep // 2
    return f"{text[:head]}…{text[-tail:]}"


def format_tool_call(tool_call: dict) -> str:
    """Compact one-line ``name(arg=value, …)`` rendering of a tool call.

    Argument values are stringified and middle-elided so a large payload (e.g.
    file content for a write, or a long command) doesn't flood the marker line
    while keeping both informative ends visible.
    """
    name = tool_call.get("name", "tool")
    args = tool_call.get("args") or {}
    parts = []
    for key, value in args.items():
        text = str(value).replace("\n", " ")
        parts.append(f"{key}={elide_middle(text)}")
    return f"{name}({', '.join(parts)})"


def record_turn_tool_calls(tool_calls) -> list:
    """Capture tool calls for the ctrl+o "expand last turn" view.

    Returns ``[{"name", "args"}]`` with the **raw, un-elided** args (unlike the
    marker line, which middle-elides for compactness). The result is stashed so
    the input reader can reprint the full call the user only saw truncated.
    Tolerant of odd shapes — a non-dict entry is skipped.
    """
    out = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        out.append({"name": tc.get("name", "tool"), "args": tc.get("args") or {}})
    return out


def normalize_tool_args(args: Any) -> Any:
    """Repair a common malformed tool-args shape from smaller models.

    Models sometimes emit ``{'query="USPTO fees"': ''}`` instead of
    ``{'query': 'USPTO fees'}`` — i.e. they pack a ``field=value`` expression into
    a single dict KEY with an empty value. Detect that exact shape and rebuild it
    into the intended ``{field: value}``, stripping surrounding quotes from the
    value. Anything that doesn't match is returned unchanged, so well-formed args
    are never touched.
    """
    if not isinstance(args, dict) or len(args) != 1:
        return args
    ((key, value),) = args.items()
    # Only attempt a repair when the value is empty (the tell-tale sign); a real
    # single-arg call has its value populated, not in the key.
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

    A pydantic arg-validation failure (raised when the model calls a tool with a
    required argument missing — e.g. ``file_edit`` without ``new_string``) is
    otherwise reported as opaque pydantic text the model tends to retry verbatim.
    Translate the common "Field required" case into a plain instruction so the
    model corrects the CALL rather than looping.
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
