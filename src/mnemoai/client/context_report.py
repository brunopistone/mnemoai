"""Where the context window actually goes (``/context``).

``/usage`` answers "what has this session spent"; this answers the different
question "what is my next turn paying for, and which part of it can I shrink".
The two big surprises it exists to expose are the ones nothing else shows: a
large steering file (re-sent verbatim every turn, and never reclaimable by
compaction) and the tool schemas (bound on every call, before a word of the
conversation).

**Exact total, estimated split.** The size of an already-sent prompt is known
precisely — the provider reports it (the same number shown as ``[Context: N]``),
so that is the total. A per-part breakdown has no such ground truth, so each part
is measured with the pre-flight estimator and then scaled onto the exact total
(:func:`scale`). The estimator applies one uniform per-provider multiplier, so the
SHARES are meaningful even where the raw numbers over-count; without the scaling
the parts would add up to roughly twice the total the user was just shown, which
reads as a bug. Before any turn has run there is nothing to scale to and the
report says the numbers are estimates.

The system prompt is broken up by SEGMENTING THE LIVE STRING
(:func:`split_system_prompt`) rather than re-deriving each block, so the report
can't drift from what the model is actually being sent — MEMORY.md edited
mid-session, a compaction rebuild that re-injects everything plus a summary, a
playbook that has since grown.

``collect``/``report`` take the ``LangGraphClient`` (the ``context_injection``
collaborator pattern); ``split_system_prompt``/``scale``/``render`` are pure, so
the whole report is unit-testable without a terminal, a model, or a config file.
"""

import json
import os
from typing import Any, List, NamedTuple, Tuple

from mnemoai.client import context_injection
from mnemoai.client.memory.steering_store import SteeringStore
from mnemoai.utils.config import config
from mnemoai.utils.logger import logger
from mnemoai.utils.tokenization import count_tokens


class Part(NamedTuple):
    """One line of the report: a labelled slice of the context window.

    ``detail`` holds indented sub-rows (label, tokens); ``group`` only affects
    layout (a blank line separates groups).
    """

    label: str
    tokens: int
    detail: Tuple[Tuple[str, int], ...] = ()
    group: str = ""


# Markers that open each block appended to the base system prompt, in the order
# they are injected (see context_injection.build_session_blocks and the
# compaction rebuild). Searched in this order with a moving cursor, so a marker
# that also appears in earlier text can't re-attribute it.
_SYSTEM_SEGMENTS = (
    ("<profile>", "Learned profile"),
    ("[Persistent Memory]", "Persistent memory (MEMORY.md)"),
    ("<available_skills>", "Skills listing"),
    ("<available_subagents>", "Sub-agent types"),
    ("[Playbook - Learned Strategies]", "Learned strategies"),
    ("<conversation_summary>", "Compaction summary"),
)

_BASE_LABEL = "System prompt"


def _find_block(text: str, marker: str, start: int) -> int:
    """Index of ``marker`` where it OPENS a block, or -1.

    The blocks are joined with a blank line, so only a match at the very start or
    right after ``\\n\\n`` opens one. Without that, the same words occurring inside
    a user's MEMORY.md or steering prose would split the prompt there and label
    everything after it wrongly.
    """
    idx = text.find(marker, start)
    while idx > 0 and text[idx - 2:idx] != "\n\n":
        idx = text.find(marker, idx + len(marker))
    return idx


def split_system_prompt(prompt: str) -> List[Tuple[str, str]]:
    """The system prompt cut into ``(label, text)`` blocks, in prompt order.

    Everything ahead of the first recognized block is the base prompt; each block
    runs to the start of the next one. An absent block is simply omitted, so this
    works the same on a fresh prompt and on one rebuilt by compaction.
    """
    text = prompt or ""
    if not text.strip():
        return []

    found: List[Tuple[int, str]] = []
    cursor = 0
    for marker, label in _SYSTEM_SEGMENTS:
        idx = _find_block(text, marker, cursor)
        if idx < 0:
            continue
        found.append((idx, label))
        cursor = idx + len(marker)

    out: List[Tuple[str, str]] = []
    base = text[: found[0][0] if found else len(text)].strip()
    if base:
        out.append((_BASE_LABEL, base))
    for i, (idx, label) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(text)
        block = text[idx:end].strip()
        if block:
            out.append((label, block))
    return out


def _message_kind(msg: Any) -> str:
    """Role of a history message: user / assistant / tool / other.

    Matches on the whole MRO, not the class name: a streamed reply is an
    ``AIMessageChunk`` (an ``AIMessage`` subclass) and an exact-name check would
    file the bulk of a live conversation under "other".
    """
    names = {c.__name__ for c in type(msg).__mro__}
    if "ToolMessage" in names:
        return "tool"
    if "HumanMessage" in names:
        return "user"
    if "AIMessage" in names:
        return "assistant"
    return "other"


def _message_text(msg: Any) -> str:
    """Everything a stored message re-sends: content, tool calls, reasoning."""
    pieces: List[str] = []
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        pieces.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                pieces.append(str(block.get("text", "")) or json.dumps(block, default=str))
            else:
                pieces.append(str(block))
    elif content:
        pieces.append(str(content))
    for call in getattr(msg, "tool_calls", None) or []:
        pieces.append(json.dumps(call, default=str))
    reasoning = (getattr(msg, "additional_kwargs", {}) or {}).get("reasoning_content")
    if reasoning:
        pieces.append(str(reasoning))
    return "\n".join(p for p in pieces if p)


def tool_schema_text(tool: Any) -> str:
    """The text a bound tool contributes to every request: name, description,
    argument schema. Duck-typed and tolerant — an odd schema costs its own row,
    not the report."""
    name = str(getattr(tool, "name", "") or "")
    description = str(getattr(tool, "description", "") or "")
    schema = getattr(tool, "args_schema", None)
    rendered = ""
    if isinstance(schema, dict):
        rendered = json.dumps(schema, default=str)
    elif schema is not None:
        for attr in ("model_json_schema", "schema"):  # pydantic v2, then v1
            fn = getattr(schema, attr, None)
            if callable(fn):
                try:
                    rendered = json.dumps(fn(), default=str)
                    break
                except Exception:  # noqa: BLE001 — a schema is never worth raising for
                    continue
        else:
            rendered = str(schema)
    return "\n".join(p for p in (name, description, rendered) if p)


def _short(path: str) -> str:
    """Home-relative path for display (``~/…``)."""
    text = str(path or "")
    home = os.path.expanduser("~")
    return "~" + text[len(home):] if home and text.startswith(home) else text


def _steering_part(client: Any) -> List[Part]:
    """The always-on steering block, with a row per instruction file.

    Per-file rows are the actionable half of this report: they name the file to
    trim. They cover the DISCOVERED files only, so ``@path`` includes show up in
    the total without a row of their own.
    """
    try:
        injected = context_injection.steering_reminder(client)
    except Exception as e:  # noqa: BLE001 — a report never breaks on a file read
        logger.debug(f"/context: steering unavailable: {e}")
        return []
    if not injected:
        return []
    try:
        sizes = SteeringStore().sizes()
    except Exception as e:  # noqa: BLE001
        logger.debug(f"/context: steering file sizes unavailable: {e}")
        sizes = []
    files = len(sizes)
    label = f"Steering: {files} file{'s' if files != 1 else ''}" if files else "Steering"
    detail = tuple((_short(str(path)), count_tokens(text)) for path, text in sizes)
    return [Part(label, count_tokens(injected), detail=detail, group="steering")]


def _tools_part(client: Any) -> List[Part]:
    """The bound tool definitions, counted as one row."""
    tools = list(getattr(client, "tools", None) or [])
    if not tools:
        return []
    total = sum(count_tokens(tool_schema_text(t)) for t in tools)
    return [
        Part(f"Tool schemas: {len(tools)} tools", total, group="tools"),
    ]


def _history_part(client: Any) -> List[Part]:
    """The conversation itself, split by who produced the tokens."""
    agent = getattr(client, "agent", None)
    messages = list(getattr(agent, "messages", None) or []) if agent else []
    if not messages:
        return []
    by_kind = {"tool": 0, "assistant": 0, "user": 0, "other": 0}
    for msg in messages:
        by_kind[_message_kind(msg)] += count_tokens(_message_text(msg))
    detail = [
        ("tool results", by_kind["tool"]),
        ("assistant replies", by_kind["assistant"]),
        ("your prompts", by_kind["user"]),
    ]
    if by_kind["other"]:
        detail.append(("other", by_kind["other"]))
    return [
        Part(
            f"Conversation: {len(messages)} messages",
            sum(by_kind.values()),
            detail=tuple((label, n) for label, n in detail if n),
            group="history",
        )
    ]


def collect(client: Any) -> List[Part]:
    """Every measurable slice of the next turn's prompt, in send order.

    Reads the AGENT's system prompt when there is one — it is the string actually
    sent, and after a compaction it differs from ``client.system_prompt``.
    """
    agent = getattr(client, "agent", None)
    system = (getattr(agent, "system_prompt", "") if agent else "") or getattr(
        client, "system_prompt", ""
    )
    parts = [
        Part(label, count_tokens(text), group="prompt")
        for label, text in split_system_prompt(system)
    ]
    parts.extend(_steering_part(client))
    parts.extend(_tools_part(client))
    parts.extend(_history_part(client))
    return parts


def scale(parts: List[Part], target: int) -> List[Part]:
    """Re-proportion estimated parts so they add up to a known exact total.

    The estimator over-counts by one uniform per-provider factor, so scaling by
    ``target / estimate`` keeps every share intact while making the rows agree
    with the total the provider reported. A missing or empty total leaves the
    parts untouched (the caller then labels them as estimates).
    """
    estimate = sum(p.tokens for p in parts)
    if target <= 0 or estimate <= 0:
        return list(parts)
    factor = target / estimate
    return [
        p._replace(
            tokens=int(round(p.tokens * factor)),
            detail=tuple((label, int(round(n * factor))) for label, n in p.detail),
        )
        for p in parts
    ]


def _fmt(n: int) -> str:
    """Thousands-separated count."""
    return f"{n:,}"


# Widest label column before clipping, so one long file path can't push the
# report past a standard terminal width.
_MAX_LABEL = 46


def _clip(label: str, width: int) -> str:
    """``label`` fitted to ``width``, keeping its TAIL.

    Paths are the only labels that get long here, and their end (the filename) is
    the part that identifies them.
    """
    return label if len(label) <= width else "…" + label[-(width - 1):]


def _bar(used: int, limit: int, width: int = 24) -> str:
    """``[███░░░]`` gauge of the window in use (empty string without a limit)."""
    if limit <= 0:
        return ""
    filled = max(0, min(width, int(round(width * used / limit))))
    # A non-empty context always shows at least one block, so "tiny" never looks
    # like "nothing".
    if used > 0 and filled == 0:
        filled = 1
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def render(
    parts: List[Part],
    total: int,
    limit: int = 0,
    high_water: int = 0,
    measured: bool = False,
) -> str:
    """Render the ``/context`` report as plain text (no ANSI).

    Kept a pure function of already-counted numbers so it can be tested without a
    client, a model, or a terminal.
    """
    if not parts:
        return (
            "Nothing is loaded into the context window yet.\n"
            "  Start a conversation and /context breaks down what it costs."
        )

    width = max(len(p.label) for p in parts)
    width = max(width, max((len(d[0]) + 2 for p in parts for d in p.detail), default=0))
    width = min(_MAX_LABEL, width)

    head = f"Context window — {_fmt(total)} token{'s' if total != 1 else ''}"
    if limit > 0:
        head += f" of {_fmt(limit)} ({round(100 * total / limit)}%)"
    out = [head]
    bar = _bar(total, limit)
    if bar:
        out.append(f"  {bar}")
    out.append("")

    group = parts[0].group
    for part in parts:
        if part.group != group:
            out.append("")
            group = part.group
        share = f"{round(100 * part.tokens / total)}%" if total > 0 else ""
        label = _clip(part.label, width)
        out.append(
            f"  {label.ljust(width)}  {_fmt(part.tokens).rjust(9)}  {share.rjust(4)}"
        )
        for sub, tokens in part.detail:
            sub = _clip(sub, width - 2)
            out.append(f"    {sub.ljust(width - 2)}  {_fmt(tokens).rjust(9)}")

    if limit > 0:
        free = max(0, limit - total)
        line = f"  Free: {_fmt(free)} tokens"
        if 0 < high_water < limit:
            line += f"  ·  compaction starts at {_fmt(high_water)}"
        out.append("")
        out.append(line)

    out.append("")
    if measured:
        out.append("  The total is the provider's exact count for the last turn;")
        out.append("  the split is estimated and scaled to it.")
    else:
        out.append("  Nothing has been sent yet, so every number here is an estimate")
        out.append("  — the provider's exact count replaces it after the first turn.")
    out.append("  Steering files and tool schemas are re-sent in full every turn and")
    out.append("  compaction can never reclaim them; the conversation is what")
    out.append("  /compact shrinks.")
    return "\n".join(out)


def report(client: Any) -> str:
    """The whole ``/context`` report for a live client."""
    parts = collect(client)
    agent = getattr(client, "agent", None)
    exact = int(getattr(agent, "_last_input_tokens", 0) or 0) if agent else 0
    total = exact or sum(p.tokens for p in parts)
    manager = getattr(client, "conversation_manager", None)
    limit = int(getattr(manager, "max_tokens", 0) or 0)
    high_water = 0
    if limit > 0:
        try:
            high_water = int(
                config.get("LLM", {}).get(
                    "COMPACT_HIGH_WATER_TOKENS", int(limit * 0.8)
                )
            )
        except (AttributeError, TypeError, ValueError):
            high_water = int(limit * 0.8)
    return render(
        scale(parts, exact),
        total,
        limit=limit,
        high_water=high_water,
        measured=bool(exact),
    )
