"""What an episode records about the tools it used — one shape, both directions.

An episode's ``tools`` metadata has exactly two readers: the recall line injected
into a prompt (``context_injection.inject_episodic_context``) and the BM25 corpus
(``ChromaEpisodicStore._get_searchable_text``). Both want the same thing — the
tool NAMES. It used to be stored as ``str(tools_used)``: the repr of the live
tool records, every argument and every full result included, re-escaped once per
nesting level. A single episode reached 1.1 MB of which most were backslashes,
kept twice on disk (metadata table + write-ahead log) and indexed a third time,
while every term of it diluted the BM25 scores nothing asked for.

So the writer stores what the readers read (:func:`format_tools`) and the reader
accepts BOTH shapes (:func:`parse_tools`), because episodes written by earlier
versions are still on disk. Names are de-duplicated in first-use order: a task
that called ``fs_read`` forty times said one thing, not forty.

Pure and config-free — the writer, the reader and the one-shot on-disk
compaction agree by sharing this module rather than by each parsing the value
its own way.
"""

import ast

# The wording for an episode that used no tools, in one place: it is stored in
# the embedded text AND rendered in the recall line, and the two must match.
NO_TOOLS = "no tools"

# Caps. A stored value is a short name list, so these only ever bite on absurd
# input; they exist so ONE oversized episode can't grow the store again.
MAX_TOOLS_CHARS = 400
MAX_TASK_CHARS = 2000
_MAX_NAMES = 12

# Above this a legacy repr isn't parsed at all: it is a value the compaction
# hasn't reached yet, and reading names out of it costs real time on every turn
# (the parse is a full Python-literal parse, per episode, per prompt).
_MAX_LEGACY_CHARS = 2_000_000


def tool_names(tools_used) -> list[str]:
    """Tool names from live tool records, de-duplicated in first-use order."""
    if not isinstance(tools_used, (list, tuple)):
        return []
    return _dedupe(r.get("name") for r in tools_used if isinstance(r, dict))


def format_tools(tools_used) -> str:
    """The value to STORE for a set of live tool records (may be "")."""
    return _join(tool_names(tools_used))


def parse_tools(value) -> list[str]:
    """Tool names out of a stored value, in either shape.

    New shape: the comma-separated name list :func:`format_tools` writes. Legacy
    shape: the repr of the live tool records, recognised by its leading ``[``/
    ``{`` — a tool name can never start with either, so one character tells the
    two apart without parsing megabytes to find out.
    """
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []
    if text[0] in "[{":
        return _dedupe(_legacy_names(text))
    return _dedupe(text.split(","))


def describe_tools(value) -> str:
    """A stored value rendered for display: the names, or ``no tools``."""
    return _join(parse_tools(value)) or NO_TOOLS


def compact_tools(value) -> str:
    """A stored value rewritten to its name list — idempotent, so re-running the
    on-disk compaction over an already-compacted store changes nothing."""
    return _join(parse_tools(value))


def clip_task(task) -> str:
    """The task text as stored: capped, since nothing reads past the first line
    of it (the recall line shows 70 chars) and an episode must stay small."""
    if not isinstance(task, str):
        return ""
    if len(task) <= MAX_TASK_CHARS:
        return task
    return task[:MAX_TASK_CHARS] + "…"


def _legacy_names(text: str):
    """Names out of a ``str(tools_used)`` repr; empty on anything unparseable."""
    if len(text) > _MAX_LEGACY_CHARS:
        return []
    try:
        records = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return []
    if not isinstance(records, (list, tuple)):
        return []
    return [r.get("name") for r in records if isinstance(r, dict)]


def _dedupe(names) -> list[str]:
    """Non-empty names, first occurrence kept, bounded by ``_MAX_NAMES``."""
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if not isinstance(name, str):
            continue
        name = name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
        if len(out) >= _MAX_NAMES:
            break
    return out


def _join(names: list[str]) -> str:
    """``", ".join`` under ``MAX_TOOLS_CHARS``, cut between names — never mid-name
    (a half name is a term that matches nothing in the BM25 corpus)."""
    out = ""
    for name in names:
        candidate = f"{out}, {name}" if out else name
        if len(candidate) > MAX_TOOLS_CHARS:
            break
        out = candidate
    return out
