"""Why a file looks the way it does (`/why`).

The other workspace reports answer "what is different" (`/diff`) and "what did we
touch" (`/files`). Neither answers the question you actually have when you open a
file a week later and don't recognize a function in it: **which prompt asked for
this?** The turn that made the change is somewhere in a transcript — scrolled
past, or in a session that ended long ago — and by the time the question comes up
the conversation that holds the answer is the one thing you no longer have.

So the answer is indexed as it happens: one line per change, naming the file, the
turn, the prompt behind it, and what the call did. Kept per project (like
sessions), because a file belongs to a project and not to a conversation.

**Derived, and it holds no content.** The full evidence stays in the session
transcript; this index is a pointer to it plus a summary short enough to stand
alone. That matters both ways round: it must be small (a project's transcripts
here run to hundreds of megabytes, so deriving this on demand by scanning them
would make `/why` take seconds — and a report that slow is a report nobody runs),
and it must OUTLIVE them, since transcripts are swept at ``SESSION_MAX_AGE_DAYS``
while a line of code lives for years. It is bounded by size rather than age for
the same reason, and disabled by the same knob that turns session recording off:
this records what the user typed, so switching that off must switch this off too.

Written from the ONE place every tool call passes through
(``agent/tool_loop.py``, after a successful invocation), which is what lets it
account for work nobody watched — a background sub-agent, a parallel wave — with
no per-caller wiring, exactly as the file ledger does. Only the two tools that
CHANGE one named file are recorded; a read changed nothing to explain.

**A ``/rewind`` does not withdraw a change here, deliberately.** Rewinding moves
the conversation only — files on disk are never touched — so the edit is still
there and the prompt that caused it is still the true answer. Deleting the record
would leave `/why` unable to explain a change that is really in the file, which
is the opposite of the point; it's the same reason the file ledger and the
read-before-write gate are left alone by a rewind.

Keyed like the ledger and that gate (``normcase(realpath(...))``) so two
spellings of one file are one file, and non-raising throughout: this is
bookkeeping, and it runs beside a tool call that must not fail because of it.
"""

import json
import os
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

from mnemoai.client.file_ledger import resolve_path
from mnemoai.client.ui.turn_view import user_prompt_text
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import provenance_path

# The tools that CHANGE one named file, and the arg that names it. `fs_read` is
# deliberately absent: `/why` explains what a file IS, and reading it changed
# nothing. Same subset as the ledger's WRITTEN group, for the same reason.
_TOOL_PATHS = {"fs_write": "path", "file_edit": "file_path"}

# The prompt is stored clipped: enough to recognize the request, not a copy of the
# conversation (which is what the transcript is for).
_MAX_PROMPT_CHARS = 160

# Size past which the oldest half of the index is dropped. A record is ~200 bytes,
# so this is tens of thousands of changes — years of them for one project.
_MAX_BYTES = 2_000_000

# Changes shown per report before the rest collapse into a count.
_MAX_ROWS = 14

# Files listed by a bare `/why` before the rest collapse into a count.
_MAX_FILES = 12

# The group heading for the running session, hoisted above the earlier ones.
_THIS_SESSION = "This session"

_GRAY = "\033[90m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_MARK = "\033[92m"
_HEADER = "\033[38;5;111m"


class Change(NamedTuple):
    """One recorded change to one file, and the prompt that asked for it."""

    path: str  # resolved key, as the ledger and the read-before-write gate key it
    tool: str
    ts: float
    turn: int  # the turn's number IN ITS OWN session (0 when unknown)
    session: str
    prompt: str
    detail: str


class ProvenanceLog:
    """Append-only per-project index of which prompt changed which file."""

    def __init__(self, cwd=None, profile: str = None) -> None:
        self.path: Optional[str] = None
        try:
            self.path = str(provenance_path(cwd, profile))
        except Exception:  # noqa: BLE001 — an index we can't open is a no-op
            logger.debug("Provenance index unavailable", exc_info=True)

    def record(
        self, tool: str, args, turn: int = 0, prompt: str = "", session: str = ""
    ) -> None:
        """Note that this turn's prompt changed a file. Never raises.

        The turn number and session id are passed in rather than held, because
        both belong to the live ``SessionLog`` — a ``/branch`` re-points the agent
        at a new file mid-run, and a stale copy here would file the rest of the
        session's changes under the conversation it forked away from.
        """
        try:
            arg = _TOOL_PATHS.get(tool)
            if arg is None or self.path is None:
                return
            raw = args.get(arg) if isinstance(args, dict) else None
            if not isinstance(raw, str) or not raw.strip():
                return
            key, _ = resolve_path(raw)
            if not key:
                return
            self._append(
                {
                    "t": "change",
                    "path": key,
                    "tool": tool,
                    "ts": round(time.time(), 3),
                    "turn": max(0, int(turn or 0)),
                    "session": str(session or ""),
                    # The one shared stripper, so the quoted prompt is what the
                    # user typed rather than the episodic block prepended to it.
                    "prompt": _clip(user_prompt_text(prompt), _MAX_PROMPT_CHARS),
                    "detail": describe_change(tool, args),
                }
            )
        except Exception:  # noqa: BLE001 — never break a tool call over bookkeeping
            logger.debug("Provenance record skipped", exc_info=True)

    def _append(self, record: Dict) -> None:
        """Add one line, then keep the file inside its cap."""
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._trim()

    def _trim(self) -> None:
        """Drop the oldest half once the index outgrows its cap.

        Age is the wrong bound here — the index exists to outlive the transcripts
        — and rewriting is cheap and rare: a turn adds a handful of short lines.
        """
        try:
            if os.path.getsize(self.path) <= _MAX_BYTES:
                return
            with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.writelines(lines[len(lines) // 2:])
            os.replace(tmp, self.path)
        except OSError:
            logger.debug("Could not trim the provenance index", exc_info=True)


def changes(target: str = "", cwd=None, profile: str = None) -> List[Change]:
    """Recorded changes, newest first — every file's, or just ``target``'s."""
    try:
        path = provenance_path(cwd, profile)
    except Exception:  # noqa: BLE001
        return []
    key = resolve_path(target)[0] if target else ""
    found: List[Change] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                change = _parse(line)
                if change is None or (key and change.path != key):
                    continue
                found.append(change)
    except OSError:
        return []
    found.reverse()
    return found


def describe_change(tool: str, args) -> str:
    """One short phrase for what a call did — the summary that stands alone.

    Read off the call's own arguments, since that is all this layer has: the tool
    result is a JSON blob and the file on disk has moved on since. Plain text —
    what is stored must not carry the colors of the report that displays it.
    """
    args = args if isinstance(args, dict) else {}
    if tool == "file_edit":
        return _replacement(args.get("old_string"), args.get("new_string"))
    command = str(args.get("command") or "")
    if command == "create":
        return f"wrote {_plural(_lines(args.get('file_text')), 'line')}"
    if command == "str_replace":
        return _replacement(args.get("old_str"), args.get("new_str"))
    if command == "insert":
        at = args.get("insert_line")
        where = f" at line {at}" if isinstance(at, int) and at > 0 else ""
        return f"+{_lines(args.get('new_str'))}{where}"
    if command == "append":
        return f"+{_lines(args.get('new_str'))} appended"
    return command or tool


def render_file(items: List[Change], subject: str, session_id: str = "") -> str:
    """One file's history: every recorded change to it, newest first.

    Pure over already-read records (no client, no disk), so it is unit-testable.
    Grouped by session, because a turn number only means something inside one.
    """
    if not items:
        return (
            f"No recorded change to {subject}.\n"
            f"  {_GRAY}Only changes this app made are indexed, and only since it "
            f"began keeping the index.{_RESET}"
        )

    out = [f"{_BOLD}Why {subject} looks like this{_RESET}", ""]
    for header, group in _by_session(items[:_MAX_ROWS], session_id):
        out.append(f"  {header}")
        for change in group:
            out.append(
                f"    {_MARK}✎{_RESET} {_GRAY}{_stamp(change, session_id)} · "
                f"{change.tool} · {change.detail}{_RESET}"
            )
            if change.prompt:
                out.append(f"        {_HEADER}>{_RESET} {change.prompt}")
        out.append("")

    total = _plural(len(items), "recorded change")
    if len(items) > _MAX_ROWS:
        total += f", {_MAX_ROWS} shown"
    out.append(f"  {_GRAY}{total}. The prompt under each is the one that asked for")
    out.append(
        f"  it. A change made outside this app, or before the index existed, "
        f"isn't here.{_RESET}"
    )
    return "\n".join(out)


def render_overview(items: List[Change], session_id: str = "") -> str:
    """Every file with a recorded change: how many, and the most recent prompt.

    Prefers THIS session's changes — the ones you are most likely asking about —
    and falls back to the project's history when this session changed nothing,
    since an empty report would be the least useful true answer available.
    """
    if not items:
        return (
            "No changes recorded for this project yet.\n"
            f"  {_GRAY}A file edited from a session here gets an entry, so `/why "
            f"<path>` can say which prompt asked for it.{_RESET}"
        )

    mine = [c for c in items if c.session and c.session == session_id]
    scope = "this session" if mine else "earlier sessions here"
    files = _by_file(mine or items)
    # The display form costs a realpath, so it is built only for the rows kept.
    shown = [(resolve_path(key)[1] or key, group) for key, group in files[:_MAX_FILES]]

    width = min(max(max(len(display) for display, _ in shown), 20), 60)
    out = [f"{_BOLD}Why these files look like this{_RESET}", "", f"  Changed in {scope}"]
    for display, group in shown:
        newest = group[0]
        count = _plural(len(group), "change")
        out.append(
            f"    {_MARK}✎{_RESET} {display.ljust(width)}  "
            f"{_GRAY}{count} · {_stamp(newest, session_id)}{_RESET}".rstrip()
        )
        if newest.prompt:
            out.append(f"        {_HEADER}>{_RESET} {newest.prompt}")
    if len(files) > _MAX_FILES:
        out.append(f"    {_GRAY}… +{len(files) - _MAX_FILES} more{_RESET}")
    out.append("")
    out.append(f"  {_GRAY}`/why <path>` for one file's whole history, across{_RESET}")
    out.append(f"  {_GRAY}sessions.{_RESET}")
    return "\n".join(out)


def report(client, target: str = "") -> str:
    """``/why`` for a client — read-only: no write, no model call, no tool."""
    agent = getattr(client, "agent", None)
    session = getattr(agent, "session_log", None)
    session_id = str(getattr(session, "session_id", "") or "")
    target = str(target or "").strip()
    try:
        items = changes(target)
    except Exception:  # noqa: BLE001 — a report that dies must not be the problem
        logger.debug("Provenance report failed", exc_info=True)
        return "Change history is unavailable."
    if target:
        return render_file(items, resolve_path(target)[1] or target, session_id)
    return render_overview(items, session_id)


def _parse(line: str) -> Optional[Change]:
    """One index line as a ``Change``, or None if it isn't one."""
    try:
        record = json.loads(line)
    except (ValueError, TypeError):
        return None
    if not isinstance(record, dict) or record.get("t") != "change":
        return None
    path = record.get("path")
    if not isinstance(path, str) or not path:
        return None
    try:
        ts = float(record.get("ts") or 0.0)
    except (TypeError, ValueError):
        ts = 0.0
    try:
        turn = max(0, int(record.get("turn") or 0))
    except (TypeError, ValueError):
        turn = 0
    return Change(
        path=path,
        tool=str(record.get("tool") or ""),
        ts=ts,
        turn=turn,
        session=str(record.get("session") or ""),
        prompt=str(record.get("prompt") or ""),
        detail=str(record.get("detail") or ""),
    )


def _by_session(items: List[Change], session_id: str) -> List[Tuple[str, List[Change]]]:
    """Changes grouped into this session's, then each earlier session's.

    This session leads unconditionally — it is what the reader is holding — and
    the rest follow newest first, each named so it can be reopened
    (``--resume <id>``): the transcript is where the whole turn still lives.
    """
    groups: List[Tuple[str, List[Change]]] = []
    index: Dict[str, int] = {}
    for change in items:
        if session_id and change.session == session_id:
            header = _THIS_SESSION
        elif change.session:
            header = f"Earlier · session {change.session}"
        else:
            header = "Earlier"
        position = index.get(header)
        if position is None:
            index[header] = len(groups)
            groups.append((header, [change]))
        else:
            groups[position][1].append(change)
    groups.sort(key=lambda g: g[0] != _THIS_SESSION)  # stable: only this one moves
    return groups


def _by_file(items: List[Change]) -> List[Tuple[str, List[Change]]]:
    """``(key, changes)`` per file, most recently changed first.

    ``items`` arrives newest-first, so each group's own head is its newest change.
    """
    grouped: Dict[str, List[Change]] = {}
    for change in items:
        grouped.setdefault(change.path, []).append(change)
    return sorted(grouped.items(), key=lambda kv: -kv[1][0].ts)


def _stamp(change: Change, session_id: str) -> str:
    """When a change happened: the clock within this session, the date outside it.

    The turn number is shown in both, since the rows are grouped by session and it
    locates the turn in that session's transcript.
    """
    if not change.ts:
        when = ""
    elif change.session and change.session == session_id:
        when = time.strftime("%H:%M", time.localtime(change.ts))
    else:
        when = time.strftime("%m-%d %H:%M", time.localtime(change.ts))
    turn = f"turn {change.turn}" if change.turn else ""
    return " · ".join(p for p in (turn, when) if p) or "unknown"


def _replacement(old, new) -> str:
    """``+3 -1`` for a replacement, in `/diff`'s wording."""
    added, removed = _lines(new), _lines(old)
    parts = []
    if added:
        parts.append(f"+{added}")
    if removed:
        parts.append(f"-{removed}")
    return " ".join(parts) or "edited"


def _lines(text) -> int:
    """Lines in a chunk of content (none for empty, one for an unterminated line)."""
    if not isinstance(text, str) or not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'s' if n != 1 else ''}"


def _clip(text: str, limit: int) -> str:
    """One line, at most ``limit`` chars, saying so when it was cut."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"
