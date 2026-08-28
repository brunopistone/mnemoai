"""Which files this session touched (`/files`).

A long session reads dozens of files and edits a handful of them, and by the time
it matters that record exists only in scrollback — scrolled past, interleaved with
everything else, and gone entirely after a compaction summarizes the turns that
did the work. So the touches are recorded as they happen: **what was changed on
disk**, what was **attached** to a prompt with `@`, and what was merely **read**.

Fed from the ONE place every tool call passes through (`agent/tool_loop.py`, after
a successful invocation — a refused or failed call touched nothing) plus the `@`
mention expansion in the UI. Deliberately narrow: only the tools that name a
single file (`fs_read`, `fs_write`, `file_edit`) are recorded, because a
`grep_search` over a tree touched no file the user could act on.

**It is a ledger, not a context inventory.** A file read early in a long session
may have had its result evicted or summarized away since; the ledger still lists
it, because "we looked at this" stays true. The report says so rather than
implying the content is still in the window.

Keyed like the read-before-write gate (`normcase(realpath(...))`) so two spellings
of one file are one row, thread-safe because sub-agents and parallel waves record
from pool threads, and bounded — a runaway session must not grow this without
limit.
"""

import os
import threading
from typing import Dict, List, Optional

READ = "read"
WRITTEN = "written"
ATTACHED = "attached"

# The tools that act on ONE named file, and the arg that names it. A tool whose
# subject is a tree (grep_search, glob_search) is deliberately absent: it produces
# no path the user can go and look at.
_TOOL_PATHS = {
    "fs_read": (READ, "path"),
    "fs_write": (WRITTEN, "path"),
    "file_edit": (WRITTEN, "file_path"),
}

# Past this many distinct files, new paths are counted but not kept. The report is
# already unreadable at that size, and a ledger is not worth unbounded memory.
_MAX_ENTRIES = 500

# Rows shown per group before the rest collapse into a count.
_MAX_ROWS = 12

_GRAY = "\033[90m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_MARKS = {WRITTEN: ("✎", "\033[92m"), ATTACHED: ("@", "\033[36m"), READ: ("·", _GRAY)}
_GROUP_TITLES = {WRITTEN: "Changed", ATTACHED: "Attached with @", READ: "Read"}


class Entry:
    """One file and what this session did to it."""

    __slots__ = ("path", "display", "counts", "seq")

    def __init__(self, path: str, display: str, seq: int) -> None:
        self.path = path
        self.display = display
        self.counts: Dict[str, int] = {READ: 0, WRITTEN: 0, ATTACHED: 0}
        self.seq = seq  # touch order, so the report can lead with the recent ones

    @property
    def kind(self) -> str:
        """The group this file belongs to — the strongest thing done to it."""
        if self.counts[WRITTEN]:
            return WRITTEN
        if self.counts[ATTACHED]:
            return ATTACHED
        return READ


class FileLedger:
    """Thread-safe record of the files this session touched."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[str, Entry] = {}
        self._seq = 0
        self._overflow = 0  # distinct files seen after the cap

    def record(self, path: str, action: str) -> None:
        """Note one touch of ``path``. Never raises — a ledger is bookkeeping."""
        try:
            if not path or action not in (READ, WRITTEN, ATTACHED):
                return
            key, display = _resolve(path)
            if not key:
                return
            with self._lock:
                self._seq += 1
                entry = self._entries.get(key)
                if entry is None:
                    if len(self._entries) >= _MAX_ENTRIES:
                        self._overflow += 1
                        return
                    entry = Entry(key, display, self._seq)
                    self._entries[key] = entry
                entry.counts[action] += 1
                entry.seq = self._seq
        except Exception:  # noqa: BLE001 — never break a tool call over this
            pass

    def record_tool(self, name: str, args) -> None:
        """Note the file a just-completed tool call acted on, if it named one."""
        mapping = _TOOL_PATHS.get(name)
        if not mapping:
            return
        action, arg = mapping
        path = (args or {}).get(arg) if isinstance(args, dict) else None
        if isinstance(path, str):
            self.record(path, action)

    def snapshot(self) -> List[Entry]:
        """Every entry, most recently touched first."""
        with self._lock:
            return sorted(self._entries.values(), key=lambda e: -e.seq)

    def changed_paths(self) -> set:
        """Resolved paths this session WROTE — what `/diff` marks as its own."""
        with self._lock:
            return {e.path for e in self._entries.values() if e.counts[WRITTEN]}

    @property
    def overflow(self) -> int:
        """Distinct files seen after the cap (counted, not kept)."""
        with self._lock:
            return self._overflow

    def reset(self) -> None:
        """Forget everything (a cleared conversation is a new session)."""
        with self._lock:
            self._entries.clear()
            self._seq = 0
            self._overflow = 0


def _resolve(path: str):
    """``(key, display)`` for a path: one key per file, a short display form.

    The key is the read-before-write gate's key, so `./x.py`, `x.py` and an
    absolute spelling are ONE row. The display form is CWD-relative when the file
    is under the directory the session runs in (which is what the user typed), and
    home-relative otherwise.
    """
    raw = str(path).strip()
    if not raw:
        return "", ""
    absolute = os.path.abspath(os.path.expanduser(raw))
    try:
        key = os.path.normcase(os.path.realpath(absolute))
    except OSError:
        key = os.path.normcase(absolute)
    return key, _display(absolute)


def _display(absolute: str) -> str:
    """A path short enough to scan a column of: CWD-relative, else ``~/…``."""
    try:
        cwd = os.getcwd()
    except OSError:
        cwd = ""
    if cwd:
        try:
            rel = os.path.relpath(absolute, cwd)
        except ValueError:  # different drive on Windows
            rel = ""
        if rel and not rel.startswith(".."):
            return rel
    home = os.path.expanduser("~")
    if home and absolute.startswith(home + os.sep):
        return "~" + absolute[len(home):]
    return absolute


def _counts_text(entry: Entry) -> str:
    """``3 edits · 1 read`` — what happened, strongest first, in the plural."""
    parts = []
    for action, word in ((WRITTEN, "edit"), (ATTACHED, "attach"), (READ, "read")):
        n = entry.counts[action]
        if not n:
            continue
        if action == ATTACHED:
            parts.append("attached" if n == 1 else f"attached {n}×")
        else:
            parts.append(f"{n} {word}{'s' if n != 1 else ''}")
    return " · ".join(parts)


def render(ledger: FileLedger) -> str:
    """The ``/files`` report: what this session touched, grouped by what happened.

    Pure over the ledger (no client, no terminal), so it is unit-testable. Groups
    lead with what CHANGED, since that's the part with consequences on disk.
    """
    entries = ledger.snapshot()
    if not entries:
        return (
            "No files touched yet in this session.\n"
            f"  {_GRAY}Files read, edited, or attached with @ appear here.{_RESET}"
        )

    grouped = {WRITTEN: [], ATTACHED: [], READ: []}
    for entry in entries:
        grouped[entry.kind].append(entry)

    # One column for every group, clamped: a single deep path must not push the
    # counts off the screen, and a short list must not scatter them.
    width = min(max(max(len(e.display) for e in entries), 20), 60)

    out = [f"{_BOLD}Files this session{_RESET}", ""]
    for kind in (WRITTEN, ATTACHED, READ):
        rows = grouped[kind]
        if not rows:
            continue
        mark, color = _MARKS[kind]
        if len(out) > 2:  # a blank line between groups, not above the first
            out.append("")
        out.append(f"  {_GROUP_TITLES[kind]} ({len(rows)})")
        for entry in rows[:_MAX_ROWS]:
            counts = _counts_text(entry)
            out.append(
                f"    {color}{mark}{_RESET} {entry.display.ljust(width)}  "
                f"{_GRAY}{counts}{_RESET}".rstrip()
            )
        if len(rows) > _MAX_ROWS:
            hidden = len(rows) - _MAX_ROWS
            out.append(f"    {_GRAY}… +{hidden} more{_RESET}")
    if ledger.overflow:
        out.append("")
        out.append(
            f"  {_GRAY}+{ledger.overflow} further file(s) not listed "
            f"(ledger holds {_MAX_ENTRIES}){_RESET}"
        )
    out.append("")
    out.append(
        f"  {_GRAY}What this session touched, most recent first. A file read "
        f"early on may have\n  had its content summarized out of the context "
        f"since — the touch still counts.{_RESET}"
    )
    return "\n".join(out)


def report(client) -> str:
    """``/files`` for a client: its agent's ledger, or why there isn't one."""
    ledger: Optional[FileLedger] = getattr(getattr(client, "agent", None), "files", None)
    if ledger is None:
        return "File tracking is unavailable (no agent running)."
    return render(ledger)
