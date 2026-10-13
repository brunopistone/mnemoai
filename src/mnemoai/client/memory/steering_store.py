"""User-authored, always-on steering instructions (``STEERING.md``/``CLAUDE.md``).

Instructions the USER writes to steer the agent every turn — conventions, commands,
"always do X" rules. It is read-only from the app's perspective (the agent never writes it),
which is the key distinction from ``MEMORY.md`` (agent-curated, bounded).

Discovery is hierarchical (see :func:`utils.paths.steering_files`): the app home
plus every directory from the CWD up to the repo root, each contributing at most
one file — ``STEERING.md``, else ``CLAUDE.md`` (:data:`utils.paths.
STEERING_FILENAMES`) — concatenated broadest→most-specific.

The content is prepended as a leading ``<steering>`` block each turn and stripped
before storage, so it is never summarized by compaction. A session snapshot is
refreshed between interactions when source files or limits change.
Pure file logic — no MCP, no LLM — so it is unit-testable on its own.

A file may pull in another with an ``@path`` reference, so a long ruleset can be
split into focused files instead of one wall of text (see :func:`references`).
"""

import os
import re
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from mnemoai.utils.config import config

# Per-file ceiling on injected instructions. Deliberately generous — this is a
# runaway guard, not a style rule: a normal hand-written file is far under it,
# while a machine-generated one can be tens of thousands of chars and is billed
# on every single turn. Overridable via ``STEERING.MAX_CHARS`` (``0`` disables
# the cap); a code default so no config edit is needed to reach existing installs.
_DEFAULT_MAX_CHARS = 45_000

# An ``@path`` reference: at the start of a line or after whitespace/an opening
# bracket or quote, never mid-word — so an email address (``a@b.com``) can't match.
# The path runs to the first whitespace and may not END on punctuation, so a
# reference that closes a sentence ("see @rules/style.md.") resolves cleanly.
_REF_RE = re.compile(r"(?:(?<=\s)|(?<=[(\[<'\"`])|^)@([\w.~/-]*[\w/])", re.MULTILINE)
# How deep a chain of references is followed (a referenced file may reference
# more), and how many files one turn's steering may pull in. Both are runaway
# guards: this content is re-sent on EVERY turn.
_MAX_INCLUDE_DEPTH = 3
_MAX_INCLUDES = 20


def references(text: str) -> List[str]:
    """The ``@path`` references in ``text``, in order, de-duplicated.

    Pure text scan — resolution and existence are the caller's problem, which is
    what keeps false positives harmless: a Python decorator (``@staticmethod``) or
    a handle (``@someone``) is matched here and then simply doesn't resolve to a
    file, so it stays in the prose untouched.
    """
    seen = set()
    out = []
    for m in _REF_RE.finditer(text or ""):
        ref = m.group(1)
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def resolve_reference(ref: str, base_dir: Path) -> Optional[Path]:
    """Resolve one ``@path`` against the referencing file's directory.

    ``~`` expands, an absolute path is taken as-is, and anything else is relative
    to the file that mentioned it (not the process cwd — a project's steering file
    must mean its own neighbours regardless of where the app was launched).
    Returns None unless it names a readable existing file: a reference to a
    directory or to something absent is left in the prose for the model to chase
    with its file tools.
    """
    try:
        path = _reference_path(ref, base_dir)
        return path if path is not None and path.is_file() else None
    except (OSError, ValueError, RuntimeError):
        # RuntimeError: expanduser() with no resolvable home.
        return None


def _reference_path(ref: str, base_dir: Path) -> Optional[Path]:
    """Keep missing include candidates observable so later creation invalidates a snapshot."""
    try:
        path = Path(ref).expanduser()
        return path if path.is_absolute() else base_dir / path
    except (OSError, ValueError, RuntimeError):
        return None


def _file_version(path: Path) -> tuple:
    """Metadata only; also detect replacements, permission changes and symlink retargeting."""
    try:
        info = path.stat()
        return (
            os.path.realpath(path), info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns, info.st_mode, os.access(path, os.R_OK),
        )
    except (OSError, ValueError) as exc:
        return (type(exc).__name__, getattr(exc, "errno", None))


@dataclass(frozen=True)
class SteeringNotice:
    path: Path
    kind: str
    detail: str

    @property
    def key(self):
        return (SteeringStore._key(self.path), self.kind)


@dataclass(frozen=True)
class SteeringFile:
    path: Path
    source_chars: int
    text: str
    loaded_chars: int


@dataclass(frozen=True)
class SteeringSnapshot:
    text: str = ""
    files: tuple = ()
    notices: tuple = ()
    dependencies: tuple = ()
    roots: tuple = ()
    max_chars: int = _DEFAULT_MAX_CHARS

    @property
    def prompt(self) -> str:
        if not self.text:
            return ""
        return (
            "<steering>\n"
            "The user's steering instructions are shown below. Adhere to them. "
            "IMPORTANT: these instructions OVERRIDE any default behavior and you "
            "MUST follow them exactly as written.\n\n"
            f"{self.text}\n"
            "</steering>\n\n"
        )


class SteeringStore:
    """Read + concatenate the discovered instruction files (never writes)."""

    def __init__(self, files: Optional[List[Path]] = None, cwd: Optional[Path] = None) -> None:
        """Initialize the store.

        Args:
            files: Explicit list of instruction-file paths (apply order).
                Defaults to :func:`utils.paths.steering_files` discovery.
            cwd: Working directory for project discovery (defaults to the process
                cwd); ignored when ``files`` is given.
        """
        if files is None:
            from mnemoai.utils.paths import steering_files

            files = steering_files(cwd=cwd)
        self.files = [Path(f) for f in files]

    def read(self) -> str:
        """One quiet snapshot; presentation never happens inside a file read."""
        return self.snapshot().text

    def snapshot(self) -> SteeringSnapshot:
        """Read each dependency once and publish immutable content plus diagnostics."""
        blocks: List[str] = []
        rows, notices, observed, texts = [], [], {}, {}

        def observe(path):
            if path not in observed:
                observed[path] = _file_version(path)
            return observed[path]

        def load(path):
            if path in texts:
                return texts[path]
            before = observe(path)
            try:
                text = self._read_text(path)
                if before != _file_version(path):
                    raise OSError("file changed during inspection")
            except (OSError, ValueError) as exc:
                text = ""
                notices.append(SteeringNotice(path, "unreadable", f"could not be read ({type(exc).__name__})"))
            texts[path] = text
            return text

        # Seeded with every discovered file so a reference to one of them (a
        # project file pointing at the global ruleset) doesn't inject it twice.
        seen = {self._key(f) for f in self.files}
        cap = self._max_chars()
        for f in self.files:
            text = load(f)
            if not text:
                continue
            body = self._bounded(f, text, cap)
            used = min(cap, len(text)) if cap > 0 else len(text)
            rows.append(SteeringFile(f, len(text), body, used))
            if used < len(text):
                notices.append(SteeringNotice(
                    f, "truncated", f"partially loaded ({used:,}/{len(text):,} chars)",
                ))
            blocks.append(f"Contents of {f} (user steering instructions):\n\n{body}")
            # What this file may still pull in shares its own ceiling, so
            # splitting a ruleset across @references can't sidestep the cap.
            # -1 = uncapped (STEERING.MAX_CHARS <= 0).
            budget = -1 if cap <= 0 else max(0, cap - len(body))
            blocks.extend(self._included_blocks(f, text, seen, budget, load, observe, notices))
        return SteeringSnapshot(
            "\n\n".join(blocks), tuple(rows), tuple(notices), tuple(observed.items()),
            tuple(self.files), cap,
        )

    def sizes(self) -> List[Tuple[Path, str]]:
        """``(file, injected text)`` per discovered file with content, apply order.

        Reads and caps exactly as :meth:`read` does, so a per-file size shown to
        the user (``/context``) is the size actually sent rather than the size on
        disk. Files pulled in by ``@path`` are not listed — they belong to the
        block as a whole, not to one row.
        """
        return [(row.path, row.text) for row in self.snapshot().files]

    def _included_blocks(
        self, source: Path, text: str, seen: set, budget: int, load, observe, notices
    ) -> List[str]:
        """Blocks for the files ``text`` pulls in with ``@path``, transitively.

        Breadth-first so a file's own references land before their sub-references,
        and bounded three ways — depth, count, and the remaining char budget of
        the steering file that started the chain — because every byte here is paid
        on every turn and compaction can never reclaim it.
        """
        blocks: List[str] = []
        # (referencing file, its text, depth)
        frontier: List[Tuple[Path, str, int]] = [(source, text, 1)]
        omitted = 0
        while frontier:
            origin, body, depth = frontier.pop(0)
            for ref in references(body):
                target = _reference_path(ref, origin.parent)
                if target is None:
                    continue
                version = observe(target)
                if len(version) != 8 or not stat.S_ISREG(version[6]):
                    continue  # not a file — leave the mention in the prose
                key = self._key(target)
                if key in seen:
                    continue
                seen.add(key)
                if len(blocks) >= _MAX_INCLUDES:
                    omitted += 1
                    continue
                included = load(target)
                if not included:
                    continue
                if budget >= 0 and len(included) > budget:
                    omitted += 1
                    continue
                blocks.append(
                    f"Contents of {target} (referenced by @{ref}):\n\n{included}"
                )
                if budget >= 0:
                    budget -= len(included)
                if depth < _MAX_INCLUDE_DEPTH:
                    frontier.append((target, included, depth + 1))
        if omitted:
            # Never silent: a dropped include would otherwise look like an
            # instruction the user wrote and the model ignored.
            notices.append(SteeringNotice(
                source, "includes", f"{omitted} referenced file(s) omitted by the include/size limit",
            ))
            blocks.append(
                f"[... {omitted} file(s) referenced from {source} were not "
                "included here (reference limit or size budget). Read them "
                "directly if they matter.]"
            )
        return blocks

    @staticmethod
    def _key(path: Path) -> str:
        """Identity for de-duplication: resolved, case-normalized path."""
        try:
            return os.path.normcase(os.path.realpath(str(path)))
        except (OSError, ValueError):
            return os.path.normcase(str(path))

    @staticmethod
    def _read_text(path: Path) -> str:
        """Read a regular file; the snapshot builder records errors without printing."""
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace").strip()

    @staticmethod
    def _max_chars() -> int:
        """Per-file ceiling from ``STEERING.MAX_CHARS`` (``<= 0`` means no cap)."""
        try:
            return int(
                (config.get("STEERING", {}) or {}).get("MAX_CHARS", _DEFAULT_MAX_CHARS)
            )
        except (AttributeError, TypeError, ValueError):
            return _DEFAULT_MAX_CHARS

    @classmethod
    def _bounded(cls, path: Path, text: str, max_chars: Optional[int] = None) -> str:
        """Cap one file's contribution, telling the model what was cut.

        This content is re-injected verbatim every turn and deliberately kept out
        of history, so compaction can never reclaim it — an oversized file is a
        permanent per-turn tax (a 55k-char file is ~14k tokens EVERY turn). The
        cap is generous and never silent: the model is told the file was
        truncated, so it can read the rest with its file tools instead of
        assuming it saw everything.
        """
        if max_chars is None:
            max_chars = cls._max_chars()
        # A non-positive cap means "no cap" — an explicit opt-out for a user who
        # wants a big file honored whole.
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return (
            text[:max_chars]
            + f"\n\n[... truncated: {path} is {len(text)} chars and only the "
            f"first {max_chars} are shown here. Read the file directly if you "
            "need the rest — do not assume the omitted part says nothing.]"
        )


class SteeringState:
    """Session-owned snapshot; refresh at interaction boundaries, never during repaint."""

    def __init__(self, files=None, cwd=None):
        self._files = files
        self._cwd = cwd
        self._lock = threading.RLock()
        self._snapshot = None
        self._reported = {}
        self.revision = 0

    @property
    def current(self) -> SteeringSnapshot:
        # A painter can keep using the last immutable snapshot while a worker
        # refreshes a slow file. It must not wait on filesystem I/O via this lock.
        snapshot = self._snapshot
        if snapshot is not None:
            return snapshot
        with self._lock:
            if self._snapshot is None:
                self.refresh()
            return self._snapshot

    def refresh(self) -> SteeringSnapshot:
        with self._lock:
            store = SteeringStore(files=self._files, cwd=self._cwd)
            previous = self._snapshot
            if (
                previous is not None
                and not any(notice.kind == "unreadable" for notice in previous.notices)
                and previous.roots == tuple(store.files)
                and previous.max_chars == store._max_chars()
                and all(_file_version(path) == stamp for path, stamp in previous.dependencies)
            ):
                return previous
            self._snapshot = store.snapshot()
            active = {notice.key for notice in self._snapshot.notices}
            self._reported = {key: value for key, value in self._reported.items() if key in active}
            self.revision += 1
            return self._snapshot

    def take_notices(self) -> tuple:
        """A condition is announced once until it changes or disappears."""
        with self._lock:
            snapshot = self.current
            new = tuple(
                notice for notice in snapshot.notices
                if self._reported.get(notice.key) != notice.detail
            )
            self._reported = {notice.key: notice.detail for notice in snapshot.notices}
            return new


_STATE_LOCK = threading.Lock()


def state_for(client) -> SteeringState:
    """Share one state across the client's request, context and UI collaborators."""
    with _STATE_LOCK:
        state = getattr(client, "_steering_state", None)
        if not isinstance(state, SteeringState):
            state = SteeringState()
            if client is not None:
                try:
                    client._steering_state = state
                except (AttributeError, TypeError):
                    pass  # read-only embedded clients can still request a snapshot
        return state
