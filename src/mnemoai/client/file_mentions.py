"""``@path`` file mentions typed into the prompt.

Pointing at a file used to mean typing its path in prose and hoping the model
chose to read it. A mention removes both halves of that: the pinned input
**completes** the path while you type it (:func:`completions`), and the file's
contents are **appended to the prompt** before the turn runs (:func:`expand`), so
the model has them whether or not it would have reached for a tool.

The syntax is the one steering files already use — ``@`` at a line start or after
whitespace, path running to the next space — and the scan is literally
:func:`memory.steering_store.references`, so there is one definition of what an
``@path`` is. As there, a reference that doesn't resolve is left in the prose
untouched: that is what keeps ``@staticmethod`` and ``@someone`` harmless.

Three deliberate limits, because an inlined file lands in the conversation as the
user's own words — which compaction can summarize but tool-result eviction can
never shrink: a per-file cap (``MENTIONS.MAX_FILE_CHARS``), a per-line total, and
a file count. Anything cut says so, so the model reads the rest with ``fs_read``
instead of assuming it saw everything. A mention is **not** a read: the
server-side read-before-write gate still wants its own ``fs_read`` before an edit.

Pure file logic (no MCP, no LLM, no prompt_toolkit) so the completer and the
dispatcher share it and it unit-tests on its own.
"""

import os
import re
import subprocess
import time
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

from mnemoai.client.memory.steering_store import references
from mnemoai.utils.config import config
from mnemoai.utils.logger import logger

# Per-file ceiling on inlined content, overridable via ``MENTIONS.MAX_FILE_CHARS``
# (``0`` disables it); a code default so no config edit is needed to reach an
# existing install. Generous — a source file fits whole; a 200k-line log doesn't.
_DEFAULT_MAX_FILE_CHARS = 20_000

# Ceilings on ONE line's mentions. Unlike steering this is paid once, but it is
# paid in history: the user cannot evict it, so `@.` on a big tree must not be
# able to fill the window.
_MAX_TOTAL_CHARS = 60_000
_MAX_FILES = 10

# A mentioned directory contributes its listing, never its files' contents —
# that's what makes `@src/` a cheap way to say "here is the shape of this dir".
_MAX_DIR_ENTRIES = 200

# Bytes sniffed for a NUL to decide "this isn't text". A binary file is named
# rather than inlined: replacement characters would be worse than a pointer.
_SNIFF_BYTES = 8192

# An in-progress mention at the cursor: same left-context rule as a complete one,
# but the fragment may be empty (just after `@`) or end on a separator, since it's
# still being typed. Backslash included so a pasted Windows path still completes.
_AT_CURSOR_RE = re.compile(r"(?:(?<=\s)|(?<=[(\[<'\"`])|^)@([\w.~/\\-]*)$")

# Directories never worth offering: build output and caches drown the real
# candidates. Only consulted by the no-git fallback walk (git already ignores them).
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "node_modules", ".venv", "venv", "dist", "build", ".tox",
    ".idea", ".vscode", ".DS_Store", "site-packages", ".next", "target",
})

# Bounds on the completion index: a walk that never ends is a hung keystroke.
_MAX_INDEX_FILES = 20_000
_MAX_WALK_DEPTH = 12
_GIT_TIMEOUT = 2.0

# The index is rebuilt at most this often per directory. Long enough that a burst
# of keystrokes costs one scan, short enough that a file you just created is
# offered by the time you look for it.
_INDEX_TTL = 10.0
_INDEX_CACHE: dict = {}


class Mention(NamedTuple):
    """One ``@path`` from a submitted line, and what became of it.

    Attributes:
        ref: The path as typed, without the ``@``.
        path: The resolved path, or None when nothing matched.
        kind: ``file`` / ``dir`` (inlined), ``missing``, or ``skipped``.
        chars: Characters actually inlined (0 when nothing was).
        summary: The human detail for the notice — "412 lines", "14 entries",
            "no such file", why it was trimmed or skipped.
    """

    ref: str
    path: Optional[Path]
    kind: str
    chars: int = 0
    summary: str = ""

    @property
    def attached(self) -> bool:
        """True when this mention put something in the prompt."""
        return self.kind in ("file", "dir")

    @property
    def label(self) -> str:
        """One line for the on-screen notice ("@src/x.py · 412 lines").

        Built here rather than in the UI so the wording is testable without a
        terminal; the caller only adds color.
        """
        return f"@{self.ref} · {self.summary}" if self.summary else f"@{self.ref}"


def _max_file_chars() -> int:
    """Per-file cap from ``MENTIONS.MAX_FILE_CHARS`` (``<= 0`` means no cap)."""
    try:
        return int(
            (config.get("MENTIONS", {}) or {}).get(
                "MAX_FILE_CHARS", _DEFAULT_MAX_FILE_CHARS
            )
        )
    except (AttributeError, TypeError, ValueError):
        return _DEFAULT_MAX_FILE_CHARS


def resolve(ref: str, cwd: Optional[Path] = None) -> Optional[Path]:
    """Resolve one mention against ``cwd``; None when it names nothing.

    ``~`` expands and an absolute path is taken as-is; anything else is relative
    to the working directory — the prompt's frame of reference, unlike a steering
    reference, which belongs to the file that wrote it. Directories resolve too
    (they contribute a listing), which is the other difference.
    """
    try:
        path = Path(ref).expanduser()
        if not path.is_absolute():
            path = Path(cwd or Path.cwd()) / path
        return path if path.exists() else None
    except (OSError, ValueError, RuntimeError):
        # RuntimeError: expanduser() with no resolvable home.
        return None


def _looks_like_a_path(ref: str) -> bool:
    """True when a *failed* mention is worth reporting to the user.

    A typo'd path silently attaching nothing is the failure mode here — the model
    then answers about a file it never saw. But ``@staticmethod`` must stay
    invisible, so only a ref with a separator or an extension earns the line.
    """
    return "/" in ref or "." in ref.strip(".")


def _read_text(path: Path, cap: int) -> Tuple[Optional[str], str, int]:
    """``(text, truncation summary, chars)`` for a file, or ``(None, reason, 0)``.

    Bounded at the read itself: ``cap`` chars need at most ``cap + 1`` bytes, so a
    huge file is never fully loaded just to be thrown away.
    """
    try:
        size = path.stat().st_size
        limit = -1 if cap <= 0 else cap + 1
        with path.open("rb") as fh:
            head = fh.read(_SNIFF_BYTES)
            if b"\x00" in head:
                return None, "not a text file", 0
            if limit < 0:
                data = head + fh.read()
            elif len(head) >= limit:
                data = head[:limit]
            else:
                data = head + fh.read(limit - len(head))
    except (OSError, ValueError) as e:
        logger.debug(f"Mention {path} could not be read: {e}")
        return None, "could not be read", 0
    # errors="replace", not strict: a stray byte in an otherwise-text file must
    # degrade, not abort the turn.
    text = data.decode("utf-8", errors="replace")
    if cap > 0 and len(text) > cap:
        return text[:cap], f"first {cap} chars of {size} bytes", cap
    return text, "", len(text)


def _dir_listing(path: Path) -> Tuple[str, int]:
    """``(listing, entry count)`` for a mentioned directory, names only.

    Directories keep their trailing ``/`` so the model can mention one back, and
    the list is bounded — a mention is a pointer, not a crawl.
    """
    try:
        entries = sorted(
            (e.name + ("/" if e.is_dir() else "") for e in os.scandir(path)),
            key=str.lower,
        )
    except OSError as e:
        logger.debug(f"Mention {path} could not be listed: {e}")
        return "", 0
    shown = entries[:_MAX_DIR_ENTRIES]
    text = "\n".join(shown)
    if len(entries) > len(shown):
        text += f"\n[... {len(entries) - len(shown)} more entries]"
    return text, len(entries)


def expand(text: str, cwd: Optional[Path] = None) -> Tuple[str, List[Mention]]:
    """Append the contents of every ``@path`` in ``text`` to it.

    Returns the prompt to send and one :class:`Mention` per reference worth
    telling the user about. The reference itself stays in the prose so the
    sentence still reads, and each file is appended once under its own header —
    the same shape steering uses, so the model meets a familiar block.

    Never raises: a mention is typed mid-sentence by a user who wants an answer,
    so anything unreadable degrades to a note and the turn goes ahead.
    """
    if not text or "@" not in text:
        return text, []
    cap = _max_file_chars()
    budget = _MAX_TOTAL_CHARS
    blocks: List[str] = []
    results: List[Mention] = []
    seen = set()
    inlined = 0
    for ref in references(text):
        path = resolve(ref, cwd)
        if path is None:
            if _looks_like_a_path(ref):
                results.append(Mention(ref, None, "missing", summary="no such file"))
            continue
        key = _key(path)
        if key in seen:
            continue
        seen.add(key)
        if inlined >= _MAX_FILES:
            results.append(
                Mention(ref, path, "skipped", summary=f"over {_MAX_FILES} files")
            )
            continue
        if path.is_dir():
            listing, count = _dir_listing(path)
            if not listing:
                results.append(
                    Mention(ref, path, "skipped", summary="empty or unreadable")
                )
                continue
            blocks.append(
                f"Contents of {path}/ (mentioned as @{ref}) — directory listing:"
                f"\n\n{listing}"
            )
            results.append(
                Mention(
                    ref, path, "dir", chars=len(listing), summary=f"{count} entries"
                )
            )
            inlined += 1
            continue
        body, cut, chars = _read_text(path, cap)
        if body is None:
            results.append(Mention(ref, path, "skipped", summary=cut))
            continue
        if chars > budget:
            # The line as a whole is out of room. Say so rather than trimming to
            # a size the user can't predict — the path is still in the prose.
            results.append(
                Mention(ref, path, "skipped", summary="over the size budget")
            )
            continue
        budget -= chars
        blocks.append(f"Contents of {path} (mentioned as @{ref}):\n\n{body}")
        if cut:
            # Never silent: an answer based on the first half of a file, presented
            # as an answer about the file, is the failure this line prevents.
            blocks.append(
                f"[... truncated: only the first {cap} chars of {path} are shown "
                "above. Read the rest directly if you need it — do not assume the "
                "omitted part says nothing.]"
            )
        results.append(
            Mention(ref, path, "file", chars=chars, summary=cut or _lines(body))
        )
        inlined += 1
    if not blocks:
        return text, results
    return text.rstrip() + "\n\n" + "\n\n".join(blocks), results


def _lines(body: str) -> str:
    """``N lines`` for a notice — what a person counts a file in."""
    n = body.count("\n") + (1 if body and not body.endswith("\n") else 0)
    return f"{n} line" if n == 1 else f"{n} lines"


def _key(path: Path) -> str:
    """De-duplication identity: resolved, case-normalized (as steering does)."""
    try:
        return os.path.normcase(os.path.realpath(str(path)))
    except (OSError, ValueError):
        return os.path.normcase(str(path))


# --- completion -----------------------------------------------------------------


def fragment_at_cursor(text_before_cursor: str) -> Optional[str]:
    """The in-progress ``@`` fragment at the cursor, or None.

    Returns "" for a bare ``@`` (offer the working directory), so the caller can
    distinguish "no mention here" (None) from "a mention with nothing typed yet".
    """
    m = _AT_CURSOR_RE.search(text_before_cursor or "")
    return m.group(1) if m else None


def _index(root: Path) -> List[str]:
    """Relative paths under ``root`` for basename matching, TTL-cached.

    ``git ls-files`` when there is a repo: it is instant and already excludes
    what ``.gitignore`` says isn't yours. Otherwise a bounded walk — the point is
    a completion menu, so a partial answer beats a stalled keystroke.
    """
    key = str(root)
    now = time.monotonic()
    hit = _INDEX_CACHE.get(key)
    if hit is not None and now - hit[0] < _INDEX_TTL:
        return hit[1]
    files = _git_index(root)
    if files is None:
        files = _walk_index(root)
    _INDEX_CACHE[key] = (now, files)
    return files


def _git_index(root: Path) -> Optional[List[str]]:
    """Tracked + untracked-but-not-ignored paths, or None when git can't answer."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_GIT_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug(f"Mention index: git unavailable in {root} ({e})")
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.decode("utf-8", errors="replace")
    return [p for p in out.split("\0") if p][:_MAX_INDEX_FILES]


def _walk_index(root: Path) -> List[str]:
    """Bounded os.walk fallback: skips caches, build output and dotted dirs."""
    files: List[str] = []
    base = str(root)
    for dirpath, dirnames, filenames in os.walk(base):
        depth = dirpath[len(base):].count(os.sep)
        if depth >= _MAX_WALK_DEPTH:
            dirnames[:] = []
        dirnames[:] = [
            d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        rel = os.path.relpath(dirpath, base)
        for name in filenames:
            if name.startswith("."):
                continue
            files.append(name if rel == "." else os.path.join(rel, name))
            if len(files) >= _MAX_INDEX_FILES:
                return files
    return files


def _dir_entries(directory: Path, prefix: str) -> List[str]:
    """Entries of ``directory`` starting with ``prefix``, dirs marked with ``/``."""
    out = []
    try:
        for entry in os.scandir(directory):
            name = entry.name
            if prefix:
                if not name.lower().startswith(prefix.lower()):
                    continue
            elif name.startswith("."):
                continue  # an empty fragment shouldn't open with dotfiles
            out.append(name + ("/" if entry.is_dir() else ""))
    except OSError:
        return []
    return sorted(out, key=str.lower)


def _rank(path: str, fragment: str) -> tuple:
    """Sort key: basename prefix, then basename hit, then shallow, then name."""
    name = path.rsplit("/", 1)[-1].lower()
    frag = fragment.lower()
    if name.startswith(frag):
        tier = 0
    elif frag in name:
        tier = 1
    else:
        tier = 2
    return (tier, path.count("/"), len(path), path.lower())


def completions(
    fragment: str, cwd: Optional[Path] = None, limit: int = 20
) -> List[Tuple[str, str]]:
    """``[(path to insert, meta)]`` for an ``@`` fragment, best match first.

    Two modes, because they answer different questions. A fragment with a
    separator (or ``~``/``/``) is a **path** being typed, so the last segment is
    completed from its parent directory — that also reaches outside the project.
    A bare fragment is a **name**, matched against the project index, so
    ``@chat_int`` finds ``src/mnemoai/client/ui/chat_interface.py`` without
    knowing where it lives. Never raises: a completion menu that throws takes the
    whole input down with it.
    """
    root = Path(cwd or Path.cwd())
    try:
        frag = fragment or ""
        if "/" in frag or frag.startswith("~") or frag.startswith("\\"):
            head, _, tail = frag.replace("\\", "/").rpartition("/")
            parent = Path(head).expanduser() if head else Path("/")
            if not parent.is_absolute():
                parent = root / parent
            prefix = f"{head}/" if head else ""
            return [
                (prefix + name, "dir" if name.endswith("/") else "file")
                for name in _dir_entries(parent, tail)
            ][:limit]
        # A bare fragment: the working directory's own entries first (what `ls`
        # would show, and what an empty fragment must offer), then anything the
        # index matches by name.
        local = [
            (name, "dir" if name.endswith("/") else "file")
            for name in _dir_entries(root, frag)
        ]
        if not frag:
            return local[:limit]
        seen = {name for name, _ in local}
        matched = sorted(
            (p for p in _index(root) if frag.lower() in p.lower() and p not in seen),
            key=lambda p: _rank(p, frag),
        )
        return (local + [(p, _parent_meta(p)) for p in matched])[:limit]
    except (OSError, ValueError) as e:
        logger.debug(f"Mention completion failed for {fragment!r}: {e}")
        return []


def _parent_meta(path: str) -> str:
    """Meta column for an index hit: where the file lives, since the label is a
    long path and the directory is what disambiguates two same-named files."""
    head, _, _tail = path.rpartition("/")
    return head or "."
