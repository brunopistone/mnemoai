"""In-process read-state registry: read-before-write + staleness gate.

The MCP server is one long-lived stdio subprocess, so this module-level dict
persists across tool calls for the whole session. We record the on-disk version
of every file the model reads (via ``fs_read``) and, before mutating an
EXISTING file (``fs_write`` / ``file_edit``), require that it was read at its
current identity/modification stamp. It refuses to edit a file that was never read, or that changed
on disk since it was last read — the model would otherwise clobber content it
never saw.

This is a server-side gate, independent of the client's confirmation prompt and
the ``path_policy`` catastrophic-write floor. It never prompts; it returns a
normal tool-error payload, so a non-TTY / directly-driven server can't deadlock.
Creating a brand-new file (absent on disk) needs no prior read — and neither does
filling an EMPTY one: the gate exists to protect content, and a zero-byte file
has none. Requiring a read there deadlocked the file outright, since a reader has
no lines to hand back and reports the empty range as an error, so the read was
never recorded and every write was refused forever.

Config-independent (only ``os`` + logger) so it stays unit-testable.
"""

import os
from typing import Optional

from mnemoai.utils.logger import logger

# A preserved mtime alone does not identify a file after replacement.
FileVersion = tuple[int, int, int, int, int]
_read_versions: dict[str, FileVersion] = {}


def _key(path: str) -> str:
    """Canonical registry key (symlinks/.. resolved) so different spellings of
    the same file — fs_read's normalize_path vs fs_write's _resolve_path — map
    to one entry. normcase additionally folds case on Windows; it is a no-op
    on POSIX."""
    return os.path.normcase(os.path.realpath(path))


def _stat(path: str) -> Optional[os.stat_result]:
    """``os.stat`` of ``path``, or None if it can't be stat'd (absent/unreadable)."""
    try:
        return os.stat(path)
    except OSError:
        return None


def _version(info: os.stat_result) -> FileVersion:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def current_version(path: str) -> Optional[FileVersion]:
    """Capture the identity and modification stamp of the file being read."""
    info = _stat(path)
    return _version(info) if info is not None else None


def record_read(path: str, expected_version: Optional[FileVersion] = None) -> bool:
    """Remember a read only when it still refers to the captured version.

    Called after every successful read (and after a successful write, so an
    immediately-following edit in the same turn isn't flagged stale). A path
    that can't be stat'd is silently ignored.
    """
    version = current_version(path)
    if version is None or (expected_version is not None and version != expected_version):
        return False
    # Store the observed version, not another later stat which could bless a
    # concurrent writer's unseen content.
    _read_versions[_key(path)] = expected_version if expected_version is not None else version
    return True


def check_write_allowed(path: str) -> Optional[dict]:
    """Gate a mutation of ``path``. Returns None if allowed, else an error payload.

    - Brand-new file (absent on disk): allowed — no prior read required.
    - Existing but EMPTY file: allowed — there is no content to clobber, and
      demanding a read of nothing is a demand that can't be met.
    - Existing file never read this session: blocked ("read it first").
    - Existing file changed on disk since last read: blocked ("read it again").
    """
    info = _stat(path)
    if info is None:
        # Doesn't exist yet (or unreadable) -> treated as a fresh create.
        return None
    if info.st_size == 0:
        # Same case as a create, in every way that matters here: nothing to lose.
        return None

    current = _version(info)
    seen = _read_versions.get(_key(path))
    if seen is None:
        logger.info("read-gate: %s modified without a prior read", path)
        return {
            "error": True,
            "must_read_first": True,
            "message": (
                f"Refusing to modify {path}: read it first with fs_read so the "
                "edit is based on its current content."
            ),
            "next_steps": [
                f'Call fs_read(path="{path}") to load the file',
                "Then retry this change against the exact text you just read",
            ],
        }
    if seen != current:
        logger.info("read-gate: %s changed on disk since last read", path)
        return {
            "error": True,
            "stale_read": True,
            "message": (
                f"Refusing to modify {path}: it changed on disk since you last "
                "read it. Read it again so your edit is based on the current "
                "content."
            ),
            "next_steps": [
                f'Call fs_read(path="{path}") to reload the current content',
                "Re-derive your edit from the fresh read, then retry",
            ],
        }
    return None


def forget(path: str) -> None:
    """Drop any recorded read-state for ``path`` (e.g. after deletion)."""
    _read_versions.pop(_key(path), None)


def reset() -> None:
    """Clear all recorded read-state (new session / test isolation)."""
    _read_versions.clear()
