"""Crash-safe file writes for the learned-state files.

Every file the app learns into -- ``MEMORY.md``, ``playbook.json``,
``metrics.json``, ``{profile}.json``, the todo lists -- was written by
truncating the target and streaming into the open handle. Two failure modes
follow from that:

* a crash (or a full disk) mid-write leaves a truncated/half-written file, and
  the next start loses that state entirely rather than the last change,
* two app instances (terminal tabs) writing at once interleave, so the loser
  isn't just overwritten -- the file can end up structurally invalid.

Writing to a sibling temp file and ``os.replace``-ing it over the target makes
the swap atomic on POSIX: a reader sees either the old file or the new one,
never a partial one. The helper alone is last-writer-wins: callers performing
read/modify/write transactions (such as MEMORY.md and configuration editors)
must also hold a shared file lock so concurrent updates are not lost.

The temp file is created in the SAME directory as the target because
``os.replace`` is only atomic within one filesystem.
"""

import json
import os
import stat
import tempfile
from typing import Any


def atomic_write_text(path: str, text: str, *, preserve_mode: bool = False) -> None:
    """Write ``text`` to ``path`` atomically (temp file + ``os.replace``).

    Args:
        path: Destination file. Its parent directory is created if absent.
        text: Full file contents.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    mode = None
    if preserve_mode:
        try:
            mode = stat.S_IMODE(os.stat(path).st_mode)
        except FileNotFoundError:
            pass
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            # Flush our buffers AND the OS page cache before the swap: without
            # this the rename can land before the bytes do, so a power loss
            # leaves an atomically-renamed but EMPTY file.
            f.flush()
            os.fsync(f.fileno())
            if mode is not None:
                os.fchmod(f.fileno(), mode)
        os.replace(tmp, path)
    except BaseException:
        # Includes KeyboardInterrupt/SystemExit on purpose: a cancelled write
        # must not leave a stray .tmp behind next to the real file.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: str, data: Any, indent: int = 2) -> None:
    """Serialize ``data`` as JSON and write it atomically.

    Args:
        path: Destination file. Its parent directory is created if absent.
        data: Any JSON-serializable object.
        indent: ``json.dump`` indent (default 2, matching the existing files).
    """
    # Serialize BEFORE opening the temp file so a non-serializable object
    # raises without leaving a temp file behind.
    atomic_write_text(path, json.dumps(data, indent=indent))
