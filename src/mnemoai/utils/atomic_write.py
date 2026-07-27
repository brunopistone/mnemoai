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
never a partial one. Concurrent writers become last-writer-wins, which is the
intended semantics for these files (they're caches of learned state, not a
ledger) -- the point is that the loser can't corrupt the winner.

The temp file is created in the SAME directory as the target because
``os.replace`` is only atomic within one filesystem.
"""

import json
import os
import tempfile
from typing import Any


def atomic_write_text(path: str, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + ``os.replace``).

    Args:
        path: Destination file. Its parent directory is created if absent.
        text: Full file contents.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            # Flush our buffers AND the OS page cache before the swap: without
            # this the rename can land before the bytes do, so a power loss
            # leaves an atomically-renamed but EMPTY file.
            f.flush()
            os.fsync(f.fileno())
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
