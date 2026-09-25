"""Cross-process exclusion for short read/modify/replace transactions."""

import errno
import os
import threading
import time
from contextlib import contextmanager

if os.name == "nt":
    import msvcrt
else:
    import fcntl

_locks = {}
_guard = threading.Lock()


@contextmanager
def file_lock(path, timeout=5.0):
    path = os.path.realpath(path)
    with _guard:
        lock = _locks.setdefault(path, threading.RLock())
    deadline = time.monotonic() + timeout
    if not lock.acquire(timeout=max(0, timeout)):
        raise TimeoutError("Timed out waiting for the memory store lock")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a+b") as handle:
            if os.name == "nt":
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
            while True:
                try:
                    if os.name == "nt":
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as e:
                    if e.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Timed out waiting for the memory store lock") from e
                    time.sleep(min(0.05, remaining))
            try:
                yield
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        lock.release()
