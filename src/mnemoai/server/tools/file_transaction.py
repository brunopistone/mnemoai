"""Serialized, failure-safe mutations of user files."""

import inspect
import json
import os
import shutil
import stat
import tempfile
import threading
from contextvars import ContextVar
from functools import wraps
from weakref import WeakValueDictionary

from mnemoai.utils.path_utils import normalize_path

_locks = WeakValueDictionary()
_registry_lock = threading.Lock()
_baseline = ContextVar("file_mutation_baseline", default=None)


def _version(path):
    try:
        info = os.stat(path)
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    except FileNotFoundError:
        return None


def serialized_mutation(path_parameter, resolve=normalize_path):
    """Hold one canonical-path lock across the gate, read, transform and write."""
    def decorate(fn):
        signature = inspect.signature(fn)

        @wraps(fn)
        def mutate(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            path = os.path.realpath(resolve(bound.arguments[path_parameter]))
            with _registry_lock:
                lock = _locks.setdefault(os.path.normcase(path), threading.RLock())
            with lock:
                bound.arguments[path_parameter] = path
                try:
                    token = _baseline.set((path, _version(path)))
                except OSError as exc:
                    return json.dumps({"error": True, "message": str(exc)})
                try:
                    return fn(*bound.args, **bound.kwargs)
                finally:
                    _baseline.reset(token)
        return mutate
    return decorate


def write_bytes(path, data):
    """Encode upstream, stage bytes beside the target, then publish atomically."""
    target = os.path.realpath(path)  # follow symlinks, never replace the link itself
    expected = _baseline.get()
    before = _version(target)
    if expected is not None and expected != (target, before):
        raise OSError("File changed during this edit; read it again before retrying.")
    info = os.stat(target) if before is not None else None
    if info is not None and not os.access(target, os.W_OK):
        raise PermissionError(f"File is not writable: {target}")
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=parent, prefix=".mnemoai-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            if info is not None:
                # Preserve permissions, timestamps, flags/xattrs where supported.
                # Ownership changes must succeed before replacing the original.
                if hasattr(os, "fchown"):
                    current = os.fstat(stream.fileno())
                    if (current.st_uid, current.st_gid) != (info.st_uid, info.st_gid):
                        os.fchown(stream.fileno(), info.st_uid, info.st_gid)
                shutil.copystat(target, temporary)
                os.fchmod(stream.fileno(), stat.S_IMODE(info.st_mode))
                os.utime(temporary, None)  # a real edit gets a fresh mtime
        if _version(target) != before:
            raise OSError("File changed during this edit; read it again before retrying.")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
