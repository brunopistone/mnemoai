"""Shell session state shared by the command tools: which shell, and where it runs.

Two facts every shell tool needs and neither should re-derive:

* **Which shell.** ``subprocess(..., shell=True)`` runs ``/bin/sh``, which is not
  bash everywhere — dash on Debian/Ubuntu, bash in POSIX mode on macOS — so
  ``[[ ]]``, arrays, ``source``, and process substitution fail or behave
  differently depending on the host. The tools are documented as bash, so bash is
  what runs them (``executable=``), falling back to the default shell only where
  no bash exists.

* **Where it runs.** Each tool call is a fresh shell, so a ``cd`` used to
  evaporate: the model would ``cd project`` in one call and the next command ran
  back at the spawn directory, silently operating on the wrong tree. The
  directory a command ENDS in becomes the starting directory of the next one, as
  in an interactive session.

The tracked directory is process-wide (one shell session per server) and
therefore lock-guarded: the server dispatches tool calls from several agents
concurrently, so this is read and written from many threads.

Pure state + path logic — no MCP, no LLM — so it is unit-testable on its own.
"""

import os
import shlex
import shutil
import threading
from typing import Optional

from mnemoai.utils.logger import logger

_lock = threading.Lock()
# The directory the next shell command starts in. None = not yet resolved; the
# first read seeds it from the process cwd (the directory the client launched in,
# inherited when the server was spawned).
_cwd: Optional[str] = None
_bash: Optional[str] = None
_bash_resolved = False

# Status variable + probe appended to a tracked command. Named unlikely-to-collide
# rather than unset afterwards: the shell exits right after, so it can't leak.
_STATUS_VAR = "__mnemoai_status"


def bash_path() -> Optional[str]:
    """Path to bash for ``executable=``, or None to accept the default shell.

    PATH first (that is the bash the user's own terminal runs — often a newer one
    than the system copy), then the system location. Resolved once per process.
    """
    global _bash, _bash_resolved
    if not _bash_resolved:
        _bash = shutil.which("bash") or ("/bin/bash" if os.path.exists("/bin/bash") else None)
        _bash_resolved = True
        if _bash is None:
            logger.warning("bash not found; shell tools fall back to /bin/sh")
    return _bash


def current_cwd() -> str:
    """Directory the next shell command should start in.

    Falls back to the process cwd if the tracked directory has since been deleted
    or renamed — otherwise one ``cd`` into a temp dir that later disappears would
    make every subsequent command fail to even start.
    """
    global _cwd
    with _lock:
        if _cwd is None:
            _cwd = os.getcwd()
        if not os.path.isdir(_cwd):
            logger.info("Tracked shell cwd %s is gone; resetting to %s", _cwd, os.getcwd())
            _cwd = os.getcwd()
        return _cwd


def set_cwd(path: str) -> bool:
    """Record ``path`` as the directory for subsequent commands; True if it took.

    Rejects anything that isn't an existing directory, so a garbled probe result
    can never strand the session in an unusable place.
    """
    global _cwd
    candidate = (path or "").strip()
    if not candidate or not os.path.isdir(candidate):
        return False
    with _lock:
        _cwd = candidate
    return True


def reset_cwd() -> None:
    """Forget the tracked directory (next read re-seeds from the process cwd)."""
    global _cwd
    with _lock:
        _cwd = None


def wrap_with_cwd_probe(command: str, probe_path: str) -> str:
    """``command`` plus a trailing ``pwd`` written to ``probe_path``.

    The probe runs after the command and the original exit status is re-raised, so
    neither the status nor stdout/stderr is disturbed — the directory is reported
    out of band instead of being parsed out of the output.

    Returns ``command`` unchanged when appending would corrupt it (a trailing line
    continuation would splice the probe into the command's own last line). No
    probe just means the directory doesn't move, which is the old behavior.
    """
    if not command or command.rstrip(" \t").endswith("\\"):
        return command
    return (
        f"{command}\n"
        f"{_STATUS_VAR}=$?\n"
        f"pwd > {shlex.quote(probe_path)} 2>/dev/null\n"
        f"exit ${_STATUS_VAR}\n"
    )


def read_cwd_probe(probe_path: str) -> Optional[str]:
    """The directory a probe recorded, or None if it never ran / is unusable.

    A command that ends the shell itself (``exit``, ``exec``) never reaches the
    probe; an empty or missing file is that case, not an error.
    """
    try:
        with open(probe_path, "r", encoding="utf-8", errors="replace") as f:
            # A pwd is one line; anything after it is noise from a command that
            # wrote to the file itself.
            value = f.readline().strip()
    except OSError:
        return None
    return value if value and os.path.isdir(value) else None
