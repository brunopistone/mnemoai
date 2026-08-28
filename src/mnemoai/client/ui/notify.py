"""Telling the user the terminal wants them back.

Everything else this app prints assumes someone is reading it. The two moments
that don't are a **long turn finishing** and a **prompt waiting for an answer**:
both happen minutes after the user looked away, and until now both were silent —
a finished turn added a line to a window nobody was watching, and a confirmation
prompt could sit there indefinitely while the work behind it was already done.

Two mechanisms, because terminals differ and neither is universal: the **bell**
(``\\a``), which every terminal has and which tmux/screen turn into a window
activity flag, and **OSC 9**, which iTerm2, WezTerm, kitty, Windows Terminal and
others raise as a real desktop notification. A terminal that doesn't understand
OSC 9 discards it. Inside tmux or screen an OSC sequence has to be wrapped in a
DCS passthrough or the multiplexer swallows it — the missing wrap is the classic
way this feature looks implemented and does nothing.

**A notification the user doesn't want is worse than none**, so: nothing off a
TTY (pipes, CI), nothing for a turn that finished faster than
``NOTIFY.AFTER_SECONDS`` (if you're still watching, a beep is an interruption),
nothing for a cancelled turn (the user's hand is on the keyboard), and a
minimum gap between two notifications so a multi-step task confirming ten writes
doesn't beep ten times.

Pure string building plus one guarded write, so the sequences are unit-testable
without a terminal.
"""

import os
import sys
import threading
import time
from typing import Optional

from mnemoai.client.ui.turn_view import format_duration
from mnemoai.utils.config import config

# Seconds a turn must last before its end is worth a notification. A code default
# so no config edit is needed to reach an existing install; `0` disables.
_DEFAULT_AFTER_SECONDS = 30.0

# Two notifications closer together than this collapse into one. A wave of
# confirmation prompts is one interruption, not eight.
_MIN_GAP_SECONDS = 10.0

_BELL = "\a"

_lock = threading.Lock()
_last_at: float = 0.0


def _cfg() -> dict:
    """The ``NOTIFY`` config section (never None)."""
    return config.get("NOTIFY", {}) or {}


def after_seconds() -> float:
    """Turn duration past which a finished turn notifies (``0`` = never)."""
    try:
        return max(0.0, float(_cfg().get("AFTER_SECONDS", _DEFAULT_AFTER_SECONDS)))
    except (TypeError, ValueError):
        return _DEFAULT_AFTER_SECONDS


def wrap_for_multiplexer(seq: str, tmux: bool = False, screen: bool = False) -> str:
    """Wrap an escape sequence so it reaches the OUTER terminal.

    tmux and screen consume sequences they don't recognize; a DCS passthrough
    (``ESC P tmux; … ESC \\``) hands them through. tmux additionally requires
    every inner ESC to be doubled. Without this the notification is emitted
    correctly and simply never arrives — the failure mode this whole module is
    about.

    Public because it is a fact about terminals, not about notifications:
    ``clipboard``'s OSC 52 needs exactly the same passthrough, and a second copy
    of it would be a second place to get the ESC doubling wrong.
    """
    if tmux:
        return f"\033Ptmux;{seq.replace(chr(27), chr(27) * 2)}\033\\"
    if screen:
        return f"\033P{seq}\033\\"
    return seq


def sequence(
    message: str = "",
    bell: bool = True,
    desktop: bool = True,
    tmux: bool = False,
    screen: bool = False,
) -> str:
    """The bytes to write for one notification (``""`` when both are off).

    The bell comes first: it is the part that works everywhere, so an OSC 9 a
    terminal chooses to ignore never costs the notification entirely.
    """
    out = _BELL if bell else ""
    if desktop and message:
        # OSC 9 ; <text> BEL — the widely-implemented one-shot notification.
        out += wrap_for_multiplexer(f"\033]9;{message}\007", tmux=tmux, screen=screen)
    return out


def _is_tty() -> bool:
    """True when stdout is a real terminal (so there is someone to notify)."""
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def notify(message: str = "", force: bool = False) -> bool:
    """Notify the user, unless it would be noise. Returns whether it fired.

    ``force`` skips only the minimum gap (used by nothing yet; the gap is the
    part that surprises), never the TTY check. Best-effort: a write that fails
    must not break the turn it was announcing.
    """
    global _last_at
    if not _is_tty():
        return False
    cfg = _cfg()
    bell = bool(cfg.get("BELL", True))
    desktop = bool(cfg.get("DESKTOP", True))
    if not (bell or desktop):
        return False
    now = time.monotonic()
    with _lock:
        if not force and _last_at and now - _last_at < _MIN_GAP_SECONDS:
            return False
        _last_at = now
    seq = sequence(
        message,
        bell=bell,
        desktop=desktop,
        tmux=bool(os.environ.get("TMUX")),
        screen=os.environ.get("TERM", "").startswith("screen")
        and not os.environ.get("TMUX"),
    )
    if not seq:
        return False
    try:
        sys.stdout.write(seq)
        sys.stdout.flush()
    except Exception:
        return False
    return True


def turn_end_message(elapsed: float) -> str:
    """The notification text for a finished turn: the wait, and where it ran.

    The directory is there because the notification arrives out of context — a
    desktop popup says nothing about WHICH terminal wants you back, and someone
    who waits on long turns is likely to have more than one open.
    """
    try:
        where = os.path.basename(os.getcwd()) or os.getcwd()
    except OSError:  # cwd deleted under us
        where = ""
    done = f"mnemoai · done in {format_duration(elapsed)}"
    return f"{done} · {where}" if where else done


def notify_turn_end(elapsed: float, summary: Optional[str] = None) -> bool:
    """Notify that a turn just finished, if it ran long enough to matter."""
    threshold = after_seconds()
    if not threshold or elapsed < threshold:
        return False
    return notify(summary or turn_end_message(elapsed))


def notify_waiting(what: str = "Waiting for your answer") -> bool:
    """Notify that the app has stopped and needs an answer.

    Not gated on a duration: a prompt blocks the work until it's answered, so
    the wait is the user's regardless of how long the turn had been running.
    """
    return notify(what)


def _reset_for_tests() -> None:
    """Forget the last-notification time (the module-level gap is process-wide)."""
    global _last_at
    with _lock:
        _last_at = 0.0
