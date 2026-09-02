"""Handing the screen back to the app: the fresh screen at launch, `/clear`'s wipe.

Two moments want a blank screen and they want it for opposite reasons, which is
the whole reason this module exists rather than one escape string at each site.

**A launch must not destroy anything.** What sits above a starting app is the
user's own shell — a build log, the last session, the command they are about to
copy — so the launch reset SCROLLS that content away instead of erasing it: a
scroll always lands in scrollback, so the screen is blank and every line is one
flick of the wheel away. Erasing (``2J``) would take the last screenful with it,
and ``3J`` would take the whole history; either turns "start clean" into data
loss the user never asked for, in an app whose UI is built around preserving
scrollback and copy-paste.

**`/clear` is the opposite.** It says "forget this conversation", so it erases
the scrollback behind it too — anything less leaves the conversation the user
just discarded still scrollable above the prompt.

Pure sequence builders plus one TTY-gated writer, so the escapes are unit-testable
without a terminal and nothing writes them into a pipe or a CI log. ``sys`` is
read through the module at call time (never bound as a default argument) because
pytest reassigns ``sys.stdout`` between fixture setup and the call itself.
"""

import shutil
import sys
from typing import Optional, TextIO

_CURSOR_HOME = "\033[H"
_ERASE_BELOW = "\033[J"
_ERASE_SCREEN = "\033[2J"
_ERASE_SCROLLBACK = "\033[3J"

# A terminal that reports an implausible height must not turn a launch into
# thousands of blank scrollback lines.
_FALLBACK_ROWS = 24
_MAX_ROWS = 200


def terminal_rows() -> int:
    """Visible rows, clamped to a sane range (24 when the size is unknown)."""
    try:
        rows = int(shutil.get_terminal_size((80, _FALLBACK_ROWS)).lines)
    except Exception:  # noqa: BLE001 — an unreadable size must not break a launch
        rows = _FALLBACK_ROWS
    return max(1, min(rows or _FALLBACK_ROWS, _MAX_ROWS))


def fresh_sequence(rows: int) -> str:
    """Blank screen, cursor home, everything above preserved in scrollback.

    Scrolls by a full screen rather than erasing, which is what keeps the prior
    output reachable; the trailing erase-below only clears cells a smaller
    terminal may have left behind.
    """
    return "\n" * max(1, int(rows or 1)) + _CURSOR_HOME + _ERASE_BELOW


def wipe_sequence() -> str:
    """Blank screen with the scrollback behind it erased as well."""
    return _ERASE_SCROLLBACK + _CURSOR_HOME + _ERASE_SCREEN


def _is_tty(stream: Optional[TextIO]) -> bool:
    """Whether `stream` is a real terminal (never raises)."""
    try:
        return bool(stream is not None and stream.isatty())
    except Exception:  # noqa: BLE001 — a wrapped stream may not implement it
        return False


def write(sequence: str, stream: Optional[TextIO] = None) -> bool:
    """Write `sequence` when the stream is a terminal; report whether it went."""
    out = stream if stream is not None else sys.stdout
    if not _is_tty(out):
        return False
    try:
        out.write(sequence)
        out.flush()
    except Exception:  # noqa: BLE001 — a reset is cosmetic, never fatal
        return False
    return True


def fresh(stream: Optional[TextIO] = None) -> bool:
    """Start at the top of a blank screen, keeping prior output in scrollback."""
    return write(fresh_sequence(terminal_rows()), stream)


def wipe(stream: Optional[TextIO] = None) -> bool:
    """Blank the screen AND discard the scrollback behind it."""
    return write(wipe_sequence(), stream)
