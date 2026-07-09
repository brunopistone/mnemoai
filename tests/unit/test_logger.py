"""Unit tests for the color log formatter (utils/logger._ColorFormatter).

Errors/warnings render in color on a TTY; everything stays plain when the
stream is redirected (so log files don't get ANSI escape codes).
"""

import io
import logging
import sys

from mnemoai.utils.logger import _RESET, _ColorFormatter, _NewlineGuardHandler

FMT = "%(name)s - %(levelname)s - %(message)s"


def _record(level):
    return logging.LogRecord("ai_app", level, "f", 1, "msg", None, None)


def test_error_is_red_on_tty():
    out = _ColorFormatter(FMT, use_color=True).format(_record(logging.ERROR))
    assert out.startswith("\033[91m") and out.endswith(_RESET)


def test_warning_is_yellow_on_tty():
    out = _ColorFormatter(FMT, use_color=True).format(_record(logging.WARNING))
    assert out.startswith("\033[93m") and out.endswith(_RESET)


def test_info_uncolored_on_tty():
    out = _ColorFormatter(FMT, use_color=True).format(_record(logging.INFO))
    assert "\033[" not in out


def test_no_color_when_not_tty():
    # Redirected/piped output must stay free of ANSI codes at every level.
    plain = _ColorFormatter(FMT, use_color=False)
    for level in (logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
        assert "\033[" not in plain.format(_record(level))


class TestNewlineGuardFollowsStderr:
    """The handler must resolve sys.stderr at emit time, not construction time.

    The pinned UI replaces sys.stderr with prompt_toolkit's patch_stdout proxy
    so logs render ABOVE the input; a handler bound to the original stderr would
    bypass that patch and stomp the pinned prompt. Regression test: swapping
    sys.stderr after the handler is built must route the record to the NEW
    stream.
    """

    def test_emit_writes_to_current_sys_stderr(self):
        original_stderr = io.StringIO()  # stream captured at construction
        handler = _NewlineGuardHandler(original_stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))

        swapped = io.StringIO()  # the "patch_stdout proxy" installed later
        saved = sys.stderr
        sys.stderr = swapped
        try:
            handler.emit(_record(logging.WARNING))
        finally:
            sys.stderr = saved

        assert "msg" in swapped.getvalue()  # went to the live stream
        assert original_stderr.getvalue() == ""  # NOT the construction-time one
