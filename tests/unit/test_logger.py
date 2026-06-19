"""Unit tests for the color log formatter (utils/logger._ColorFormatter).

Errors/warnings render in color on a TTY; everything stays plain when the
stream is redirected (so log files don't get ANSI escape codes).
"""

import logging

from mnemoai.utils.logger import _RESET, _ColorFormatter

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
