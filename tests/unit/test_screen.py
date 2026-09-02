"""Unit tests for the launch fresh screen and `/clear`'s wipe.

The property worth pinning is the DIFFERENCE between the two: a launch must not
erase what the shell left above it (it scrolls, so everything stays in
scrollback), while `/clear` must erase the scrollback as well. No real terminal
is involved — the writer takes the stream, so a pipe and a CI log are covered too.
"""

import shutil

import pytest

from mnemoai.client.ui import screen

ESC = "\033"


class _FakeTTY:
    """Stands in for stdout: records raw writes, claims to be a terminal."""

    def __init__(self, tty=True, fail=False):
        self.text = ""
        self._tty = tty
        self._fail = fail

    def write(self, text):
        if self._fail:
            raise OSError("closed")
        self.text += text

    def flush(self):
        pass

    def isatty(self):
        return self._tty


def test_fresh_sequence_scrolls_and_never_erases_scrollback():
    seq = screen.fresh_sequence(30)
    assert seq.startswith("\n" * 30)
    assert seq.endswith(f"{ESC}[H{ESC}[J")
    # 3J would drop the history, 2J the last screenful — both are data loss here.
    assert f"{ESC}[3J" not in seq
    assert f"{ESC}[2J" not in seq


def test_wipe_sequence_discards_scrollback():
    assert screen.wipe_sequence() == f"{ESC}[3J{ESC}[H{ESC}[2J"


@pytest.mark.parametrize("rows", [0, -5, None])
def test_fresh_sequence_always_scrolls_at_least_one_row(rows):
    assert screen.fresh_sequence(rows).startswith("\n")


def test_terminal_rows_clamps_an_implausible_height(monkeypatch):
    monkeypatch.setattr(
        shutil, "get_terminal_size", lambda *_: shutil.os.terminal_size((80, 99999))
    )
    assert screen.terminal_rows() == screen._MAX_ROWS


def test_terminal_rows_falls_back_when_the_size_is_unreadable(monkeypatch):
    def boom(*_):
        raise OSError("no tty")

    monkeypatch.setattr(shutil, "get_terminal_size", boom)
    assert screen.terminal_rows() == screen._FALLBACK_ROWS


def test_nothing_is_written_to_a_non_tty():
    out = _FakeTTY(tty=False)
    assert screen.fresh(out) is False
    assert screen.wipe(out) is False
    assert out.text == ""


def test_fresh_writes_to_a_tty():
    out = _FakeTTY()
    assert screen.fresh(out) is True
    assert out.text == screen.fresh_sequence(screen.terminal_rows())


def test_a_failing_stream_is_reported_not_raised():
    assert screen.fresh(_FakeTTY(fail=True)) is False


def test_stdout_is_resolved_at_call_time(monkeypatch):
    """pytest reassigns sys.stdout during the call phase, so a default arg lies."""
    out = _FakeTTY()
    monkeypatch.setattr(screen.sys, "stdout", out)
    assert screen.wipe() is True
    assert out.text == screen.wipe_sequence()
