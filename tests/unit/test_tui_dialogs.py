"""Unit tests for the TUI dialog helpers' non-TTY fallbacks.

select_from_list and confirm_inline present prompt_toolkit dialogs on a TTY but
degrade to plain input() when stdin isn't a TTY (pipes / CI / tests). These tests
drive the non-TTY branch with input() mocked — no modal, no terminal.
"""

import builtins
import sys

import pytest

from mnemoai.client.ui import tui


@pytest.fixture
def not_a_tty(monkeypatch):
    # The helpers do `import sys` locally, so patch the real sys.std* streams.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)


class TestSelectFromList:
    OPTIONS = [("/a/one.json", "one"), ("/a/two.json", "two")]

    def test_empty_returns_none(self):
        assert tui.select_from_list("Pick", []) is None

    def test_pick_by_number(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda *_: "2")
        assert tui.select_from_list("Pick", self.OPTIONS) == "/a/two.json"

    def test_blank_cancels(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda *_: "")
        assert tui.select_from_list("Pick", self.OPTIONS) is None

    def test_out_of_range_cancels(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda *_: "9")
        assert tui.select_from_list("Pick", self.OPTIONS) is None

    def test_non_number_cancels(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda *_: "abc")
        assert tui.select_from_list("Pick", self.OPTIONS) is None

    def test_eof_cancels(self, not_a_tty, monkeypatch):
        def _raise(*_):
            raise EOFError

        monkeypatch.setattr(builtins, "input", _raise)
        assert tui.select_from_list("Pick", self.OPTIONS) is None


class TestConfirmInline:
    @pytest.mark.parametrize("answer,expected", [
        ("y", True), ("yes", True), ("Y", True),
        ("n", False), ("no", False), ("", False), ("garbage", False),
    ])
    def test_answers(self, not_a_tty, monkeypatch, answer, expected):
        monkeypatch.setattr(builtins, "input", lambda *_: answer)
        assert tui.confirm_inline("Clear?") is expected

    def test_eof_is_no(self, not_a_tty, monkeypatch):
        def _raise(*_):
            raise EOFError

        monkeypatch.setattr(builtins, "input", _raise)
        assert tui.confirm_inline("Clear?") is False
