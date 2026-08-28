"""Unit tests for the terminal notification (client/ui/notify.py).

The whole feature is a handful of escape bytes, and the way it fails is by
looking implemented and doing nothing — an OSC sequence eaten by tmux, a bell
sent into a pipe, a beep for a turn the user was watching anyway. So the
sequences are asserted byte for byte, and each suppression rule has a test of
its own.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from mnemoai.client.ui import notify

ESC = "\033"
BEL = "\007"


class _FakeTTY:
    """A stdout that claims to be a terminal and remembers what was written."""

    def __init__(self, fail: bool = False) -> None:
        self.written: list = []
        self.flushed = 0
        self._fail = fail

    def isatty(self) -> bool:
        return True

    def write(self, data: str) -> int:
        if self._fail:
            raise OSError("broken pipe")
        self.written.append(data)
        return len(data)

    def flush(self) -> None:
        self.flushed += 1

    @property
    def text(self) -> str:
        return "".join(self.written)


def _use_stdout(monkeypatch, stream) -> None:
    """Point the module at ``stream`` instead of the real stdout.

    Patches the module's ``sys`` reference rather than ``sys.stdout`` itself:
    pytest's capture re-assigns the real ``sys.stdout`` when it resumes at the
    start of the call phase, which silently undoes anything a FIXTURE set.
    """
    monkeypatch.setattr(notify, "sys", SimpleNamespace(stdout=stream))


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """No inherited gap, no inherited config, no inherited multiplexer env."""
    notify._reset_for_tests()
    monkeypatch.setattr(notify, "_cfg", lambda: {})
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    yield
    notify._reset_for_tests()


@pytest.fixture
def tty(monkeypatch):
    """stdout replaced by a fake terminal."""
    fake = _FakeTTY()
    _use_stdout(monkeypatch, fake)
    return fake


class TestSequence:
    def test_bell_plus_osc9(self):
        assert notify.sequence("done") == f"\a{ESC}]9;done{BEL}"

    def test_the_bell_comes_first(self):
        # The part every terminal implements leads, so an OSC 9 that gets
        # discarded never costs the notification entirely.
        assert notify.sequence("done").startswith("\a")

    def test_bell_only(self):
        assert notify.sequence("done", desktop=False) == "\a"

    def test_desktop_only(self):
        assert notify.sequence("done", bell=False) == f"{ESC}]9;done{BEL}"

    def test_no_message_means_no_desktop_half(self):
        # OSC 9 with empty text is a notification with nothing in it.
        assert notify.sequence("") == "\a"

    def test_both_off_is_nothing(self):
        assert notify.sequence("done", bell=False, desktop=False) == ""


class TestMultiplexerWrap:
    def test_plain_terminal_is_untouched(self):
        assert notify.wrap_for_multiplexer("x") == "x"

    def test_tmux_doubles_every_escape(self):
        # The doubling is the part that is easy to omit and impossible to notice
        # without a tmux session: without it the sequence is swallowed whole.
        out = notify.wrap_for_multiplexer(f"{ESC}]9;hi{BEL}", tmux=True)
        assert out == f"{ESC}Ptmux;{ESC}{ESC}]9;hi{BEL}{ESC}\\"

    def test_screen_wraps_without_doubling(self):
        out = notify.wrap_for_multiplexer(f"{ESC}]9;hi{BEL}", screen=True)
        assert out == f"{ESC}P{ESC}]9;hi{BEL}{ESC}\\"

    def test_tmux_wins_over_screen(self):
        out = notify.wrap_for_multiplexer("x", tmux=True, screen=True)
        assert out.startswith(f"{ESC}Ptmux;")

    def test_the_bell_is_not_wrapped(self):
        # Only the OSC half needs a passthrough; a bell reaches the outer
        # terminal on its own (and becomes the window's activity flag). Note the
        # OSC terminator IS a BEL, so the sequence legitimately holds two.
        out = notify.sequence("hi", tmux=True)
        assert out.startswith("\a")
        assert out[1:] == notify.wrap_for_multiplexer(
            f"{ESC}]9;hi{BEL}", tmux=True
        )


class TestNotify:
    def test_it_writes_and_flushes(self, tty):
        assert notify.notify("hello") is True
        assert tty.text == f"\a{ESC}]9;hello{BEL}"
        assert tty.flushed == 1

    def test_nothing_off_a_tty(self, monkeypatch):
        fake = _FakeTTY()
        monkeypatch.setattr(fake, "isatty", lambda: False)
        _use_stdout(monkeypatch, fake)
        assert notify.notify("hello") is False
        assert fake.written == []

    def test_a_stdout_without_isatty_is_not_a_tty(self, monkeypatch):
        _use_stdout(monkeypatch, object())
        assert notify.notify("hello") is False

    def test_both_mechanisms_off_sends_nothing(self, tty, monkeypatch):
        monkeypatch.setattr(notify, "_cfg", lambda: {"BELL": False, "DESKTOP": False})
        assert notify.notify("hello") is False
        assert tty.written == []

    def test_bell_disabled_keeps_the_desktop_half(self, tty, monkeypatch):
        monkeypatch.setattr(notify, "_cfg", lambda: {"BELL": False})
        assert notify.notify("hello") is True
        assert tty.text == f"{ESC}]9;hello{BEL}"

    def test_a_second_notification_inside_the_gap_is_dropped(self, tty):
        assert notify.notify("one") is True
        assert notify.notify("two") is False
        assert len(tty.written) == 1  # one write per notification

    def test_force_skips_the_gap(self, tty):
        assert notify.notify("one") is True
        assert notify.notify("two", force=True) is True

    def test_force_does_not_skip_the_tty_check(self, monkeypatch):
        monkeypatch.setattr(notify, "_is_tty", lambda: False)
        assert notify.notify("one", force=True) is False

    def test_past_the_gap_it_fires_again(self, tty, monkeypatch):
        monkeypatch.setattr(notify, "_MIN_GAP_SECONDS", 0.0)
        assert notify.notify("one") is True
        assert notify.notify("two") is True
        assert len(tty.written) == 2

    def test_a_failed_write_is_not_an_error(self, monkeypatch):
        _use_stdout(monkeypatch, _FakeTTY(fail=True))
        assert notify.notify("hello") is False  # best-effort, never raises

    def test_tmux_env_wraps_the_sequence(self, tty, monkeypatch):
        monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,123,0")
        notify.notify("hello")
        assert f"{ESC}Ptmux;" in tty.text

    def test_screen_term_wraps_the_sequence(self, tty, monkeypatch):
        monkeypatch.setenv("TERM", "screen.xterm-256color")
        notify.notify("hello")
        assert tty.text.startswith(f"\a{ESC}P{ESC}]9;")

    def test_inside_tmux_the_tmux_form_wins(self, tty, monkeypatch):
        # tmux sets TERM=screen-* itself, so the two conditions overlap and the
        # screen form (no ESC doubling) would be swallowed.
        monkeypatch.setenv("TERM", "screen")
        monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,123,0")
        notify.notify("hello")
        assert f"{ESC}Ptmux;" in tty.text


class TestAfterSeconds:
    def test_default_when_unset(self):
        assert notify.after_seconds() == notify._DEFAULT_AFTER_SECONDS

    def test_a_configured_value_wins(self, monkeypatch):
        monkeypatch.setattr(notify, "_cfg", lambda: {"AFTER_SECONDS": 5})
        assert notify.after_seconds() == 5.0

    def test_zero_disables(self, monkeypatch):
        monkeypatch.setattr(notify, "_cfg", lambda: {"AFTER_SECONDS": 0})
        assert notify.after_seconds() == 0.0

    def test_garbage_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setattr(notify, "_cfg", lambda: {"AFTER_SECONDS": "soon"})
        assert notify.after_seconds() == notify._DEFAULT_AFTER_SECONDS

    def test_a_negative_value_reads_as_disabled(self, monkeypatch):
        monkeypatch.setattr(notify, "_cfg", lambda: {"AFTER_SECONDS": -10})
        assert notify.after_seconds() == 0.0


class TestTurnEnd:
    def test_a_short_turn_stays_silent(self, tty, monkeypatch):
        monkeypatch.setattr(notify, "_cfg", lambda: {"AFTER_SECONDS": 30})
        assert notify.notify_turn_end(4.0) is False
        assert tty.written == []

    def test_a_long_turn_notifies(self, tty, monkeypatch):
        monkeypatch.setattr(notify, "_cfg", lambda: {"AFTER_SECONDS": 30})
        assert notify.notify_turn_end(95.0) is True
        assert "done in 1m35s" in tty.text

    def test_zero_threshold_never_notifies(self, tty, monkeypatch):
        monkeypatch.setattr(notify, "_cfg", lambda: {"AFTER_SECONDS": 0})
        assert notify.notify_turn_end(600.0) is False
        assert tty.written == []

    def test_an_explicit_summary_replaces_the_text(self, tty, monkeypatch):
        monkeypatch.setattr(notify, "_cfg", lambda: {"AFTER_SECONDS": 1})
        notify.notify_turn_end(60.0, summary="all 12 tests pass")
        assert f"{ESC}]9;all 12 tests pass{BEL}" in tty.text

    def test_the_message_names_the_directory(self):
        # A desktop popup arrives out of context: it has to say which terminal.
        msg = notify.turn_end_message(12.0)
        assert Path.cwd().name in msg
        assert "done in 12s" in msg


class TestWaiting:
    def test_it_is_not_duration_gated(self, tty, monkeypatch):
        # A prompt holds the work until it's answered, so the AFTER_SECONDS
        # threshold (which is about a turn's length) must not silence it.
        monkeypatch.setattr(notify, "_cfg", lambda: {"AFTER_SECONDS": 0})
        assert notify.notify_waiting() is True
        assert tty.text.startswith("\a")

    def test_it_carries_its_own_text(self, tty):
        notify.notify_waiting("mnemoai · a question is waiting")
        assert f"{ESC}]9;mnemoai · a question is waiting{BEL}" in tty.text


class TestDocumentedDefault:
    def test_the_shipped_config_quotes_the_code_default(self):
        # The examples are what users read; the code default is what applies
        # when nobody edits config. A drift between them is a docs bug.
        examples = list(
            (Path(notify.__file__).parents[2] / "utils").glob("config.yaml*.example")
        )
        assert examples, "no config examples found"
        for ex in examples:
            quoted = re.search(
                r"^NOTIFY:\s*\n\s*AFTER_SECONDS:\s*(\d+)",
                ex.read_text(),
                re.MULTILINE,
            )
            assert quoted, f"{ex.name} does not document NOTIFY.AFTER_SECONDS"
            assert int(quoted.group(1)) == notify._DEFAULT_AFTER_SECONDS, ex.name
