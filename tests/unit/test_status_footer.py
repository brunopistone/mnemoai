"""The context size is shown ONCE — in the footer on a TTY, else per turn.

The pinned UI carries `model · provider  dir  ▓▓░░ 90.1k · 9%` under the input,
so the old per-turn `[Context: N tokens]` line would only repeat it into
scrollback. Off a TTY there is no footer and that line is the only signal, so the
print survives — gated on one flag, in one place. These tests pin the gate, the
single print site, and the wiring that sets the flag.
"""

import inspect
from types import SimpleNamespace

from mnemoai.client.client import LangGraphClient
from mnemoai.client.ui.chat_interface import ChatInterface


def _client(**attrs):
    c = LangGraphClient.__new__(LangGraphClient)
    c.agent = None
    for k, v in attrs.items():
        setattr(c, k, v)
    return c


class TestPrintContextSize:
    def test_prints_when_there_is_no_footer(self, capsys):
        c = _client()
        c._count_context_tokens = lambda: 90096
        c._print_context_size()
        assert "[Context: 90096 tokens]" in capsys.readouterr().out

    def test_silent_while_the_footer_shows_it(self, capsys):
        c = _client(status_footer_active=True)
        c._count_context_tokens = lambda: 90096
        c._print_context_size()
        assert capsys.readouterr().out == ""

    def test_missing_flag_defaults_to_printing(self, capsys):
        """A client built before the UI ran (or by a test) still reports."""
        c = _client()
        assert not hasattr(c, "status_footer_active")
        c._count_context_tokens = lambda: 12
        c._print_context_size()
        assert "[Context: 12 tokens]" in capsys.readouterr().out

    def test_the_flag_defaults_off_on_a_real_client(self):
        src = inspect.getsource(LangGraphClient.__init__)
        assert "self.status_footer_active: bool = False" in src


class TestSinglePrintSite:
    """Every path that reported the context size goes through the gate."""

    def test_no_ungated_context_print_remains(self):
        src = inspect.getsource(LangGraphClient)
        printed = [
            line for line in src.split("\n")
            if "[Context:" in line and "print(" in line
        ]
        # Exactly one: the f-string inside _print_context_size itself.
        assert len(printed) == 1, printed
        assert printed[0] in inspect.getsource(
            LangGraphClient._print_context_size
        )

    def test_turn_resume_and_load_all_use_it(self):
        for name in ("query", "resume_session", "load_conversation"):
            src = inspect.getsource(getattr(LangGraphClient, name))
            assert "_print_context_size()" in src, name


class TestResumeSaysNothingAboutTheSummary:
    """A resume reads the same whether or not the session was compacted.

    The old notice ("N earlier messages carried as a summary, as they were when
    this session ended") described an internal detail the user can't act on — and
    the conversation above it was replayed in full anyway, so it contradicted what
    was on screen.
    """

    def test_the_notice_is_gone(self):
        src = inspect.getsource(LangGraphClient.resume_session)
        assert "carried" not in src
        assert "compacted_away" not in src


class TestPinnedLoopWiring:
    def test_the_pinned_loop_supplies_a_footer(self):
        assert "footer_text=" in inspect.getsource(ChatInterface._run_pinned_loop)

    def test_the_plain_loop_does_not(self):
        """Off a TTY nothing paints a footer, so the per-turn line must stay."""
        src = inspect.getsource(ChatInterface._plain_loop)
        assert "status_footer_active" not in src

    def test_the_flag_is_set_at_construction_from_the_tty_check(self, monkeypatch):
        """`--resume` replays BEFORE the loop starts, so the loop is too late.

        `main._resume_session` builds the ChatInterface, then restores — and the
        restore printed the context line while the flag was still False, so a
        resumed session showed it right above the footer that already had it.
        """
        import mnemoai.client.ui.chat_interface as ci

        for on_tty in (True, False):
            monkeypatch.setattr(ci, "_dialog_is_tty", lambda: on_tty)
            client = SimpleNamespace()
            ci.ChatInterface(client)
            assert client.status_footer_active is on_tty
