"""Unit tests for `/copy` — what gets copied, and how it reaches the clipboard.

No real clipboard is touched: the helper lookup and the terminal write are both
stubbed, so these run identically on a headless CI box and on a desktop.
"""

import base64
import subprocess
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from mnemoai.client.ui import clipboard

ESC = "\033"
BEL = "\007"


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


@pytest.fixture(autouse=True)
def _no_real_clipboard(monkeypatch):
    """No test may reach a helper binary or the real terminal by accident."""
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: None)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")


def use_stdout(monkeypatch, stream):
    """Patch the MODULE's ``sys``, not ``sys.stdout``.

    pytest's capture re-assigns ``sys.stdout`` when it resumes at the start of the
    call phase, which silently undoes a fixture-level patch of the attribute.
    """
    monkeypatch.setattr(clipboard, "sys", SimpleNamespace(stdout=stream))


def fake_helper(monkeypatch, available=("pbcopy",), returncode=0, boom=None):
    """Make ``available`` helpers exist and record what they were handed."""
    calls = []
    monkeypatch.setattr(
        clipboard.shutil, "which", lambda name: f"/bin/{name}" if name in available else None
    )

    def run(argv, input=None, text=None, capture_output=None, timeout=None):
        calls.append((argv, input))
        if boom:
            raise boom
        return SimpleNamespace(returncode=returncode, stdout="", stderr="")

    monkeypatch.setattr(clipboard.subprocess, "run", run)
    return calls


class TestAnswerText:
    def test_the_last_answer_wins(self):
        messages = [
            HumanMessage(content="q1"),
            AIMessage(content="first"),
            HumanMessage(content="q2"),
            AIMessage(content="second"),
        ]
        assert clipboard.answer_text(messages) == "second"

    def test_a_streamed_chunk_is_an_answer(self):
        # AIMessageChunk is an AIMessage subclass; a class-NAME check misses it,
        # which is the bug that once exported a transcript with no answers in it.
        assert clipboard.answer_text([AIMessageChunk(content="streamed")]) == "streamed"

    def test_tool_calls_and_results_are_not_answers(self):
        messages = [
            AIMessage(content="the answer"),
            AIMessage(content="", tool_calls=[
                {"name": "fs_read", "args": {"path": "x"}, "id": "1"}
            ]),
            ToolMessage(content="file contents", tool_call_id="1"),
        ]
        assert clipboard.answer_text(messages) == "the answer"

    def test_block_list_content_is_flattened(self):
        msg = AIMessage(content=[{"type": "text", "text": "block "}, {"type": "text", "text": "text"}])
        assert clipboard.answer_text([msg]) == "block text"

    def test_back_reaches_an_earlier_answer(self):
        messages = [AIMessage(content="older"), AIMessage(content="newer")]
        assert clipboard.answer_text(messages, back=2) == "older"
        assert clipboard.answer_text(messages, back=3) == ""

    def test_back_below_one_is_the_latest(self):
        assert clipboard.answer_text([AIMessage(content="x")], back=0) == "x"

    def test_no_messages(self):
        assert clipboard.answer_text([]) == ""
        assert clipboard.answer_text(None) == ""


class TestLastCodeBlock:
    def test_the_last_block_and_its_language(self):
        text = "intro\n```py\nfirst()\n```\nmiddle\n```bash\nls -la\n```\nend"
        assert clipboard.last_code_block(text) == ("ls -la", "bash")

    def test_a_tilde_fence_works(self):
        assert clipboard.last_code_block("~~~js\nlet a = 1\n~~~") == ("let a = 1", "js")

    def test_a_block_with_no_language(self):
        assert clipboard.last_code_block("```\nplain\n```") == ("plain", "")

    def test_an_unterminated_fence_still_yields_its_code(self):
        # A turn cut short mid-block: the code is on the screen, so refusing to
        # copy it would be a lie about what's there.
        assert clipboard.last_code_block("```py\nhalf_written(")[0] == "half_written("

    def test_indentation_inside_the_block_is_preserved(self):
        code, _ = clipboard.last_code_block("```py\ndef f():\n    return 1\n```")
        assert code == "def f():\n    return 1"

    def test_an_info_string_with_extras_takes_the_language_only(self):
        _, lang = clipboard.last_code_block("```python title=x.py\npass\n```")
        assert lang == "python"

    def test_no_block(self):
        assert clipboard.last_code_block("just prose") == ("", "")
        assert clipboard.last_code_block("") == ("", "")


class TestOsc52:
    def test_the_payload_is_base64_of_the_text(self):
        seq = clipboard.osc52_sequence("hello")
        assert seq.startswith(f"{ESC}]52;c;")
        assert seq.endswith(BEL)
        payload = seq[len(f"{ESC}]52;c;"):-1]
        assert base64.b64decode(payload).decode() == "hello"

    def test_non_ascii_survives(self):
        seq = clipboard.osc52_sequence("héllo ✓")
        payload = seq[len(f"{ESC}]52;c;"):-1]
        assert base64.b64decode(payload).decode("utf-8") == "héllo ✓"

    def test_inside_tmux_the_sequence_is_wrapped(self, monkeypatch):
        monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
        seq = clipboard.osc52_sequence("x")
        assert seq.startswith(f"{ESC}Ptmux;{ESC}{ESC}]52;")
        assert seq.endswith(f"{ESC}\\")

    def test_inside_screen_the_sequence_is_wrapped(self, monkeypatch):
        monkeypatch.setenv("TERM", "screen.xterm-256color")
        seq = clipboard.osc52_sequence("x")
        assert seq.startswith(f"{ESC}P{ESC}]52;")

    def test_an_oversized_payload_is_refused_not_truncated(self, monkeypatch):
        tty = _FakeTTY()
        use_stdout(monkeypatch, tty)
        monkeypatch.setattr(clipboard, "_MAX_OSC52_CHARS", 10)
        ok, how = clipboard.copy("x" * 50)
        assert not ok and how == ""
        assert tty.text == ""  # nothing was sent

    def test_nothing_is_written_off_a_tty(self, monkeypatch):
        tty = _FakeTTY(tty=False)
        use_stdout(monkeypatch, tty)
        assert clipboard.copy("x") == (False, "")
        assert tty.text == ""

    def test_a_failed_write_is_reported_not_raised(self, monkeypatch):
        use_stdout(monkeypatch, _FakeTTY(fail=True))
        assert clipboard.copy("x") == (False, "")


class TestCopyTransportOrder:
    def test_a_local_helper_is_preferred_and_gets_the_text(self, monkeypatch):
        calls = fake_helper(monkeypatch, available=("pbcopy",))
        tty = _FakeTTY()
        use_stdout(monkeypatch, tty)
        ok, how = clipboard.copy("payload")
        assert (ok, how) == (True, "pbcopy")
        assert calls == [(["/bin/pbcopy"], "payload")]
        assert tty.text == ""  # the terminal was not asked

    def test_helpers_are_tried_in_order(self, monkeypatch):
        calls = fake_helper(monkeypatch, available=("xclip", "xsel"))
        use_stdout(monkeypatch, _FakeTTY())
        ok, how = clipboard.copy("x")
        assert (ok, how) == (True, "xclip")
        assert calls[0][0] == ["/bin/xclip", "-selection", "clipboard"]

    def test_a_helper_that_fails_falls_through_to_the_terminal(self, monkeypatch):
        fake_helper(monkeypatch, available=("xclip",), returncode=1)
        tty = _FakeTTY()
        use_stdout(monkeypatch, tty)
        ok, how = clipboard.copy("x")
        assert (ok, how) == (True, "the terminal")
        assert f"{ESC}]52;c;" in tty.text

    def test_a_helper_that_times_out_falls_through(self, monkeypatch):
        fake_helper(
            monkeypatch,
            available=("xclip",),
            boom=subprocess.TimeoutExpired(cmd="xclip", timeout=5),
        )
        use_stdout(monkeypatch, _FakeTTY())
        assert clipboard.copy("x") == (True, "the terminal")

    def test_over_ssh_the_terminal_goes_first(self, monkeypatch):
        # A helper on the far end would copy to the clipboard of a machine nobody
        # is sitting at — the whole reason OSC 52 exists.
        monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 22 5.6.7.8 22")
        calls = fake_helper(monkeypatch, available=("xclip",))
        tty = _FakeTTY()
        use_stdout(monkeypatch, tty)
        ok, how = clipboard.copy("x")
        assert (ok, how) == (True, "the terminal")
        assert calls == []

    def test_empty_text_is_not_a_copy(self, monkeypatch):
        use_stdout(monkeypatch, _FakeTTY())
        assert clipboard.copy("") == (False, "")


def client_with(*answers, code=None):
    """A stand-in client whose agent history ends with ``answers``."""
    messages = []
    for text in answers:
        messages.append(HumanMessage(content="q"))
        messages.append(AIMessage(content=text))
    if code is not None:
        messages.append(AIMessage(content=code))
    return SimpleNamespace(agent=SimpleNamespace(messages=messages))


class TestReport:
    def test_it_copies_the_answer_and_says_what_it_did(self, monkeypatch):
        calls = fake_helper(monkeypatch)
        use_stdout(monkeypatch, _FakeTTY())
        out = clipboard.report(client_with("the answer"))
        assert "Copied the answer" in out
        assert "via pbcopy" in out
        assert calls[0][1] == "the answer"

    def test_the_notice_names_the_size(self, monkeypatch):
        fake_helper(monkeypatch)
        use_stdout(monkeypatch, _FakeTTY())
        out = clipboard.report(client_with("a\nb\nc"))
        assert "3 lines" in out and "5 chars" in out

    def test_a_one_line_answer_is_singular(self, monkeypatch):
        fake_helper(monkeypatch)
        use_stdout(monkeypatch, _FakeTTY())
        assert "1 line," in clipboard.report(client_with("x"))

    def test_copy_code_narrows_to_the_last_block(self, monkeypatch):
        calls = fake_helper(monkeypatch)
        use_stdout(monkeypatch, _FakeTTY())
        client = client_with(code="prose\n```sh\nmake test\n```\nmore prose")
        out = clipboard.report(client, "code")
        assert calls[0][1] == "make test"
        assert "the last sh block" in out

    def test_copy_code_without_a_block_says_so(self, monkeypatch):
        fake_helper(monkeypatch)
        use_stdout(monkeypatch, _FakeTTY())
        out = clipboard.report(client_with("no code here"), "code")
        assert "no code block" in out

    def test_a_numeric_argument_reaches_an_earlier_answer(self, monkeypatch):
        calls = fake_helper(monkeypatch)
        use_stdout(monkeypatch, _FakeTTY())
        out = clipboard.report(client_with("older", "newer"), "2")
        assert calls[0][1] == "older"
        assert "answer -2" in out

    def test_an_unknown_option_is_rejected_with_the_usage(self, monkeypatch):
        out = clipboard.report(client_with("x"), "sideways")
        assert "/copy [code|N]" in out

    def test_an_empty_conversation_says_there_is_nothing_to_copy(self, monkeypatch):
        out = clipboard.report(SimpleNamespace(agent=SimpleNamespace(messages=[])))
        assert "Nothing to copy" in out

    def test_asking_further_back_than_the_history_says_so(self, monkeypatch):
        out = clipboard.report(client_with("only one"), "3")
        assert "3 answers back" in out

    def test_no_clipboard_at_all_points_at_export(self, monkeypatch):
        use_stdout(monkeypatch, _FakeTTY(tty=False))  # no helper, no terminal
        out = clipboard.report(client_with("x"))
        assert "Could not reach a clipboard" in out
        assert "/export" in out

    def test_a_missing_agent_is_tolerated(self):
        assert "Nothing to copy" in clipboard.report(SimpleNamespace(agent=None))
        assert "Nothing to copy" in clipboard.report(object())


class TestWiring:
    def test_copy_is_a_builtin_command(self):
        from mnemoai.client.ui.chat_interface import ChatInterface
        from mnemoai.client.user_commands import BUILTIN_COMMANDS

        assert "copy" in BUILTIN_COMMANDS
        assert any(cmd == "/copy" for cmd, _ in ChatInterface._COMMANDS)
        assert any(
            cmd.startswith("/copy")
            for _, items in ChatInterface._COMMAND_GROUPS
            for cmd, _ in items
        )
