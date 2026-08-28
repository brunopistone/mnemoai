"""Unit tests for ChatInterface._dispatch command routing.

_dispatch is the shared command/query handler used by both the inline-TUI loop
and the non-TTY plain loop. These tests exercise its pure routing decisions
(the _EXIT sentinel, slash-command dispatch to the right handler) with a stub
client — no LLM, no prompt_toolkit app, no TTY.
"""

import re

import pytest

from mnemoai.client import user_commands
from mnemoai.client.ui.chat_interface import ChatInterface
from mnemoai.client.user_commands import UserCommandStore


class _StubClient:
    """Minimal client capturing which methods _dispatch calls."""

    def __init__(self):
        self.plan_mode_active = False
        self.episodic_memory = None
        self.reflector = None
        self.session_id = "sess_20260101_000000"
        self.calls = []
        self.query_return = "ok"

    def clear_context(self):
        self.calls.append("clear_context")

    def query(self, q):
        self.calls.append(("query", q))
        return self.query_return

    def save_conversation(self, ts, path=None):
        self.calls.append(("save", ts, path))

    def compact_conversation(self, focus=""):
        self.calls.append(("compact", focus))
        return True

    def context_report(self):
        self.calls.append("context_report")
        return "Context window — 10 tokens"

    def files_report(self):
        self.calls.append("files_report")
        return "Files this session"

    def diff_report(self, path=""):
        self.calls.append(("diff_report", path))
        return "Uncommitted changes"

    def copy_last(self, arg=""):
        self.calls.append(("copy_last", arg))
        return "Copied the answer"


@pytest.fixture
def ci():
    c = ChatInterface.__new__(ChatInterface)
    c.client = _StubClient()
    return c


@pytest.fixture(autouse=True)
def _empty_app_home(tmp_path, monkeypatch):
    """No user-defined slash commands unless a test writes one.

    ``_dispatch`` consults ``~/.mnemoai/commands/`` for a slash line it doesn't
    recognize, so without this the suite would depend on what the developer
    running it has authored.
    """
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path / "home"))


def test_exit_and_quit_return_sentinel(ci):
    assert ci._dispatch("/exit") is ChatInterface._EXIT
    assert ci._dispatch("/quit") is ChatInterface._EXIT
    # case-insensitive
    assert ci._dispatch("/QUIT") is ChatInterface._EXIT


def test_plan_toggles_client_flag(ci):
    assert not ci.client.plan_mode_active
    assert ci._dispatch("/plan") is None
    assert ci.client.plan_mode_active is True
    ci._dispatch("/plan")
    assert ci.client.plan_mode_active is False


def test_save_routes_with_optional_path(ci):
    ci._dispatch("/save")
    ci._dispatch("/save /tmp/foo.json")
    saves = [c for c in ci.client.calls if isinstance(c, tuple) and c[0] == "save"]
    assert saves[0][2] is None
    assert saves[1][2] == "/tmp/foo.json"


def test_compact_passes_focus(ci):
    ci._dispatch("/compact keep the API design")
    assert ("compact", "keep the API design") in ci.client.calls


def test_plain_query_calls_client_query(ci):
    assert ci._dispatch("what time is it?") is None
    assert ("query", "what time is it?") in ci.client.calls


def test_blank_query_is_noop(ci):
    assert ci._dispatch("   ") is None
    assert not any(
        isinstance(c, tuple) and c[0] == "query" for c in ci.client.calls
    )


def test_cancelled_query_prints_stopped(ci, capsys):
    # A cancelled turn must resolve the transient "(cancelling…)" to a final
    # "stopped" line (not just silently swallow the response).
    ci.client.query_return = "Operation was cancelled."
    ci._dispatch("do a long thing")
    out = capsys.readouterr().out
    assert "stopped" in out


def test_normal_query_does_not_print_stopped(ci, capsys):
    ci.client.query_return = "here is your answer"
    ci._dispatch("a question")
    assert "stopped" not in capsys.readouterr().out


def test_every_turn_ends_with_a_done_marker(ci, capsys):
    # A streamed answer just stops mid-page: without this line nothing says the
    # turn is over, and the idle prompt looks the same either way.
    ci.client.query_return = "here is your answer"
    ci._dispatch("a question")
    out = re.sub(r"\033\[[0-9;]*m", "", capsys.readouterr().out)
    assert re.search(r"· done in \d+\w* · \d\d:\d\d", out)


def test_a_cancel_raised_into_the_frame_is_still_resolved(ci, capsys):
    # Esc between steps arrives as a bare KeyboardInterrupt, not as the
    # "Operation was cancelled." response — it must not be the one turn that ends
    # with nothing under the UI's transient "(cancelling…)".
    def _boom(_q):
        raise KeyboardInterrupt

    ci.client.query = _boom
    assert ci._dispatch("do a long thing") is None
    assert "stopped" in capsys.readouterr().out


def test_a_slash_command_gets_no_turn_marker(ci, capsys):
    # It marks a TURN, not every line the app prints.
    ci._dispatch("/context")
    assert "done in" not in capsys.readouterr().out


def test_help_prints_the_command_reference(ci, capsys):
    assert ci._dispatch("/help") is None
    out = capsys.readouterr().out
    assert "/context" in out and "Ctrl+J" in out
    # It must not reach the model — the box is rendered locally.
    assert not any(isinstance(c, tuple) and c[0] == "query" for c in ci.client.calls)


def test_context_prints_the_client_report(ci, capsys):
    assert ci._dispatch("/context") is None
    assert "context_report" in ci.client.calls
    assert "Context window" in capsys.readouterr().out


def test_files_prints_the_ledger_report(ci, capsys):
    assert ci._dispatch("/files") is None
    assert "files_report" in ci.client.calls
    assert "Files this session" in capsys.readouterr().out


def test_diff_passes_an_optional_path(ci, capsys):
    assert ci._dispatch("/diff") is None
    assert ci._dispatch("/diff  src/app.py ") is None
    assert ("diff_report", "") in ci.client.calls
    assert ("diff_report", "src/app.py") in ci.client.calls
    assert "Uncommitted changes" in capsys.readouterr().out


def test_copy_passes_its_argument(ci, capsys):
    assert ci._dispatch("/copy") is None
    assert ci._dispatch("/copy code") is None
    assert ("copy_last", "") in ci.client.calls
    assert ("copy_last", "code") in ci.client.calls
    assert "Copied the answer" in capsys.readouterr().out


def test_the_workspace_commands_never_reach_the_model(ci):
    # Each is answered locally: a report that costs a turn is a report nobody runs.
    for line in ("/files", "/diff", "/copy"):
        ci._dispatch(line)
    assert not any(isinstance(c, tuple) and c[0] == "query" for c in ci.client.calls)


class _BoomEpisodic:
    """Episodic memory whose storage always fails (e.g. ChromaDB code 1032)."""

    def store_episode(self, *a, **k):
        raise RuntimeError(
            "Query error: Database error: (code: 1032) attempt to write a "
            "readonly database"
        )


def test_episodic_storage_failure_does_not_crash_turn(ci, capsys):
    # The answer already succeeded; a best-effort episodic-store failure must be
    # swallowed (logged), NOT surfaced as a turn "Error:" line.
    ci.client.episodic_memory = _BoomEpisodic()
    ci.client.query_return = "here is your answer"
    # __store_current_episode_immediately checks tools/length + success markers;
    # simplest is to make the store itself raise, which it does above. Route
    # through the real immediate-storage branch:
    ci._dispatch("please do the thing")
    out = capsys.readouterr().out
    assert "Error:" not in out          # the turn did NOT error out
    assert "readonly database" not in out


def _user_command(tmp_path, name, body):
    """Point ci at a commands dir holding one authored command file."""
    root = tmp_path / "commands"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.md").write_text(body)
    user_commands._SCAN_CACHE.clear()
    return UserCommandStore(root=root)


def test_a_user_command_expands_into_the_prompt(ci, tmp_path):
    # The model never learns a command was involved: the expansion IS the prompt,
    # so the turn is an ordinary one afterwards.
    ci._user_commands = _user_command(tmp_path, "deploy", "Ship $ARGUMENTS carefully.")
    ci._dispatch("/deploy staging")
    assert ("query", "Ship staging carefully.") in ci.client.calls


def test_the_expansion_is_announced(ci, tmp_path, capsys):
    # The prompt the model answers is the file's body, so without this line the
    # transcript shows an answer to a question that appears nowhere.
    ci._user_commands = _user_command(tmp_path, "deploy", "Ship it.")
    ci._dispatch("/deploy")
    out = re.sub(r"\033\[[0-9;]*m", "", capsys.readouterr().out)
    assert "/deploy" in out and "deploy.md" in out


def test_a_builtin_always_wins_over_a_user_command(ci, tmp_path, capsys):
    # Expansion is checked AFTER every built-in, so a file can never shadow one.
    ci._user_commands = _user_command(tmp_path, "help", "not the command reference")
    assert ci._dispatch("/help") is None
    out = capsys.readouterr().out
    assert "Ctrl+J" in out                       # the real /help box rendered
    assert "not the command reference" not in out
    assert not any(isinstance(c, tuple) and c[0] == "query" for c in ci.client.calls)


def test_an_unknown_slash_line_is_sent_as_typed(ci, tmp_path):
    # An unknown /thing keeps its current meaning (prose), not an error.
    ci._user_commands = _user_command(tmp_path, "deploy", "Ship it.")
    ci._dispatch("/nope now")
    assert ("query", "/nope now") in ci.client.calls


def test_a_failing_store_leaves_the_line_as_typed(ci):
    class _Boom:
        def expand(self, line):
            raise OSError("commands dir is on fire")

    ci._user_commands = _Boom()
    ci._dispatch("/deploy")
    assert ("query", "/deploy") in ci.client.calls


def test_a_mention_attaches_the_file_to_the_prompt(ci, tmp_path, monkeypatch):
    # The point of a mention: the content is there whether or not the model would
    # have chosen to read it.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.md").write_text("remember the milk\n")
    ci._dispatch("summarize @notes.md")
    sent = next(c[1] for c in ci.client.calls if isinstance(c, tuple) and c[0] == "query")
    assert sent.startswith("summarize @notes.md")
    assert "remember the milk" in sent


def test_a_mention_is_announced(ci, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.md").write_text("a\nb\n")
    ci._dispatch("summarize @notes.md")
    out = re.sub(r"\033\[[0-9;]*m", "", capsys.readouterr().out)
    assert "@notes.md · 2 lines" in out


def test_a_typoed_mention_says_so_and_still_asks(ci, tmp_path, monkeypatch, capsys):
    # Attaching nothing looks exactly like attaching the right file, so the line
    # is the only way to tell — but the question still goes through.
    monkeypatch.chdir(tmp_path)
    ci._dispatch("summarize @notes.mdd")
    out = re.sub(r"\033\[[0-9;]*m", "", capsys.readouterr().out)
    assert "@notes.mdd · no such file" in out
    assert ("query", "summarize @notes.mdd") in ci.client.calls


def test_a_builtin_argument_is_never_expanded(ci, tmp_path, monkeypatch):
    # Mentions are expanded after every built-in, so a path typed as an argument
    # keeps its literal meaning.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.md").write_text("hi\n")
    ci._dispatch("/save @notes.md")
    saves = [c for c in ci.client.calls if isinstance(c, tuple) and c[0] == "save"]
    assert saves[0][2] == "@notes.md"


def test_a_failing_expansion_leaves_the_line_as_typed(ci, monkeypatch):
    from mnemoai.client.ui import chat_interface as ci_mod

    def _boom(_text):
        raise OSError("filesystem is on fire")

    monkeypatch.setattr(ci_mod.file_mentions, "expand", _boom)
    ci._dispatch("summarize @notes.md")
    assert ("query", "summarize @notes.md") in ci.client.calls
