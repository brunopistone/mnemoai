"""Unit tests for ChatInterface._dispatch command routing.

_dispatch is the shared command/query handler used by both the inline-TUI loop
and the non-TTY plain loop. These tests exercise its pure routing decisions
(the _EXIT sentinel, slash-command dispatch to the right handler) with a stub
client — no LLM, no prompt_toolkit app, no TTY.
"""

import pytest

from mnemoai.client.ui.chat_interface import ChatInterface


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


@pytest.fixture
def ci():
    c = ChatInterface.__new__(ChatInterface)
    c.client = _StubClient()
    return c


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
    # "Stopped" line (not just silently swallow the response).
    ci.client.query_return = "Operation was cancelled."
    ci._dispatch("do a long thing")
    out = capsys.readouterr().out
    assert "Stopped" in out


def test_normal_query_does_not_print_stopped(ci, capsys):
    ci.client.query_return = "here is your answer"
    ci._dispatch("a question")
    assert "Stopped" not in capsys.readouterr().out


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
