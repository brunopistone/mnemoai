"""Unit tests for the confirmation 'allow for this session' option.

`_confirm_tool` gates bash/write/memory tools with a y/N/a prompt. Answering
'a' trusts that whole category for the rest of the session, so repeated calls
don't re-prompt. These tests drive the prompt with a stubbed input + a forced
TTY and a config that enables the gate.
"""

import builtins

import pytest

from mnemoai.client.agent.agent import LangGraphAgent


def _agent(monkeypatch, answers):
    """An agent whose _confirm_tool sees a TTY, an enabled gate, and queued input."""
    a = LangGraphAgent.__new__(LangGraphAgent)
    a._trusted_confirm_categories = set()
    a._stop_spinner = lambda: None

    # Force the interactive branch.
    import mnemoai.client.agent.agent as mod

    monkeypatch.setattr(mod.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(mod.config, "get", lambda k, d=None: True)

    queued = list(answers)

    def fake_input(prompt=""):
        return queued.pop(0)

    monkeypatch.setattr(builtins, "input", fake_input)
    return a


def test_yes_proceeds_once_but_reprompts(monkeypatch):
    a = _agent(monkeypatch, ["y", "n"])
    assert a._confirm_tool("execute_bash", {"command": "ls"}) is True
    # Not trusted -> a second call prompts again (and we answer 'n').
    assert a._confirm_tool("execute_bash", {"command": "rm x"}) is False
    assert "bash" not in a._trusted_confirm_categories


def test_allow_trusts_category_for_session(monkeypatch):
    a = _agent(monkeypatch, ["a"])  # only ONE input queued
    assert a._confirm_tool("execute_bash", {"command": "ls"}) is True
    assert "bash" in a._trusted_confirm_categories
    # Subsequent calls must NOT consume input (would IndexError if they did).
    assert a._confirm_tool("execute_bash", {"command": "wc -l f"}) is True
    assert a._confirm_tool("execute_bash", {"command": "cat f"}) is True


def test_allow_is_category_scoped(monkeypatch):
    # Trusting bash must not auto-approve a file write.
    a = _agent(monkeypatch, ["a", "n"])
    assert a._confirm_tool("execute_bash", {"command": "ls"}) is True
    assert "bash" in a._trusted_confirm_categories
    # write is a different category -> still prompts (answered 'n').
    assert a._confirm_tool("fs_write", {"path": "/x", "command": "create"}) is False


def test_no_declines(monkeypatch):
    a = _agent(monkeypatch, ["n"])
    assert a._confirm_tool("execute_bash", {"command": "ls"}) is False


class TestToolMarkerElision:
    """Tool-call markers middle-elide long values, keeping both ends."""

    def test_short_value_unchanged(self):
        assert LangGraphAgent._elide_middle("ls -la", 72) == "ls -la"

    def test_long_value_keeps_head_and_tail(self):
        cmd = (
            "python3 /Users/x/sagemaker_training_cost.py SFT --model llama-3-8b "
            "--tokens 1000000000 --epochs 1"
        )
        out = LangGraphAgent._elide_middle(cmd, 72)
        assert "…" in out
        assert out.startswith("python3 /Users/x/")  # head kept
        assert out.endswith("--epochs 1")  # tail kept (the meaningful part)
        assert len(out) <= 72

    def test_format_tool_call_shows_trailing_args(self):
        tc = {
            "name": "execute_bash",
            "args": {"command": "python3 " + "x" * 200 + " --final-flag", "timeout": 30},
        }
        out = LangGraphAgent._format_tool_call(tc)
        # The trailing timeout arg must still be present (not pushed off the end).
        assert "timeout=30" in out
        assert "--final-flag" in out  # command tail preserved by middle elision
