"""Unit tests for the execute_bash confirmation gate (LangGraphAgent._confirm_tool).

The gate is a hard, client-side prompt before any shell command runs (the MCP
server is a piped subprocess and can't prompt). These tests exercise the pure
decision logic with input()/config/TTY mocked — no real agent graph or LLM.
"""

import builtins

import pytest

import mnemoai.client.agent.agent as agent_mod
from mnemoai.client.agent.agent import LangGraphAgent


@pytest.fixture
def agent():
    a = LangGraphAgent.__new__(LangGraphAgent)
    a._stop_spinner = lambda: None
    return a


def _run(agent, monkeypatch, tool, args, answer, *, toggle=True, tty=True):
    # Both confirmation toggles read the same `toggle` value here.
    monkeypatch.setattr(
        agent_mod.config, "get",
        lambda k, d=None: toggle
        if k in ("REQUIRE_BASH_CONFIRMATION", "REQUIRE_WRITE_CONFIRMATION")
        else d,
    )
    monkeypatch.setattr(agent_mod.sys.stdin, "isatty", lambda: tty)
    monkeypatch.setattr(builtins, "input", lambda prompt="": answer)
    return agent._confirm_tool(tool, args)


def test_non_bash_tool_never_prompts(agent, monkeypatch):
    # A 'no' answer would block — but non-bash tools aren't gated at all.
    assert _run(agent, monkeypatch, "read_file", {}, "n") is True


@pytest.mark.parametrize("answer", ["y", "yes", "Y", "YES"])
def test_bash_yes_proceeds(agent, monkeypatch, answer):
    assert _run(agent, monkeypatch, "execute_bash", {"command": "ls"}, answer) is True


@pytest.mark.parametrize("answer", ["n", "no", "", "maybe", "sure"])
def test_bash_non_yes_declines(agent, monkeypatch, answer):
    # Strict: only explicit yes proceeds; everything else (incl. empty) declines.
    assert _run(agent, monkeypatch, "execute_bash", {"command": "rm -rf /"}, answer) is False


def test_toggle_off_proceeds_without_prompt(agent, monkeypatch):
    assert _run(agent, monkeypatch, "execute_bash", {"command": "x"}, "n", toggle=False) is True


def test_non_interactive_auto_proceeds(agent, monkeypatch):
    # No TTY (tests/CI/pipes): can't prompt, so don't block.
    assert _run(agent, monkeypatch, "execute_bash", {"command": "x"}, "n", tty=False) is True


def test_eof_declines(agent, monkeypatch):
    # Ctrl-D / closed stdin during the prompt is treated as a decline.
    monkeypatch.setattr(
        agent_mod.config, "get",
        lambda k, d=None: True if k == "REQUIRE_BASH_CONFIRMATION" else d,
    )
    monkeypatch.setattr(agent_mod.sys.stdin, "isatty", lambda: True)

    def _raise(prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", _raise)
    assert agent._confirm_tool("execute_bash", {"command": "x"}) is False


# --- File-write tools (fs_write / file_edit) ---


@pytest.mark.parametrize("tool", ["fs_write", "file_edit"])
def test_write_tool_yes_proceeds(agent, monkeypatch, tool):
    assert _run(agent, monkeypatch, tool, {"path": "/tmp/x", "command": "create"}, "y") is True


@pytest.mark.parametrize("tool", ["fs_write", "file_edit"])
def test_write_tool_no_declines(agent, monkeypatch, tool):
    assert _run(agent, monkeypatch, tool, {"path": "/tmp/x"}, "n") is False


def test_fs_write_dry_run_preview_not_gated(agent, monkeypatch):
    # The dry_run=True preview performs no write, so it is never gated even when
    # the user would decline.
    assert _run(agent, monkeypatch, "fs_write",
                {"path": "/tmp/x", "command": "create", "dry_run": True}, "n") is True


def test_write_toggle_off_proceeds(agent, monkeypatch):
    assert _run(agent, monkeypatch, "fs_write", {"path": "/tmp/x"}, "n", toggle=False) is True


def test_write_non_interactive_auto_proceeds(agent, monkeypatch):
    assert _run(agent, monkeypatch, "file_edit", {"path": "/tmp/x"}, "n", tty=False) is True


# --- Memory tool (REQUIRE_MEMORY_CONFIRMATION, default OFF) ---


def _run_memory(agent, monkeypatch, args, answer, *, toggle, tty=True):
    monkeypatch.setattr(
        agent_mod.config, "get",
        lambda k, d=None: toggle if k == "REQUIRE_MEMORY_CONFIRMATION" else d,
    )
    monkeypatch.setattr(agent_mod.sys.stdin, "isatty", lambda: tty)
    monkeypatch.setattr(builtins, "input", lambda prompt="": answer)
    return agent._confirm_tool("memory", args)


def test_memory_default_off_proceeds_without_prompt(agent, monkeypatch):
    # Default is off (auto-save): even a 'no' answer proceeds (no prompt fires).
    assert _run_memory(agent, monkeypatch,
                       {"action": "add", "text": "x"}, "n", toggle=False) is True


def test_memory_toggle_on_gates_writes(agent, monkeypatch):
    assert _run_memory(agent, monkeypatch,
                       {"action": "add", "text": "x"}, "n", toggle=True) is False
    assert _run_memory(agent, monkeypatch,
                       {"action": "replace", "old_text": "a", "text": "b"}, "y",
                       toggle=True) is True


def test_memory_non_write_action_not_gated(agent, monkeypatch):
    # An unknown/read action touches no file, so it proceeds even with gate on.
    assert _run_memory(agent, monkeypatch,
                       {"action": "view"}, "n", toggle=True) is True
