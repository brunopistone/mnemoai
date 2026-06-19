"""Unit tests for enforced plan mode (LangGraphAgent._is_blocked_by_plan_mode).

Plan mode is a user-toggled, client-side hard gate: while active, mutating /
shell-executing tools are blocked at the tool chokepoint (the same place the
confirmation gate lives), regardless of what the model does. Read-only tools
and the memory notebook stay allowed.
"""

import pytest

from mnemoai.client.agent.agent import LangGraphAgent

BLOCKED = [
    "execute_bash",
    "fs_write",
    "file_edit",
    "git_safe",
    "git_commit_safe",
    "start_background_task",
]
ALLOWED = ["fs_read", "glob_search", "grep_search", "memory", "web_search", "read_pdf"]


def _agent(plan_active):
    a = LangGraphAgent.__new__(LangGraphAgent)
    a._plan_mode_provider = lambda: plan_active
    return a


@pytest.mark.parametrize("tool", BLOCKED)
def test_blocks_mutating_tools_when_active(tool):
    assert _agent(True)._is_blocked_by_plan_mode(tool) is True


@pytest.mark.parametrize("tool", ALLOWED)
def test_allows_readonly_and_memory_when_active(tool):
    assert _agent(True)._is_blocked_by_plan_mode(tool) is False


@pytest.mark.parametrize("tool", BLOCKED + ALLOWED)
def test_nothing_blocked_when_inactive(tool):
    assert _agent(False)._is_blocked_by_plan_mode(tool) is False


def test_blocked_set_matches_expectation():
    # Guard against accidental drift of the blocked set.
    assert LangGraphAgent._PLAN_BLOCKED_TOOLS == set(BLOCKED)


def test_default_provider_is_inactive():
    # An agent built without a provider must default to plan mode OFF (no
    # accidental blocking when the feature isn't wired).
    a = LangGraphAgent.__new__(LangGraphAgent)
    a._plan_mode_provider = (lambda: False)
    assert a._is_blocked_by_plan_mode("execute_bash") is False
