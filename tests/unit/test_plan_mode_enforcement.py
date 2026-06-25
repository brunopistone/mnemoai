"""Unit tests for enforced plan mode (LangGraphAgent._is_blocked_by_plan_mode).

Plan mode is a user-toggled, client-side hard gate: while active, mutating /
shell-executing tools are blocked at the tool chokepoint (the same place the
confirmation gate lives), regardless of what the model does. Read-only tools
and the memory notebook stay allowed.
"""

import pytest

from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.utils.paths import plans_dir

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


# --- Conditional plan-mode allowances ---

READONLY_CMDS = [
    "ls -la",
    "cat file.py",
    "grep -rn foo src/",
    "rg pattern",
    "git status",
    "git log --oneline",
    "git diff HEAD~1",
    "git show abc123",
    "find . -name '*.py'",
    "wc -l file.txt",
    "head -20 file.py",
    "grep -i pattern file.py",  # -i here = case-insensitive, read-only
    "sed -n 1,10p file.py",  # sed without -i is read-only
    "find . -name '*.py' -type f",  # find without -delete/-exec is read-only
]
MUTATING_CMDS = [
    "rm -rf /tmp/x",
    "echo hi > out.txt",
    "cat a.txt >> b.txt",
    "git commit -m x",
    "git push",
    "git checkout -b new",
    "pip install foo",
    "ls && rm x",
    "cat x | tee y",
    "touch newfile",
    "mkdir d",
    "$(rm x)",
    "git tag v1",
    "git stash",
    "sed -i s/a/b/ f.txt",
    "sed -i.bak s/a/b/ f.txt",
    "find . -name '*.py' -delete",
    "find . -type f -exec rm {} +",
]


@pytest.mark.parametrize("cmd", READONLY_CMDS)
def test_readonly_bash_allowed_in_plan_mode(cmd):
    a = _agent(True)
    assert a._is_blocked_by_plan_mode("execute_bash", {"command": cmd}) is False


@pytest.mark.parametrize("cmd", MUTATING_CMDS)
def test_mutating_bash_blocked_in_plan_mode(cmd):
    a = _agent(True)
    assert a._is_blocked_by_plan_mode("execute_bash", {"command": cmd}) is True


def test_empty_bash_blocked_in_plan_mode():
    a = _agent(True)
    assert a._is_blocked_by_plan_mode("execute_bash", {"command": ""}) is True


@pytest.mark.parametrize("tool", ["fs_write", "file_edit"])
def test_plan_file_write_allowed(tool):
    a = _agent(True)
    plan_path = str(plans_dir() / "my-plan.md")
    assert a._is_blocked_by_plan_mode(tool, {"path": plan_path}) is False


@pytest.mark.parametrize("tool", ["fs_write", "file_edit"])
def test_non_plan_file_write_blocked(tool):
    a = _agent(True)
    # Right dir, wrong extension.
    assert (
        a._is_blocked_by_plan_mode(tool, {"path": str(plans_dir() / "x.txt")}) is True
    )
    # Plausible plan name but outside the plans dir.
    assert (
        a._is_blocked_by_plan_mode(tool, {"path": "/tmp/elsewhere/plan.md"}) is True
    )
    # No path at all.
    assert a._is_blocked_by_plan_mode(tool, {}) is True
