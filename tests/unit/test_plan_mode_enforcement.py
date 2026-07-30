"""Unit tests for enforced plan mode (LangGraphAgent._is_blocked_by_plan_mode).

Plan mode is a user-toggled, client-side hard gate: while active, mutating /
shell-executing tools are blocked at the tool chokepoint (the same place the
confirmation gate lives), regardless of what the model does. Read-only tools
and the memory notebook stay allowed.
"""

import pytest

from mnemoai.client.agent import plan_policy
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


class TestBothWriteToolArgSpellings:
    """The write tools disagree on their target's argument name: ``fs_write``
    takes ``path``, ``file_edit`` takes ``file_path``. Client-side readers must
    accept both — reading only ``path`` meant ``file_edit`` presented NO target,
    so ``is_plan_file("")`` was false and plan mode blocked every ``file_edit``
    including one writing the plan itself.
    """

    def _plan_file(self):
        from mnemoai.utils.paths import plans_dir

        return str(plans_dir() / "plan_20260730_120000.md")

    def test_file_edit_may_write_the_plan_file(self):
        # The regression: this was blocked, so an approved plan couldn't be saved
        # through file_edit at all.
        assert (
            _agent(True)._is_blocked_by_plan_mode(
                "file_edit", {"file_path": self._plan_file()}
            )
            is False
        )

    def test_fs_write_may_write_the_plan_file(self):
        assert (
            _agent(True)._is_blocked_by_plan_mode(
                "fs_write", {"path": self._plan_file()}
            )
            is False
        )

    def test_either_spelling_works_for_either_tool(self):
        for tool in ("file_edit", "fs_write"):
            for key in ("path", "file_path"):
                assert (
                    _agent(True)._is_blocked_by_plan_mode(
                        tool, {key: self._plan_file()}
                    )
                    is False
                ), f"{tool} with {key} was blocked"

    def test_a_non_plan_file_is_still_blocked(self):
        for key in ("path", "file_path"):
            assert (
                _agent(True)._is_blocked_by_plan_mode("file_edit", {key: "/tmp/src.py"})
                is True
            )

    def test_a_call_with_no_target_is_blocked(self):
        # Fail safe: an unparseable call must not be treated as the plan file.
        assert _agent(True)._is_blocked_by_plan_mode("file_edit", {}) is True

    def test_write_target_reads_both_names(self):
        assert plan_policy.write_target({"path": "/a"}) == "/a"
        assert plan_policy.write_target({"file_path": "/b"}) == "/b"
        assert plan_policy.write_target({}) == ""
        assert plan_policy.write_target(None) == ""

    def test_write_target_prefers_a_populated_key(self):
        # A tool that sends both (or an empty one) must not yield "".
        assert plan_policy.write_target({"path": "", "file_path": "/b"}) == "/b"


class TestTheConfirmationPromptNamesTheFile:
    """A prompt that hides its target defeats the gate: the user was asked to
    approve a bare "edit" with no filename."""

    def _detail_for(self, tool_args, monkeypatch):
        import sys

        from mnemoai.client.agent.agent import LangGraphAgent

        seen = []
        agent = LangGraphAgent.__new__(LangGraphAgent)
        agent._trusted_confirm_categories = set()
        agent._confirm_lock = None
        agent._headless_tl = None
        agent._spawn_depth_plain = 0
        agent._spawn_depth_tl = None
        agent._prompt_confirm = lambda h, d, c: seen.append(d) or True
        agent._spinner_snapshot = lambda: (False, "x")
        agent._stop_spinner = lambda *a, **k: None
        agent._start_spinner = lambda *a, **k: None
        agent._is_preapproved_bash = lambda c: False
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        agent._confirm_tool("file_edit", tool_args)
        return seen[0] if seen else ""

    def test_file_edit_prompt_shows_the_path(self, monkeypatch):
        detail = self._detail_for(
            {"file_path": "/tmp/target.py", "old_string": "a", "new_string": "b"},
            monkeypatch,
        )
        assert "/tmp/target.py" in detail

    def test_the_spinner_label_shows_the_path(self):
        from mnemoai.client.agent.agent import LangGraphAgent

        agent = LangGraphAgent.__new__(LangGraphAgent)
        label = agent._tool_progress_label("file_edit", {"file_path": "/tmp/x.py"})
        assert "/tmp/x.py" in label
