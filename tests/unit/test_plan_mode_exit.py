"""Unit tests for the plan-mode approval → execute flow.

The `exit_plan_mode` tool is intercepted client-side: the agent shows the plan
via `_plan_approval_ui` and, on approval, calls `_exit_plan_mode_provider` (which
flips plan mode off + persists the plan) and hands the approved plan back so the
model executes it. Without a UI hook (non-TTY/tests) it auto-approves.
"""

import pathlib
import tempfile
from unittest.mock import patch

import mnemoai.client.client as client_mod
from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.client import LangGraphClient


def _agent():
    return LangGraphAgent.__new__(LangGraphAgent)


class TestExitPlanModeVerdict:
    def test_no_ui_auto_approves(self):
        # Bare object (non-TTY/tests): no _plan_approval_ui → auto-approve.
        a = _agent()
        out = a._handle_exit_plan_mode("# Plan\n1. do x")
        assert "APPROVED" in out
        assert "do x" in out  # approved plan handed back for execution

    def test_approve_calls_provider_and_returns_plan(self):
        a = _agent()
        seen = {}
        a._plan_approval_ui = lambda plan: ("approve", plan)
        a._exit_plan_mode_provider = lambda plan: seen.__setitem__("plan", plan)
        out = a._handle_exit_plan_mode("the plan text")
        assert seen["plan"] == "the plan text"  # provider flips mode + persists
        assert "APPROVED" in out and "the plan text" in out

    def test_keep_planning_does_not_call_provider(self):
        a = _agent()
        called = {"n": 0}
        a._plan_approval_ui = lambda plan: ("keep_planning", plan)
        a._exit_plan_mode_provider = lambda plan: called.__setitem__("n", called["n"] + 1)
        out = a._handle_exit_plan_mode("draft")
        assert called["n"] == 0  # plan mode stays ON — provider not called
        assert "keep planning" in out.lower()

    def test_edited_plan_flows_to_provider_and_message(self):
        # The UI may return an edited plan; approval must use the edited text.
        a = _agent()
        seen = {}
        a._plan_approval_ui = lambda plan: ("approve", "EDITED plan")
        a._exit_plan_mode_provider = lambda plan: seen.__setitem__("plan", plan)
        out = a._handle_exit_plan_mode("original plan")
        assert seen["plan"] == "EDITED plan"
        assert "EDITED plan" in out


class TestPreApprovedBash:
    def test_allowed_bash_registered_on_approve(self):
        a = _agent()
        out = a._handle_exit_plan_mode("plan", ["pytest", "npm run build"])
        assert a._preapproved_bash == ["pytest", "npm run build"]
        # The go-ahead message tells the model these won't prompt.
        assert "pre-approved" in out and "pytest" in out

    def test_keep_planning_does_not_register(self):
        a = _agent()
        a._plan_approval_ui = lambda plan: ("keep_planning", plan)
        a._handle_exit_plan_mode("plan", ["pytest"])
        assert getattr(a, "_preapproved_bash", []) == []

    def test_blank_and_none_entries_dropped(self):
        a = _agent()
        a._handle_exit_plan_mode("plan", ["  ", "", "make test"])
        assert a._preapproved_bash == ["make test"]

    def test_no_allowed_bash_leaves_list_empty(self):
        a = _agent()
        out = a._handle_exit_plan_mode("plan")
        assert getattr(a, "_preapproved_bash", []) == []
        assert "pre-approved" not in out


class TestIsPreApprovedBash:
    def test_exact_and_prefix_match(self):
        a = _agent()
        a._preapproved_bash = ["pytest", "npm run build"]
        assert a._is_preapproved_bash("pytest") is True
        assert a._is_preapproved_bash("pytest tests/unit") is True  # prefix + space
        assert a._is_preapproved_bash("npm run build") is True

    def test_non_match_and_partial_word_rejected(self):
        a = _agent()
        a._preapproved_bash = ["pytest"]
        assert a._is_preapproved_bash("rm -rf /") is False
        # Must not match a different command that merely starts with the letters.
        assert a._is_preapproved_bash("pytest-cov") is False

    def test_empty_approvals_never_match(self):
        a = _agent()
        a._preapproved_bash = []
        assert a._is_preapproved_bash("pytest") is False

    def test_blank_command_never_matches(self):
        a = _agent()
        a._preapproved_bash = ["pytest"]
        assert a._is_preapproved_bash("") is False


class TestApprovePlan:
    def test_flips_flag_and_persists(self):
        c = LangGraphClient.__new__(LangGraphClient)
        c.plan_mode_active = True
        c.session_id = "user_20260704_120000"
        d = pathlib.Path(tempfile.mkdtemp())
        with patch.object(client_mod, "plans_dir", lambda: d):
            c._approve_plan("# Plan\n- step one")
        assert c.plan_mode_active is False
        files = list(d.glob("plan_*.md"))
        assert len(files) == 1
        assert files[0].name == "plan_20260704_120000.md"
        assert "step one" in files[0].read_text()

    def test_persist_failure_still_flips_flag(self):
        # A write error must not leave plan mode stuck ON.
        c = LangGraphClient.__new__(LangGraphClient)
        c.plan_mode_active = True
        c.session_id = "user_20260704_120000"

        class _BadDir:
            def __truediv__(self, other):
                raise OSError("disk full")

        with patch.object(client_mod, "plans_dir", lambda: _BadDir()):
            c._approve_plan("plan")
        assert c.plan_mode_active is False


class TestExecutePlanRoutePin:
    """After a plan is approved, per-turn routing must not narrow the toolset —
    execution may call any tool. Regression: an approved implementation plan
    re-classified as a read-only route (e.g. a docs question) found its
    file_edit/fs_write/execute_bash tools unbound, so it couldn't apply them."""

    def test_approve_sets_route_pin(self):
        a = _agent()
        a._handle_exit_plan_mode("# Plan\n- edit files")
        assert a._execute_plan_route is True

    def test_keep_planning_does_not_pin(self):
        a = _agent()
        a._plan_approval_ui = lambda plan: ("keep_planning", plan)
        a._handle_exit_plan_mode("draft")
        assert getattr(a, "_execute_plan_route", False) is False

    def test_effective_route_forces_full_when_pinned(self):
        a = _agent()
        a._execute_plan_route = True
        # Even though the classifier chose a read-only route, binding is 'full'.
        assert a._effective_route({"route": "knowledge"}) == "full"

    def test_effective_route_uses_state_when_not_pinned(self):
        a = _agent()
        a._execute_plan_route = False
        assert a._effective_route({"route": "knowledge"}) == "knowledge"

    def test_route_model_full_binding_when_pinned(self):
        a = _agent()
        a._execute_plan_route = True
        a.models_by_route = {"knowledge": "KNOWLEDGE_MODEL", "full": "FULL_MODEL"}
        a.model_with_tools = "ALL"
        # A read-only classification must still bind the full-route model.
        assert a._get_route_model({"route": "knowledge"}) == "FULL_MODEL"

    def test_route_tools_full_binding_when_pinned(self):
        a = _agent()
        a._execute_plan_route = True
        a.tools_by_route = {"knowledge": ["fs_read"], "full": ["fs_read", "file_edit"]}
        a.tools = ["fs_read", "file_edit"]
        assert a._get_route_tools({"route": "knowledge"}) == ["fs_read", "file_edit"]

    def test_route_after_classify_skips_orchestrator_when_pinned(self):
        # During plan execution, a 'full' route must go straight to the agent
        # (not re-decompose into read-only workers).
        a = _agent()
        a._execute_plan_route = True
        assert a._route_after_classify({"route": "full", "messages": []}) == "agent"

    def test_clear_messages_resets_pin(self):
        a = _agent()
        a._messages = []
        a._thinking = None
        a._last_input_tokens = None
        a._preapproved_bash = ["pytest"]
        a._execute_plan_route = True
        a.clear_messages()
        assert a._execute_plan_route is False
        assert a._preapproved_bash == []
