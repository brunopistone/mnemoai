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
