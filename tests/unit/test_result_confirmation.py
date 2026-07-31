"""Unit tests for the post-call confirmation gate (confirmation_gate.confirm_result).

Some tools can only tell a call is dangerous once they inspect the world (is this
branch pushed? is this a protected branch?), so they refuse and answer
``requires_confirmation`` instead of acting. That payload was handed to the MODEL,
whose documented next step was to re-call with ``allow_dangerous=True`` — the
model approved its own dangerous call and no human was ever asked.

Both halves are tested here: the refusal payload is resolved with the user, and
``allow_dangerous`` set by the model is itself confirmed before the call runs.
"""

import json

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import create_model

from mnemoai.client.agent.agent import LangGraphAgent

REFUSAL = json.dumps(
    {
        "error": True,
        "requires_confirmation": True,
        "message": "This command has potential risks",
        "warnings": [
            {"type": "hard_reset", "message": "Hard reset will discard all changes."}
        ],
        "command": "git reset --hard HEAD~1",
    }
)
ARGS = {"command": "reset --hard HEAD~1"}


class FakeTool:
    """Mimics git_safe: refuses unless allow_dangerous is set."""

    args_schema = create_model(
        "GitSafeArgs",
        command=(str, ...),
        allow_dangerous=(bool, False),
        reason=(str, ""),
    )

    def __init__(self):
        self.calls = []

    def invoke(self, args):
        self.calls.append(dict(args))
        if not args.get("allow_dangerous"):
            return REFUSAL
        return json.dumps({"success": True, "stdout": "HEAD is now at abc123"})


class NoOverrideTool(FakeTool):
    """A tool that asks for confirmation but exposes no override parameter."""

    args_schema = create_model("PlainArgs", command=(str, ...))


def _agent(monkeypatch, answer, headless=False):
    """An agent whose gate sees a TTY and an enabled toggle, answering `answer`."""
    a = LangGraphAgent.__new__(LangGraphAgent)
    a._trusted_confirm_categories = set()
    a._stop_spinner = lambda: None
    a._is_headless = lambda: headless
    a.prompts = []

    def prompt(header, detail, category):
        a.prompts.append((header, detail, category))
        return answer

    a._prompt_confirm = prompt

    import mnemoai.client.agent.confirmation_gate as gate

    monkeypatch.setattr(gate.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(gate.config, "get", lambda k, d=None: True)
    return a


class TestRefusalIsResolvedWithTheUser:
    def test_approval_retries_with_the_override(self, monkeypatch):
        a, tool = _agent(monkeypatch, True), FakeTool()
        out = a._confirm_tool_result(tool, "git_safe", ARGS, tool.invoke(ARGS))
        assert json.loads(out)["success"] is True
        assert len(tool.calls) == 2
        assert tool.calls[1]["allow_dangerous"] is True
        assert tool.calls[1]["reason"]  # a reason is supplied for the audit trail

    def test_prompt_shows_the_real_warning(self, monkeypatch):
        a, tool = _agent(monkeypatch, True), FakeTool()
        a._confirm_tool_result(tool, "git_safe", ARGS, tool.invoke(ARGS))
        header, detail, category = a.prompts[0]
        assert "Hard reset will discard all changes." in detail
        assert "git reset --hard HEAD~1" in detail
        assert category == "git"

    def test_decline_does_not_rerun_the_tool(self, monkeypatch):
        a, tool = _agent(monkeypatch, False), FakeTool()
        out = a._confirm_tool_result(tool, "git_safe", ARGS, tool.invoke(ARGS))
        payload = json.loads(out)
        assert payload["declined_by_user"] is True
        assert "Do NOT retry" in payload["message"]
        assert len(tool.calls) == 1  # only the original refused call

    def test_headless_subagent_cannot_approve(self, monkeypatch):
        a, tool = _agent(monkeypatch, True, headless=True), FakeTool()
        out = a._confirm_tool_result(tool, "git_safe", ARGS, tool.invoke(ARGS))
        assert json.loads(out)["declined_by_user"] is True
        assert a.prompts == []  # never prompted: no TTY of its own

    def test_trusted_category_skips_the_prompt(self, monkeypatch):
        a, tool = _agent(monkeypatch, False), FakeTool()
        a._trusted_confirm_categories.add("git")
        out = a._confirm_tool_result(tool, "git_safe", ARGS, tool.invoke(ARGS))
        assert json.loads(out)["success"] is True
        assert a.prompts == []

    def test_no_override_parameter_passes_the_payload_through(self, monkeypatch):
        a, tool = _agent(monkeypatch, True), NoOverrideTool()
        out = a._confirm_tool_result(tool, "plain", ARGS, tool.invoke(ARGS))
        assert json.loads(out)["requires_confirmation"] is True
        assert a.prompts == []  # nothing we could re-run differently

    def test_already_approved_retry_does_not_loop(self, monkeypatch):
        a, tool = _agent(monkeypatch, True), FakeTool()
        args = dict(ARGS, allow_dangerous=True)
        out = a._confirm_tool_result(tool, "git_safe", args, REFUSAL)
        assert json.loads(out)["requires_confirmation"] is True
        assert a.prompts == []


class TestOrdinaryResultsAreUntouched:
    def test_success_payload_passes_through(self, monkeypatch):
        a, tool = _agent(monkeypatch, True), FakeTool()
        result = '{"success": true, "content": "hi"}'
        assert a._confirm_tool_result(tool, "fs_read", {}, result) == result
        assert a.prompts == []

    def test_non_json_result_passes_through(self, monkeypatch):
        a, tool = _agent(monkeypatch, True), FakeTool()
        assert a._confirm_tool_result(tool, "fs_read", {}, "plain text") == "plain text"

    def test_non_string_result_passes_through(self, monkeypatch):
        a, tool = _agent(monkeypatch, True), FakeTool()
        assert a._confirm_tool_result(tool, "x", {}, {"ok": 1}) == {"ok": 1}


class TestModelCannotSelfApprove:
    """`allow_dangerous=True` from the model is itself a confirmable request."""

    def test_flag_triggers_the_pre_call_prompt(self, monkeypatch):
        a = _agent(monkeypatch, False)
        args = {"command": "push origin main --force", "allow_dangerous": True,
                "reason": "user confirmed"}
        assert a._confirm_tool("git_safe", args) is False
        assert "Override safety check?" in a.prompts[0][0]
        assert "push origin main --force" in a.prompts[0][1]

    def test_approved_flag_proceeds(self, monkeypatch):
        a = _agent(monkeypatch, True)
        args = {"command": "reset --hard", "allow_dangerous": True, "reason": "why"}
        assert a._confirm_tool("git_safe", args) is True

    def test_headless_subagent_flag_is_denied(self, monkeypatch):
        a = _agent(monkeypatch, True, headless=True)
        args = {"command": "reset --hard", "allow_dangerous": True, "reason": "why"}
        assert a._confirm_tool("git_safe", args) is False

    def test_toggle_off_lets_the_override_through(self, monkeypatch):
        a = _agent(monkeypatch, False)
        import mnemoai.client.agent.confirmation_gate as gate

        monkeypatch.setattr(
            gate.config,
            "get",
            lambda k, d=None: False if k == "REQUIRE_GIT_CONFIRMATION" else True,
        )
        args = {"command": "reset --hard", "allow_dangerous": True, "reason": "why"}
        assert a._confirm_tool("git_safe", args) is True
        assert a.prompts == []


class TestTheGateIsActuallyReachable:
    """Behavior coverage above proves the gate WORKS; these prove it RUNS.

    Verified by mutation: unwiring the call from ``_invoke_tool`` left every other
    test in this file green — the exact "green suite, dead feature" shape, since
    they all call ``confirm_result`` directly.
    """

    def test_invoke_tool_routes_results_through_the_gate(self, monkeypatch):
        seen = {}

        def _spy(agent, tool, tool_name, tool_args, result):
            seen["called"] = (tool_name, result)
            return "gated"

        monkeypatch.setattr(
            "mnemoai.client.agent.confirmation_gate.confirm_result", _spy
        )

        class _Tool:
            name = "git_safe"

            def invoke(self, args):
                return REFUSAL

        agent = LangGraphAgent.__new__(LangGraphAgent)
        agent._SELF_REPORTING_TOOLS = frozenset()
        agent._start_spinner = lambda *a, **k: None
        agent._stop_spinner = lambda *a, **k: None
        agent._tool_progress_label = lambda *a: "x"

        out = agent._invoke_tool(_Tool(), "git_safe", ARGS)
        assert out == "gated"
        assert seen["called"] == ("git_safe", REFUSAL)

    def test_a_quiet_subagent_call_is_gated_too(self, monkeypatch):
        # The quiet path skips the spinner; it must NOT skip the gate, or a
        # sub-agent becomes the way around it.
        calls = []
        monkeypatch.setattr(
            "mnemoai.client.agent.confirmation_gate.confirm_result",
            lambda a, t, n, ar, r: calls.append(n) or "gated",
        )

        class _Tool:
            name = "git_safe"

            def invoke(self, args):
                return REFUSAL

        agent = LangGraphAgent.__new__(LangGraphAgent)
        agent._SELF_REPORTING_TOOLS = frozenset()
        assert agent._invoke_tool(_Tool(), "git_safe", ARGS, quiet=True) == "gated"
        assert calls == ["git_safe"]

    def test_both_tool_chokepoints_call_invoke_tool(self):
        # _execute_tools (main loop) and _run_worker_loop (sub-agents/orchestrator)
        # must both go through _invoke_tool, which is where the gate lives. This
        # DRIVES both loops with a tool that counts direct .invoke() calls: the
        # earlier version asserted `"_invoke_tool(" in inspect.getsource(...)`,
        # which never ran either chokepoint, short-circuited on the first of the
        # two, and broke the moment the shared loop moved to its own module.
        direct = []

        class _Tool:
            name = "git_safe"

            def invoke(self, args):
                direct.append(args)  # a bypass — the gate never sees this
                return REFUSAL

        for path in ("main", "worker"):
            gated = []
            a = LangGraphAgent.__new__(LangGraphAgent)
            a.verbose = False
            a.tools = [_Tool()]
            a.tools_by_route = None
            a._start_spinner = lambda *x, **k: None
            a._stop_spinner = lambda *x, **k: None
            a._effective_route = lambda state: None
            a._run_spawn_batch = lambda tool_calls: {}
            a._is_blocked_by_plan_mode = lambda *x: False
            a._confirm_tool = lambda *x: True

            def _invoke_tool(tool, name, args, quiet=False, _seen=gated):
                _seen.append(name)
                return "gated"

            a._invoke_tool = _invoke_tool
            ai = AIMessage(
                content="", tool_calls=[{"name": "git_safe", "args": ARGS, "id": "t1"}]
            )

            if path == "main":
                out = a._execute_tools({"messages": [ai]})["messages"]
            else:
                a.system_prompt = "SYS"
                a.callbacks = []
                a._extract_visible = lambda c: c if isinstance(c, str) else ""
                a._worker_messages_seen = lambda: None
                turns = iter([ai, AIMessage(content="done")])
                a._stream_response = lambda *x, **k: (next(turns), False)
                out = a._run_worker_loop(object(), [], "task", quiet=True)[1]

            results = [m for m in out if isinstance(m, ToolMessage)]
            assert [m.content for m in results] == ["gated"], path
            assert gated == ["git_safe"], f"{path} bypasses _invoke_tool"

        assert direct == []  # neither chokepoint called the tool directly
