"""Behavior coverage for the two tool-execution chokepoints.

``_execute_tools`` (main loop) and ``_run_worker_loop`` (sub-agents/orchestrator)
are where every tool call passes the plan-mode block, the pre-call confirmation
gate and ``_invoke_tool``. Both were pinned only by a source-text substring
assertion (``"_invoke_tool(" in inspect.getsource(...)``): no test ever CALLED
``_execute_tools``, and deleting the ``if not self._confirm_tool(...)`` branch
from either chokepoint left the whole unit suite green — a destructive-tool gate
with no test protecting it.

These drive both chokepoints for real and assert on the ToolMessage each branch
produces, so unwiring a gate fails here instead of shipping. Both now run the same
``tool_loop.run_tool_calls``, but the cases are deliberately kept per-path rather
than parameterized: the shared loop is what must be verifiable from each side, and
these tests are also what would catch it being re-forked.
"""

import logging
from contextlib import contextmanager

from langchain_core.messages import AIMessage, ToolMessage

from mnemoai.client import hooks
from mnemoai.client.agent import tool_loop
from mnemoai.client.agent.agent import LangGraphAgent


@contextmanager
def capture_logs(level):
    """Collect ``ai_app`` records emitted inside the block.

    Not ``caplog``: the app logger sets ``propagate = False``
    (``utils/logger.py:120``), so pytest's root handler never sees these records
    and every assertion against ``caplog.text`` would vacuously compare against
    "". Attach to the logger itself instead.
    """
    logger = logging.getLogger("ai_app")

    class _Capture(logging.Handler):
        def __init__(self):
            super().__init__(level)
            self.lines = []

        def emit(self, record):
            self.lines.append(record.getMessage())

    handler = _Capture()
    logger.addHandler(handler)
    try:
        yield handler.lines
    finally:
        logger.removeHandler(handler)


class _Tool:
    """A minimal stand-in for a bound MCP tool."""

    def __init__(self, name, fn=None):
        self.name = name
        self._fn = fn

    def invoke(self, args):
        return self._fn(args) if self._fn else f"{self.name}-out"


def _agent(tools):
    """A bare agent wired just enough to run _execute_tools."""
    a = LangGraphAgent.__new__(LangGraphAgent)
    a.verbose = False
    a.tools = tools
    a.tools_by_route = None
    a._start_spinner = lambda *x, **k: None
    a._stop_spinner = lambda *x, **k: None
    a._effective_route = lambda state: None
    a._run_spawn_batch = lambda tool_calls: {}
    a._is_blocked_by_plan_mode = lambda *x: False
    a._confirm_tool = lambda *x: True

    def _invoke_tool(tool, name, args, quiet=False):
        return tool.invoke(args)

    a._invoke_tool = _invoke_tool
    return a


def _run(agent, name="grep_search", args=None, tool_id="t1"):
    """Drive one tool call through the main-loop chokepoint."""
    ai = AIMessage(
        content="", tool_calls=[{"name": name, "args": args or {}, "id": tool_id}]
    )
    return agent._execute_tools({"messages": [ai]})["messages"]


class TestMainLoopChokepoint:
    """_execute_tools: one branch per ToolMessage it can emit."""

    def test_a_successful_call_returns_the_result(self):
        out = _run(_agent([_Tool("grep_search")]))
        assert [(m.name, m.content, m.tool_call_id) for m in out] == [
            ("grep_search", "grep_search-out", "t1")
        ]
        assert all(isinstance(m, ToolMessage) for m in out)

    def test_the_result_goes_through_invoke_tool(self):
        # _invoke_tool is where the POST-call confirmation gate lives, so the
        # chokepoint must not call tool.invoke() directly.
        a = _agent([_Tool("git_safe")])
        seen = []

        def _invoke_tool(tool, name, args, quiet=False):
            seen.append(name)
            return "gated"

        a._invoke_tool = _invoke_tool
        assert [m.content for m in _run(a, name="git_safe")] == ["gated"]
        assert seen == ["git_safe"]

    def test_a_denied_confirmation_does_not_run_the_tool(self):
        ran = []
        a = _agent([_Tool("execute_bash", lambda args: ran.append(args) or "ok")])
        a._confirm_tool = lambda *x: False
        out = _run(a, name="execute_bash", args={"command": "rm -rf /tmp/x"})
        assert [m.content for m in out] == ["User declined to run this command."]
        assert ran == []  # the gate ran BEFORE the tool, not after

    def test_the_confirmation_gate_sees_the_normalized_call(self):
        # A malformed shape, NOT already-normal args: smaller models emit the
        # whole `field="value"` packed into the dict key. Normalizing must happen
        # BEFORE the gates, or the prompt shows the user an arg dict with no
        # `command` in it — and `_is_blocked_by_plan_mode`, which matches on the
        # same args, can't classify what it can't read.
        a = _agent([_Tool("execute_bash")])
        asked, planned = [], []
        a._confirm_tool = lambda name, args: asked.append((name, args)) or True
        a._is_blocked_by_plan_mode = (
            lambda name, args: planned.append((name, args)) or False
        )
        _run(a, name="execute_bash", args={'command="ls -la"': ""})
        assert asked == [("execute_bash", {"command": "ls -la"})]
        assert planned == [("execute_bash", {"command": "ls -la"})]

    def test_plan_mode_blocks_before_confirming(self):
        # A blocked tool must never even prompt: the block is above the gate.
        a = _agent([_Tool("fs_write")])
        a._is_blocked_by_plan_mode = lambda *x: True
        a._confirm_tool = lambda *x: (_ for _ in ()).throw(
            AssertionError("blocked tool must not reach the confirm gate")
        )
        out = _run(a, name="fs_write", args={"path": "/tmp/x", "command": "create"})
        assert "plan mode is active" in out[0].content

    def test_an_unknown_tool_reports_itself(self):
        out = _run(_agent([]))
        assert [m.content for m in out] == ["Tool not found: grep_search"]

    def test_a_raising_tool_becomes_an_error_message(self):
        def _boom(args):
            raise RuntimeError("nope")

        out = _run(_agent([_Tool("grep_search", _boom)]))
        assert "nope" in out[0].content
        assert out[0].tool_call_id == "t1"  # the pair stays valid for the provider

    def test_an_empty_str_exception_still_yields_a_message(self):
        # A bare TimeoutError has an empty str(); both the ToolMessage and the
        # log line must still name the failure rather than come back blank.
        def _timeout(args):
            raise TimeoutError()

        with capture_logs(logging.ERROR) as lines:
            out = _run(_agent([_Tool("execute_bash", _timeout)]), name="execute_bash")
        assert out[0].content.strip()
        assert "Timeout" in out[0].content
        assert any("TimeoutError" in line for line in lines), lines

    def test_a_result_is_truncated_at_the_source(self):
        big = "x" * 500
        a = _agent([_Tool("fs_read", lambda args: big)])
        a._max_tool_result_chars = 100
        out = _run(a, name="fs_read")
        assert len(out[0].content) < len(big)
        # A shorter-than-`big` result is ALSO what a mis-wired stub produces: the
        # loop's `except Exception` turns a TypeError into a plausible tool error,
        # which satisfied the length assertion above while testing nothing.
        assert "x" in out[0].content
        assert "unexpected keyword argument" not in out[0].content

    def test_every_tool_call_gets_exactly_one_reply(self):
        # An unanswered tool_call_id is a provider-level error, so each call must
        # produce one message whatever branch it takes.
        a = _agent([_Tool("grep_search"), _Tool("fs_write")])
        a._is_blocked_by_plan_mode = lambda name, args: name == "fs_write"
        ai = AIMessage(
            content="",
            tool_calls=[
                {"name": "grep_search", "args": {}, "id": "a"},
                {"name": "fs_write", "args": {}, "id": "b"},
                {"name": "nope", "args": {}, "id": "c"},
            ],
        )
        out = a._execute_tools({"messages": [ai]})["messages"]
        assert [m.tool_call_id for m in out] == ["a", "b", "c"]

    def test_a_client_side_tool_short_circuits(self):
        # exit_plan_mode / spawn_agent / ask_user_question are handled here, not
        # via MCP: they must not be looked up in the tool list at all.
        a = _agent([])
        a._client_side_tool_message = lambda name, args, tid, spawn=None: ToolMessage(
            content="handled here", tool_call_id=tid, name=name
        )
        out = _run(a, name="exit_plan_mode", args={"plan": "do it"})
        assert [m.content for m in out] == ["handled here"]

    def test_a_message_without_tool_calls_is_a_no_op(self):
        a = _agent([])
        assert a._execute_tools({"messages": [AIMessage(content="hi")]}) == {
            "messages": []
        }


def _worker_agent(tools, turns):
    """A bare agent wired to run _run_worker_loop over canned model turns."""
    a = LangGraphAgent.__new__(LangGraphAgent)
    a.system_prompt = "SYS"
    a.callbacks = []
    a.verbose = False
    a.tools = tools
    a._extract_visible = lambda c: c if isinstance(c, str) else ""
    a._is_blocked_by_plan_mode = lambda *x: False
    a._confirm_tool = lambda *x: True
    a._invoke_tool = lambda tool, name, args, quiet=False: tool.invoke(args)
    state = {"i": 0}

    def _stream_response(msgs, config, model=None, quiet=False, **k):
        r = turns[state["i"]]
        state["i"] += 1
        return r, False

    a._stream_response = _stream_response
    a._worker_messages_seen = lambda: None
    return a


def _tool_turn(name, args=None, tool_id="w1"):
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args or {}, "id": tool_id}]
    )


class TestWorkerLoopChokepoint:
    """_run_worker_loop: the same gates, on the sub-agent/orchestrator path.

    A sub-agent must not be a way around the confirmation gate, so these mirror
    the main-loop cases above rather than trusting the shared helper names.
    """

    def test_a_successful_call_feeds_the_result_back(self):
        a = _worker_agent(
            [_Tool("grep_search")],
            [_tool_turn("grep_search"), AIMessage(content="done")],
        )
        text, saveable = a._run_worker_loop(object(), [], "task", quiet=True)
        assert text == "done"
        results = [m for m in saveable if isinstance(m, ToolMessage)]
        assert [(m.name, m.content) for m in results] == [
            ("grep_search", "grep_search-out")
        ]

    def test_a_denied_confirmation_does_not_run_the_tool(self):
        ran = []
        a = _worker_agent(
            [_Tool("execute_bash", lambda args: ran.append(args) or "ok")],
            [
                _tool_turn("execute_bash", {"command": "rm -rf /tmp/x"}),
                AIMessage(content="stopped"),
            ],
        )
        a._confirm_tool = lambda *x: False
        text, saveable = a._run_worker_loop(object(), [], "task", quiet=True)
        assert text == "stopped"
        declined = [m for m in saveable if isinstance(m, ToolMessage)]
        assert [m.content for m in declined] == ["User declined to run this command."]
        assert ran == []

    def test_plan_mode_blocks_before_confirming(self):
        a = _worker_agent(
            [_Tool("fs_write")],
            [_tool_turn("fs_write"), AIMessage(content="planned")],
        )
        a._is_blocked_by_plan_mode = lambda *x: True
        a._confirm_tool = lambda *x: (_ for _ in ()).throw(
            AssertionError("blocked tool must not reach the confirm gate")
        )
        _, saveable = a._run_worker_loop(object(), [], "task", quiet=True)
        blocked = [m for m in saveable if isinstance(m, ToolMessage)]
        assert "plan mode is active" in blocked[0].content

    def test_the_result_goes_through_invoke_tool(self):
        a = _worker_agent(
            [_Tool("git_safe")], [_tool_turn("git_safe"), AIMessage(content="done")]
        )
        seen = []
        a._invoke_tool = (
            lambda tool, name, args, quiet=False: seen.append(name) or "gated"
        )
        _, saveable = a._run_worker_loop(object(), [], "task", quiet=True)
        assert seen == ["git_safe"]
        gated = [m for m in saveable if isinstance(m, ToolMessage)]
        assert [m.content for m in gated] == ["gated"]

    def test_an_unknown_tool_reports_itself_and_warns(self):
        # The warning is the divergence the shared loop closed: the worker copy
        # answered the model but logged nothing, so a sub-agent calling a tool it
        # wasn't given left no trace in the log at all.
        a = _worker_agent([], [_tool_turn("nope"), AIMessage(content="done")])
        with capture_logs(logging.WARNING) as lines:
            _, saveable = a._run_worker_loop(object(), [], "task", quiet=True)
        missing = [m for m in saveable if isinstance(m, ToolMessage)]
        assert [m.content for m in missing] == ["Tool not found: nope"]
        assert any("Tool not found: nope" in line for line in lines), lines

    def test_an_empty_str_exception_still_yields_a_message(self):
        # Same empty-str() TimeoutError case as the main loop. This one is why
        # the duplication matters: the `str(e) or repr(e)` fix was applied to the
        # foreground copy only, so for a whole release the worker logged
        # "Worker tool error:" and nothing else.
        def _timeout(args):
            raise TimeoutError()

        a = _worker_agent(
            [_Tool("execute_bash", _timeout)],
            [_tool_turn("execute_bash"), AIMessage(content="done")],
        )
        with capture_logs(logging.ERROR) as lines:
            _, saveable = a._run_worker_loop(object(), [], "task", quiet=True)
        errs = [m for m in saveable if isinstance(m, ToolMessage)]
        assert errs[0].content.strip()
        assert "Timeout" in errs[0].content
        assert any("TimeoutError" in line for line in lines), lines


class TestTheTwoPathsCannotDivergeAgain:
    """Both chokepoints must resolve to the ONE shared loop.

    The duplication is what let the `str(e) or repr(e)` fix land in the foreground
    copy alone (a release where a worker's tool timeout logged a blank error) and
    what left the worker's tool-not-found unlogged. Re-forking either path would
    make every gate above testable twice and fixable once — so assert the single
    implementation, not just its behavior.
    """

    def test_both_chokepoints_dispatch_into_the_shared_loop(self, monkeypatch):
        callers = []
        monkeypatch.setattr(
            tool_loop,
            "run_tool_calls",
            lambda agent, calls, tools, messages, **k: callers.append(
                (k.get("log_label"), [c["name"] for c in calls])
            ),
        )

        a = _agent([_Tool("grep_search")])
        _run(a)

        w = _worker_agent(
            [_Tool("grep_search")],
            [_tool_turn("grep_search"), AIMessage(content="done")],
        )
        w._run_worker_loop(object(), [], "task", quiet=True)

        assert [names for _, names in callers] == [["grep_search"], ["grep_search"]]
        # Distinct labels: with sub-agents running concurrently, an operator can't
        # otherwise tell the foreground's log lines from a worker's.
        labels = [label for label, _ in callers]
        assert labels[0] != labels[1]

    def test_the_worker_passes_quiet_through_to_invoke_tool(self):
        # quiet is a real parameter of the shared loop, not a branch — it must
        # still reach _invoke_tool on the sub-agent path.
        seen = []
        w = _worker_agent(
            [_Tool("grep_search")],
            [_tool_turn("grep_search"), AIMessage(content="done")],
        )

        def _invoke_tool(tool, name, args, quiet=False):
            seen.append(quiet)
            return "ok"

        w._invoke_tool = _invoke_tool
        w._run_worker_loop(object(), [], "task", quiet=True)
        assert seen == [True]


def _hooked(agent, *outcomes):
    """Feed canned hook outcomes to an agent and record each event it fires."""
    fired = []
    queue = list(outcomes)

    def _run_hooks(event, name, args, response=None, quiet=False):
        fired.append((event, name, response))
        return queue.pop(0) if queue else hooks.Outcome()

    agent._run_hooks = _run_hooks
    return fired


class TestHooksAtTheChokepoint:
    """Where a user hook sits in the gate order, and what it can reach.

    The order is the security property: a hook's ``deny`` is honored anywhere, but
    its ``allow`` satisfies exactly one gate — the confirmation prompt. A config
    file must never be a way to widen what the app may do, so an ``allow`` cannot
    unblock plan mode (and cannot touch the server-side floors, which live in the
    MCP subprocess and are tested there).
    """

    def test_a_deny_blocks_the_call_and_never_prompts(self):
        ran = []
        a = _agent([_Tool("execute_bash", lambda args: ran.append(args) or "ok")])
        a._confirm_tool = lambda *x: (_ for _ in ()).throw(
            AssertionError("a hook-denied tool must not reach the confirm gate")
        )
        _hooked(a, hooks.Outcome(decision="deny", reason="no writes under /secrets"))
        out = _run(a, name="execute_bash", args={"command": "rm -rf /secrets"})
        assert "no writes under /secrets" in out[0].content
        assert ran == []

    def test_the_deny_message_tells_the_model_not_to_retry(self):
        # A hook is a standing rule, not a transient failure: a model that reads
        # it as flakiness burns the turn re-calling the same blocked tool.
        a = _agent([_Tool("fs_write")])
        _hooked(a, hooks.Outcome(decision="deny", reason="protected path"))
        content = _run(a, name="fs_write", args={"path": "/secrets/x"})[0].content
        assert "do not" in content.lower() and "retry" in content.lower()

    def test_an_allow_satisfies_the_confirmation_prompt(self):
        a = _agent([_Tool("execute_bash")])
        a._confirm_tool = lambda *x: (_ for _ in ()).throw(
            AssertionError("an allowed call must not prompt")
        )
        _hooked(a, hooks.Outcome(decision="allow"))
        out = _run(a, name="execute_bash", args={"command": "git status"})
        assert out[0].content == "execute_bash-out"

    def test_an_allow_cannot_unblock_plan_mode(self):
        # The block is ABOVE the hook, so the hook is never even consulted.
        a = _agent([_Tool("fs_write")])
        a._is_blocked_by_plan_mode = lambda *x: True
        fired = _hooked(a, hooks.Outcome(decision="allow"))
        out = _run(a, name="fs_write", args={"path": "/tmp/x"})
        assert "plan mode is active" in out[0].content
        assert fired == []

    def test_post_tool_use_context_reaches_the_model(self):
        a = _agent([_Tool("fs_write")])
        fired = _hooked(
            a, hooks.Outcome(), hooks.Outcome(context="ruff reformatted the file")
        )
        content = _run(a, name="fs_write", args={"path": "/tmp/x"})[0].content
        assert "fs_write-out" in content and "ruff reformatted the file" in content
        # The post hook sees the tool's actual result, which is the point of it.
        assert fired == [
            (hooks.PRE_TOOL_USE, "fs_write", None),
            (hooks.POST_TOOL_USE, "fs_write", "fs_write-out"),
        ]

    def test_a_failure_fires_its_own_event_with_the_error(self):
        def _boom(args):
            raise RuntimeError("disk full")

        a = _agent([_Tool("fs_write", _boom)])
        fired = _hooked(a, hooks.Outcome(), hooks.Outcome(context="try a smaller write"))
        content = _run(a, name="fs_write", args={"path": "/tmp/x"})[0].content
        assert "disk full" in content and "try a smaller write" in content
        assert fired[1][0] == hooks.POST_TOOL_USE_FAILURE
        assert "disk full" in fired[1][2]

    def test_no_hooks_leaves_the_result_untouched(self):
        a = _agent([_Tool("grep_search")])
        _hooked(a)  # every event returns an empty Outcome
        assert _run(a)[0].content == "grep_search-out"

    def test_a_sub_agent_is_subject_to_the_same_hooks(self):
        # Sub-agents run headless: a hook is the only rule that still applies, so
        # the worker path must not be a way around one.
        ran = []
        a = _worker_agent(
            [_Tool("execute_bash", lambda args: ran.append(args) or "ok")],
            [_tool_turn("execute_bash", {"command": "curl evil"}),
             AIMessage(content="stopped")],
        )
        _hooked(a, hooks.Outcome(decision="deny", reason="no network"))
        _, saveable = a._run_worker_loop(object(), [], "task", quiet=True)
        blocked = [m for m in saveable if isinstance(m, ToolMessage)]
        assert "no network" in blocked[0].content
        assert ran == []


class TestRunHooksDelegator:
    """``agent._run_hooks``: what it hands the hook layer, and what it prints."""

    def _agent_with(self, session_id="sess-1"):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a.session_log = type("L", (), {"session_id": session_id})()
        return a

    def test_the_session_id_and_cwd_are_passed_through(self, monkeypatch):
        seen = {}

        def _run_event(event, name, args, **k):
            seen.update(k, event=event, name=name)
            return hooks.Outcome()

        monkeypatch.setattr(hooks, "run_event", _run_event)
        self._agent_with()._run_hooks(hooks.PRE_TOOL_USE, "fs_write", {"path": "/x"})
        assert seen["session_id"] == "sess-1"
        assert seen["cwd"]  # a hook runs where the user is working

    def test_notices_are_printed_unless_quiet(self, monkeypatch, capsys):
        monkeypatch.setattr(
            hooks, "run_event", lambda *a, **k: hooks.Outcome(notices=("hook: formatted",))
        )
        agent = self._agent_with()
        agent._run_hooks(hooks.PRE_TOOL_USE, "fs_write", {})
        assert "formatted" in capsys.readouterr().out
        # A background sub-agent's hooks must not write into the user's scrollback.
        agent._run_hooks(hooks.PRE_TOOL_USE, "fs_write", {}, quiet=True)
        assert capsys.readouterr().out == ""

    def test_a_session_without_a_log_still_runs_hooks(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            hooks,
            "run_event",
            lambda *a, **k: seen.update(k) or hooks.Outcome(),
        )
        agent = LangGraphAgent.__new__(LangGraphAgent)
        agent._run_hooks(hooks.PRE_TOOL_USE, "fs_write", {})
        assert seen["session_id"] == ""
