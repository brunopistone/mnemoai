"""Unit tests for Phase 4: background sub-agents + resume + headless confirm.

Covers the registry (client/agent/background_agents.py) and the agent-side
launch/drain/resume + headless auto-deny — on a bare agent with stubbed loops,
no LLM.
"""

import threading

from langchain_core.messages import AIMessage, HumanMessage

from mnemoai.client.agent import subagents
from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.agent.background_agents import BackgroundAgentRegistry


class TestRegistry:
    def test_register_is_running(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mnemoai.client.agent.background_agents.tasks_dir", lambda: tmp_path
        )
        reg = BackgroundAgentRegistry()
        rec = reg.register("explore", "map routing", "investigate routing")
        assert rec.status == "running"
        assert rec.agent_id == "explore-1"
        assert reg.any_running()

    def test_complete_marks_done(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mnemoai.client.agent.background_agents.tasks_dir", lambda: tmp_path
        )
        reg = BackgroundAgentRegistry()
        rec = reg.register("explore", "d", "p")
        reg.complete(rec.agent_id, "the report")
        got = reg.get(rec.agent_id)
        assert got.status == "done" and got.result == "the report"
        assert not reg.any_running()

    def test_complete_failed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mnemoai.client.agent.background_agents.tasks_dir", lambda: tmp_path
        )
        reg = BackgroundAgentRegistry()
        rec = reg.register("plan", "d", "p")
        reg.complete(rec.agent_id, "boom", failed=True)
        assert reg.get(rec.agent_id).status == "failed"

    def test_drain_returns_each_completion_once(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mnemoai.client.agent.background_agents.tasks_dir", lambda: tmp_path
        )
        reg = BackgroundAgentRegistry()
        r1 = reg.register("explore", "d1", "p1")
        r2 = reg.register("explore", "d2", "p2")
        reg.complete(r1.agent_id, "res1")
        first = reg.drain_completed_unnotified()
        assert [r.agent_id for r in first] == [r1.agent_id]
        # r1 already notified → not returned again; r2 still running.
        assert reg.drain_completed_unnotified() == []
        reg.complete(r2.agent_id, "res2")
        second = reg.drain_completed_unnotified()
        assert [r.agent_id for r in second] == [r2.agent_id]

    def test_ids_are_unique_per_type_counter(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mnemoai.client.agent.background_agents.tasks_dir", lambda: tmp_path
        )
        reg = BackgroundAgentRegistry()
        a = reg.register("explore", "d", "p")
        b = reg.register("plan", "d", "p")
        assert a.agent_id == "explore-1" and b.agent_id == "plan-2"

    def test_any_undelivered_tracks_finished_unnotified(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mnemoai.client.agent.background_agents.tasks_dir", lambda: tmp_path
        )
        reg = BackgroundAgentRegistry()
        rec = reg.register("explore", "d", "p")
        assert not reg.any_undelivered()  # still running
        reg.complete(rec.agent_id, "res")
        assert reg.any_undelivered()  # finished, not yet surfaced
        reg.drain_completed_unnotified()  # marks notified
        assert not reg.any_undelivered()


def _agent(tmp_path):
    a = LangGraphAgent.__new__(LangGraphAgent)
    a._headless_tl = threading.local()
    a._bg_agents = BackgroundAgentRegistry()
    a._spawn_depth_tl = threading.local()
    a.verbose = False
    a.tools = []
    return a


class TestHeadlessConfirm:
    def _agent(self):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a._headless_tl = threading.local()
        a._trusted_confirm_categories = set()
        a._CONFIRM_BASH_TOOLS = LangGraphAgent._CONFIRM_BASH_TOOLS
        a._CONFIRM_WRITE_TOOLS = LangGraphAgent._CONFIRM_WRITE_TOOLS
        a._CONFIRM_MEMORY_TOOLS = LangGraphAgent._CONFIRM_MEMORY_TOOLS
        a._preapproved_bash = []
        return a

    def test_headless_auto_denies_untrusted_destructive(self, monkeypatch):
        from mnemoai.client.agent import agent as agent_mod

        monkeypatch.setattr(agent_mod.config, "get", lambda k, d=None: True)
        a = self._agent()
        a._set_headless(True)
        # execute_bash is destructive + untrusted + headless → auto-deny.
        assert a._confirm_tool("execute_bash", {"command": "rm -rf x"}) is False

    def test_headless_allows_pretrusted_category(self, monkeypatch):
        from mnemoai.client.agent import agent as agent_mod

        monkeypatch.setattr(agent_mod.config, "get", lambda k, d=None: True)
        a = self._agent()
        a._set_headless(True)
        a._trusted_confirm_categories = {"bash"}  # pre-approved this session
        assert a._confirm_tool("execute_bash", {"command": "ls"}) is True

    def test_headless_flag_is_thread_local(self):
        a = self._agent()
        a._set_headless(True)
        seen = {}

        def _worker():
            seen["child"] = a._is_headless()  # a fresh thread is NOT headless

        t = threading.Thread(target=_worker)
        t.start()
        t.join()
        assert a._is_headless() is True
        assert seen["child"] is False

    def test_non_destructive_tool_always_proceeds(self):
        a = self._agent()
        a._set_headless(True)
        assert a._confirm_tool("grep_search", {}) is True


class TestBackgroundLaunchAndDrain:
    def _agent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mnemoai.client.agent.background_agents.tasks_dir", lambda: tmp_path
        )
        a = _agent(tmp_path)
        return a

    def test_launch_returns_immediately_with_id(self, tmp_path, monkeypatch):
        a = self._agent(tmp_path, monkeypatch)
        done = threading.Event()

        # Stub the runner so the daemon finishes deterministically.
        def _run_one(agent, prompt, label, drive_spinner=True):
            done.set()
            return "BG REPORT"

        a._run_one_subagent = _run_one
        agent = subagents.get_subagent("explore")
        ack = a._launch_background_subagent(agent, "investigate", "map it")
        assert "Started background sub-agent" in ack
        assert "explore-1" in ack
        done.wait(timeout=5)
        # After the daemon completes, the registry has the result.
        rec = a._bg_agents.get("explore-1")
        assert rec is not None and rec.result == "BG REPORT"

    def test_drain_completions_returns_wrapped_messages(self, tmp_path, monkeypatch):
        a = self._agent(tmp_path, monkeypatch)
        rec = a._bg_agents.register("explore", "map it", "investigate")
        a._bg_agents.complete(rec.agent_id, "the findings")
        msgs = a.drain_background_completions()
        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert "the findings" in msgs[0].content
        assert rec.agent_id in msgs[0].content
        # Drained once → not delivered again.
        assert a.drain_background_completions() == []

    def test_headless_set_during_background_run(self, tmp_path, monkeypatch):
        a = self._agent(tmp_path, monkeypatch)
        captured = {}
        done = threading.Event()

        def _run_one(agent, prompt, label, drive_spinner=True):
            captured["headless"] = a._is_headless()
            done.set()
            return "R"

        a._run_one_subagent = _run_one
        a._launch_background_subagent(
            subagents.get_subagent("explore"), "p", "l"
        )
        done.wait(timeout=5)
        assert captured["headless"] is True  # daemon thread ran headless


class TestDeliveryOnlyTurn:
    """invoke('') runs a delivery-only turn when a background completion is
    pending, and returns '' (no-op) when nothing is pending."""

    def _agent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mnemoai.client.agent.background_agents.tasks_dir", lambda: tmp_path
        )
        a = LangGraphAgent.__new__(LangGraphAgent)
        a._headless_tl = threading.local()
        a._bg_agents = BackgroundAgentRegistry()
        a._messages = []
        a.system_prompt = "SYS"
        a.recursion_limit = 50
        a._thinking = None
        a._compact_provider = None
        # Capture what the graph ran on; return a simple assistant message.
        captured = {}

        class _Graph:
            def invoke(self, state, config=None):
                captured["messages"] = state["messages"]
                return {"messages": state["messages"] + [AIMessage(content="ok")],
                        "thinking": None}

        a.graph = _Graph()
        a._captured = captured
        return a

    def test_empty_prompt_no_pending_is_noop(self, tmp_path, monkeypatch):
        a = self._agent(tmp_path, monkeypatch)
        assert a.invoke("") == ""
        assert "messages" not in a._captured  # graph never ran

    def test_empty_prompt_with_pending_delivers(self, tmp_path, monkeypatch):
        a = self._agent(tmp_path, monkeypatch)
        rec = a._bg_agents.register("explore", "map it", "task")
        a._bg_agents.complete(rec.agent_id, "the findings")
        out = a.invoke("")  # delivery-only turn
        assert out == "ok"
        # The completion report was the input to the graph (no empty user turn).
        ran = a._captured["messages"]
        assert any(
            isinstance(m, HumanMessage) and "the findings" in m.content for m in ran
        )
        # No empty HumanMessage was appended.
        assert all(
            not (isinstance(m, HumanMessage) and not m.content.strip()) for m in ran
        )

    def test_has_undelivered_reflects_state(self, tmp_path, monkeypatch):
        a = self._agent(tmp_path, monkeypatch)
        assert not a.has_undelivered_background()
        rec = a._bg_agents.register("explore", "d", "p")
        a._bg_agents.complete(rec.agent_id, "r")
        assert a.has_undelivered_background()


class TestResumeAgent:
    def _agent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mnemoai.client.agent.background_agents.tasks_dir", lambda: tmp_path
        )
        a = _agent(tmp_path)
        a._run_one_subagent = (
            lambda agent, prompt, label, drive_spinner=True: f"RESUMED<{prompt[:20]}>"
        )
        return a

    def test_resume_unknown_id(self, tmp_path, monkeypatch):
        a = self._agent(tmp_path, monkeypatch)
        out = a._handle_resume_agent("nope-9", "keep going")
        assert "Unknown agent_id" in out

    def test_resume_empty_prompt(self, tmp_path, monkeypatch):
        a = self._agent(tmp_path, monkeypatch)
        rec = a._bg_agents.register("explore", "d", "p")
        a._bg_agents.complete(rec.agent_id, "prior")
        assert "non-empty" in a._handle_resume_agent(rec.agent_id, "  ")

    def test_resume_running_is_refused(self, tmp_path, monkeypatch):
        a = self._agent(tmp_path, monkeypatch)
        rec = a._bg_agents.register("explore", "d", "p")  # still running
        out = a._handle_resume_agent(rec.agent_id, "more")
        assert "still running" in out

    def test_resume_from_disk_after_restart(self, tmp_path, monkeypatch):
        # Simulate a restart: a fresh registry (empty in-memory) but a persisted
        # record on disk from a prior session. resume must find it via disk.
        monkeypatch.setattr(
            "mnemoai.client.agent.background_agents.tasks_dir", lambda: tmp_path
        )
        import json
        (tmp_path / "subagent_explore-9.json").write_text(json.dumps({
            "agent_id": "explore-9",
            "agent_type": "explore",
            "description": "map routing",
            "prompt": "the original task",
            "status": "done",
            "result": "the prior findings",
        }))
        a = self._agent(tmp_path, monkeypatch)  # fresh empty registry
        assert a._bg_agents.get("explore-9") is None  # not in memory
        seen = {}
        a._run_one_subagent = (
            lambda agent, prompt, label, drive_spinner=True: seen.update(p=prompt)
            or "OK"
        )
        out = a._handle_resume_agent(
            "explore-9", "now check tests", run_in_background=False
        )
        assert "not shown to the user" in out
        assert "the original task" in seen["p"]
        assert "the prior findings" in seen["p"]
        assert "now check tests" in seen["p"]

    def test_resume_unknown_when_no_disk_record(self, tmp_path, monkeypatch):
        a = self._agent(tmp_path, monkeypatch)
        out = a._handle_resume_agent("ghost-1", "go")
        assert "no live or persisted record" in out

    def test_resume_ignores_still_running_disk_record(self, tmp_path, monkeypatch):
        # A stale record left 'running' by a crashed process is not resumable.
        monkeypatch.setattr(
            "mnemoai.client.agent.background_agents.tasks_dir", lambda: tmp_path
        )
        import json
        (tmp_path / "subagent_explore-8.json").write_text(json.dumps({
            "agent_id": "explore-8", "agent_type": "explore", "description": "d",
            "prompt": "p", "status": "running", "result": None,
        }))
        a = self._agent(tmp_path, monkeypatch)
        assert a._bg_agents.load_from_disk("explore-8") is None

    def test_resume_threads_prior_context(self, tmp_path, monkeypatch):
        a = self._agent(tmp_path, monkeypatch)
        seen = {}
        a._run_one_subagent = (
            lambda agent, prompt, label, drive_spinner=True: seen.update(p=prompt)
            or "OK"
        )
        rec = a._bg_agents.register("explore", "map routing", "original task")
        a._bg_agents.complete(rec.agent_id, "prior report text")
        out = a._handle_resume_agent(
            rec.agent_id, "now check tests", run_in_background=False
        )
        assert "not shown to the user" in out
        # The resume prompt carries the original task, prior report, and follow-up.
        assert "original task" in seen["p"]
        assert "prior report text" in seen["p"]
        assert "now check tests" in seen["p"]

    def test_resume_defaults_to_background(self, tmp_path, monkeypatch):
        # Resuming defaults to background (matches the original background run):
        # returns an ack immediately, launches on a daemon thread.
        a = self._agent(tmp_path, monkeypatch)
        launched = {}
        a._launch_background_subagent = (
            lambda agent, prompt, label: launched.update(p=prompt, label=label)
            or f"Started background sub-agent 'x' ({label})."
        )
        rec = a._bg_agents.register("explore", "map routing", "original task")
        a._bg_agents.complete(rec.agent_id, "prior findings")
        out = a._handle_resume_agent(rec.agent_id, "keep going")  # no flag → bg
        assert "Started background sub-agent" in out
        # It went through the background launcher, re-briefed with prior context.
        assert "prior findings" in launched["p"]
        assert "keep going" in launched["p"]
        assert "resumed" in launched["label"]


class TestToolInterceptionCoverage:
    """spawn_agent AND resume_agent must be intercepted at BOTH tool chokepoints
    (the top-level _execute_tools AND the orchestrator/worker _run_worker_loop),
    else the orchestrator path can't resume/spawn (it hit 'tool not found')."""

    def _agent_stub_worker_loop(self):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a.verbose = False
        a.callbacks = []
        a.system_prompt = "SYS"
        a.tools = []
        a._extract_visible = lambda c: c if isinstance(c, str) else ""
        a._normalize_tool_args = lambda args: args
        a._is_blocked_by_plan_mode = lambda *x: False
        # Record which client-side handlers the worker loop invokes.
        calls = {}
        a._handle_resume_agent = (
            lambda agent_id, prompt, **kw: calls.setdefault(
                "resume", (agent_id, prompt)
            )
            or "RESUMED"
        )
        a._handle_spawn_agent = (
            lambda *aa, **kw: calls.setdefault("spawn", aa) or "SPAWNED"
        )
        a._calls = calls
        return a

    def test_worker_loop_intercepts_resume_agent(self):
        a = self._agent_stub_worker_loop()
        turns = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "resume_agent",
                        "id": "t1",
                        "args": {"agent_id": "explore-1", "prompt": "keep going"},
                    }
                ],
            ),
            AIMessage(content="final"),
        ]
        state = {"i": 0}

        def _stream(msgs, config, model=None, quiet=False, **k):
            r = turns[state["i"]]
            state["i"] += 1
            return r, False

        a._stream_response = _stream
        text, _ = a._run_worker_loop(object(), [], "resume it", quiet=True)
        assert text == "final"
        # resume_agent was handled client-side (not "tool not found").
        assert a._calls.get("resume") == ("explore-1", "keep going")

    def test_resume_agent_in_always_available_tools(self):
        assert "resume_agent" in LangGraphAgent._ALWAYS_AVAILABLE_TOOLS
