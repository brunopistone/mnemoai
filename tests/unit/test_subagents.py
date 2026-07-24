"""Unit tests for spawnable sub-agents (spawn_agent).

Covers the pure registry (client/agent/subagents.py) and the client-side
`_handle_spawn_agent` orchestration on a bare agent with a fake model — no LLM.
A sub-agent runs an isolated model↔tool loop and returns only its final report.
"""

import pytest
from langchain_core.messages import AIMessage

from mnemoai.client.agent import subagents
from mnemoai.client.agent.agent import LangGraphAgent


@pytest.fixture(autouse=True)
def _isolate_agents_dir(tmp_path, monkeypatch):
    """Point ``agents_dir()`` at an empty temp dir so the registry tests see only
    built-ins — never whatever custom agents happen to live in the real
    ``~/.mnemoai/agents/`` on the machine running the suite. Tests that exercise
    custom loading pass their own root to ``_load_custom_subagents`` directly."""
    empty = tmp_path / "agents"
    empty.mkdir()
    monkeypatch.setattr(subagents, "agents_dir", lambda: empty)


class TestRegistry:
    def test_builtin_types_present(self):
        names = {a.name for a in subagents.list_subagents()}
        assert names == {"general-purpose", "explore", "plan"}

    def test_get_is_case_insensitive(self):
        assert subagents.get_subagent("EXPLORE").name == "explore"
        assert subagents.get_subagent("  Plan ").name == "plan"

    def test_unknown_returns_none(self):
        assert subagents.get_subagent("nope") is None
        assert subagents.get_subagent("") is None

    def test_explore_and_plan_are_readonly(self):
        for name in ("explore", "plan"):
            tools = subagents.get_subagent(name).tools
            assert tools is not None
            for banned in ("fs_write", "file_edit", "execute_bash", "git_commit_safe"):
                assert banned not in tools

    def test_general_purpose_has_all_tools(self):
        assert subagents.get_subagent("general-purpose").tools is None

    def test_system_prompt_falls_back_when_absent(self, monkeypatch):
        # No prompts.yaml entry → the in-code fallback is used (crash-guard).
        monkeypatch.setattr(subagents.config, "prompt", lambda k, d=None: None)
        p = subagents.subagent_system_prompt(subagents.get_subagent("explore"))
        assert "read-only" in p.lower()

    def test_system_prompt_prefers_config(self, monkeypatch):
        monkeypatch.setattr(subagents.config, "prompt", lambda k, d=None: "CUSTOM")
        assert subagents.subagent_system_prompt(subagents.get_subagent("plan")) == "CUSTOM"


class TestParseTools:
    def test_none_and_star_mean_all(self):
        assert subagents._parse_tools(None) is None
        assert subagents._parse_tools("*") is None
        assert subagents._parse_tools("all") is None
        assert subagents._parse_tools("") is None
        assert subagents._parse_tools(["*"]) is None
        assert subagents._parse_tools([]) is None

    def test_csv_string(self):
        assert subagents._parse_tools("fs_read, grep_search ,glob_search") == [
            "fs_read", "grep_search", "glob_search"
        ]

    def test_yaml_list(self):
        assert subagents._parse_tools(["fs_read", "grep_search"]) == [
            "fs_read", "grep_search"
        ]


class TestCustomLoader:
    def _write(self, root, name, text):
        (root / name).write_text(text)

    def test_absent_dir_yields_empty(self, tmp_path):
        assert subagents._load_custom_subagents(tmp_path / "nope") == []

    def test_loads_valid_custom_agent(self, tmp_path):
        self._write(
            tmp_path, "reviewer.md",
            "---\nname: reviewer\ndescription: Reviews code\ntools: fs_read, grep_search\n---\n"
            "You are a code reviewer.\n",
        )
        agents = subagents._load_custom_subagents(tmp_path)
        assert len(agents) == 1
        a = agents[0]
        assert a.name == "reviewer"
        assert a.description == "Reviews code"
        assert a.tools == ["fs_read", "grep_search"]
        assert a.source == "custom"
        assert subagents.subagent_system_prompt(a) == "You are a code reviewer."

    def test_name_defaults_to_filename_stem(self, tmp_path):
        self._write(
            tmp_path, "MyAgent.md",
            "---\ndescription: does things\n---\nbody here\n",
        )
        agents = subagents._load_custom_subagents(tmp_path)
        assert agents[0].name == "myagent"  # stem, lowercased

    def test_tolerant_scan_skips_bad_files(self, tmp_path):
        self._write(tmp_path, "good.md",
                    "---\ndescription: ok\n---\nbody\n")
        self._write(tmp_path, "no_front.md", "just a body, no frontmatter\n")
        self._write(tmp_path, "no_desc.md", "---\nname: x\n---\nbody\n")
        self._write(tmp_path, "empty_body.md", "---\ndescription: d\n---\n   \n")
        self._write(tmp_path, "notmd.txt", "---\ndescription: d\n---\nbody\n")
        names = {a.name for a in subagents._load_custom_subagents(tmp_path)}
        assert names == {"good"}

    def test_custom_overrides_builtin(self, tmp_path, monkeypatch):
        self._write(
            tmp_path, "explore.md",
            "---\ndescription: my explore\n---\ncustom explore prompt\n",
        )
        monkeypatch.setattr(subagents, "agents_dir", lambda: tmp_path)
        agents = {a.name: a for a in subagents.list_subagents()}
        assert agents["explore"].source == "custom"
        assert agents["explore"].description == "my explore"


class TestAvailableSubagentsBlock:
    def test_lists_builtins(self):
        block = subagents.available_subagents_block()
        assert "<available_subagents>" in block
        assert "general-purpose" in block and "explore" in block and "plan" in block
        assert "spawn_agent" in block

    def test_tool_scope_annotation(self):
        block = subagents.available_subagents_block()
        # general-purpose has the full toolset → "(Tools: all)"
        assert "(Tools: all)" in block
        # read-only types list their allowlist (e.g. grep_search), not writes
        assert "grep_search" in block


class _Tool:
    def __init__(self, name):
        self.name = name


def _agent(tools):
    from mnemoai.client.agent.agent_activity import AgentActivityStore

    a = LangGraphAgent.__new__(LangGraphAgent)
    a.tools = [_Tool(n) for n in tools]
    a._spawn_depth = 0
    a.verbose = False
    a.model = _FakeModel()
    a.callbacks = []  # spinner start/stop become harmless no-ops
    a._activity = AgentActivityStore()  # _run_one_subagent opens an activity run
    return a


class _FakeModel:
    def bind_tools(self, tools):
        # Record the bound subset so tests can assert what the sub-agent got.
        self.bound = [t.name for t in tools]
        return self


class TestSubagentTools:
    def test_readonly_type_excludes_write_and_spawn(self):
        a = _agent(["fs_read", "fs_write", "execute_bash", "grep_search", "spawn_agent"])
        subset = {t.name for t in a._subagent_tools(subagents.get_subagent("explore"))}
        assert "grep_search" in subset and "fs_read" in subset
        assert "fs_write" not in subset and "execute_bash" not in subset
        assert "spawn_agent" not in subset  # no nested spawning

    def test_general_purpose_gets_all_but_spawn(self):
        a = _agent(["fs_read", "fs_write", "execute_bash", "spawn_agent"])
        subset = {t.name for t in a._subagent_tools(subagents.get_subagent("general-purpose"))}
        assert subset == {"fs_read", "fs_write", "execute_bash"}
        assert "spawn_agent" not in subset

    def test_meta_tools_always_included(self):
        # fs_read/describe_image are added even if not in the type's allowlist.
        a = _agent(["fs_read", "describe_image", "grep_search"])
        subset = {t.name for t in a._subagent_tools(subagents.get_subagent("explore"))}
        assert "fs_read" in subset and "describe_image" in subset


class TestCustomLoaderDenylistModel:
    def _write(self, root, name, text):
        (root / name).write_text(text)

    def test_loads_disallowed_tools_and_model(self, tmp_path):
        self._write(
            tmp_path, "safe.md",
            "---\ndescription: safe agent\ntools: fs_read, grep_search, execute_bash\n"
            "disallowed-tools: execute_bash, fs_write\nmodel: haiku-cheap\n---\nbody\n",
        )
        a = subagents._load_custom_subagents(tmp_path)[0]
        assert a.disallowed_tools == ["execute_bash", "fs_write"]
        assert a.model == "haiku-cheap"

    def test_disallowed_tools_underscore_variant(self, tmp_path):
        self._write(
            tmp_path, "u.md",
            "---\ndescription: d\ndisallowed_tools: fs_write\n---\nbody\n",
        )
        a = subagents._load_custom_subagents(tmp_path)[0]
        assert a.disallowed_tools == ["fs_write"]

    def test_absent_denylist_and_model_are_none(self, tmp_path):
        self._write(tmp_path, "p.md", "---\ndescription: d\n---\nbody\n")
        a = subagents._load_custom_subagents(tmp_path)[0]
        assert a.disallowed_tools is None
        assert a.model is None


class TestSubagentDenylist:
    def test_denylist_removes_tool_after_allowlist(self):
        a = _agent(["fs_read", "grep_search", "execute_bash"])
        agent_type = subagents.SubAgentType(
            name="x", description="d",
            tools=["fs_read", "grep_search", "execute_bash"],
            prompt_key="", fallback_prompt="", inline_prompt="p", source="custom",
            disallowed_tools=["execute_bash"],
        )
        subset = {t.name for t in a._subagent_tools(agent_type)}
        assert "grep_search" in subset and "execute_bash" not in subset

    def test_denylist_can_remove_meta_tool(self):
        a = _agent(["fs_read", "describe_image", "grep_search"])
        agent_type = subagents.SubAgentType(
            name="x", description="d", tools=["grep_search"],
            prompt_key="", fallback_prompt="", inline_prompt="p", source="custom",
            disallowed_tools=["describe_image"],
        )
        subset = {t.name for t in a._subagent_tools(agent_type)}
        assert "describe_image" not in subset  # denylist wins over meta

    def test_denylist_on_all_tools_type(self):
        a = _agent(["fs_read", "fs_write", "execute_bash", "spawn_agent"])
        agent_type = subagents.SubAgentType(
            name="x", description="d", tools=None,  # all
            prompt_key="", fallback_prompt="", inline_prompt="p", source="custom",
            disallowed_tools=["fs_write"],
        )
        subset = {t.name for t in a._subagent_tools(agent_type)}
        assert "fs_write" not in subset and "execute_bash" in subset
        assert "spawn_agent" not in subset  # still no nested spawning

    def test_deny_all_sentinel_removes_everything(self):
        # disallowed-tools: "*" means deny EVERYTHING (lockdown), not "deny
        # nothing" — _parse_denylist yields the ["*"] sentinel.
        assert subagents._parse_denylist("*") == ["*"]
        assert subagents._parse_denylist("all") == ["*"]
        a = _agent(["fs_read", "grep_search", "execute_bash"])
        agent_type = subagents.SubAgentType(
            name="x", description="d", tools=None,
            prompt_key="", fallback_prompt="", inline_prompt="p", source="custom",
            disallowed_tools=["*"],
        )
        assert a._subagent_tools(agent_type) == []  # nothing survives

    def test_parse_denylist_absent_is_none(self):
        assert subagents._parse_denylist(None) is None
        assert subagents._parse_denylist("") is None
        assert subagents._parse_denylist(["fs_write", "execute_bash"]) == [
            "fs_write", "execute_bash"
        ]


class TestSubagentModelOverride:
    def _custom(self, model):
        return subagents.SubAgentType(
            name="x", description="d", tools=None,
            prompt_key="", fallback_prompt="", inline_prompt="p", source="custom",
            model=model,
        )

    def test_override_uses_factory(self):
        a = _agent(["fs_read"])
        requested = {}
        sentinel = object()

        def _factory(name):
            requested["name"] = name
            return sentinel

        a._subagent_model_factory = _factory
        assert a._subagent_base_model(self._custom("cheap")) is sentinel
        assert requested["name"] == "cheap"

    def test_no_override_uses_callback_free_model(self):
        a = _agent(["fs_read"])
        called = {"factory": False}

        def _factory(name):
            called["factory"] = True
            return object()

        a._subagent_model_factory = _factory
        # A built-in type has model=None → factory must NOT be consulted.
        a._subagent_base_model(subagents.get_subagent("explore"))
        assert called["factory"] is False

    def test_factory_failure_falls_back(self):
        a = _agent(["fs_read"])
        a._subagent_model_factory = lambda name: None  # build failed
        # Falls back to the (callback-free) parent model, no crash.
        assert a._subagent_base_model(self._custom("bad")) is a.model


class TestHandleSpawnAgent:
    def _spawn_agent(self):
        a = _agent(["fs_read", "grep_search"])
        # Stub the isolated loop: return a fixed report + no messages.
        a._run_worker_loop = (
            lambda model, tools, prompt, **kw: (f"REPORT for: {prompt}", [])
        )
        return a

    def test_returns_final_report_with_isolation_note(self):
        a = self._spawn_agent()
        out = a._handle_spawn_agent("explore", "find the config loader")
        assert "REPORT for: find the config loader" in out
        assert "not shown to the user" in out  # parent told to summarize

    def test_nested_spawn_blocked(self):
        a = self._spawn_agent()
        a._spawn_depth = 1
        out = a._handle_spawn_agent("explore", "x")
        assert "cannot spawn" in out

    def test_unknown_type_lists_available(self):
        a = self._spawn_agent()
        out = a._handle_spawn_agent("wizard", "x")
        assert "Unknown agent_type" in out and "explore" in out

    def test_empty_prompt_rejected(self):
        a = self._spawn_agent()
        assert "non-empty prompt" in a._handle_spawn_agent("explore", "   ")

    def test_depth_restored_after_run(self):
        a = self._spawn_agent()
        a._handle_spawn_agent("explore", "x")
        assert a._spawn_depth == 0  # decremented back

    def test_depth_restored_on_error(self):
        a = self._spawn_agent()

        def _boom(*args, **kwargs):
            raise RuntimeError("loop blew up")

        a._run_worker_loop = _boom
        out = a._handle_spawn_agent("explore", "x")
        assert "failed" in out.lower()
        assert a._spawn_depth == 0  # finally restored it

    def test_system_prompt_passed_to_loop(self, monkeypatch):
        a = self._spawn_agent()
        monkeypatch.setattr(
            subagents.config, "prompt", lambda k, d=None: "EXPLORE-SYS-PROMPT"
        )
        seen = {}
        a._run_worker_loop = (
            lambda model, tools, prompt, system_prompt=None, **kw: (
                seen.update(sp=system_prompt) or ("ok", [])
            )
        )
        a._handle_spawn_agent("explore", "x")
        assert seen["sp"] == "EXPLORE-SYS-PROMPT"

    def test_gets_generous_iteration_budget(self):
        # A sub-agent must get the main-loop budget (recursion_limit), not the
        # orchestrator-worker default of 10 — otherwise exploration starves.
        a = self._spawn_agent()
        a.recursion_limit = 200
        seen = {}
        a._run_worker_loop = (
            lambda model, tools, prompt, max_iterations=10, **kw: (
                seen.update(mi=max_iterations) or ("ok", [])
            )
        )
        a._handle_spawn_agent("explore", "x")
        assert seen["mi"] == 200

    def test_iteration_budget_defaults_when_unset(self):
        # Bare agent with no recursion_limit falls back to 200 (not 10).
        a = self._spawn_agent()  # no recursion_limit attribute
        seen = {}
        a._run_worker_loop = (
            lambda model, tools, prompt, max_iterations=10, **kw: (
                seen.update(mi=max_iterations) or ("ok", [])
            )
        )
        a._handle_spawn_agent("explore", "x")
        assert seen["mi"] == 200

    def test_always_available_includes_spawn_agent(self):
        assert "spawn_agent" in LangGraphAgent._ALWAYS_AVAILABLE_TOOLS


class TestWorkerLoopSystemPromptOverride:
    """_run_worker_loop uses the given system_prompt (sub-agent's) over the
    parent's self.system_prompt, and runs on an isolated message list."""

    def test_override_used_over_parent_prompt(self):
        from langchain_core.messages import SystemMessage

        a = LangGraphAgent.__new__(LangGraphAgent)
        a.system_prompt = "PARENT PROMPT"
        a.callbacks = []
        a._start_spinner = lambda *x, **k: None
        a._stop_spinner = lambda: None
        a.verbose = False
        captured = {}

        def _stream(msgs, config, model=None, quiet=False, **k):
            captured["sys"] = msgs[0].content if isinstance(msgs[0], SystemMessage) else None
            return AIMessage(content="done"), False

        a._stream_response = _stream
        a._extract_visible = lambda c: c if isinstance(c, str) else ""
        text, _ = a._run_worker_loop(object(), [], "task", system_prompt="SUB PROMPT")
        assert captured["sys"] == "SUB PROMPT"
        assert text == "done"


class TestQuietWorkerLoop:
    """A quiet (spawned) sub-agent STILL streams (so it keeps the idle-timeout +
    retry) but silently — via _stream_once_quiet, never the display path
    (_stream_once) and never any print/shared formatter."""

    def _agent_for_quiet(self, turns):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a.system_prompt = "SYS"
        a.callbacks = []
        a.verbose = False
        a.tools = []
        a._extract_visible = lambda c: c if isinstance(c, str) else ""
        # _run_worker_loop calls _stream_response; capture that it was told to be
        # quiet, and hand back the next canned turn (no real model/stream).
        state = {"i": 0, "quiet_seen": []}

        def _stream_response(msgs, config, model=None, quiet=False, **k):
            state["quiet_seen"].append(quiet)
            r = turns[state["i"]]
            state["i"] += 1
            return r, False

        a._stream_response = _stream_response
        a._quiet_state = state
        return a

    def test_quiet_streams_silently(self):
        a = self._agent_for_quiet([AIMessage(content="quiet-done")])
        progressed = []
        text, _ = a._run_worker_loop(
            object(), [], "task", quiet=True, progress=progressed.append
        )
        assert text == "quiet-done"
        assert progressed == []  # no tool calls → no progress ticks
        # The loop streamed with quiet=True (keeps idle-timeout+retry, no display).
        assert a._quiet_state["quiet_seen"] == [True]

    def test_cancel_stops_worker_loop_before_streaming(self):
        # A cancelled turn must abort a sub-agent worker at the top of the loop
        # (pool/daemon threads can't receive the injected KeyboardInterrupt, so
        # they must poll _cancelled()) — else it keeps looping after "Stopped".
        import threading

        a = self._agent_for_quiet([AIMessage(content="should-not-run")])
        a._cancel_event = threading.Event()
        a._cancel_event.set()  # already cancelled
        a._last_visible_from = lambda msgs: ""
        text, saveable = a._run_worker_loop(object(), [], "task", quiet=True)
        assert text == "(cancelled)"
        # It bailed at the top of the loop — never called _stream_response.
        assert a._quiet_state["quiet_seen"] == []

    def test_no_cancel_event_runs_normally(self):
        # A bare agent without a _cancel_event attr must not crash (_cancelled
        # guards with getattr) and runs as usual.
        a = self._agent_for_quiet([AIMessage(content="done")])
        text, _ = a._run_worker_loop(object(), [], "task", quiet=True)
        assert text == "done"

    def test_per_run_stop_aborts_even_when_turn_not_cancelled(self):
        # x / stop-all set a PER-RUN cancel (via the activity sink) — the worker
        # must stop even though the global turn cancel is NOT set (this is what
        # lets a BACKGROUND agent be stopped across turns).
        import threading

        from mnemoai.client.agent.agent_activity import AgentActivityStore

        a = self._agent_for_quiet([AIMessage(content="should-not-run")])
        a._cancel_event = threading.Event()  # global NOT set
        a._last_visible_from = lambda msgs: ""
        store = AgentActivityStore()
        sink = store.open_run("explore", "d", "background")
        store.request_stop(sink._run_id)  # per-run stop only
        text, _ = a._run_worker_loop(object(), [], "task", quiet=True, activity=sink)
        assert text == "(cancelled)"
        assert a._quiet_state["quiet_seen"] == []  # never streamed

    def test_should_continue_ends_the_graph_when_cancelled(self):
        # "Stop all" cancels the turn; the graph must END even with pending tool
        # calls, else it loops into another model call after tools return and the
        # turn keeps going (the model resumes with the stopped agents' partials).
        import threading

        a = LangGraphAgent.__new__(LangGraphAgent)
        a._cancel_event = threading.Event()
        ai = AIMessage(content="")
        ai.tool_calls = [{"name": "x", "args": {}, "id": "c0", "type": "tool_call"}]
        state = {"messages": [ai]}
        assert a._should_continue(state) == "continue"  # not cancelled → loop
        a._cancel_event.set()
        assert a._should_continue(state) == "end"  # cancelled → stop the turn

    def test_quiet_progress_counts_tool_calls(self):
        turns = [
            AIMessage(
                content="",
                tool_calls=[{"name": "grep_search", "id": "t1", "args": {}}],
            ),
            AIMessage(content="final"),
        ]
        a = self._agent_for_quiet(turns)
        a._normalize_tool_args = lambda args: args
        a._truncate_tool_result = lambda s: s
        a._is_blocked_by_plan_mode = lambda *x: False
        a._confirm_tool = lambda *x: True
        invoked_quiet = []
        a._invoke_tool = (
            lambda tool, name, args, quiet=False: invoked_quiet.append(quiet)
            or "TOOL_OUT"
        )
        tool = _Tool("grep_search")

        notes = []
        text, _ = a._run_worker_loop(
            object(), [tool], "task", quiet=True, progress=notes.append
        )
        assert text == "final"
        assert notes and "1 tool call" in notes[0]
        assert a._quiet_state["quiet_seen"] == [True, True]  # both turns quiet
        # The tool ran quiet too → _invoke_tool won't touch the shared spinner.
        assert invoked_quiet == [True]

    def test_invoke_tool_quiet_touches_no_spinner(self):
        # Regression: per-tool spinner start/stop inside a quiet sub-agent would
        # clobber the batch's shared "N sub-agents running…" line (the flicker).
        a = LangGraphAgent.__new__(LangGraphAgent)
        a._SELF_REPORTING_TOOLS = set()
        touched = []
        a._start_spinner = lambda *x, **k: touched.append("start")
        a._stop_spinner = lambda *x, **k: touched.append("stop")

        class _T:
            def invoke(self, args):
                return "OUT"

        assert a._invoke_tool(_T(), "grep_search", {}, quiet=True) == "OUT"
        assert touched == []  # quiet path never touches the spinner
        # non-quiet path DOES manage the spinner (baseline).
        a._tool_progress_label = lambda n, args: "lbl"
        a._invoke_tool(_T(), "grep_search", {}, quiet=False)
        assert touched  # started/stopped for the visible path


class TestSpawnConcurrency:
    """Thread-local spawn depth + parallel batch aggregation."""

    def test_spawn_depth_is_thread_local(self):
        import threading

        a = LangGraphAgent.__new__(LangGraphAgent)
        a._spawn_depth_tl = threading.local()
        # Main thread sets depth 1 (as if inside a spawn).
        a._spawn_depth = 1
        seen = {}

        def _worker():
            seen["child"] = a._spawn_depth  # a fresh thread must see 0

        t = threading.Thread(target=_worker)
        t.start()
        t.join()
        assert a._spawn_depth == 1  # main thread unchanged
        assert seen["child"] == 0  # sibling thread isolated

    def test_batch_runs_multiple_spawns_and_keys_by_id(self):
        a = _agent(["fs_read"])
        a._max_subagent_concurrency = 4
        # Stub the per-spawn runner so no real model/loop is needed.
        a._handle_spawn_agent = (
            lambda at, prompt, description="", in_batch=False:
            f"RESULT[{prompt}]"
        )
        # Only FOREGROUND spawns (run_in_background=False) join the wait-batch;
        # background is the default, so foreground must be explicit.
        calls = [
            {"name": "spawn_agent", "id": "a",
             "args": {"prompt": "one", "run_in_background": False}},
            {"name": "spawn_agent", "id": "b",
             "args": {"prompt": "two", "run_in_background": False}},
            {"name": "grep_search", "id": "c", "args": {}},  # ignored by batch
        ]
        results = a._run_spawn_batch(calls)
        assert results == {"a": "RESULT[one]", "b": "RESULT[two]"}

    def test_batch_excludes_background_spawns(self):
        # Background is the default: an omitted (or true) run_in_background is NOT
        # batched (those launch detached via the inline path), so the batch is a
        # no-op even with several such spawns.
        a = _agent(["fs_read"])
        a._max_subagent_concurrency = 4
        calls = [
            {"name": "spawn_agent", "id": "a", "args": {"prompt": "one"}},  # default bg
            {"name": "spawn_agent", "id": "b",
             "args": {"prompt": "two", "run_in_background": True}},
        ]
        assert a._run_spawn_batch(calls) == {}

    def test_batch_noop_for_single_spawn(self):
        a = _agent(["fs_read"])
        a._max_subagent_concurrency = 4
        calls = [{"name": "spawn_agent", "id": "a",
                  "args": {"prompt": "only", "run_in_background": False}}]
        assert a._run_spawn_batch(calls) == {}  # inline path handles a lone spawn

    def test_batch_noop_when_concurrency_disabled(self):
        a = _agent(["fs_read"])
        a._max_subagent_concurrency = 1  # forced sequential
        calls = [
            {"name": "spawn_agent", "id": "a",
             "args": {"prompt": "one", "run_in_background": False}},
            {"name": "spawn_agent", "id": "b",
             "args": {"prompt": "two", "run_in_background": False}},
        ]
        assert a._run_spawn_batch(calls) == {}


class TestOrchestratorScheduling:
    """_run_subtasks_scheduled runs independent subtasks concurrently and waits
    for declared dependencies, threading only depended-on results into a subtask."""

    def _agent(self):
        from mnemoai.client.agent.agent_activity import AgentActivityStore

        a = LangGraphAgent.__new__(LangGraphAgent)
        a._max_subagent_concurrency = 4
        a.verbose = False
        a.callbacks = []
        a.tools = []
        a.tools_by_route = None  # → every subtask uses the full toolset
        a.model = object()
        a.model_with_tools = object()
        a._activity = AgentActivityStore()  # orchestrator opens an activity run
        # Record the prompt each subtask received + return a marker result.
        seen = {}

        def _loop(model, tools, prompt, **kw):
            # index encoded in the description "s<i>"
            return f"RESULT<{prompt.splitlines()[-1]}>", []

        a._run_worker_loop = _loop
        a._seen = seen
        return a

    def test_independent_subtasks_all_run(self):
        a = self._agent()
        subtasks = [
            {"description": "s0", "category": "full", "depends_on": []},
            {"description": "s1", "category": "full", "depends_on": []},
        ]
        results = a._run_subtasks_scheduled(subtasks)
        assert set(results.keys()) == {0, 1}
        assert results[0]["task"] == "s0" and results[1]["task"] == "s1"

    def test_parallel_wave_workers_run_headless(self):
        # Concurrent workers must be headless (auto-deny destructive tools) —
        # stacking interactive confirm prompts on one terminal is unworkable.
        import threading

        a = self._agent()
        a._headless_tl = threading.local()
        headless_seen = []

        def _loop(model, tools, prompt, **kw):
            headless_seen.append(a._is_headless())
            return "R", []

        a._run_worker_loop = _loop
        subtasks = [
            {"description": "s0", "category": "full", "depends_on": []},
            {"description": "s1", "category": "full", "depends_on": []},
        ]
        a._run_subtasks_scheduled(subtasks)  # 2 ready → parallel wave
        assert headless_seen == [True, True]  # both ran headless
        # And the flag is cleared on the main thread afterward.
        assert a._is_headless() is False

    def test_lone_wave_worker_is_not_headless(self):
        # A single-ready wave runs inline on the main thread and CAN prompt.
        import threading

        a = self._agent()
        a._headless_tl = threading.local()
        headless_seen = []

        def _loop(model, tools, prompt, **kw):
            headless_seen.append(a._is_headless())
            return "R", []

        a._run_worker_loop = _loop
        # A dependency chain → each wave has exactly one ready subtask.
        subtasks = [
            {"description": "s0", "category": "full", "depends_on": []},
            {"description": "s1", "category": "full", "depends_on": [0]},
        ]
        a._run_subtasks_scheduled(subtasks)
        assert headless_seen == [False, False]  # neither ran headless

    def test_dependency_threads_result_into_dependent(self):
        a = self._agent()
        # s1 depends on s0 → s1's prompt must include s0's completed result.
        prompts = []

        def _loop(model, tools, prompt, **kw):
            prompts.append(prompt)
            # Return a distinctive result for s0 so we can find it in s1's prompt.
            return ("RESULT_S0" if "s0" in prompt else "RESULT_S1"), []

        a._run_worker_loop = _loop
        subtasks = [
            {"description": "s0", "category": "full", "depends_on": []},
            {"description": "s1", "category": "full", "depends_on": [0]},
        ]
        results = a._run_subtasks_scheduled(subtasks)
        assert set(results.keys()) == {0, 1}
        # s0 ran first with no context; s1 ran with s0's result threaded in.
        s1_prompt = next(p for p in prompts if p.rstrip().endswith("s1"))
        assert "Context from completed steps" in s1_prompt
        assert "RESULT_S0" in s1_prompt

    def test_broken_cycle_still_completes(self):
        # Both declare a dep that can never resolve within the graph (sanitized
        # away in practice, but guard against a hand-built cyclic input).
        a = self._agent()
        subtasks = [
            {"description": "s0", "category": "full", "depends_on": [1]},
            {"description": "s1", "category": "full", "depends_on": [0]},
        ]
        results = a._run_subtasks_scheduled(subtasks)
        assert set(results.keys()) == {0, 1}  # forced through, no deadlock

    def test_independent_subtask_gets_no_context(self):
        a = self._agent()
        captured = {}

        def _loop(model, tools, prompt, **kw):
            captured[prompt] = True
            return "R", []

        a._run_worker_loop = _loop
        subtasks = [{"description": "solo", "category": "full", "depends_on": []}]
        a._run_subtasks_scheduled(subtasks)
        # A lone independent subtask's prompt is just its description.
        assert "solo" in captured and all(
            "Context from completed steps" not in p for p in captured
        )

    def test_history_threaded_into_each_worker(self):
        # The real prior-message LIST is passed to each worker via history=.
        a = self._agent()
        seen = []

        def _loop(model, tools, prompt, **kw):
            seen.append(kw.get("history"))
            return "R", []

        a._run_worker_loop = _loop
        from langchain_core.messages import AIMessage, HumanMessage

        hist = [HumanMessage(content="draft"), AIMessage(content="THE ISSUE")]
        subtasks = [
            {"description": "s0", "category": "full", "depends_on": []},
            {"description": "s1", "category": "full", "depends_on": []},
        ]
        a._run_subtasks_scheduled(subtasks, hist)
        # Every worker received the SAME real message list (not a text block).
        assert all(h == hist for h in seen)

    def test_history_default_none_no_change(self):
        # Default None (existing callers) → workers get history=None (isolation).
        a = self._agent()
        seen = []
        a._run_worker_loop = lambda model, tools, prompt, **kw: (
            seen.append(kw.get("history")), ("R", [])
        )[1]
        subtasks = [{"description": "solo", "category": "full", "depends_on": []}]
        a._run_subtasks_scheduled(subtasks)  # no history arg
        assert seen == [None]


class TestPriorHistory:
    """_prior_history returns the REAL prior message list (uncapped): drops the
    leading system prompt(s) + the trailing current-query HumanMessage, and
    repairs tool-pairing. No text rendering, no count/token cap."""

    def _agent(self):
        a = LangGraphAgent.__new__(LangGraphAgent)
        return a

    def test_single_turn_returns_empty(self):
        from langchain_core.messages import HumanMessage, SystemMessage

        a = self._agent()
        msgs = [SystemMessage(content="SYS"), HumanMessage(content="hi")]
        assert a._prior_history(msgs) == []

    def test_drops_system_and_current_query_keeps_prior_turns(self):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        a = self._agent()
        prior_ai = AIMessage(content="# Title\nheterogeneous ModelTrainer bug — full issue")
        msgs = [
            SystemMessage(content="SYS"),
            HumanMessage(content="draft a github issue"),
            prior_ai,
            HumanMessage(content="write in a .md file under ~/Desktop"),  # current
        ]
        hist = a._prior_history(msgs)
        # System prompt dropped; current query dropped; prior turns kept as REAL
        # messages (the drafted artifact survives verbatim, not truncated text).
        assert not any(isinstance(m, SystemMessage) for m in hist)
        assert prior_ai in hist
        assert all(getattr(m, "content", "") != "write in a .md file under ~/Desktop" for m in hist)
        assert "heterogeneous ModelTrainer bug" in hist[-1].content  # full, uncapped

    def test_empty_messages(self):
        a = self._agent()
        assert a._prior_history([]) == []

    def test_sanitizes_orphaned_tool_pair(self):
        # An assistant tool_call with no matching ToolMessage (orphan from a
        # compaction slice) must be repaired so a strict provider won't 400.
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        a = self._agent()
        orphan = AIMessage(content="", tool_calls=[{"name": "grep", "args": {}, "id": "x"}])
        msgs = [
            SystemMessage(content="SYS"),
            HumanMessage(content="do a thing"),
            orphan,  # tool_call with no following ToolMessage
            HumanMessage(content="current"),
        ]
        hist = a._prior_history(msgs)
        # sanitize_tool_pairs drops the orphaned tool-call turn (no result).
        assert orphan not in hist


class TestWorkerLoopHistory:
    """_run_worker_loop inserts history BETWEEN the system prompt and the
    subtask, and excludes injected context from the saved messages."""

    def _agent(self):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a.system_prompt = "WORKER_SYS"
        a.callbacks = []
        a._start_spinner = lambda *x, **k: None
        a._stop_spinner = lambda *x, **k: None
        return a

    def test_history_inserted_between_system_and_subtask(self):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        a = self._agent()
        captured = {}

        def _stream_response(messages, config, **kw):
            captured["messages"] = list(messages)
            return AIMessage(content="done"), None

        a._stream_response = _stream_response
        a._extract_visible = lambda c: c if isinstance(c, str) else ""
        a._extract_thinking = lambda r: None
        hist = [HumanMessage(content="draft"), AIMessage(content="THE ISSUE")]
        result, saveable = a._run_worker_loop(
            object(), [], "write it to a file", quiet=True, history=hist
        )
        msgs = captured["messages"]
        # [System(WORKER_SYS), Human(draft), AI(THE ISSUE), Human(subtask)]
        assert isinstance(msgs[0], SystemMessage)
        assert msgs[1] == hist[0] and msgs[2] == hist[1]
        assert isinstance(msgs[3], HumanMessage) and "write it to a file" in msgs[3].content
        # saveable EXCLUDES the system prompt AND the injected history (only this
        # worker's own turns are saved — no per-worker history duplication).
        assert hist[0] not in saveable and hist[1] not in saveable
        assert not any(isinstance(m, SystemMessage) for m in saveable)

    def test_no_history_is_byte_equivalent(self):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        a = self._agent()
        captured = {}
        a._stream_response = lambda messages, config, **kw: (
            captured.update(messages=list(messages)), (AIMessage(content="done"), None)
        )[1]
        a._extract_visible = lambda c: c if isinstance(c, str) else ""
        a._extract_thinking = lambda r: None
        a._run_worker_loop(object(), [], "solo task", quiet=True)  # history=None
        msgs = captured["messages"]
        # Old shape preserved: [System, Human(subtask)] exactly — isolation intact.
        assert len(msgs) == 2
        assert isinstance(msgs[0], SystemMessage) and isinstance(msgs[1], HumanMessage)


class TestOrchestrateEndToEnd:
    """_orchestrate assembles the final answer. Regression: a MULTI-subtask run
    must not crash (the aggregate branch once fell inside `if len==1`, leaving
    final_content unbound → UnboundLocalError for every multi-subtask task)."""

    def _agent(self, subtasks):
        from langchain_core.messages import HumanMessage

        from mnemoai.client.agent.agent_activity import AgentActivityStore

        a = LangGraphAgent.__new__(LangGraphAgent)
        a._max_subagent_concurrency = 4
        a.verbose = False
        a.callbacks = []
        a.tools = []
        a.tools_by_route = None
        a.model = object()
        a.model_with_tools = object()
        a._activity = AgentActivityStore()
        a._steer_queue = []
        a._steer_lock = None
        a._external_tools_prompt_block = lambda: ""
        a._decompose_task = lambda q, p, cats, history=None: subtasks
        a._run_worker_loop = (
            lambda model, tools, prompt, **kw: (f"RESULT[{prompt[:20]}]", [])
        )
        a._aggregate_results = (
            lambda query, results, prompt, steering=None:
            "AGGREGATED(" + "+".join(r["result"] for r in results) + ")"
        )
        a._state = {"messages": [HumanMessage(content="do a multi-part task")]}
        return a

    def test_multi_subtask_does_not_crash_and_aggregates(self):
        subtasks = [
            {"description": "s0", "category": "full", "depends_on": []},
            {"description": "s1", "category": "full", "depends_on": []},
        ]
        a = self._agent(subtasks)
        out = a._orchestrate(a._state)  # must NOT raise UnboundLocalError
        final = out["messages"][-1]
        assert "AGGREGATED(" in final.content  # aggregator ran for multi-subtask

    def test_single_subtask_returns_result_without_aggregating(self):
        subtasks = [{"description": "only", "category": "full", "depends_on": []}]
        a = self._agent(subtasks)
        # Aggregator must NOT be called for a single subtask.
        a._aggregate_results = lambda *x, **k: (_ for _ in ()).throw(
            AssertionError("aggregator must not run for a single subtask")
        )
        out = a._orchestrate(a._state)
        assert "RESULT[" in out["messages"][-1].content

    def test_aggregation_failure_falls_back_to_concatenation(self):
        subtasks = [
            {"description": "s0", "category": "full", "depends_on": []},
            {"description": "s1", "category": "full", "depends_on": []},
        ]
        a = self._agent(subtasks)
        a._stop_spinner = lambda *x, **k: None
        a._aggregate_results = lambda *x, **k: (_ for _ in ()).throw(
            RuntimeError("synthesis boom")
        )
        out = a._orchestrate(a._state)  # must not crash — concatenates instead
        content = out["messages"][-1].content
        assert "### s0" in content and "### s1" in content

    def test_orchestrate_uses_pre_decomposed_subtasks(self):
        # When the ``decompose`` node already produced subtasks (they're in
        # state), _orchestrate reuses them and NEVER re-decomposes.
        subtasks = [
            {"description": "s0", "category": "full", "depends_on": []},
            {"description": "s1", "category": "full", "depends_on": []},
        ]
        a = self._agent(subtasks)
        a._decompose_task = lambda *x, **k: (_ for _ in ()).throw(
            AssertionError("must not re-decompose when subtasks are in state")
        )
        state = dict(a._state)
        state["subtasks"] = subtasks
        out = a._orchestrate(state)  # must not call _decompose_task
        assert "AGGREGATED(" in out["messages"][-1].content


class TestDecomposeNode:
    """The `decompose` graph node stashes subtasks into state and adds no
    messages (so a fallback to the streaming agent keeps history pristine)."""

    def _agent(self, subtasks):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a._external_tools_prompt_block = lambda: ""
        self._captured = {}

        def _decompose_task(q, p, cats, history=None):
            self._captured["query"] = q
            self._captured["history"] = history
            return subtasks

        a._decompose_task = _decompose_task
        return a

    def test_stashes_subtasks_and_adds_no_messages(self):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        subtasks = [
            {"description": "s0", "category": "full", "depends_on": []},
            {"description": "s1", "category": "full", "depends_on": []},
        ]
        a = self._agent(subtasks)
        prior_ai = AIMessage(content="prior artifact")
        state = {
            "messages": [
                SystemMessage(content="SYS"),
                HumanMessage(content="first ask"),
                prior_ai,
                HumanMessage(content="the current multi-part request"),
            ]
        }
        out = a._decompose(state)
        assert out == {"subtasks": subtasks}
        assert "messages" not in out  # agent history stays pristine on fallback
        # Decomposed against the current query with the REAL prior history.
        assert self._captured["query"] == "the current multi-part request"
        assert prior_ai in self._captured["history"]

    def test_no_human_message_returns_empty_subtasks(self):
        from langchain_core.messages import SystemMessage

        a = self._agent([{"description": "x", "category": "full", "depends_on": []}])
        out = a._decompose({"messages": [SystemMessage(content="SYS")]})
        assert out == {"subtasks": []}  # -> routes to agent


class TestRouteAfterDecompose:
    """_route_after_decompose sends only a genuine multi-step plan (>=2
    subtasks) to the orchestrator; everything else streams via the agent."""

    def _agent(self):
        return LangGraphAgent.__new__(LangGraphAgent)

    def test_two_subtasks_go_to_orchestrator(self):
        a = self._agent()
        assert a._route_after_decompose({"subtasks": [{}, {}]}) == "orchestrator"

    def test_single_subtask_goes_to_agent(self):
        # The core user decision: the old "Step 1/1" now streams via the agent.
        a = self._agent()
        assert a._route_after_decompose({"subtasks": [{}]}) == "agent"

    def test_zero_or_missing_goes_to_agent(self):
        a = self._agent()
        assert a._route_after_decompose({"subtasks": []}) == "agent"
        assert a._route_after_decompose({}) == "agent"


class TestRouteAfterClassify:
    """_route_after_classify sends a non-trivial 'full' task to the decompose
    node (not straight to the orchestrator); trivial/plan turns skip it."""

    def _agent(self):
        return LangGraphAgent.__new__(LangGraphAgent)

    def test_nontrivial_full_goes_to_decompose(self):
        from langchain_core.messages import HumanMessage

        a = self._agent()
        a._execute_plan_route = False
        state = {
            "route": "full",
            "messages": [
                HumanMessage(
                    content="refactor the auth module across all files and update the tests"
                )
            ],
        }
        assert a._route_after_classify(state) == "decompose"

    def test_trivial_full_goes_to_agent(self):
        from langchain_core.messages import HumanMessage

        a = self._agent()
        a._execute_plan_route = False
        state = {"route": "full", "messages": [HumanMessage(content="hi")]}
        assert a._route_after_classify(state) == "agent"  # no decompose LLM call

    def test_plan_exec_pinned_goes_to_agent(self):
        from langchain_core.messages import HumanMessage

        a = self._agent()
        a._execute_plan_route = True  # approved-plan execution pins the full route
        state = {
            "route": "full",
            "messages": [
                HumanMessage(
                    content="refactor the auth module across all files and update the tests"
                )
            ],
        }
        assert a._route_after_classify(state) == "agent"


class TestGraphShape:
    """The compiled graph gains a `decompose` node between classifier and the
    agent-vs-orchestrator branch when orchestration is on."""

    def test_decompose_node_present_when_orchestration_enabled(self):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a.router = object()  # any truthy router
        a.orchestrator_enabled = True
        graph = a._build_graph()
        nodes = set(graph.get_graph().nodes)
        assert "decompose" in nodes
        assert "orchestrator" in nodes and "agent" in nodes

    def test_no_decompose_node_without_orchestration(self):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a.router = object()
        a.orchestrator_enabled = False
        graph = a._build_graph()
        nodes = set(graph.get_graph().nodes)
        assert "decompose" not in nodes and "orchestrator" not in nodes
