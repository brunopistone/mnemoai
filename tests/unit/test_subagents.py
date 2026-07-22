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
    a = LangGraphAgent.__new__(LangGraphAgent)
    a.tools = [_Tool(n) for n in tools]
    a._spawn_depth = 0
    a.verbose = False
    a.model = _FakeModel()
    a.callbacks = []  # spinner start/stop become harmless no-ops
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
        a = LangGraphAgent.__new__(LangGraphAgent)
        a._max_subagent_concurrency = 4
        a.verbose = False
        a.callbacks = []
        a.tools = []
        a.tools_by_route = None  # → every subtask uses the full toolset
        a.model = object()
        a.model_with_tools = object()
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


class TestOrchestrateEndToEnd:
    """_orchestrate assembles the final answer. Regression: a MULTI-subtask run
    must not crash (the aggregate branch once fell inside `if len==1`, leaving
    final_content unbound → UnboundLocalError for every multi-subtask task)."""

    def _agent(self, subtasks):
        from langchain_core.messages import HumanMessage

        a = LangGraphAgent.__new__(LangGraphAgent)
        a._max_subagent_concurrency = 4
        a.verbose = False
        a.callbacks = []
        a.tools = []
        a.tools_by_route = None
        a.model = object()
        a.model_with_tools = object()
        a._steer_queue = []
        a._steer_lock = None
        a._external_tools_prompt_block = lambda: ""
        a._decompose_task = lambda q, p, cats: subtasks
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
