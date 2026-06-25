"""Unit tests for external MCP server config loading and tool merging.

Covers the pure-logic parts — no real subprocesses are launched:
- ``load_external_servers`` parsing (valid / missing / malformed / disabled /
  bad-entry / env-merge-and-coerce)
- ``MultiMCPClient`` tool merging with collision namespacing
"""

import json

from mnemoai.client.mcp_config import load_external_servers


def _write(tmp_path, data):
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps(data) if not isinstance(data, str) else data)
    return p


def test_load_valid_server(tmp_path):
    p = _write(tmp_path, {
        "mcpServers": {
            "brave": {"command": "npx", "args": ["-y", "x"], "env": {"K": "v"}}
        }
    })
    servers = load_external_servers(p)
    assert len(servers) == 1
    s = servers[0]
    assert s.name == "brave"
    assert s.params.command == "npx"
    assert s.params.args == ["-y", "x"]
    # env override present AND merged over the process environment.
    assert s.params.env["K"] == "v"
    assert "PATH" in s.params.env


def test_missing_file_returns_empty(tmp_path):
    assert load_external_servers(tmp_path / "nope.json") == []


def test_malformed_json_returns_empty(tmp_path):
    p = _write(tmp_path, "{ not valid json")
    assert load_external_servers(p) == []


def test_disabled_entry_skipped(tmp_path):
    p = _write(tmp_path, {
        "mcpServers": {
            "a": {"command": "x", "disabled": True},
            "b": {"command": "y"},
        }
    })
    assert [s.name for s in load_external_servers(p)] == ["b"]


def test_entry_without_command_skipped(tmp_path):
    p = _write(tmp_path, {
        "mcpServers": {"bad": {"args": []}, "good": {"command": "z"}}
    })
    assert [s.name for s in load_external_servers(p)] == ["good"]


def test_bad_args_type_skipped(tmp_path):
    p = _write(tmp_path, {
        "mcpServers": {"x": {"command": "c", "args": "not-a-list"}}
    })
    assert load_external_servers(p) == []


def test_env_values_coerced_to_strings(tmp_path):
    p = _write(tmp_path, {
        "mcpServers": {"n": {"command": "c", "env": {"PORT": 8080}}}
    })
    s = load_external_servers(p)[0]
    assert s.params.env["PORT"] == "8080"


def test_missing_mcpservers_key_returns_empty(tmp_path):
    p = _write(tmp_path, {"somethingElse": {}})
    assert load_external_servers(p) == []


# --- MultiMCPClient tool merging / collision namespacing --------------------


class _FakeTool:
    def __init__(self, name, description=""):
        self.name = name
        self.description = description
        self.mcp_tool = type("T", (), {"name": name})()


class _FakeWrapper:
    def __init__(self, tools):
        self._tools = tools

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def list_tools_sync(self):
        return self._tools

    def shutdown(self):
        pass


def _multi(members):
    from mnemoai.client.mcp_tool_wrapper import MultiMCPClient

    m = MultiMCPClient.__new__(MultiMCPClient)
    m._members = members
    m._tools = []
    return m


def test_merge_namespaces_only_collisions():
    m = _multi([
        ("builtin", _FakeWrapper([_FakeTool("read_file"), _FakeTool("execute_bash")])),
        ("brave", _FakeWrapper([_FakeTool("brave_search"), _FakeTool("read_file")])),
    ])
    tools = m.list_tools_sync()
    names = [t.name for t in tools]
    # Built-in names untouched; the colliding external one is namespaced.
    assert names == ["read_file", "execute_bash", "brave_search", "brave__read_file"]


def test_collided_tool_keeps_server_side_name():
    m = _multi([
        ("builtin", _FakeWrapper([_FakeTool("read_file")])),
        ("ext", _FakeWrapper([_FakeTool("read_file")])),
    ])
    tools = m.list_tools_sync()
    collided = next(t for t in tools if t.name == "ext__read_file")
    # The server is still called with the original name, not the namespaced one.
    assert collided.mcp_tool.name == "read_file"


def test_failing_server_is_skipped_not_fatal():
    class _Boom(_FakeWrapper):
        def list_tools_sync(self):
            raise RuntimeError("server died")

    m = _multi([
        ("builtin", _FakeWrapper([_FakeTool("ok_tool")])),
        ("broken", _Boom([])),
    ])
    tools = m.list_tools_sync()
    # The broken server is skipped; the built-in tool still loads.
    assert [t.name for t in tools] == ["ok_tool"]


# --- Orchestrator awareness of external tools -------------------------------


def _agent_with_external(external):
    """A bare LangGraphAgent with only external_tools set (no graph build)."""
    from mnemoai.client.agent.agent import LangGraphAgent

    a = LangGraphAgent.__new__(LangGraphAgent)
    a.external_tools = external
    return a


def test_orchestrator_block_empty_without_external_tools():
    # No external tools -> the decomposition prompt is left unchanged.
    a = _agent_with_external([])
    assert a._external_tools_prompt_block() == ""


def test_orchestrator_block_lists_tools_and_routes_to_full():
    a = _agent_with_external([
        _FakeTool("brave_search"),
        _FakeTool("jira_create_issue"),
    ])
    block = a._external_tools_prompt_block()
    assert "brave_search" in block and "jira_create_issue" in block
    # Steers the decomposer to the 'full' category for these tools.
    assert '"full"' in block
    assert "<external_tools>" in block


def test_external_tools_appended_to_every_route():
    # The route-building logic appends external tools (those not named in any
    # route) to EVERY route — including the no-tools 'simple_qa' route — and
    # 'full' (None) still gets everything. External tools are user-configured
    # capabilities, so a short factual question ("what time is it?") that
    # classifies as simple_qa must still be able to reach an external server.
    from mnemoai.client.agent.agent import LangGraphAgent

    class _StubModel:
        # bind_tools just records and returns self — these tests check tool
        # routing, not model wiring, so we avoid real schema validation.
        def bind_tools(self, tools):
            return self

    tools = [_FakeTool("read_file"), _FakeTool("brave_search")]
    routes = {"simple_qa": [], "code": ["read_file"], "full": None}
    model = _StubModel()

    agent = LangGraphAgent(
        model=model, tools=tools, router=object(), tool_routes=routes,
    )
    by_route = {k: [t.name for t in v] for k, v in agent.tools_by_route.items()}
    # External 'brave_search' reaches every route despite not being listed.
    assert "brave_search" in by_route["code"]
    assert by_route["simple_qa"] == ["brave_search"]   # external reachable here too
    assert "brave_search" in by_route["full"]          # full binds everything
    assert agent.external_tools and agent.external_tools[0].name == "brave_search"


def test_memory_meta_tool_reachable_on_every_route():
    # The 'memory' meta tool must be bound on EVERY route, including the
    # no-tools simple_qa route (a "remember this" request classifies there),
    # and must NOT be counted as an external tool.
    from mnemoai.client.agent.agent import LangGraphAgent

    class _StubModel:
        def bind_tools(self, tools):
            return self

    tools = [_FakeTool("read_file"), _FakeTool("memory"), _FakeTool("brave_search")]
    routes = {"simple_qa": [], "code": ["read_file"], "full": None}
    agent = LangGraphAgent(
        model=_StubModel(), tools=tools, router=object(), tool_routes=routes,
    )
    by_route = {k: [t.name for t in v] for k, v in agent.tools_by_route.items()}
    # 'memory' (meta) is bound on the no-tools route; the external 'brave_search'
    # rides along too, but no built-in non-meta tool does.
    assert "memory" in by_route["simple_qa"]
    assert "read_file" not in by_route["simple_qa"]
    assert "memory" in by_route["code"]
    assert "memory" in by_route["full"]
    # memory is a meta tool, not external; brave_search still is.
    assert [t.name for t in agent.external_tools] == ["brave_search"]


def test_describe_image_reachable_on_every_route():
    # describe_image is a meta tool: an image can be referenced in any query
    # ("what's in this image?" classifies as simple_qa/knowledge), so the vision
    # tool must be bound on every route — otherwise the model falls back to a
    # text reader on a binary file.
    from mnemoai.client.agent.agent import LangGraphAgent

    class _StubModel:
        def bind_tools(self, tools):
            return self

    tools = [
        _FakeTool("fs_read"),
        _FakeTool("describe_image"),
        _FakeTool("memory"),
    ]
    routes = {
        "simple_qa": [],
        "knowledge": ["fs_read"],
        "code": ["fs_read"],
        "full": None,
    }
    agent = LangGraphAgent(
        model=_StubModel(), tools=tools, router=object(), tool_routes=routes,
    )
    by_route = {k: [t.name for t in v] for k, v in agent.tools_by_route.items()}
    for route in ("simple_qa", "knowledge", "code", "full"):
        assert "describe_image" in by_route[route], route
    # It's a meta tool, not an external one.
    assert "describe_image" not in [t.name for t in agent.external_tools]


def test_route_table_has_no_orphans_or_stale_refs():
    """Every route-named tool must exist, and every real tool must be reachable.

    Guards the route audit: a tool named in a route but not registered (stale),
    or a registered tool in no route and not a meta tool (orphan, only reachable
    via 'full'), is a routing bug.
    """
    from mnemoai.client.agent.agent import LangGraphAgent
    from mnemoai.client.agent.router import ROUTE_TOOLS

    # The full registered tool surface (server/tools/*). Kept here as the
    # source of truth for the audit; update alongside ROUTE_TOOLS when tools
    # are added/removed.
    registered = {
        "add_plan_file", "add_plan_risk", "add_plan_step", "approve_plan",
        "cancel_background_task", "clear_completed_tasks", "clear_documents",
        "describe_image", "enter_plan_mode", "execute_bash", "exit_plan_mode",
        "file_edit", "fs_read", "fs_write", "get_plan_status", "get_task_output",
        "get_task_status", "git_commit_safe", "git_safe", "git_status_safe",
        "glob_search", "grep_search", "list_background_tasks", "list_documents",
        "memory", "present_plan", "search_in_documents", "start_background_task",
        "todo_clear", "todo_read", "todo_write", "wait_for_task", "web_crawler",
        "web_search",
    }
    meta = LangGraphAgent._ALWAYS_AVAILABLE_TOOLS

    named = set()
    for tools in ROUTE_TOOLS.values():
        if tools is not None:
            named |= set(tools)

    stale = named - registered
    assert not stale, f"routes name nonexistent tools: {sorted(stale)}"

    orphans = registered - named - meta
    assert not orphans, f"tools reachable only via 'full': {sorted(orphans)}"
