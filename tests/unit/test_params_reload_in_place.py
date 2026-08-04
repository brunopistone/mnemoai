"""Unit tests: /params reloads in place instead of restarting the process.

/params edits ONLY inference knobs (temperature, top_p, …) — never the provider,
model name, or connection — so nothing the MCP subprocess fixed at boot can have
changed and there is no reason to re-exec. Re-exec'ing discarded the conversation,
which was worst right after a `--resume`: the restored history lived only in the
new session file, and that file was abandoned turn-less by the restart.

Pure logic: a stub client/agent, no LLM, no prompt_toolkit, no TTY.
"""

import pytest

from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.ui.chat_interface import ChatInterface


class _Bound:
    """Stand-in for a tool-bound model, remembering what it was bound from."""

    def __init__(self, parent, tools):
        self.parent = parent
        self.tools = tools


class _Model:
    def __init__(self, tag):
        self.tag = tag

    def bind_tools(self, tools):
        return _Bound(self, tools)


class _Tool:
    def __init__(self, name):
        self.name = name


class _Log:
    def __init__(self):
        self.discarded = 0

    def discard_if_empty(self):
        self.discarded += 1
        return True


class _Client:
    """Client stub recording whether the in-place path or a restart was taken."""

    def __init__(self, reload_ok=True):
        self.plan_mode_active = False
        self.episodic_memory = None
        self.reflector = None
        self.session_id = "sess_20260101_000000"
        self.reload_ok = reload_ok
        self.reload_calls = 0
        self.agent = type("A", (), {"session_log": _Log()})()

    def reload_inference_params(self):
        self.reload_calls += 1
        return self.reload_ok


@pytest.fixture
def ci():
    c = ChatInterface.__new__(ChatInterface)
    c.client = _Client()
    c.restarts = 0
    c._restart_in_place = lambda: setattr(c, "restarts", c.restarts + 1)
    return c


class TestParamsDoesNotRestart:
    def test_params_reloads_in_place_and_keeps_the_conversation(self, ci, monkeypatch):
        monkeypatch.setattr(
            "mnemoai.client.ui.chat_interface.run_params_override", lambda: "/cfg.yaml"
        )
        assert ci._dispatch("/params") is None
        assert ci.client.reload_calls == 1
        # The whole point: no re-exec, so in-memory history survives.
        assert ci.restarts == 0

    def test_cancelled_params_dialog_changes_nothing(self, ci, monkeypatch):
        # run_params_override returns None when cancelled or unchanged.
        monkeypatch.setattr(
            "mnemoai.client.ui.chat_interface.run_params_override", lambda: None
        )
        ci._dispatch("/params")
        assert ci.client.reload_calls == 0
        assert ci.restarts == 0

    def test_failed_reload_falls_back_to_restart(self, monkeypatch):
        # A half-applied config is worse than a restart: if the rebuild fails we
        # must not keep running as though the new params were live.
        c = ChatInterface.__new__(ChatInterface)
        c.client = _Client(reload_ok=False)
        c.restarts = 0
        c._restart_in_place = lambda: setattr(c, "restarts", c.restarts + 1)
        monkeypatch.setattr(
            "mnemoai.client.ui.chat_interface.run_params_override", lambda: "/cfg.yaml"
        )
        c._dispatch("/params")
        assert c.restarts == 1

    def test_features_still_restarts(self, ci, monkeypatch):
        # /features can flip toggles that gate MCP tool REGISTRATION at server boot
        # (ENABLE_RAG/WEB_SEARCH/WEB_CRAWL/MEMORY/SKILLS), so it still needs the
        # full restart — this test pins the deliberate asymmetry with /params.
        monkeypatch.setattr(
            "mnemoai.client.ui.chat_interface.run_features_override",
            lambda: "/cfg.yaml",
        )
        ci._dispatch("/features")
        assert ci.restarts == 1
        assert ci.client.reload_calls == 0


class TestRebindModel:
    """Reassigning agent.model alone is not enough — derived bindings go stale."""

    def _agent(self, with_routes=True):
        a = LangGraphAgent.__new__(LangGraphAgent)
        old = _Model("old")
        a.model = old
        a.tools = [_Tool("fs_read"), _Tool("execute_bash")]
        a.model_with_tools = old.bind_tools(a.tools)
        if with_routes:
            a.tools_by_route = {
                "code": [a.tools[0]],
                "full": a.tools,
                "simple_qa": [],
            }
            a.models_by_route = {
                r: (old.bind_tools(t) if t else old)
                for r, t in a.tools_by_route.items()
            }
        else:
            a.tools_by_route = None
            a.models_by_route = None
        return a

    def test_tool_bound_model_is_rebuilt_from_the_new_model(self):
        a = self._agent()
        new = _Model("new")
        a.rebind_model(new)
        assert a.model is new
        # The tool-bound model is a SEPARATE object; a stale one would keep
        # calling the old model with the old inference params.
        assert a.model_with_tools.parent is new
        assert a.model_with_tools.tools == a.tools

    def test_every_route_binding_is_rebuilt(self):
        a = self._agent()
        new = _Model("new")
        a.rebind_model(new)
        for route, bound in a.models_by_route.items():
            # An empty route subset binds nothing and uses the bare model.
            if a.tools_by_route[route]:
                assert bound.parent is new, route
                assert bound.tools == a.tools_by_route[route], route
            else:
                assert bound is new, route

    def test_route_subsets_are_preserved_not_widened(self):
        # Rebinding must not accidentally give a narrow route the full toolset —
        # that would defeat per-route tool pruning.
        a = self._agent()
        a.rebind_model(_Model("new"))
        assert [t.name for t in a.models_by_route["code"].tools] == ["fs_read"]

    def test_works_when_routing_is_disabled(self):
        a = self._agent(with_routes=False)
        new = _Model("new")
        a.rebind_model(new)
        assert a.model is new
        assert a.model_with_tools.parent is new
        assert a.models_by_route is None


class TestRestartDiscardsTurnlessSession:
    """os.execv runs no atexit/finally, so the restart must clean up itself."""

    def test_restart_discards_an_empty_session_file(self, monkeypatch):
        c = ChatInterface.__new__(ChatInterface)
        c.client = _Client()
        log = c.client.agent.session_log

        calls = {"shutdown": 0, "execv": 0}
        c.client.mcp_client = type(
            "M", (), {"shutdown": lambda self: calls.__setitem__("shutdown", 1)}
        )()
        monkeypatch.setattr(
            "mnemoai.client.ui.chat_interface.os.execv",
            lambda *a: calls.__setitem__("execv", 1),
        )
        c._restart_in_place()
        # Cleanup happens BEFORE the process is replaced, or it never happens.
        assert log.discarded == 1
        assert calls["execv"] == 1

    def test_cleanup_failure_never_blocks_the_restart(self, monkeypatch):
        c = ChatInterface.__new__(ChatInterface)
        c.client = _Client()

        class _Boom:
            def discard_if_empty(self):
                raise OSError("disk gone")

        c.client.agent.session_log = _Boom()
        c.client.mcp_client = type("M", (), {"shutdown": lambda self: None})()
        seen = {}
        monkeypatch.setattr(
            "mnemoai.client.ui.chat_interface.os.execv",
            lambda *a: seen.setdefault("execv", True),
        )
        c._restart_in_place()
        assert seen.get("execv") is True
