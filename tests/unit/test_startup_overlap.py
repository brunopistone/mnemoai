"""Unit tests: startup builds the model WHILE it boots the MCP servers.

The two have nothing to say to each other — the model needs no tool list, the
servers need no model — but they used to run one after the other, so the user
waited for the sum of a subprocess spawn (plus an ``npx`` package resolution) and
a provider SDK import with credential resolution before the first prompt.

The overlap is proven with a ``Barrier``: it can only be passed if both are in
flight at the same moment, so a sequential startup fails the test rather than
merely being slower. The rest of the file pins what must NOT have changed —
the model is finished before anything reads it, and a provider that can't be
built still fails startup with its own exception.
"""

import threading

import pytest

import mnemoai.client.client as client_mod
from mnemoai.client.client import LangGraphClient

_BARRIER_TIMEOUT = 5.0


class _Config:
    """Every toggle off, so startup takes the same path on any machine."""

    def get(self, key, default=None):
        return default

    def validate_prompts(self, **kwargs):
        return None


class _Agent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _MCP:
    """The MCP client stand-in: `with` is the boot, then the tool list."""

    def __init__(self, barrier=None):
        self._barrier = barrier
        self.entered = False

    def __enter__(self):
        if self._barrier is not None:
            self._barrier.wait(timeout=_BARRIER_TIMEOUT)
        self.entered = True
        return self

    def __exit__(self, *a):
        return False

    def list_tools_sync(self):
        return []

    def set_cancel_probe(self, probe):
        self.probe = probe


class _Controller:
    """The provider stand-in: `initialize_model` is the slow, joinable half."""

    def __init__(self, barrier=None, fail=None):
        self._barrier = barrier
        self._fail = fail
        self.built = False
        self.model = object()

    def initialize_model(self, callbacks=None):
        if self._barrier is not None:
            self._barrier.wait(timeout=_BARRIER_TIMEOUT)
        if self._fail is not None:
            raise self._fail
        self.built = True

    def get_model(self):
        assert self.built, "the model was read before it was finished"
        return self.model


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A real client with only the two slow collaborators replaced."""
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))
    monkeypatch.setattr(client_mod, "config", _Config())
    monkeypatch.setattr(client_mod, "LangGraphAgent", _Agent)

    c = LangGraphClient.__new__(LangGraphClient)
    c.callback_handler = object()
    c.tools = []
    c.agent = None
    c.model = None
    c.session_id = "test_session"
    c.plan_mode_active = False
    c.auto_approve_mode = "off"
    # Startup steps that are not what these tests are about.
    c._initialize_chunk_cache = lambda: None
    c._system_prompt_with_playbook = lambda: "system"
    c._attach_session_log = lambda: None
    c._area_model = lambda area: None
    c._area_usage_name = lambda area: "model"
    c.model_name_for_log = lambda: "model"
    return c


def test_the_model_is_built_while_the_servers_boot(client):
    barrier = threading.Barrier(2)
    client.mcp_client = _MCP(barrier=barrier)
    client.llm_controller = _Controller(barrier=barrier)

    client.start()

    assert client.mcp_client.entered is True
    assert client.llm_controller.built is True


def test_the_model_is_finished_before_anything_reads_it(client):
    # get_model() asserts on this; the agent must never be handed a half-built
    # model just because the tool list arrived first.
    client.mcp_client = _MCP()
    client.llm_controller = _Controller()

    client.start()

    assert client.model is client.llm_controller.model
    assert client.agent.kwargs["model"] is client.llm_controller.model


def test_a_provider_that_cannot_be_built_still_fails_startup(client):
    client.mcp_client = _MCP()
    client.llm_controller = _Controller(fail=RuntimeError("no credentials"))

    with pytest.raises(RuntimeError, match="no credentials"):
        client.start()

    # And it failed on the way to the agent, not after building a broken one.
    assert client.agent is None


def test_the_tools_still_reach_the_agent(client):
    class _WithTools(_MCP):
        def list_tools_sync(self):
            return ["tool_a", "tool_b"]

    client.mcp_client = _WithTools()
    client.llm_controller = _Controller()

    client.start()

    assert client.tools == ["tool_a", "tool_b"]
    assert client.agent.kwargs["tools"] == ["tool_a", "tool_b"]


def test_the_cancel_probe_is_still_installed(client):
    # It is the last step of the boot, so it is the one an early return or a
    # mis-scoped block would silently drop.
    client.mcp_client = _MCP()
    client.llm_controller = _Controller()

    client.start()

    assert callable(client.mcp_client.probe)


def test_no_worker_thread_outlives_the_boot(client):
    # The model is built on a thread; a pool left running would keep an idle
    # thread parked for the whole session and delay exit.
    client.mcp_client = _MCP()
    client.llm_controller = _Controller()
    before = {t.name for t in threading.enumerate()}

    client.start()

    for _ in range(50):
        leftover = {t.name for t in threading.enumerate()} - before
        if not any(name.startswith("model") for name in leftover):
            return
        threading.Event().wait(0.02)
    pytest.fail(f"a model-init thread outlived start(): {leftover}")
