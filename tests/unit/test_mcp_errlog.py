"""Unit test: MCP server subprocess stderr is routed to a log file, not the
terminal, so a noisy server (e.g. `npm notice` from an npx server) can't corrupt
the pinned UI. The stderr still lands in ~/.mnemoai/logs/mcp.log for debugging.
"""

import asyncio

import anyio
import pytest
from mcp import StdioServerParameters

import mnemoai.client.mcp_tool_wrapper as mod
from mnemoai.client.mcp_tool_wrapper import MCPClientWrapper


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))
    return tmp_path


def test_connect_passes_errlog_to_stdio_client(tmp_home, monkeypatch):
    captured = {}

    class _FakeCM:
        async def __aenter__(self):
            return ("read", "write")

        async def __aexit__(self, *a):
            return False

    def _fake_stdio_client(server, errlog=None):
        captured["errlog"] = errlog
        return _FakeCM()

    class _FakeSession:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def initialize(self):
            return None

    monkeypatch.setattr(mod, "stdio_client", _fake_stdio_client)
    monkeypatch.setattr(mod, "ClientSession", _FakeSession)

    w = MCPClientWrapper(StdioServerParameters(command="true", args=[]))
    asyncio.run(w._connect())

    # stderr was routed to a writable file handle (not the default inherited one)
    errlog = captured["errlog"]
    assert errlog is not None
    assert hasattr(errlog, "write")
    # ...and that file lives under the app-home logs dir.
    assert errlog.name == str(tmp_home / "logs" / "mcp.log")

    # Disconnect closes the handle (no leaked fd).
    asyncio.run(w._disconnect())
    assert w._errlog is None
    assert errlog.closed


class TestSessionContextsAreTaskAffine:
    """Both async contexts must be entered AND exited on the same asyncio task.

    ``stdio_client`` and ``ClientSession`` are anyio task groups, whose cancel
    scopes are task-affine. The old design entered them in ``_connect()`` and
    exited them in ``_disconnect()`` — each submitted via its own
    ``run_coroutine_threadsafe`` call, hence a DIFFERENT task — so every
    reconnect after a server crash raised ``RuntimeError: Attempted to exit
    cancel scope in a different task than it was entered in``, dumped an
    unretrieved-task traceback into the user's terminal, and left the dead
    subprocess's pipes unreaped.

    A single long-lived ``_serve_session`` task now owns them for the whole
    connection; ``_disconnect`` only signals it.
    """

    def _wrapper(self, tmp_home, monkeypatch, tasks):
        """A wrapper whose fake contexts record which task entered/exited them."""

        class _Ctx:
            def __init__(self, label):
                self.label = label

            async def __aenter__(self):
                tasks[f"{self.label}_enter"] = asyncio.current_task()
                return ("read", "write")

            async def __aexit__(self, *a):
                tasks[f"{self.label}_exit"] = asyncio.current_task()
                return False

        class _Session:
            def __init__(self, *a):
                pass

            async def __aenter__(self):
                tasks["session_enter"] = asyncio.current_task()
                return self

            async def __aexit__(self, *a):
                tasks["session_exit"] = asyncio.current_task()
                return False

            async def initialize(self):
                return None

        monkeypatch.setattr(mod, "stdio_client", lambda *a, **k: _Ctx("stdio"))
        monkeypatch.setattr(mod, "ClientSession", _Session)
        return MCPClientWrapper(StdioServerParameters(command="x", args=[]))

    def test_enter_and_exit_happen_on_the_same_task(self, tmp_home, monkeypatch):
        tasks = {}
        w = self._wrapper(tmp_home, monkeypatch, tasks)

        async def drive():
            # Separate awaits, mirroring the two distinct _run_coroutine calls
            # that the real caller makes.
            await w._connect()
            await w._disconnect()

        asyncio.run(drive())
        assert tasks["stdio_enter"] is tasks["stdio_exit"]
        assert tasks["session_enter"] is tasks["session_exit"]

    def test_contexts_are_not_owned_by_the_calling_task(self, tmp_home, monkeypatch):
        # The whole point: ownership lives in _serve_session, NOT in whichever
        # task happened to call _connect.
        tasks = {}
        w = self._wrapper(tmp_home, monkeypatch, tasks)

        async def drive():
            tasks["caller"] = asyncio.current_task()
            await w._connect()
            await w._disconnect()

        asyncio.run(drive())
        assert tasks["stdio_enter"] is not tasks["caller"]

    def test_disconnect_from_a_different_task_does_not_raise(self, tmp_home, monkeypatch):
        """Connect on one task, disconnect on another — the exact failing shape.

        Uses a REAL ``anyio`` task group rather than a plain fake: only the real
        cancel scope enforces task affinity, so a hand-rolled async context
        manager passes even against the buggy code and would make this test a
        no-op. Verified: with the old ``_connect``/``_disconnect`` pair this
        raises ``Attempted to exit cancel scope in a different task…``.
        """

        class _AnyioCtx:
            async def __aenter__(self):
                self._tg = anyio.create_task_group()
                await self._tg.__aenter__()
                return ("read", "write")

            async def __aexit__(self, *a):
                return await self._tg.__aexit__(*a)

        class _Session:
            def __init__(self, *a):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def initialize(self):
                return None

        monkeypatch.setattr(mod, "stdio_client", lambda *a, **k: _AnyioCtx())
        monkeypatch.setattr(mod, "ClientSession", _Session)
        w = MCPClientWrapper(StdioServerParameters(command="x", args=[]))

        async def drive():
            await asyncio.create_task(w._connect())
            await asyncio.create_task(w._disconnect())  # different task

        asyncio.run(drive())  # must not raise
        assert w._connected is False

    def test_state_is_cleared_after_disconnect(self, tmp_home, monkeypatch):
        w = self._wrapper(tmp_home, monkeypatch, {})

        async def drive():
            await w._connect()
            assert w._connected is True and w._session is not None
            await w._disconnect()

        asyncio.run(drive())
        assert w._connected is False
        assert w._session is None
        assert w._session_task is None

    def test_disconnect_without_connect_is_a_noop(self, tmp_home, monkeypatch):
        w = self._wrapper(tmp_home, monkeypatch, {})
        asyncio.run(w._disconnect())  # must not raise
        assert w._connected is False

    def test_a_startup_failure_surfaces_to_the_caller(self, tmp_home, monkeypatch):
        # MultiMCPClient relies on this to skip a bad external server; if the
        # exception stayed inside the session task, a broken server would look
        # like a successful connection.
        class _Boom:
            async def __aenter__(self):
                raise RuntimeError("server exited immediately")

            async def __aexit__(self, *a):
                return False

        monkeypatch.setattr(mod, "stdio_client", lambda *a, **k: _Boom())
        w = MCPClientWrapper(StdioServerParameters(command="x", args=[]))
        with pytest.raises(RuntimeError, match="server exited immediately"):
            asyncio.run(w._connect())
        assert w._connected is False
