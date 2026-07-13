"""Unit test: MCP server subprocess stderr is routed to a log file, not the
terminal, so a noisy server (e.g. `npm notice` from an npx server) can't corrupt
the pinned UI. The stderr still lands in ~/.mnemoai/logs/mcp.log for debugging.
"""

import asyncio

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
