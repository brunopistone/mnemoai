"""Real MCP sessions without a subprocess retain tool safety and isolation."""

import asyncio
import json
import os
import shlex
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import anyio.to_thread
import pytest
from langchain_core.tools import ToolException
from mcp import StdioServerParameters
from mcp.server.fastmcp import FastMCP

from mnemoai.client.local_mcp import LocalMCPClient
from mnemoai.client.mcp_config import ExternalServer
from mnemoai.client.mcp_tool_wrapper import MultiMCPClient
from mnemoai.server import factory, tool_state
from mnemoai.server.tools import background_tasks, read_state, tools_manager
from mnemoai.server.tools.thread_offload import ThreadedToolServer, run_blocking
from mnemoai.utils import execution_context


@pytest.fixture
def clients(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        tools_manager,
        "config",
        {
            "ENABLE_MEMORY": False,
            "ENABLE_SKILLS": False,
            "ENABLE_RAG": False,
        },
    )
    monkeypatch.setattr(background_tasks, "TASK_OUTPUT_DIR", str(tmp_path / "tasks"))
    opened = []

    def start():
        client = LocalMCPClient()
        opened.append(client)
        client.__enter__()
        client.list_tools_sync()
        return client

    yield start
    for client in reversed(opened):
        client.shutdown()
        assert client._loop is None


def call(client, name, **arguments):
    return json.loads(client.call_tool_sync(name, arguments))


def test_local_tools_validate_arguments_without_stdio(clients, monkeypatch, tmp_path):
    from mnemoai.client import mcp_tool_wrapper

    def forbidden(*args, **kwargs):
        pytest.fail("Built-in tools attempted to launch a stdio subprocess")

    monkeypatch.setattr(mcp_tool_wrapper, "stdio_client", forbidden)
    client = clients()
    schemas = [tool.mcp_tool.model_dump() for tool in client._tools]
    expected = asyncio.run(factory.create_server().list_tools())
    assert schemas == [tool.model_dump() for tool in expected]
    with pytest.raises(ToolException, match="validation error"):
        call(client, "fs_write", path=str(tmp_path / "file"), command="bogus")
    assert not (tmp_path / "file").exists()


def test_local_and_external_stdio_tools_keep_collision_routing(clients, tmp_path):
    script = tmp_path / "external.py"
    script.write_text(
        "from mcp.server.fastmcp import FastMCP\n"
        "server = FastMCP('external', log_level='WARNING')\n"
        "@server.tool()\n"
        "def fs_read(path: str) -> str:\n"
        "    return 'external:' + path\n"
        "server.run(transport='stdio')\n"
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(script)],
        env=os.environ.copy(),
    )
    client = MultiMCPClient(None, [ExternalServer("extra", params)])
    try:
        client.start_background()
        tools = {tool.name: tool for tool in client.finish_startup()}
        target = tmp_path / "local.txt"
        target.write_text("local file")
        assert "local file" in tools["fs_read"].invoke({"path": str(target)})
        assert tools["extra__fs_read"].invoke({"path": "value"}) == "external:value"
    finally:
        client.shutdown()


def test_read_permissions_are_per_server_and_detect_external_edits(clients, tmp_path):
    first, second = clients(), clients()
    path = tmp_path / "shared.txt"
    path.write_text("original")
    # A client-side read cannot authorize either tool server.
    read_state.record_read(str(path))
    assert call(first, "fs_write", path=str(path), command="append", new_str="!")[
        "must_read_first"
    ]
    assert "original" in call(first, "fs_read", path=str(path))["content"]
    assert call(second, "fs_write", path=str(path), command="append", new_str="!")[
        "must_read_first"
    ]
    assert call(first, "fs_write", path=str(path), command="append", new_str="!")[
        "success"
    ]
    path.write_text("external")
    assert call(first, "fs_write", path=str(path), command="append", new_str="!")[
        "stale_read"
    ]
    assert path.read_text() == "external"


def test_async_tool_uses_its_own_loop_and_preserves_workspace(clients, tmp_path):
    client = clients()
    directory = tmp_path / "async"
    directory.mkdir()
    (directory / "file.txt").write_text("async workspace")
    tool = next(tool for tool in client._tools if tool.name == "fs_read")

    async def read():
        with execution_context.use(execution_context.Workspace(str(directory))):
            return await asyncio.wait_for(tool.ainvoke({"path": "file.txt"}), timeout=3)

    assert "async workspace" in json.loads(asyncio.run(read()))["content"]


def test_async_call_cannot_reopen_a_closed_server(clients, monkeypatch):
    client = clients()
    client.shutdown()

    def forbidden():
        pytest.fail("A closed client tried to recreate its server")

    monkeypatch.setattr(factory, "create_server", forbidden)
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(client.call_tool("fs_read", {"path": "unused"}))


def test_reconnect_requires_a_new_read(clients, tmp_path):
    client = clients()
    path = tmp_path / "file.txt"
    path.write_text("original")
    call(client, "fs_read", path=str(path))
    client._run_coroutine(client._disconnect())
    result = call(client, "fs_write", path=str(path), command="append", new_str="!")
    assert result["must_read_first"]
    assert path.read_text() == "original"


def test_shell_directories_and_workspace_metadata_are_isolated(clients, tmp_path):
    first, second = clients(), clients()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert call(first, "execute_bash", command=f"cd {a}")["cwd"] == str(a)
    assert call(second, "execute_bash", command="pwd")["stdout"].strip() == str(
        tmp_path
    )
    with execution_context.use(execution_context.Workspace(str(b))):
        assert call(first, "execute_bash", command="pwd")["stdout"].strip() == str(b)
    assert call(first, "execute_bash", command="pwd")["stdout"].strip() == str(a)


def test_background_jobs_are_owned_by_their_server(clients, tmp_path):
    first, second = clients(), clients()
    task = call(first, "start_background_task", command="printf finished")
    assert task["success"]
    task_id = task["task_id"]
    assert call(second, "get_task_status", task_id=task_id)["error"]
    result = call(first, "wait_for_task", task_id=task_id, timeout_seconds=5)
    assert result["status"] == "completed"
    assert task_id not in background_tasks._background_tasks


def test_blocking_call_does_not_block_other_tools_and_cancels_once(
    clients, monkeypatch
):
    entered, release, cancelled = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    calls = []
    server = FastMCP("concurrency")
    threaded = ThreadedToolServer(server)

    @threaded.tool()
    def blocking() -> str:
        calls.append("blocking")
        entered.set()
        release.wait(5)
        return "finished"

    @threaded.tool()
    def quick() -> str:
        return "responsive"

    monkeypatch.setattr(factory, "create_server", lambda: server)
    client = clients()
    client._cancel_probe = cancelled.is_set
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.call_tool_sync, "blocking", {})
        try:
            assert entered.wait(5)
            assert client.call_tool_sync("quick", {}) == "responsive"
            cancelled.set()
            with pytest.raises(KeyboardInterrupt):
                pending.result(timeout=3)
            assert calls == ["blocking"]
        finally:
            release.set()
            client._cancel_probe = None


def test_cancelled_blocking_tool_does_not_hold_shutdown_open(clients, monkeypatch):
    entered, release, cancelled = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    server = FastMCP("shutdown")
    threaded = ThreadedToolServer(server)

    @threaded.tool()
    def blocking() -> str:
        entered.set()
        release.wait(5)
        return "finished"

    monkeypatch.setattr(factory, "create_server", lambda: server)
    client = clients()
    client._cancel_probe = cancelled.is_set
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(client.call_tool_sync, "blocking", {})
        try:
            assert entered.wait(5)
            cancelled.set()
            with pytest.raises(KeyboardInterrupt):
                pending.result(timeout=3)
            stopped = pool.submit(client.shutdown)
            stopped.result(timeout=1)
            assert not release.is_set()
        finally:
            release.set()


def test_cancelled_shell_stops_its_process_group(clients, tmp_path):
    client = clients()
    pid_file, late_file = tmp_path / "pid", tmp_path / "late"
    cancelled = threading.Event()
    client._cancel_probe = cancelled.is_set
    command = (
        f"printf '%s' $$ > {shlex.quote(str(pid_file))}; "
        f"sleep 30; printf late > {shlex.quote(str(late_file))}"
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            client.call_tool_sync, "execute_bash", {"command": command, "timeout": 40}
        )
        try:
            deadline = time.monotonic() + 5
            while not pid_file.exists() and time.monotonic() < deadline:
                cancelled.wait(0.01)
            assert pid_file.exists()
            pid = int(pid_file.read_text())
            cancelled.set()
            with pytest.raises(KeyboardInterrupt):
                pending.result(timeout=3)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.01)
            else:
                pytest.fail("The cancelled shell is still running")
            assert not late_file.exists()
        finally:
            cancelled.set()


def test_cancelled_read_cannot_grant_write_permission(clients, monkeypatch, tmp_path):
    from mnemoai.server.tools import fs_read

    entered, release, recorded, cancelled = (threading.Event() for _ in range(4))
    original_read, original_record = fs_read.read_lines, fs_read.record_read

    def slow_read(*args, **kwargs):
        entered.set()
        release.wait(5)
        return original_read(*args, **kwargs)

    def record(*args, **kwargs):
        try:
            return original_record(*args, **kwargs)
        finally:
            recorded.set()

    monkeypatch.setattr(fs_read, "read_lines", slow_read)
    monkeypatch.setattr(fs_read, "record_read", record)
    client = clients()
    client._cancel_probe = cancelled.is_set
    target = tmp_path / "file.txt"
    target.write_text("original")
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.call_tool_sync, "fs_read", {"path": str(target)})
        try:
            assert entered.wait(5)
            cancelled.set()
            with pytest.raises(KeyboardInterrupt):
                pending.result(timeout=3)
        finally:
            release.set()
            client._cancel_probe = None
        assert recorded.wait(3)
    result = call(client, "fs_write", path=str(target), command="append", new_str="!")
    assert result["must_read_first"]
    assert target.read_text() == "original"


def test_cancelled_waiting_tool_never_runs_and_does_not_cancel_peer(
    clients, monkeypatch
):
    entered, queued, release = threading.Event(), threading.Event(), threading.Event()
    writes = []
    server = FastMCP("queued cancellation")
    threaded = ThreadedToolServer(server)

    @threaded.tool()
    def hold() -> str:
        entered.set()
        release.wait(5)
        return "peer finished"

    @threaded.tool()
    async def mutate() -> str:
        queued.set()
        await run_blocking(lambda: writes.append("unexpected"))
        return "mutated"

    monkeypatch.setattr(factory, "create_server", lambda: server)
    client = clients()

    async def one_worker():
        anyio.to_thread.current_default_thread_limiter().total_tokens = 1

    client._run_coroutine(one_worker())
    with ThreadPoolExecutor(max_workers=1) as pool:
        peer = pool.submit(client.call_tool_sync, "hold", {})
        try:
            assert entered.wait(3)

            async def cancel_waiter():
                pending = asyncio.create_task(client.call_tool("mutate", {}))
                assert await asyncio.to_thread(queued.wait, 3)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending

            asyncio.run(cancel_waiter())
            release.set()
            assert peer.result(timeout=3) == "peer finished"

            async def drained():
                while client._local_state.pending:
                    await asyncio.sleep(0.01)

            client._run_coroutine(asyncio.wait_for(drained(), timeout=3))
            assert writes == []
        finally:
            release.set()


def test_rag_cache_is_scoped_to_the_tool_server(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from mnemoai.server.tools.rag import session

    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))
    monkeypatch.setattr(session, "_rag_session", None)
    first, second = tool_state.ToolServerState(), tool_state.ToolServerState()
    value = SimpleNamespace(session_id="first")
    with tool_state.use(first):
        session.set_rag_session(value)
    with tool_state.use(second):
        assert session.get_rag_session() is None
    with tool_state.use(first):
        assert session.get_rag_session() is value
    assert session._rag_session is None
