"""Workspace identity must survive real MCP requests and remain worker-local."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from mcp import StdioServerParameters

import mnemoai
from mnemoai.client.mcp_tool_wrapper import MultiMCPClient
from mnemoai.server.tools import shell_state
from mnemoai.server.tools.fs_write import _resolve_path
from mnemoai.utils import execution_context


def test_direct_file_tools_use_the_tracked_shell_directory(tmp_path):
    shell_state.reset_cwd()
    try:
        shell_state.set_cwd(str(tmp_path))
        assert _resolve_path("new.txt") == str(tmp_path / "new.txt")
        assert shell_state.resolve_path("new.txt") == str(tmp_path / "new.txt")
    finally:
        shell_state.reset_cwd()


def test_worker_fork_cannot_change_the_parent_workspace(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    agent = SimpleNamespace(_workspace=execution_context.Workspace(str(tmp_path)))
    with execution_context.fork(agent) as workspace:
        assert shell_state.set_cwd(str(child))
        assert workspace.cwd == str(child)
    assert agent._workspace.cwd == str(tmp_path)
    assert execution_context.current() is None


def test_real_mcp_routes_parallel_file_git_and_shell_calls_to_the_same_workspace(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    env = {
        **os.environ, "MNEMOAI_HOME": str(tmp_path / "home"), "MNEMOAI_CONFIG": "/dev/null",
        "PYTHONPATH": str(Path(mnemoai.__file__).resolve().parents[1]),
    }
    client = MultiMCPClient(StdioServerParameters(
        command=sys.executable, args=["-m", "mnemoai.server.server"], env=env,
    ))
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        root.mkdir()
        subprocess.run(["git", "init", "-q", str(root)], check=True)
    try:
        with client:
            builtin = client._members[0][1]

            def run(root):
                workspace = execution_context.Workspace(str(root))
                with execution_context.use(workspace):
                    result = json.loads(builtin.call_tool_sync("execute_bash", {"command": "pwd"}))
                    assert result["cwd"] == str(root)
                    assert result["stdout"].strip() == str(root)
                    created = json.loads(builtin.call_tool_sync(
                        "fs_write", {"path": f"{root.name}.txt", "command": "create", "file_text": root.name},
                    ))
                    assert created["success"], created
                    status = json.loads(builtin.call_tool_sync("git_safe", {"command": "status --short"}))
                    assert f"{root.name}.txt" in status["stdout"]
                    assert (root / f"{root.name}.txt").read_text() == root.name
                return workspace

            with ThreadPoolExecutor(max_workers=2) as pool:
                workspaces = list(pool.map(run, [a, b]))
            assert [item.cwd for item in workspaces] == [str(a), str(b)]
            assert not list(work.glob("*.txt"))
    finally:
        client.shutdown()
