"""Unit tests for execute_bash timeout + process-group handling.

These exercise the real subprocess behavior (no LLM involved): partial output
capture, prompt timeout, and that the whole process group is killed so a
spawned grandchild does not outlive the timeout.
"""

import asyncio
import json
import os
import tempfile
import time

import pytest

from mnemoai.server.tools.execute_bash import register_execute_bash_tools


class _CapturingMCP:
    """Minimal stand-in for FastMCP that captures registered tool functions."""

    def __init__(self):
        self.registered = {}

    def tool(self):
        def decorator(func):
            self.registered[func.__name__] = func
            return func

        return decorator


@pytest.fixture
def execute_bash():
    mcp = _CapturingMCP()
    register_execute_bash_tools(mcp)
    return mcp.registered["execute_bash"]


def run(coro):
    return asyncio.run(coro)


class TestExecuteBash:
    def test_basic_command_returns_stdout(self, execute_bash):
        result = json.loads(run(execute_bash("echo hello", timeout=5)))
        assert result["stdout"].strip() == "hello"
        assert result["exit_status"] == 0

    def test_nonzero_exit_status_captured(self, execute_bash):
        result = json.loads(run(execute_bash("exit 3", timeout=5)))
        assert result["exit_status"] == 3

    def test_stderr_captured(self, execute_bash):
        result = json.loads(run(execute_bash("echo oops 1>&2", timeout=5)))
        assert "oops" in result["stderr"]

    def test_timeout_returns_promptly_with_error(self, execute_bash):
        start = time.time()
        result = json.loads(run(execute_bash("sleep 10", timeout=1)))
        elapsed = time.time() - start
        assert result["error"] is True
        assert "timed out" in result["message"].lower()
        # Should return shortly after the 1s timeout, not after 10s.
        assert elapsed < 5

    def test_timeout_returns_partial_output(self, execute_bash):
        # Print before sleeping; on timeout we should still capture the line.
        cmd = "echo partial_line; sleep 10"
        result = json.loads(run(execute_bash(cmd, timeout=1)))
        assert result["error"] is True
        assert "partial_line" in result["stdout"]

    def test_timeout_kills_process_group_no_orphan_grandchild(self, execute_bash):
        # A grandchild writes a marker file after 3s. We time out at 1s and the
        # process-group kill must prevent the marker from ever being written.
        marker = os.path.join(tempfile.gettempdir(), f"_pg_test_{os.getpid()}.txt")
        if os.path.exists(marker):
            os.remove(marker)
        try:
            cmd = f"(sleep 3; echo alive > {marker}) & echo started; sleep 10"
            result = json.loads(run(execute_bash(cmd, timeout=1)))
            assert result["error"] is True
            # Wait past the grandchild's 3s write window.
            time.sleep(4)
            assert not os.path.exists(marker), "grandchild survived the timeout kill"
        finally:
            if os.path.exists(marker):
                os.remove(marker)
