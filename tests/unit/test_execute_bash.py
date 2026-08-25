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

from mnemoai.server.tools import shell_state
from mnemoai.server.tools.execute_bash import (
    MAX_OUTPUT_CHARS,
    _truncate_output,
    register_execute_bash_tools,
)


@pytest.fixture(autouse=True)
def _clean_tracked_cwd():
    """The shell's tracked directory is process-wide state; isolate each test."""
    shell_state.reset_cwd()
    yield
    shell_state.reset_cwd()


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


def run(result):
    """Resolve a tool result: await a coroutine, else pass the value through.

    A tool with a blocking body is a plain ``def`` (server/tools/thread_offload.py
    offloads it to a thread at registration), so calling it directly here returns
    the string rather than a coroutine.
    """
    return asyncio.run(result) if asyncio.iscoroutine(result) else result


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


class TestShellIsBash:
    """The tool is documented as bash, so bash is what runs it. Under /bin/sh
    these constructs fail or behave differently depending on the host (dash on
    Debian/Ubuntu, bash in POSIX mode on macOS)."""

    def test_double_bracket_test_works(self, execute_bash):
        result = json.loads(run(execute_bash("[[ abc == a* ]] && echo matched", timeout=5)))
        assert result["stdout"].strip() == "matched"
        assert result["exit_status"] == 0

    def test_arrays_work(self, execute_bash):
        cmd = "arr=(one two three); echo ${arr[1]}"
        result = json.loads(run(execute_bash(cmd, timeout=5)))
        assert result["stdout"].strip() == "two"

    def test_process_substitution_works(self, execute_bash):
        result = json.loads(run(execute_bash("cat <(echo inner)", timeout=5)))
        assert result["stdout"].strip() == "inner"


class TestTrackedWorkingDirectory:
    """A `cd` used to evaporate: each call is a fresh shell, so the next command
    ran back at the spawn directory — silently operating on the wrong tree."""

    def test_cwd_is_reported(self, execute_bash):
        result = json.loads(run(execute_bash("echo hi", timeout=5)))
        assert result["cwd"] == os.getcwd()

    def test_a_cd_carries_over_to_the_next_call(self, execute_bash, tmp_path):
        target = tmp_path / "project"
        target.mkdir()
        first = json.loads(run(execute_bash(f"cd {target}", timeout=5)))
        assert os.path.realpath(first["cwd"]) == os.path.realpath(str(target))
        second = json.loads(run(execute_bash("pwd", timeout=5)))
        assert os.path.realpath(second["stdout"].strip()) == os.path.realpath(str(target))

    def test_a_relative_path_resolves_against_the_tracked_directory(
        self, execute_bash, tmp_path
    ):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "marker.txt").write_text("found me")
        run(execute_bash(f"cd {tmp_path}", timeout=5))
        result = json.loads(run(execute_bash("cat sub/marker.txt", timeout=5)))
        assert "found me" in result["stdout"]

    def test_a_failed_cd_leaves_the_directory_alone(self, execute_bash):
        before = json.loads(run(execute_bash("pwd", timeout=5)))["cwd"]
        result = json.loads(run(execute_bash("cd /definitely/not/here", timeout=5)))
        assert result["exit_status"] != 0
        assert result["cwd"] == before

    def test_a_subshell_cd_does_not_move_the_session(self, execute_bash, tmp_path):
        # `(cd x)` deliberately scopes the change; the session must agree.
        run(execute_bash(f"(cd {tmp_path})", timeout=5))
        assert json.loads(run(execute_bash("pwd", timeout=5)))["cwd"] == os.getcwd()

    def test_the_probe_does_not_pollute_stdout(self, execute_bash, tmp_path):
        result = json.loads(run(execute_bash(f"cd {tmp_path} && echo only-this", timeout=5)))
        assert result["stdout"].strip() == "only-this"
        assert result["stderr"].strip() == ""

    def test_a_command_that_exits_the_shell_keeps_the_old_directory(
        self, execute_bash, tmp_path
    ):
        # The probe never runs, which must fail open rather than lose the cwd.
        result = json.loads(run(execute_bash(f"cd {tmp_path}; exit 7", timeout=5)))
        assert result["exit_status"] == 7
        assert result["cwd"] == os.getcwd()

    def test_a_timed_out_command_does_not_move_the_directory(self, execute_bash, tmp_path):
        result = json.loads(run(execute_bash(f"cd {tmp_path}; sleep 10", timeout=1)))
        assert result["error"] is True
        assert result["cwd"] == os.getcwd()


class TestStdinIsNotInherited:
    """The server's stdin IS the MCP protocol pipe. A command reading stdin would
    otherwise eat the client's JSON-RPC stream, and one waiting on input that can
    never arrive would burn the whole timeout instead of seeing EOF."""

    def test_a_command_reading_stdin_gets_eof_immediately(self, execute_bash):
        start = time.time()
        result = json.loads(run(execute_bash("cat", timeout=10)))
        assert result.get("exit_status") == 0
        assert result["stdout"] == ""
        assert time.time() - start < 5  # EOF, not a wait on a pipe


class TestOutputCap:
    """execute_bash caps the output it RETURNS (server-side, before the result
    leaves the tool) so a chatty command can't blow up the token/context budget."""

    def test_truncate_leaves_small_output_untouched(self):
        small = "hello\nworld\n"
        assert _truncate_output(small) == small

    def test_truncate_caps_large_output_and_keeps_head_and_tail(self):
        big = "A" * 20_000 + "B" * 20_000  # 40k > MAX_OUTPUT_CHARS (30k)
        out = _truncate_output(big)
        assert len(out) <= MAX_OUTPUT_CHARS  # true ceiling (marker counted in)
        assert out.startswith("A")
        assert out.endswith("B")  # tail preserved (often the status/error)
        assert "output truncated" in out

    def test_truncate_is_a_true_ceiling_for_small_overflow(self):
        # Just-over-cap input must not GROW past the cap once the marker is added.
        big = "x" * (MAX_OUTPUT_CHARS + 1)
        out = _truncate_output(big)
        assert len(out) <= MAX_OUTPUT_CHARS

    def test_execute_bash_return_is_bounded(self, execute_bash):
        # 100k of output must come back bounded at the cap, not verbatim.
        result = json.loads(run(execute_bash("yes A | head -c 100000", timeout=10)))
        assert len(result["stdout"]) <= MAX_OUTPUT_CHARS  # true ceiling
        assert "output truncated" in result["stdout"]

    def test_execute_bash_small_output_verbatim(self, execute_bash):
        result = json.loads(run(execute_bash("echo hi", timeout=5)))
        assert result["stdout"].strip() == "hi"
        assert "truncated" not in result["stdout"]
