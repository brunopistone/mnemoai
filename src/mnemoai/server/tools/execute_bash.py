"""Execute bash commands tool."""

import json
import os
import signal
import subprocess

from mcp.server.fastmcp import FastMCP

from mnemoai.utils.logger import logger

from ..error_handler import tool_error_handler
from .safety import classify_shell_command

# Cap the output we RETURN so a chatty command can't blow up the token/context
# budget or the tool-result JSON. We keep the head and the tail (the start plus 
# the final status/error lines, which are usually the interesting ones) and drop 
# the middle. The child's full output is still bounded only by its own behaviour 
# + the timeout; this guards what leaves the server, ahead of the client's downstream 
# MAX_TOOL_RESULT_CHARS compaction.
MAX_OUTPUT_CHARS = 30_000


def _truncate_output(text: str) -> str:
    """Middle-truncate ``text`` to ``MAX_OUTPUT_CHARS``, keeping head + tail.

    ``MAX_OUTPUT_CHARS`` is a true ceiling: the marker is counted against the
    budget so the returned string never grows past the cap.
    """
    if not text or len(text) <= MAX_OUTPUT_CHARS:
        return text
    total_lines = text.count("\n") + 1
    dropped = len(text) - MAX_OUTPUT_CHARS
    marker = (
        f"\n\n... [output truncated: ~{dropped} of {len(text)} chars omitted "
        f"from the middle; ~{total_lines} total lines] ...\n\n"
    )
    keep = max(0, (MAX_OUTPUT_CHARS - len(marker)) // 2)
    return text[:keep] + marker + text[-keep:]


def register_execute_bash_tools(mcp: FastMCP) -> None:
    """Register bash execution tool.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    @tool_error_handler
    async def execute_bash(command: str, timeout: int = 30) -> str:
        """Execute bash/shell commands and return the output.

        Use this tool when users ask to run commands, list directories recursively, check system info, or perform shell operations.

        Safety rules - Do not execute commands that:
        - Delete files: rm, rmdir, unlink
        - Format disks: mkfs, dd
        - Modify system: shutdown, reboot, halt
        - Change permissions dangerously: chmod 777, chown
        - Overwrite files destructively: > redirection without confirmation

        ONLY use safe read-only commands like: ls, find, cat, grep, df, ps, etc.

        Args:
            command: The bash command to execute (e.g., "ls -la", "find /path -print")
            timeout: Maximum execution time in seconds (default: 30)

        Returns:
            JSON string with stdout, stderr, and exit_status
        """
        logger.debug(f"Tool execute_bash called with command: {command}")

        # Server-side hard floor against catastrophic, irreversible commands
        # (rm -rf /, mkfs, dd to a device, shutdown, fork bomb, …). This enforces
        # the docstring's safety rules below the client layer, since the MCP
        # server can be driven directly. Ordinary destructive-but-scoped commands
        # are NOT blocked here — they stay gated by the client confirmation.
        verdict = classify_shell_command(command)
        if verdict.blocked:
            logger.warning(
                "execute_bash blocked command (rule=%s): %s", verdict.rule, command
            )
            return json.dumps(
                {
                    "error": True,
                    "blocked": True,
                    "message": verdict.reason,
                    "command": command,
                }
            )

        # start_new_session puts the shell (and its children) in their own
        # process group so a timeout can kill the whole tree, not just the shell.
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return json.dumps(
                {
                    "stdout": _truncate_output(stdout),
                    "stderr": _truncate_output(stderr),
                    "exit_status": proc.returncode,
                }
            )
        except subprocess.TimeoutExpired:
            # Kill the entire process group, then reap so we don't orphan
            # grandchildren, and return whatever partial output we captured.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            stdout, stderr = proc.communicate()
            return json.dumps(
                {
                    "error": True,
                    "message": f"Command timed out after {timeout} seconds",
                    "stdout": _truncate_output(stdout),
                    "stderr": _truncate_output(stderr),
                }
            )
        except Exception as e:
            logger.error(f"Error executing bash command: {e}")
            proc.kill()
            return json.dumps({"error": True, "message": str(e)})
