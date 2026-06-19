"""Execute bash commands tool."""

import subprocess
import json
from mcp.server.fastmcp import FastMCP
import os
import signal

from mnemoai.utils.logger import logger
from ..error_handler import tool_error_handler


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
                    "stdout": stdout,
                    "stderr": stderr,
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
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )
        except Exception as e:
            logger.error(f"Error executing bash command: {e}")
            proc.kill()
            return json.dumps({"error": True, "message": str(e)})
