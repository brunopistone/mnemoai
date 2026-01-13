"""Execute bash commands tool."""

import subprocess
import json
from mcp.server.fastmcp import FastMCP
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from utils.logger import logger
from .error_handler import tool_error_handler


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

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=timeout
            )

            return json.dumps(
                {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_status": result.returncode,
                }
            )

        except subprocess.TimeoutExpired:
            return json.dumps(
                {"error": True, "message": f"Command timed out after {timeout} seconds"}
            )
        except Exception as e:
            logger.error(f"Error executing bash command: {e}")
            return json.dumps({"error": True, "message": str(e)})
