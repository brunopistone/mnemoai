"""Todo list management for tracking multi-step tasks."""

import json
import os
import re

from mcp.server.fastmcp import FastMCP

from mnemoai.utils.atomic_write import atomic_write_json
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import instance_id, profile_dir


def _sanitize_scope(scope: str) -> str:
    """Reduce a scope label to a filename-safe token (default 'default').

    Non-alphanumerics collapse to '_' and the token is capped at 64 chars, so
    labels differing only in punctuation/length may alias to one file (last
    writer wins — never corrupts, thanks to atomic writes). Pass stable, simple
    labels for isolation.
    """
    token = re.sub(r"[^A-Za-z0-9_-]", "_", (scope or "").strip()) or "default"
    return token[:64]


def _todo_file(scope: str = "default") -> str:
    """Path to the todo list for ``scope``, namespaced per app instance.

    Namespaced by :func:`instance_id` (inherited by the MCP subprocess via
    ``MNEMOAI_INSTANCE_ID``) so concurrent app instances (terminal tabs) don't
    clobber each other's list; ``scope`` further isolates lists WITHIN one
    instance for callers that pass a stable label. No per-agent id is threaded
    through MCP, so same-scope concurrent agents are last-writer-wins (never
    corrupt, thanks to atomic writes).
    """
    return str(
        profile_dir()
        / "todos"
        / f"current_todos_{instance_id()}_{_sanitize_scope(scope)}.json"
    )


def _atomic_write_json(path: str, data) -> None:
    """Write JSON to ``path`` atomically so a concurrent reader never sees a
    half-written list. Delegates to the shared ``utils.atomic_write`` helper
    (same guarantee is now used for the learned-state files)."""
    atomic_write_json(path, data)


def register_todo_tools(mcp: FastMCP) -> None:
    """Register todo list tools.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    def todo_write(todos: str, scope: str = "default") -> str:
        """Update the current todo list.

        Use this for tasks with 3 or more steps; skip it for simple, single-step
        requests.

        Use this to:
        - Break down complex tasks into steps
        - Track progress on multi-step tasks
        - Mark tasks as in_progress or completed

        IMPORTANT:
        - Exactly ONE task must be in_progress at any time
        - Mark tasks completed IMMEDIATELY after finishing
        - Update the list frequently to show progress

        Args:
            todos: JSON string of todo items with format:
                [
                    {
                        "content": "Task description (imperative form)",
                        "status": "pending|in_progress|completed",
                        "activeForm": "Present continuous form (e.g., 'Running tests')"
                    }
                ]

        Returns:
            JSON string with success status and current todos
        """
        try:
            # Parse todos JSON string
            todos_list = json.loads(todos)

            # Validate todo format
            for todo in todos_list:
                if (
                    "content" not in todo
                    or "status" not in todo
                    or "activeForm" not in todo
                ):
                    return json.dumps(
                        {
                            "error": True,
                            "message": "Each todo must have 'content', 'status', and 'activeForm'",
                        }
                    )

                if todo["status"] not in ["pending", "in_progress", "completed"]:
                    return json.dumps(
                        {
                            "error": True,
                            "message": f"Invalid status: {todo['status']}. Must be: pending, in_progress, or completed",
                        }
                    )

            # Count tasks by status
            in_progress_count = sum(
                1 for t in todos_list if t["status"] == "in_progress"
            )

            # Warn if not exactly one in_progress task (when there are uncompleted tasks)
            uncompleted = sum(1 for t in todos_list if t["status"] != "completed")
            if uncompleted > 0 and in_progress_count != 1:
                logger.warning(
                    f"Expected exactly 1 in_progress task, but found {in_progress_count}. "
                    f"You should have exactly one task in_progress at a time."
                )

            # Atomic write (temp + os.replace) so concurrent orchestrator waves /
            # background sub-agents can't read a half-written list.
            _atomic_write_json(_todo_file(scope), todos_list)

            # Build status summary
            pending = sum(1 for t in todos_list if t["status"] == "pending")
            completed = sum(1 for t in todos_list if t["status"] == "completed")

            return json.dumps(
                {
                    "success": True,
                    "total": len(todos_list),
                    "pending": pending,
                    "in_progress": in_progress_count,
                    "completed": completed,
                    "message": f"Todo list updated: {completed}/{len(todos_list)} completed",
                }
            )

        except json.JSONDecodeError as e:
            return json.dumps(
                {"error": True, "message": f"Invalid JSON format: {str(e)}"}
            )
        except Exception as e:
            logger.error(f"Error in todo_write: {str(e)}", exc_info=True)
            return json.dumps(
                {"error": True, "message": f"Error updating todos: {str(e)}"}
            )

    @mcp.tool()
    def todo_read(scope: str = "default") -> str:
        """Read the current todo list.

        Args:
            scope: Optional list label (must match the ``scope`` used to write).

        Returns:
            JSON string with current todos
        """
        try:
            todo_file = _todo_file(scope)
            if not os.path.exists(todo_file):
                return json.dumps({"todos": [], "message": "No active todo list"})

            with open(todo_file, "r") as f:
                todos_list = json.load(f)

            return json.dumps({"todos": todos_list, "count": len(todos_list)})
        except Exception as e:
            logger.error(f"Error in todo_read: {str(e)}", exc_info=True)
            return json.dumps(
                {"error": True, "message": f"Error reading todos: {str(e)}"}
            )

    @mcp.tool()
    def todo_clear(scope: str = "default") -> str:
        """Clear the current todo list.

        Use this when starting a completely new task or when
        the current todo list is no longer relevant.

        Args:
            scope: Optional list label (must match the ``scope`` used to write).

        Returns:
            JSON string with success status
        """
        try:
            todo_file = _todo_file(scope)
            if os.path.exists(todo_file):
                os.remove(todo_file)

            return json.dumps({"success": True, "message": "Todo list cleared"})
        except Exception as e:
            logger.error(f"Error in todo_clear: {str(e)}", exc_info=True)
            return json.dumps(
                {"error": True, "message": f"Error clearing todos: {str(e)}"}
            )
