"""Todo list management for tracking multi-step tasks."""

import json
import os
from typing import Dict, List

from mcp.server.fastmcp import FastMCP

from mnemoai.utils.config import config
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import profile_dir

TODO_FILE = str(profile_dir() / "todos" / "current_todos.json")


def register_todo_tools(mcp: FastMCP) -> None:
    """Register todo list tools.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    async def todo_write(todos: str) -> str:
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

            # Write to file
            os.makedirs(os.path.dirname(TODO_FILE), exist_ok=True)
            with open(TODO_FILE, "w") as f:
                json.dump(todos_list, f, indent=2)

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
    async def todo_read() -> str:
        """Read the current todo list.

        Returns:
            JSON string with current todos
        """
        try:
            if not os.path.exists(TODO_FILE):
                return json.dumps({"todos": [], "message": "No active todo list"})

            with open(TODO_FILE, "r") as f:
                todos_list = json.load(f)

            return json.dumps({"todos": todos_list, "count": len(todos_list)})
        except Exception as e:
            logger.error(f"Error in todo_read: {str(e)}", exc_info=True)
            return json.dumps(
                {"error": True, "message": f"Error reading todos: {str(e)}"}
            )

    @mcp.tool()
    async def todo_clear() -> str:
        """Clear the current todo list.

        Use this when starting a completely new task or when
        the current todo list is no longer relevant.

        Returns:
            JSON string with success status
        """
        try:
            if os.path.exists(TODO_FILE):
                os.remove(TODO_FILE)

            return json.dumps({"success": True, "message": "Todo list cleared"})
        except Exception as e:
            logger.error(f"Error in todo_clear: {str(e)}", exc_info=True)
            return json.dumps(
                {"error": True, "message": f"Error clearing todos: {str(e)}"}
            )
