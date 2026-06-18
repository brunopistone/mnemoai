"""Background task execution system.

Allows running long-running tasks in the background:
- Start tasks asynchronously
- Check task status
- Get task output when complete
- Cancel running tasks
"""

import json
import os
import subprocess
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Optional
from mcp.server.fastmcp import FastMCP
from utils.paths import tasks_dir

# Store for background tasks
_background_tasks: Dict[str, dict] = {}
_task_lock = threading.Lock()

# Task output directory (under the app home, created on demand)
TASK_OUTPUT_DIR = str(tasks_dir())


def ensure_task_dir():
    """Ensure the task output directory exists."""
    os.makedirs(TASK_OUTPUT_DIR, exist_ok=True)


def get_task_output_file(task_id: str) -> str:
    """Get the output file path for a task."""
    return os.path.join(TASK_OUTPUT_DIR, f"{task_id}.log")


def run_background_command(task_id: str, command: str, cwd: str):
    """Run a command in the background and capture output."""
    output_file = get_task_output_file(task_id)

    try:
        with _task_lock:
            _background_tasks[task_id]["status"] = "running"
            _background_tasks[task_id]["started_at"] = datetime.now().isoformat()

        # Run the command
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            text=True,
        )

        with _task_lock:
            _background_tasks[task_id]["pid"] = process.pid

        # Capture output
        output_lines = []
        with open(output_file, "w") as f:
            for line in process.stdout:
                f.write(line)
                f.flush()
                output_lines.append(line)

        process.wait()

        with _task_lock:
            _background_tasks[task_id]["status"] = (
                "completed" if process.returncode == 0 else "failed"
            )
            _background_tasks[task_id]["return_code"] = process.returncode
            _background_tasks[task_id]["completed_at"] = datetime.now().isoformat()
            _background_tasks[task_id]["output_preview"] = "".join(
                output_lines[-20:]
            )  # Last 20 lines

    except Exception as e:
        with _task_lock:
            _background_tasks[task_id]["status"] = "error"
            _background_tasks[task_id]["error"] = str(e)
            _background_tasks[task_id]["completed_at"] = datetime.now().isoformat()


def register_background_tasks_tools(mcp: FastMCP) -> None:
    """Register background task tools."""

    @mcp.tool()
    async def start_background_task(
        command: str, description: str = "", working_directory: str = ""
    ) -> str:
        """Start a command running in the background.

        Use this for long-running operations like:
        - Running test suites
        - Building projects
        - Installing dependencies
        - Running linters on entire codebase

        Args:
            command: Shell command to execute
            description: Brief description of what this task does
            working_directory: Directory to run in (default: current directory)

        Returns:
            JSON string with task_id to check status later

        Examples:
            start_background_task(command="npm test", description="Running tests")
            start_background_task(command="pip install -r requirements.txt")
            start_background_task(command="cargo build --release", description="Building release")
        """
        ensure_task_dir()

        task_id = str(uuid.uuid4())[:8]
        cwd = working_directory if working_directory else os.getcwd()

        # Expand user directory
        cwd = os.path.expanduser(cwd)

        if not os.path.exists(cwd):
            return json.dumps(
                {"error": True, "message": f"Working directory does not exist: {cwd}"},
                indent=2,
            )

        # Create task record
        task = {
            "id": task_id,
            "command": command,
            "description": description or command[:50],
            "working_directory": cwd,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "output_file": get_task_output_file(task_id),
        }

        with _task_lock:
            _background_tasks[task_id] = task

        # Start background thread
        thread = threading.Thread(
            target=run_background_command, args=(task_id, command, cwd), daemon=True
        )
        thread.start()

        return json.dumps(
            {
                "success": True,
                "task_id": task_id,
                "message": f"Task started in background: {description or command[:50]}",
                "check_status": f'get_task_status(task_id="{task_id}")',
                "get_output": f'get_task_output(task_id="{task_id}")',
            },
            indent=2,
        )

    @mcp.tool()
    async def get_task_status(task_id: str) -> str:
        """Get the status of a background task.

        Args:
            task_id: The task ID returned by start_background_task

        Returns:
            JSON string with task status and details
        """
        with _task_lock:
            task = _background_tasks.get(task_id)

        if not task:
            return json.dumps(
                {
                    "error": True,
                    "message": f"Task not found: {task_id}",
                    "suggestion": "Use list_background_tasks() to see all tasks",
                },
                indent=2,
            )

        return json.dumps(
            {
                "success": True,
                "task_id": task_id,
                "status": task["status"],
                "command": task["command"],
                "description": task["description"],
                "created_at": task["created_at"],
                "started_at": task.get("started_at"),
                "completed_at": task.get("completed_at"),
                "return_code": task.get("return_code"),
                "error": task.get("error"),
                "output_file": task["output_file"],
            },
            indent=2,
        )

    @mcp.tool()
    async def get_task_output(task_id: str, tail_lines: int = 50) -> str:
        """Get the output of a background task.

        Args:
            task_id: The task ID
            tail_lines: Number of lines from the end to return (default: 50)

        Returns:
            JSON string with task output
        """
        with _task_lock:
            task = _background_tasks.get(task_id)

        if not task:
            return json.dumps(
                {"error": True, "message": f"Task not found: {task_id}"}, indent=2
            )

        output_file = task["output_file"]
        output = ""

        if os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    lines = f.readlines()
                    if tail_lines > 0:
                        output = "".join(lines[-tail_lines:])
                    else:
                        output = "".join(lines)
            except Exception as e:
                output = f"Error reading output: {e}"

        return json.dumps(
            {
                "success": True,
                "task_id": task_id,
                "status": task["status"],
                "output": output,
                "output_file": output_file,
                "total_lines": len(lines) if "lines" in dir() else 0,
            },
            indent=2,
        )

    @mcp.tool()
    async def list_background_tasks(include_completed: bool = True) -> str:
        """List all background tasks.

        Args:
            include_completed: Include completed/failed tasks (default: True)

        Returns:
            JSON string with list of tasks
        """
        with _task_lock:
            tasks = []
            for task_id, task in _background_tasks.items():
                if include_completed or task["status"] in ["pending", "running"]:
                    tasks.append(
                        {
                            "id": task_id,
                            "status": task["status"],
                            "description": task["description"],
                            "created_at": task["created_at"],
                            "return_code": task.get("return_code"),
                        }
                    )

        # Sort by creation time
        tasks.sort(key=lambda x: x["created_at"], reverse=True)

        running_count = sum(1 for t in tasks if t["status"] == "running")
        completed_count = sum(1 for t in tasks if t["status"] == "completed")
        failed_count = sum(1 for t in tasks if t["status"] == "failed")

        return json.dumps(
            {
                "success": True,
                "tasks": tasks,
                "summary": {
                    "total": len(tasks),
                    "running": running_count,
                    "completed": completed_count,
                    "failed": failed_count,
                },
            },
            indent=2,
        )

    @mcp.tool()
    async def cancel_background_task(task_id: str) -> str:
        """Cancel a running background task.

        Args:
            task_id: The task ID to cancel

        Returns:
            JSON string confirming cancellation
        """
        with _task_lock:
            task = _background_tasks.get(task_id)

        if not task:
            return json.dumps(
                {"error": True, "message": f"Task not found: {task_id}"}, indent=2
            )

        if task["status"] != "running":
            return json.dumps(
                {
                    "error": True,
                    "message": f"Task is not running (status: {task['status']})",
                },
                indent=2,
            )

        pid = task.get("pid")
        if pid:
            try:
                os.kill(pid, 9)  # SIGKILL
                with _task_lock:
                    _background_tasks[task_id]["status"] = "cancelled"
                    _background_tasks[task_id][
                        "completed_at"
                    ] = datetime.now().isoformat()
                return json.dumps(
                    {
                        "success": True,
                        "message": f"Task {task_id} cancelled",
                        "pid": pid,
                    },
                    indent=2,
                )
            except ProcessLookupError:
                return json.dumps(
                    {"error": True, "message": "Process already terminated"}, indent=2
                )
            except Exception as e:
                return json.dumps(
                    {"error": True, "message": f"Error cancelling task: {e}"}, indent=2
                )

        return json.dumps({"error": True, "message": "No PID found for task"}, indent=2)

    @mcp.tool()
    async def wait_for_task(task_id: str, timeout_seconds: int = 300) -> str:
        """Wait for a background task to complete.

        Args:
            task_id: The task ID to wait for
            timeout_seconds: Maximum time to wait (default: 300 = 5 minutes)

        Returns:
            JSON string with task result when complete
        """
        start_time = time.time()

        while True:
            with _task_lock:
                task = _background_tasks.get(task_id)

            if not task:
                return json.dumps(
                    {"error": True, "message": f"Task not found: {task_id}"}, indent=2
                )

            if task["status"] in ["completed", "failed", "error", "cancelled"]:
                # Get output
                output_file = task["output_file"]
                output = ""
                if os.path.exists(output_file):
                    try:
                        with open(output_file, "r") as f:
                            lines = f.readlines()
                            output = "".join(lines[-50:])  # Last 50 lines
                    except:
                        pass

                return json.dumps(
                    {
                        "success": task["status"] == "completed",
                        "task_id": task_id,
                        "status": task["status"],
                        "return_code": task.get("return_code"),
                        "output": output,
                        "duration": task.get("completed_at", "")
                        + " - "
                        + task.get("started_at", ""),
                    },
                    indent=2,
                )

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                return json.dumps(
                    {
                        "error": True,
                        "message": f"Timeout after {timeout_seconds} seconds",
                        "task_id": task_id,
                        "status": task["status"],
                        "suggestion": f'Task still running. Use get_task_status(task_id="{task_id}") to check later.',
                    },
                    indent=2,
                )

            # Wait a bit before checking again
            time.sleep(1)

    @mcp.tool()
    async def clear_completed_tasks() -> str:
        """Remove completed, failed, and cancelled tasks from the list.

        Returns:
            JSON string confirming cleanup
        """
        with _task_lock:
            to_remove = [
                task_id
                for task_id, task in _background_tasks.items()
                if task["status"] in ["completed", "failed", "error", "cancelled"]
            ]

            for task_id in to_remove:
                # Remove output file
                output_file = _background_tasks[task_id]["output_file"]
                if os.path.exists(output_file):
                    try:
                        os.remove(output_file)
                    except:
                        pass
                del _background_tasks[task_id]

        return json.dumps(
            {
                "success": True,
                "message": f"Removed {len(to_remove)} completed tasks",
                "removed_count": len(to_remove),
            },
            indent=2,
        )
