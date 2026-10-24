"""Unit tests for background_tasks: process-group cancel + output accounting.

Exercise the real subprocess/thread machinery (no LLM): a cancel must kill the
whole process group (not orphan grandchildren) and must not have its "cancelled"
status clobbered to "failed" by the reader thread; get_task_output reports the
true total_lines. Distinct from test_background_agents.py (which covers the
model-initiated sub-agent registry, not background shell tasks).
"""

import asyncio
import json
import time
import tracemalloc

from mnemoai.server.tools.background_tasks import register_background_tasks_tools
from mnemoai.utils import execution_context


class _CapturingMCP:
    def __init__(self):
        self.registered = {}

    def tool(self):
        def decorator(func):
            self.registered[func.__name__] = func
            return func

        return decorator


def run(result):
    """Resolve a tool result: await a coroutine, else pass the value through.

    A tool with a blocking body is a plain ``def`` (server/tools/thread_offload.py
    offloads it to a thread at registration), so calling it directly here returns
    the string rather than a coroutine.
    """
    return asyncio.run(result) if asyncio.iscoroutine(result) else result


class TestBackgroundTasks:
    def test_default_tail_does_not_load_the_complete_log(self, tmp_path, monkeypatch):
        import mnemoai.server.tools.background_tasks as bt

        tools = self._tools(tmp_path, monkeypatch)
        log = tmp_path / "large.log"
        with log.open("w") as stream:
            for i in range(10000):
                stream.write(f"{i}: " + "x" * 1000 + "\n")
        monkeypatch.setitem(bt._background_tasks, "tail-test", {
            "output_file": str(log), "status": "completed",
        })
        tracemalloc.start()
        try:
            result = json.loads(run(tools["get_task_output"]("tail-test")))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert result["total_lines"] == 10000
        assert len(result["output"].splitlines()) == 50
        assert "9999: " in result["output"]
        assert peak < 3_000_000

    def test_relative_working_directory_is_resolved_in_the_owning_workspace(self, tmp_path, monkeypatch):
        server = tmp_path / "server"
        worker = tmp_path / "worker"
        (server / "subdir").mkdir(parents=True)
        (worker / "subdir").mkdir(parents=True)
        monkeypatch.chdir(server)
        tools = self._tools(tmp_path / "logs", monkeypatch)
        with execution_context.use(execution_context.Workspace(str(worker))):
            response = json.loads(run(tools["start_background_task"](
                "pwd", working_directory="subdir",
            )))
        result = json.loads(run(tools["wait_for_task"](response["task_id"], timeout_seconds=5)))
        assert result["success"]
        assert result["output"].strip() == str(worker / "subdir")

    def _tools(self, tmp_path, monkeypatch):
        # Point the task dir at a temp location so we don't touch the real one.
        import mnemoai.server.tools.background_tasks as bt

        monkeypatch.setattr(bt, "TASK_OUTPUT_DIR", str(tmp_path))
        mcp = _CapturingMCP()
        register_background_tasks_tools(mcp)
        return mcp.registered

    def _wait_for_pid(self, task_id):
        import mnemoai.server.tools.background_tasks as bt

        for _ in range(50):
            if bt._background_tasks.get(task_id, {}).get("pid"):
                return
            time.sleep(0.1)

    def test_cancel_kills_grandchild_process_group(self, tmp_path, monkeypatch):
        tools = self._tools(tmp_path, monkeypatch)
        marker = tmp_path / "alive.txt"
        # Grandchild writes a marker after 3s; we cancel almost immediately.
        cmd = f"(sleep 3; echo alive > {marker}) & echo started; sleep 10"
        started = json.loads(run(tools["start_background_task"](cmd, "pg test")))
        task_id = started["task_id"]
        # Poll the pid, not the status (status flips to "running" first).
        self._wait_for_pid(task_id)
        cancel = json.loads(run(tools["cancel_background_task"](task_id)))
        assert cancel.get("success") is True
        time.sleep(4)  # past the grandchild's write window
        assert not marker.exists(), "grandchild survived group cancel"

    def test_cancel_status_not_clobbered_to_failed(self, tmp_path, monkeypatch):
        tools = self._tools(tmp_path, monkeypatch)
        started = json.loads(
            run(tools["start_background_task"]("sleep 30", "cancel status"))
        )
        task_id = started["task_id"]
        self._wait_for_pid(task_id)
        cancel = json.loads(run(tools["cancel_background_task"](task_id)))
        assert cancel.get("success") is True
        # Give the worker thread time to observe EOF + wait() and (previously)
        # clobber the status. It must remain "cancelled", not become "failed".
        time.sleep(1.5)
        st = json.loads(run(tools["get_task_status"](task_id)))
        assert st["status"] == "cancelled"

    def test_total_lines_reported(self, tmp_path, monkeypatch):
        tools = self._tools(tmp_path, monkeypatch)
        started = json.loads(
            run(tools["start_background_task"]("printf 'l1\\nl2\\nl3\\n'", "lines"))
        )
        task_id = started["task_id"]
        run(tools["wait_for_task"](task_id, timeout_seconds=10))
        out = json.loads(run(tools["get_task_output"](task_id)))
        assert out["total_lines"] == 3

    def test_total_lines_zero_when_no_output_file(self, tmp_path, monkeypatch):
        tools = self._tools(tmp_path, monkeypatch)
        # Register a fake task whose log file never gets created.
        import mnemoai.server.tools.background_tasks as bt

        bt._background_tasks["ghost"] = {
            "status": "running",
            "output_file": str(tmp_path / "does_not_exist.log"),
        }
        try:
            out = json.loads(run(tools["get_task_output"]("ghost")))
            assert out["total_lines"] == 0
        finally:
            bt._background_tasks.pop("ghost", None)
