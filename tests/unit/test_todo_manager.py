"""Unit tests for todo_manager scoping + atomic writes

Concurrent orchestrator waves / background sub-agents share one server process,
so the todo list is namespaced per app instance (MNEMOAI_INSTANCE_ID) with an
optional scope label, and writes are atomic (temp + os.replace) so a reader
never sees a half-written file. No LLM.
"""

import asyncio
import json
import os

import pytest

import mnemoai.server.tools.todo_manager as tm


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


_VALID = json.dumps(
    [{"content": "a", "status": "in_progress", "activeForm": "Doing a"}]
)


@pytest.fixture
def tools(tmp_path, monkeypatch):
    # Deterministic instance id + a temp profile dir so files land under tmp.
    monkeypatch.setenv("MNEMOAI_INSTANCE_ID", "testinst")
    monkeypatch.setattr(tm, "profile_dir", lambda: tmp_path)
    monkeypatch.setattr(tm, "instance_id", lambda: "testinst")
    mcp = _CapturingMCP()
    tm.register_todo_tools(mcp)
    return mcp.registered


class TestTodoScoping:
    def test_write_read_roundtrip(self, tools):
        assert json.loads(run(tools["todo_write"](_VALID)))["success"] is True
        r = json.loads(run(tools["todo_read"]()))
        assert r["count"] == 1
        assert r["todos"][0]["content"] == "a"

    def test_scope_isolation(self, tools):
        run(tools["todo_write"](_VALID, scope="waveA"))
        # waveB is a different, empty list.
        r = json.loads(run(tools["todo_read"](scope="waveB")))
        assert r.get("todos") == []
        # waveA still has the entry.
        assert json.loads(run(tools["todo_read"](scope="waveA")))["count"] == 1

    def test_clear_removes_only_that_scope(self, tools):
        run(tools["todo_write"](_VALID, scope="waveA"))
        run(tools["todo_write"](_VALID))  # default
        run(tools["todo_clear"](scope="waveA"))
        assert json.loads(run(tools["todo_read"](scope="waveA"))).get("todos") == []
        assert json.loads(run(tools["todo_read"]()))["count"] == 1  # default intact

    def test_atomic_write_leaves_no_tmp(self, tools, tmp_path):
        run(tools["todo_write"](_VALID))
        todos_dir = tmp_path / "todos"
        leftovers = [p for p in todos_dir.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []
        # The written file is valid JSON.
        written = next(p for p in todos_dir.iterdir() if p.suffix == ".json")
        json.loads(written.read_text())

    def test_per_instance_namespacing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tm, "profile_dir", lambda: tmp_path)
        monkeypatch.setattr(tm, "instance_id", lambda: "instA")
        a = tm._todo_file("default")
        monkeypatch.setattr(tm, "instance_id", lambda: "instB")
        b = tm._todo_file("default")
        assert a != b

    def test_scope_sanitization_prevents_traversal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tm, "profile_dir", lambda: tmp_path)
        monkeypatch.setattr(tm, "instance_id", lambda: "i")
        p = tm._todo_file("../../etc/passwd")
        # No path separators leak into the filename; stays under todos/.
        assert os.sep not in os.path.basename(p)
        assert str(tmp_path / "todos") in p

    def test_read_missing_scope_is_empty(self, tools):
        r = json.loads(run(tools["todo_read"](scope="never-written")))
        assert r["todos"] == []

    def test_one_in_progress_warning_preserved(self, tools, monkeypatch):
        two = json.dumps(
            [
                {"content": "a", "status": "in_progress", "activeForm": "A"},
                {"content": "b", "status": "in_progress", "activeForm": "B"},
            ]
        )
        warnings = []
        monkeypatch.setattr(tm.logger, "warning", lambda msg, *a: warnings.append(msg))
        res = json.loads(run(tools["todo_write"](two)))
        assert res["success"] is True  # still writes
        assert any("in_progress" in w for w in warnings)

    def test_invalid_json_errors(self, tools):
        res = json.loads(run(tools["todo_write"]("not json")))
        assert res["error"] is True
