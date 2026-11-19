"""Early recovery from stale observations without limiting useful tool work."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from mnemoai.client.agent.agent import INTERRUPTED_MARKER, LangGraphAgent
from mnemoai.client.agent.task_continuation import IncompleteResult
from mnemoai.client.agent.tool_progress import ToolProgressGuard
from mnemoai.client.session_log import SessionLog, read_session


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path / "home"))


def test_alternating_unchanged_reads_are_detected(tmp_path):
    path = tmp_path / "file"
    path.write_text("unchanged")
    guard = ToolProgressGuard()
    for _ in range(3):
        guard.observe("fs_read", {"path": str(path)}, "unchanged")
        guard.observe("execute_bash", {"command": "wc -l file"}, '{"stdout":"1","exit_status":0}')
    assert guard.repeated("fs_read", {"path": str(path)})
    assert guard.repeated("execute_bash", {"command": "wc -l file"})


def test_changed_file_or_result_is_new_information(tmp_path):
    path = tmp_path / "file"
    path.write_text("old")
    guard = ToolProgressGuard()
    args = {"path": str(path)}
    for _ in range(3):
        guard.observe("fs_read", args, "old")
    path.write_text("new contents")
    assert not guard.repeated("fs_read", args)
    guard.observe("fs_read", args, "new contents")
    assert not guard.repeated("fs_read", args)
    for value in range(100):
        guard.observe("web_search", {"query": "changing"}, str(value))
        assert not guard.repeated("web_search", {"query": "changing"})


def test_new_information_permits_verifying_a_previously_unchanged_command():
    guard = ToolProgressGuard()
    args = {"command": "wc -l file"}
    for _ in range(3):
        guard.observe("execute_bash", args, '{"stdout":"1","exit_status":0}')
    assert guard.repeated("execute_bash", args)
    guard.observe("fs_read", {"path": "new-evidence"}, "New information")
    assert not guard.repeated("execute_bash", args)
    guard.observe("execute_bash", args, '{"stdout":"1","exit_status":0}')
    assert guard.repeated("execute_bash", args)


def test_changing_read_ranges_without_new_content_does_not_evade_the_guard():
    guard = ToolProgressGuard()
    for end in range(3, 9):
        notice = guard.observe("fs_read", {"path": "file", "end_line": end}, json.dumps({
            "content": "1\tOnly line\n", "total_lines": 1, "end_line": 1,
            "lines_requested": end, "message": f"Read using limit {end}",
        }))
    assert notice
    assert guard.exhausted


def test_default_read_arguments_and_cosmetic_write_summaries_are_not_new_calls():
    guard = ToolProgressGuard()
    for _ in range(3):
        guard.observe("fs_read", {"path": "file", "mode": "Line", "start_line": 1}, '{"content":"same"}')
    assert guard.repeated("fs_read", {"path": "file"})
    for i in range(3):
        guard.observe("fs_write", {"path": "file", "summary": f"Label {i}"}, '{"error":true}')
    assert guard.repeated("fs_write", {"path": "file", "summary": "Another label"})


def test_simple_shell_quoting_is_normalized_but_expansions_are_not():
    guard = ToolProgressGuard()
    for _ in range(3):
        guard.observe("execute_bash", {"command": 'cat "file"'}, '{"stdout":"same","exit_status":0}')
    assert guard.repeated("execute_bash", {"command": "cat file"})
    for _ in range(3):
        guard.observe("execute_bash", {"command": "cat '*'"}, '{"stdout":"literal","exit_status":0}')
    assert not guard.repeated("execute_bash", {"command": "cat *"})


def test_successful_writes_tests_and_polling_are_not_read_loops():
    guard = ToolProgressGuard()
    for _ in range(20):
        for name, args in [
            ("fs_write", {"path": "file", "command": "append", "new_str": "another line"}),
            ("execute_bash", {"command": "pytest tests/test_flaky.py"}),
            ("get_task_status", {"task_id": "running"}),
            ("wait_for_task", {"task_id": "running"}),
        ]:
            guard.observe(name, args, '{"success":true,"status":"running"}')
            assert not guard.repeated(name, args)

    for _ in range(20):
        guard.observe("wait_for_task", {"task_id": "running"}, '{"success":false,"status":"running"}')
        assert not guard.repeated("wait_for_task", {"task_id": "running"})


def test_failed_polling_and_explicit_no_change_are_detected():
    guard = ToolProgressGuard()
    for _ in range(3):
        guard.observe("get_task_status", {"task_id": "missing"}, '{"error":true,"message":"not found"}')
        guard.observe("fs_write", {"path": "file"}, '{"success":true,"changed":false}')
    assert guard.repeated("get_task_status", {"task_id": "missing"})
    assert guard.repeated("fs_write", {"path": "file"})
    guard.reset()
    assert not guard.repeated("get_task_status", {"task_id": "missing"})


class Read:
    name = "fs_read"

    def __init__(self):
        self.calls = 0

    def invoke(self, args):
        self.calls += 1
        return '{"content":"unchanged","total_lines":1}'


class RepeatingModel:
    def __init__(self, recover=False):
        self.requests = []
        self.recover = recover

    def bind_tools(self, tools, **kwargs):
        return self

    def stream(self, messages, config=None):
        self.requests.append(list(messages))
        blocked = any(isinstance(m, ToolMessage) and "This call was not executed" in m.content for m in messages)
        if blocked and self.recover:
            yield AIMessageChunk(content="The previous read already supplied the answer.")
            return
        yield AIMessageChunk(content="", tool_call_chunks=[{
            "id": f"call-{len(self.requests)}", "name": "fs_read",
            "args": '{"path":"file"}', "index": 0,
        }])


def test_main_loop_recovers_without_repeating_the_tool(tmp_path):
    model, reader = RepeatingModel(recover=True), Read()
    agent = LangGraphAgent(model, [reader], "SYS", verbose=False)
    agent.recursion_limit = 200
    answer = agent.invoke("Read the file and report its contents.")
    assert "already supplied" in answer
    assert reader.calls == 3 and len(model.requests) == 5
    assert not agent._last_turn_stopped


def test_persistent_loop_stops_early_and_retains_all_results(tmp_path):
    model, reader = RepeatingModel(), Read()
    agent = LangGraphAgent(model, [reader], "SYS", verbose=False)
    agent.recursion_limit = 200
    agent.session_log = SessionLog(cwd=str(tmp_path))
    answer = agent.invoke("Read the file and report its contents.")
    assert "task is incomplete" in answer
    assert reader.calls == 3 and len(model.requests) == 6
    assert agent._last_turn_stopped
    replies = [m for m in agent.messages if isinstance(m, ToolMessage)]
    assert len(replies) == 6 and len({m.tool_call_id for m in replies}) == 6
    assert all(m.content != INTERRUPTED_MARKER for m in agent.messages)
    assert "execution guard" in str(read_session(agent.session_log.path)["all_messages"])


def test_concurrent_workers_have_independent_guards_and_do_not_stop_parent():
    agent = LangGraphAgent(RepeatingModel(), [Read()], "SYS", verbose=False)
    agent._last_turn_stopped = False

    def worker(_):
        model, reader = RepeatingModel(), Read()
        result, messages = agent._run_worker_loop(
            model, [reader], "Read a file", max_iterations=100, quiet=True,
        )
        return result, reader.calls, len(model.requests), messages

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, range(2)))
    for result, calls, requests, messages in results:
        assert isinstance(result, IncompleteResult) and "task is incomplete" in result
        assert calls == 3 and requests == 6
        assert len([m for m in messages if isinstance(m, ToolMessage)]) == 6
    assert agent._last_turn_stopped is False


def test_new_guidance_resets_recovery_budget():
    from mnemoai.client.agent import mid_turn

    guard = ToolProgressGuard()
    for _ in range(3):
        guard.observe("fs_read", {"path": "same"}, '{"error":true}')
    guard.refuse("fs_read")
    agent = SimpleNamespace(_tool_loop_guard=guard, _mid_turn_queue=["The file has been restored; retry."])
    assert mid_turn.drain(agent)
    assert guard.refusals == 0 and not guard.repeated("fs_read", {"path": "same"})


def test_completed_poll_results_do_not_exempt_an_unproductive_loop():
    guard = ToolProgressGuard()
    for _ in range(3):
        guard.observe("get_task_status", {"task_id": "done"}, '{"status":"completed","success":true}')
    assert guard.repeated("get_task_status", {"task_id": "done"})


def test_malformed_poll_metadata_and_paths_do_not_crash_observation():
    guard = ToolProgressGuard()
    guard.observe("get_task_status", {}, '{"tasks":null,"status":[]}')
    guard.observe("fs_read", {"path": "bad\0path"}, '{"error":true}')
    assert not guard.repeated("fs_read", {"path": "bad\0path"})


def test_recovery_advice_preserves_result_classification():
    from mnemoai.client.agent.tool_loop import _with_progress
    from mnemoai.utils.tool_results import is_error_result, is_unchanged_result

    error = _with_progress('{"error":true,"message":"Denied"}', "Correct the request")
    assert is_error_result(error)
    assert json.loads(error)["message"] == "Denied"
    unchanged = _with_progress('{"success":true,"changed":false}', "Already satisfied")
    assert is_unchanged_result(unchanged) and not is_error_result(unchanged)


def test_guard_stop_retains_results_of_workers_already_run_in_the_batch(tmp_path):
    agent = LangGraphAgent(RepeatingModel(), [Read()], "SYS", verbose=False)
    guard = agent._tool_loop_guard = ToolProgressGuard()
    args = {"path": str(tmp_path / "file")}
    for _ in range(3):
        guard.observe("fs_read", args, '{"content":"unchanged"}')
    guard.refuse("fs_read")
    guard.refuse("fs_read")
    # Foreground spawn batches execute before the sequential tool reply loop.
    completed = {"worker-1": "Completed first worker", "worker-2": "Completed second worker"}
    agent._run_spawn_batch = lambda calls: completed
    calls = [
        {"id": "read", "name": "fs_read", "args": args},
        *[{"id": key, "name": "spawn_agent", "args": {"prompt": key}} for key in completed],
    ]
    result = agent._execute_tools({"messages": [AIMessage(content="", tool_calls=calls)]})
    replies = {m.tool_call_id: m.content for m in result["messages"] if isinstance(m, ToolMessage)}
    assert agent._last_turn_stopped
    assert "not executed" in replies["read"]
    for key, content in completed.items():
        assert replies[key] == content
