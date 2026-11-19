"""Long tasks retain their goal and completed actions across execution segments."""

import inspect
import json
import threading

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from mnemoai.client.agent.agent import (
    INTERRUPTED_MARKER,
    LangGraphAgent,
    _ContextOverflow,
)
from mnemoai.client.agent.task_continuation import (
    IncompleteResult,
    ProgressLedger,
    TaskContext,
    compact_worker,
)
from mnemoai.client.client import LangGraphClient
from mnemoai.client.managers.agent_conversation_manager import AgentConversationManager
from mnemoai.client.session_log import SessionLog, read_session


def _agent(tmp_path, *, repeat=False, steps=7):
    calls, requests = [], []

    class Write:
        name = "fs_write"

        def invoke(self, args):
            calls.append(args["index"])
            (tmp_path / f"item-{args['index']}").write_text("written once")
            return f"Stored item {args['index']}"

    class Model:
        def bind_tools(self, tools, **kwargs):
            return self

        def stream(self, messages, config=None):
            requests.append(list(messages))
            if len(calls) >= steps and not repeat:
                yield AIMessageChunk(content="FINISHED")
            else:
                index = 0 if repeat else len(calls)
                yield AIMessageChunk(content="", tool_call_chunks=[{
                    "id": f"call-{len(requests)}", "name": "fs_write",
                    "args": json.dumps({"index": index}), "index": 0,
                }])

    agent = LangGraphAgent(Model(), [Write()], "SYS", verbose=False,
                           auto_approve_provider=lambda: "all")
    agent.session_log = SessionLog(cwd=str(tmp_path))
    return agent, calls, requests


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path / "home"))


@pytest.mark.parametrize("limit", [3, 4, 5])
def test_long_task_continues_after_checkpoints_and_compaction_without_replay(tmp_path, limit):
    agent, calls, requests = _agent(tmp_path)
    agent.recursion_limit = limit
    compacted = []

    def compact(force=False):
        if not calls:
            return False
        # A successful summary may omit the original request. The live task must
        # still carry it, along with always-on instructions and newer guidance.
        agent._messages = agent.messages[-2:]
        agent.system_prompt = "SYS\nSummary of completed work."
        compacted.append(len(calls))
        agent.session_log.log_compaction(summary="Completed work.", kept=agent.messages)
        if len(compacted) == 1:
            assert agent.accept_mid_turn("Preserve my unrelated edits.")
        return True

    agent._compact_provider = compact
    answer = agent.invoke("<steering>Use fixture tools only.</steering>\nROOT REQUEST")
    assert answer == "FINISHED"
    assert calls == list(range(7))
    assert compacted
    for messages in requests:
        assert any("ROOT REQUEST" in str(m.content) for m in messages)
        assert any("Use fixture tools only." in str(m.content) for m in messages)
    assert any("Preserve my unrelated edits." in str(m.content) for m in requests[-1])
    assert not agent._last_turn_stopped
    assert agent._task_context is None
    transcript = agent._last_turn_messages
    assert sum(isinstance(m, ToolMessage) for m in transcript) == 7
    assert sum(isinstance(m, HumanMessage) and m.content == "ROOT REQUEST" for m in transcript) == 1
    records = [json.loads(line) for line in agent.session_log.path.read_text().splitlines()]
    assert sum(r.get("t") == "turn" for r in records) == 1
    restored = read_session(agent.session_log.path)
    assert len(restored["messages"]) == len(agent.messages)
    assert "<active-task>" not in str(restored["all_messages"])


def test_unchanged_tool_loop_stops_and_is_never_marked_complete(tmp_path):
    agent, calls, _ = _agent(tmp_path, repeat=True)
    agent.recursion_limit = 4
    answer = agent.invoke("Do a finite job.")
    assert "step limit" in answer
    assert len(calls) <= 4
    assert agent._last_turn_stopped
    assert all(m.content != INTERRUPTED_MARKER for m in agent.messages)


def test_refused_tools_do_not_renew_execution_budget(tmp_path):
    agent, calls, requests = _agent(tmp_path, repeat=True)
    agent.recursion_limit = 4
    agent._confirm_tool = lambda *args: False
    agent.invoke("Try the operation.")
    assert calls == []
    assert len(requests) <= 2
    assert agent._last_turn_stopped


def test_guard_advice_and_unchanged_writes_do_not_renew_execution_budget():
    ledger = ProgressLedger()

    def messages(result):
        return [
            AIMessage(content="", tool_calls=[{"id": "read", "name": "fs_read", "args": {"path": "file"}}]),
            ToolMessage(content=json.dumps(result), tool_call_id="read"),
        ]

    ledger.observe(messages({"content": "same"}))
    assert ledger.take_progress()
    ledger.observe(messages({"content": "same", "progress_notice": "Reuse the result"}))
    assert not ledger.take_progress()
    ledger.observe([
        AIMessage(content="", tool_calls=[{"id": "write", "name": "fs_write", "args": {"path": "file"}}]),
        ToolMessage(content='{"success":true,"changed":false}', tool_call_id="write"),
    ])
    assert not ledger.take_progress()


def test_cancel_during_checkpoint_does_not_start_more_tools(tmp_path):
    agent, calls, _ = _agent(tmp_path)
    agent.recursion_limit = 3

    def compact(force=False):
        if calls:
            agent.request_cancel()
        return False
    agent._compact_provider = compact
    with pytest.raises(KeyboardInterrupt):
        agent.invoke("Do the fixture work.")
    assert calls == [0]
    assert agent._last_turn_stopped
    assert agent._task_context is None


def test_context_overflow_compacts_current_tool_work_before_retrying(tmp_path):
    agent, calls, requests = _agent(tmp_path, steps=3)
    invoke = agent.tools[0].invoke
    agent.tools[0].invoke = lambda args: invoke(args) + " retained evidence" * 500
    stream = agent._stream_response
    overflow = []
    compacted = []

    def model(messages, *args, **kwargs):
        if len(calls) == 2 and not overflow:
            overflow.append(True)
            raise _ContextOverflow("fixture overflow")
        return stream(messages, *args, **kwargs)

    def compact(force=False):
        if not force:
            return False
        compacted.append(sum(isinstance(m, ToolMessage) for m in agent.messages))
        agent._messages = agent.messages[-2:]
        agent.system_prompt = "Completed items 0 and 1."
        return True

    agent._stream_response = model
    agent._compact_provider = compact
    assert agent.invoke("ORIGINAL TASK") == "FINISHED"
    assert calls == [0, 1, 2]
    assert compacted == [2]
    assert all(any("ORIGINAL TASK" in str(m.content) for m in request) for request in requests)
    assert sum(isinstance(m, ToolMessage) for m in agent._last_turn_messages) == 3


def test_task_guidance_survives_lossy_history_without_mutating_it():
    context = TaskContext("Original goal")
    context.add_guidance(["A status request is not cancellation."])
    history = [HumanMessage(content="Summary only")]
    outgoing = context.apply(history)
    assert history == [HumanMessage(content="Summary only")]
    assert isinstance(outgoing[0], SystemMessage)
    assert "Original goal" in outgoing[0].content
    assert "A status request is not cancellation." in outgoing[0].content
    assert outgoing[-1] == history[-1]
    assert context.apply(outgoing) == outgoing


@pytest.mark.parametrize("prompt", ["hello", "ciao!", "thank you", "Read README.md"])
def test_present_request_needs_no_continuation_reminder(prompt):
    history = [SystemMessage(content="Base instructions"), HumanMessage(content=prompt)]
    assert TaskContext(prompt).apply(history) == history


def test_context_and_request_stay_separate_in_real_agent_requests(tmp_path):
    agent, calls, requests = _agent(tmp_path, steps=0)
    user = "Review this file without changing it."
    injected = "<episodic-context>Historical tool observations only.</episodic-context>\n\n"
    assert agent.invoke(injected + user, request=user) == "FINISHED"
    outgoing = requests[0]
    assert [m.content for m in outgoing if isinstance(m, HumanMessage)] == [user]
    assert any(injected in str(m.content) for m in outgoing if isinstance(m, SystemMessage))
    assert "<active-task>" not in str(outgoing)
    assert all("<episodic-context>" not in str(m.content) for m in agent.messages)


def test_multiblock_system_context_is_not_mutated_or_repeated():
    original = SystemMessage(content=[{"type": "text", "text": "base", "cache_control": {"type": "ephemeral"}}])
    history = [original, HumanMessage(content="current")]
    task = TaskContext("original task", context="trusted context")
    outgoing = task.apply(history)
    assert original.content == [{"type": "text", "text": "base", "cache_control": {"type": "ephemeral"}}]
    assert outgoing[0].content[0] == original.content[0]
    assert task.apply(outgoing) == outgoing


def test_follow_through_uses_a_provider_compatible_control_event():
    from langchain_anthropic.chat_models import _format_messages

    task = TaskContext("Complete the original work")
    messages = [
        SystemMessage(content="Base instructions"), HumanMessage(content=task.request),
        AIMessage(content="Here is the status."), task.follow_through(),
    ]
    outgoing = task.apply(messages)
    options = {"model": "claude-sonnet-4-5"} if "model" in inspect.signature(_format_messages).parameters else {}
    system, body = _format_messages(outgoing, **options)
    assert system and body[-1]["role"] == "user"
    assert sum(isinstance(m, SystemMessage) for m in outgoing) == 1
    assert LangGraphAgent._last_human_query(outgoing) == task.request
    assert outgoing[-1].additional_kwargs["mnemoai_control"]


def test_user_authored_reserved_markup_is_not_discarded(tmp_path):
    agent, calls, requests = _agent(tmp_path, steps=0)
    user = "Explain this text: <episodic-context>example data</episodic-context>"
    assert agent.invoke(user, request=user) == "FINISHED"
    assert LangGraphAgent._last_human_query(requests[0]) == user
    assert agent.messages[0].content == user


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_preflight_failure_keeps_input_and_closes_acceptance_window(tmp_path, error):
    agent, calls, _ = _agent(tmp_path)
    def compact(force=False):
        raise error("fixture")
    agent._compact_provider = compact
    with pytest.raises(error):
        agent.invoke("Keep this request in the transcript.")
    assert calls == []
    assert not agent.accept_mid_turn("Too late")
    assert agent._last_turn_stopped
    records = [json.loads(line) for line in agent.session_log.path.read_text().splitlines()]
    assert sum(r.get("t") == "turn" for r in records) == 1
    assert "Keep this request" in str(read_session(agent.session_log.path)["messages"])


def test_fresh_job_ids_do_not_disguise_repeated_delegation():
    progress = ProgressLedger()
    for index in range(2):
        progress.observe([
            AIMessage(content="", tool_calls=[{
                "id": str(index), "name": "spawn_agent", "args": {"prompt": "same task"},
            }]),
            ToolMessage(content=f"Started job-{index}", tool_call_id=str(index)),
        ])
        assert progress.take_progress() is (index == 0)


def test_status_reply_does_not_end_the_original_tool_task(tmp_path):
    agent, calls, requests = _agent(tmp_path, steps=3)
    run = agent._run_tool_calls
    stream = agent._stream_response
    replied = []

    def tools(*args, **kwargs):
        run(*args, **kwargs)
        if calls == [0]:
            assert agent.accept_mid_turn("What is the status?")

    def answer_status(messages, *args, **kwargs):
        if calls == [0] and not replied:
            replied.append(True)
            return AIMessage(content="One item is finished; two remain."), False
        return stream(messages, *args, **kwargs)

    agent._run_tool_calls = tools
    agent._stream_response = answer_status
    assert agent.invoke("Complete all three items.") == "FINISHED"
    assert calls == [0, 1, 2]
    assert sum("task-continuation" in str(m.content) for m in agent.messages) == 0


@pytest.mark.parametrize("overflow", [False, True])
def test_worker_continues_with_private_compaction_and_keeps_full_trace(tmp_path, overflow):
    agent, calls, requests = _agent(tmp_path)
    agent._messages = [HumanMessage(content="PARENT HISTORY")]
    parent = list(agent.messages)
    compactions = []
    if overflow:
        original = agent._stream_response
        raised = []
        def stream(messages, *args, **kwargs):
            if calls == [0] and not raised:
                raised.append(True)
                raise _ContextOverflow("fixture context overflow")
            return original(messages, *args, **kwargs)
        agent._stream_response = stream
    def compact(messages, *, force=False, cancelled):
        compactions.append(force)
        return [SystemMessage(content="SYS"), HumanMessage(content="Summary of completed items.")] + messages[-2:]
    agent._worker_compact_provider = compact
    result, trace = agent._run_worker_loop(
        agent.model, agent.tools, "WORKER GOAL", max_iterations=2, quiet=True,
    )
    assert result == "FINISHED" and not isinstance(result, IncompleteResult)
    assert calls == list(range(7))
    assert sum(isinstance(m, ToolMessage) for m in trace) == 7
    assert compactions and (not overflow or True in compactions)
    assert agent.messages == parent
    assert all("PARENT HISTORY" not in str(messages) for messages in requests)
    assert all(any("WORKER GOAL" in str(m.content) for m in messages) for messages in requests)


def test_worker_unchanged_activity_is_reported_incomplete(tmp_path):
    agent, calls, _ = _agent(tmp_path, repeat=True)
    sink = agent._activity.open_run("explore", "repeat", "background")
    result, _ = agent._run_worker_loop(
        agent.model, agent.tools, "Do the finite job.", max_iterations=2,
        quiet=True, activity=sink,
    )
    assert isinstance(result, IncompleteResult)
    sink.finish_ok(result)
    assert agent._activity.snapshot()[-1].status == "failed"
    assert len(calls) == 4


def test_worker_summary_cancellation_does_not_accept_late_context():
    cancelled = threading.Event()
    finished = threading.Event()
    def summarizer(messages, *, force, cancelled):
        finished.wait(2)
        return [HumanMessage(content="late context")]
    def stop():
        cancelled.set()
        finished.set()
    timer = threading.Timer(0.02, stop)
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            compact_worker(summarizer, [], force=True, cancelled=cancelled.is_set, timeout=2)
    finally:
        finished.set()
        timer.join()


@pytest.mark.parametrize("overflow", [False, True])
@pytest.mark.parametrize("cancel_parent", [False, True])
def test_stop_during_worker_compaction_respects_cancellation_scope(tmp_path, overflow, cancel_parent):
    agent, calls, _ = _agent(tmp_path)
    stop_worker = threading.Event()
    sink = agent._activity.open_run("explore", "cancel", "foreground", cancel=stop_worker)
    original = agent._stream_response
    if overflow:
        def stream(messages, *args, **kwargs):
            if calls:
                raise _ContextOverflow("fixture overflow")
            return original(messages, *args, **kwargs)
        agent._stream_response = stream
    def compact(messages, *, force=False, cancelled):
        agent.request_cancel() if cancel_parent else stop_worker.set()
        return [HumanMessage(content="Must never become active")]
    agent._worker_compact_provider = compact
    def run():
        return agent._run_worker_loop(
            agent.model, agent.tools, "WORKER GOAL", max_iterations=2 if overflow else 1,
            quiet=True, activity=sink,
        )
    if cancel_parent:
        with pytest.raises(KeyboardInterrupt):
            run()
    else:
        result, trace = run()
        sink.finish_ok(result)
        assert agent._activity.snapshot()[-1].status == "stopped"
        assert not agent._cancelled()
        assert all("Must never become active" not in str(m.content) for m in trace)
    assert calls == [0]


def test_worker_summary_does_not_mutate_parent_context_or_token_count(tmp_path):
    agent, _, _ = _agent(tmp_path)
    agent._messages = [HumanMessage(content="PARENT")]
    agent._last_input_tokens = 123
    client = LangGraphClient.__new__(LangGraphClient)
    client.agent = agent
    client.conversation_manager = AgentConversationManager(max_tokens=1000)
    client._worker_compaction_slot = threading.BoundedSemaphore(1)
    calls = []
    class SummaryModel:
        async def ainvoke(self, messages, **kwargs):
            calls.append(messages)
            return AIMessage(content="Earlier worker actions completed.", usage_metadata={
                "input_tokens": 10, "output_tokens": 4, "total_tokens": 14,
            })
    client._summary_model = lambda: SummaryModel()
    messages = [SystemMessage(content="WORKER SYSTEM")]
    messages.extend(HumanMessage(content=f"Item {i} " + "detail " * 100) for i in range(30))
    before = list(messages)
    fresh = client._compact_worker_context(messages, force=True)
    assert fresh is not None and len(fresh) < len(messages)
    assert fresh[0].content == "WORKER SYSTEM"
    assert calls
    assert messages == before
    assert agent.messages == [HumanMessage(content="PARENT")]
    assert agent._last_input_tokens == 123
    assert client.conversation_manager.summary_text == ""
