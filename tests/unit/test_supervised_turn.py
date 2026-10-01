"""Supervisor dialogue uses the chat executor, one user turn, and unchanged gates."""

import json
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from mnemoai.client import review
from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.agent.supervised_turn import budget_for, budget_scope
from mnemoai.client.client import LangGraphClient
from mnemoai.client.memory.reflector import current_turn_messages
from mnemoai.client.session_log import SessionLog, read_session, turn_summaries
from mnemoai.client.ui.turn_view import user_prompt_text
from mnemoai.client.work_review import WorkReview
from mnemoai.utils.review_protocol import (
    DATA_MARKER,
    FEEDBACK_PREFIX,
    ReviewStopped,
    feedback_summary,
    is_feedback,
)


def result(kind="pass"):
    return {"verdict": kind, "summary": "Check the requested value.", "findings": [] if kind == "pass" else [
        {"issue": "VALUE is not 2.", "evidence_ids": ["task", "answer"],
         "verification": "Inspect VALUE, correct it if needed, or explain counterevidence."},
    ]}


class Judge:
    def __init__(self, results):
        self.results = iter(results)
        self.packets = []

    def invoke(self, messages, config=None):
        self.packets.append(json.loads(messages[1].content))
        return AIMessage(content=json.dumps(next(self.results)))


class Model:
    def bind_tools(self, tools):
        return self


class WriteTool:
    name = "fs_write"

    def invoke(self, args):
        Path(args["path"]).write_text(args["file_text"])
        return '{"success":true}'


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path / "app"))
    monkeypatch.setitem(review.config._config_data, "REVIEW", {"MAX_CHANGE_REVIEWS": 0})
    # These tests isolate the final correction engine. Strategy/change lifecycle
    # ordering is exercised end to end in test_work_review.
    def prepared(work, state, steps):
        work.options = {**review.settings(), "MAX_CHANGE_REVIEWS": 0}
        work.remaining = work.options["TOTAL_TIMEOUT"]
        work.kind = "code"
        work.capture = review.Capture(work.task)
        work.agent._review_capture = work.capture
        work.agent._completion_supervisor = work.complete
        return 0
    monkeypatch.setattr(WorkReview, "prepare", prepared)
    client = LangGraphClient.__new__(LangGraphClient)
    client.agent = LangGraphAgent(Model(), [WriteTool()], system_prompt="Follow the user.", verbose=False)
    client.agent.session_log = SessionLog(cwd=str(tmp_path))
    client.agent.provenance = SimpleNamespace(record=Mock())
    client.agent._confirm_tool = Mock(return_value=True)
    client.agent._run_hooks = lambda *a, **kw: SimpleNamespace(allowed=False, denied=False, context="", notices=[])
    client.reviewer = review.Reviewer(enabled=True)
    client.llm_controller = SimpleNamespace(model_name="fixture", model_type="ollama")
    client.spinner, client.callback_handler = Mock(), Mock()
    client.spinner_lock = threading.Lock()
    client.mcp_client = nullcontext()
    client.episodic_memory = client.reflector = None
    client.plan_mode_active = False
    client.auto_approve_mode = "off"
    client._steering_reminder = lambda: ""
    client._summary_model = lambda: None
    client._profile_turn = lambda: None
    client._print_context_size = lambda: None
    async def manage(*args):
        pass
    client.conversation_manager = SimpleNamespace(manage_messages=manage)
    return client, tmp_path


def feed_actor(client, responses):
    responses = iter(responses)
    requests = []
    def stream(messages, *args, **kwargs):
        requests.append(list(messages))
        return next(responses), False
    client.agent._stream_response = stream
    return requests


def test_revise_chat_edits_recheck_pass_is_one_original_user_turn(setup, capsys):
    client, root = setup
    target = root / "value.py"
    target.write_text("VALUE = 1\n")
    judge = Judge([result("revise"), result()])
    client._area_model = lambda area: judge
    requests = feed_actor(client, [
        AIMessage(content="I think VALUE is correct."),
        AIMessage(content="", tool_calls=[{"id": "edit-1", "name": "fs_write",
                                         "args": {"path": str(target), "file_text": "VALUE = 2\n"}}]),
        AIMessage(content="Corrected VALUE to 2."),
    ])
    answer = client.query("Set VALUE to 2.")
    assert answer == "Corrected VALUE to 2."
    assert target.read_text() == "VALUE = 2\n"
    assert client.reviewer.last["verdict"] == "pass" and client.reviewer.last["revisions"] == 1
    assert len(judge.packets) == 2
    assert any(is_feedback(m.content) for m in requests[1])
    client.agent._confirm_tool.assert_called_once()
    assert client.auto_approve_mode == "off" and not client.agent._is_headless()
    call = client.agent.provenance.record.call_args
    assert call.kwargs["turn"] == 1 and call.kwargs["prompt"] == "Set VALUE to 2."
    saved = read_session(client.agent.session_log.path)
    assert saved["turns"] == 1 and saved["exchanges"] == 1
    assert len(turn_summaries(client.agent.session_log.path)) == 1
    assert current_turn_messages(client.agent.messages)[0].content == "Set VALUE to 2."
    assert sum(bool(user_prompt_text(m.content)) for m in client.agent.messages if isinstance(m, HumanMessage)) == 1
    assert "review" in client.agent.session_log.path.read_text()
    output = capsys.readouterr().out
    assert output.count("Final answer · chat model") == 1
    assert "Corrected VALUE to 2." in output.split("Final answer · chat model")[-1]
    assert "Chat model · draft (review pending)" in output
    assert "Suggested verification" not in output, "full feedback must stay collapsed"
    assert "Suggested verification" in client.reviewer.view.details
    assert "\033[90mPeer review" in output


def test_chat_can_dispute_finding_without_editing(setup):
    client, root = setup
    target = root / "value.py"
    target.write_text("VALUE = 2\n")
    judge = Judge([result("revise"), result()])
    client._area_model = lambda area: judge
    feed_actor(client, [AIMessage(content="VALUE is already 2."),
                        AIMessage(content="Counterevidence: the file already has VALUE = 2; no edit needed.")])
    assert "Counterevidence" in client.query("Keep VALUE equal to 2.")
    assert target.read_text() == "VALUE = 2\n"
    client.agent._confirm_tool.assert_not_called()
    assert client.reviewer.last["verdict"] == "pass"
    assert any(i["kind"] == "prior_review_exchange" for i in judge.packets[1]["evidence"])


def test_denied_edit_is_never_approved_by_the_reviewer(setup, capsys):
    client, root = setup
    target = root / "value.py"
    target.write_text("VALUE = 1\n")
    client.agent._confirm_tool.return_value = False
    client._area_model = lambda area: Judge([result("revise")])
    # Return the same judge instance; call count must not cause a fresh loop.
    judge = Judge([result("revise"), {**result(), "verdict": "inconclusive", "summary": "User declined the edit."}])
    client._area_model = lambda area: judge
    feed_actor(client, [
        AIMessage(content="No change yet."),
        AIMessage(content="", tool_calls=[{"id": "edit-1", "name": "fs_write",
                                         "args": {"path": str(target), "file_text": "VALUE = 2\n"}}]),
        AIMessage(content="The user declined; no change made."),
    ])
    client.query("Set VALUE to 2.")
    assert target.read_text() == "VALUE = 1\n"
    assert client.reviewer.last["verdict"] == "inconclusive"
    client.agent._confirm_tool.assert_called_once()
    output = capsys.readouterr().out
    assert "Final answer · chat model" not in output
    assert "Chat model answer · review incomplete" in output


def test_round_limit_does_not_reset_on_disagreement(setup, monkeypatch):
    client, _ = setup
    monkeypatch.setitem(review.config._config_data, "REVIEW", {"MAX_ROUNDS": 2})
    judge = Judge([result("revise")] * 3)
    client._area_model = lambda area: judge
    requests = feed_actor(client, [AIMessage(content=f"response {n}") for n in range(3)])
    assert client.query("Set VALUE to 2.") == "response 2"
    assert len(requests) == len(judge.packets) == 3
    assert client.reviewer.last["revisions"] == 2
    assert "limit" in client.reviewer.last["summary"]


def test_zero_rounds_preserves_one_shot_review(setup, monkeypatch):
    client, _ = setup
    monkeypatch.setitem(review.config._config_data, "REVIEW", {"MAX_ROUNDS": 0})
    client._area_model = lambda area: Judge([result("revise")])
    requests = feed_actor(client, [AIMessage(content="original answer")])
    assert client.query("task") == "original answer"
    assert len(requests) == 1 and client.reviewer.last["revisions"] == 0


def test_cancellation_during_correction_keeps_work_and_one_turn(setup, capsys):
    client, _ = setup
    client._area_model = lambda area: Judge([result("revise")])
    count = 0
    def stream(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise KeyboardInterrupt
        return AIMessage(content="initial answer"), False
    client.agent._stream_response = stream
    assert client.query("task") == "Operation was cancelled."
    saved = read_session(client.agent.session_log.path)
    assert saved["turns"] == 1
    assert any("initial answer" in str(m) for m in saved["messages"])
    assert client.reviewer.last["verdict"] == "inconclusive"
    output = capsys.readouterr().out
    assert "Final answer · chat model" not in output
    assert "Suggested verification" not in output


def test_time_budget_stops_new_tools_but_keeps_completed_calls(setup):
    client, root = setup
    budget = review.LoopBudget(5, 10)
    target = root / "must-not-exist"
    budget.deadline = 0
    messages = []
    with budget_scope(client.agent, budget):
        client.agent._run_tool_calls(
            [{"id": "call", "name": "fs_write", "args": {"path": str(target), "file_text": "no"}}],
            client.agent.tools, messages,
        )
    assert not target.exists() and "budget" in messages[0].content
    client.agent._confirm_tool.assert_not_called()


def test_approval_does_not_extend_expired_budget_or_run_failure_hooks(setup):
    client, root = setup
    target = root / "not-started"
    events = []
    client.agent._run_hooks = lambda event, *a, **kw: (
        events.append(event) or SimpleNamespace(allowed=False, denied=False, context="", notices=[])
    )
    def approve(*args):
        budget_for(client.agent).deadline = 0
        return True
    client.agent._confirm_tool = approve
    messages = []
    with budget_scope(client.agent, review.LoopBudget(10, 10)):
        client.agent._run_tool_calls(
            [{"id": "write", "name": "fs_write", "args": {"path": str(target), "file_text": "no"}}],
            client.agent.tools, messages,
        )
    assert not target.exists()
    assert events == ["PreToolUse"]
    assert "budget" in messages[0].content


def test_step_budget_is_shared_with_initial_graph_and_keeps_completed_edit(setup):
    client, root = setup
    target = root / "value.py"
    client.agent.recursion_limit = 3
    client._area_model = lambda area: Judge([result("revise")])
    requests = feed_actor(client, [
        AIMessage(content="initial answer"),
        AIMessage(content="", tool_calls=[{"id": "write", "name": "fs_write",
                                         "args": {"path": str(target), "file_text": "VALUE = 2"}}]),
    ])
    client.query("Set VALUE to 2.")
    assert len(requests) == 2
    assert target.read_text() == "VALUE = 2"
    assert client.reviewer.last["verdict"] == "inconclusive"
    assert "step budget" in client.reviewer.last["summary"]
    assert read_session(client.agent.session_log.path)["turns"] == 1


def test_real_user_can_steer_a_correction_without_losing_the_message(setup):
    client, _ = setup
    judge = Judge([result("revise"), result()])
    client._area_model = lambda area: judge
    calls = 0
    def stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert client.agent.accept_mid_turn("Also keep the existing comment.")
        return AIMessage(content=f"answer {calls}"), False
    client.agent._stream_response = stream
    assert client.query("Set VALUE to 2.") == "answer 3"
    assert any(i["kind"] == "user_followup" for i in judge.packets[1]["evidence"])
    assert any(m.content == "Also keep the existing comment." for m in client.agent.messages)


def test_shared_deadline_reaches_the_model_wait_even_when_idle_timeout_is_off(setup):
    client, _ = setup
    agent = client.agent
    agent._stream_idle_timeout = 0
    release = threading.Event()
    class Slow:
        def stream(self, *args, **kwargs):
            release.wait(2)
            yield AIMessage(content="late")
    try:
        with budget_scope(agent, review.LoopBudget(0.03, 10)):
            with pytest.raises(ReviewStopped):
                list(agent._iter_stream_with_idle_timeout(Slow(), [HumanMessage(content="task")], {}))
    finally:
        release.set()


def test_shared_step_budget_cannot_be_replenished(setup):
    client, _ = setup
    budget = review.LoopBudget(10, 1)
    budget.take_step()
    with pytest.raises(ReviewStopped):
        budget.take_step()


def test_detached_launch_is_blocked_only_inside_automatic_corrections(setup):
    client, _ = setup
    client.agent._client_side_tool_message = Mock(return_value=None)
    messages = []
    with budget_scope(client.agent, review.LoopBudget(10, 10)):
        client.agent._run_tool_calls(
            [{"id": "spawn", "name": "spawn_agent", "args": {"run_in_background": True}}],
            [], messages,
        )
    client.agent._client_side_tool_message.assert_not_called()
    assert "detached" in messages[0].content
    client.agent._run_tool_calls(
        [{"id": "normal-spawn", "name": "spawn_agent", "args": {"run_in_background": True}}],
        [], [],
    )
    client.agent._client_side_tool_message.assert_called_once()


def test_feedback_is_not_mistaken_for_a_user_prompt():
    assert user_prompt_text(FEEDBACK_PREFIX + "\ndata") == ""


def test_imported_feedback_summary_cannot_emit_terminal_controls():
    raw = FEEDBACK_PREFIX + DATA_MARKER + json.dumps({"summary": "\x1b[2J" + "x" * 2000})
    shown = feedback_summary(raw)
    assert "\x1b" not in shown and len(shown) == 1000
