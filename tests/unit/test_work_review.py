"""Artifact strategy/checkpoint ordering and fail-open review outage handling."""

import copy
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
from mnemoai.client.client import LangGraphClient
from mnemoai.client.session_log import SessionLog, read_session
from mnemoai.client.ui import review_view
from mnemoai.client.work_review import WorkReview
from mnemoai.utils.config import config
from mnemoai.utils.review_protocol import DATA_MARKER, is_feedback


def verdict(kind="pass"):
    return {"verdict": kind, "summary": "Checkpoint checked.", "findings": [] if kind == "pass" else [
        {"issue": "Preserve the requested value.", "evidence_ids": ["task", "answer"],
         "verification": "Inspect the changed artifact and check its value."},
    ]}


class Model:
    callbacks = None

    def bind_tools(self, tools):
        return self

    def model_copy(self, update=None):
        return copy.copy(self)


class Writer:
    name = "fs_write"

    def invoke(self, args):
        Path(args["path"]).write_text(args["file_text"])
        return '{"success":true}'


class Judge:
    def __init__(self, respond=None):
        self.respond = respond or (lambda checkpoint, packet: verdict())
        self.calls = []

    def invoke(self, messages, config=None):
        packet = json.loads(messages[1].content)
        checkpoint = packet["checkpoint"]
        self.calls.append(checkpoint)
        return AIMessage(content=json.dumps(self.respond(checkpoint, packet)))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path / "home"))
    monkeypatch.setitem(config._config_data, "REVIEW", {})
    monkeypatch.setitem(config._config_data, "LLM", {"MAX_RETRIES": 3, "RETRY_DELAY": 0})
    client = LangGraphClient.__new__(LangGraphClient)
    client.agent = LangGraphAgent(Model(), [Writer()], system_prompt="Follow the user.", verbose=False)
    client.agent.recursion_limit = 30
    client.agent.session_log = SessionLog(cwd=str(tmp_path))
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


def feed(client, actions, *, kind="code", strategies=None):
    actions = iter(actions)
    strategies = iter(strategies) if strategies else None
    seen = {"strategy": [], "work": []}
    def stream(messages, *args, **kwargs):
        if messages[0].content == config.require_prompt("WORK_STRATEGY_PROMPT"):
            seen["strategy"].append(json.loads(messages[1].content))
            item = next(strategies) if strategies is not None else {
                "kind": kind, "strategy": "Inspect existing material, make the scoped change, and verify it." if kind != "none" else "",
            }
            return AIMessage(content=json.dumps(item)), False
        seen["work"].append(list(messages))
        return next(actions), False
    client.agent._stream_response = stream
    return seen


def write(path, value="VALUE = 2"):
    return AIMessage(content="", tool_calls=[{
        "id": str(path), "name": "fs_write", "args": {"path": str(path), "file_text": value},
    }])


@pytest.mark.parametrize("kind", ["code", "document"])
def test_strategy_precedes_edits_and_changes_are_reviewed_before_completion(setup, kind):
    client, root = setup
    target = root / "artifact.txt"
    stages = []
    def inspect(checkpoint, packet):
        stages.append((checkpoint, target.exists()))
        if checkpoint == "changes":
            assert any(item["kind"] == "current_file" and "VALUE = 2" in item["content"]
                       for item in packet["evidence"])
            return verdict("revise")
        if checkpoint == "completion":
            assert any(item["kind"] == "tool" and item["name"] == "fs_write"
                       for item in packet["evidence"]), "checkpoint transitions must retain write evidence"
        return verdict()
    judge = Judge(inspect)
    client._area_model = lambda _: judge
    seen = feed(client, [write(target), AIMessage(content="Verified VALUE = 2.")], kind=kind)
    assert client.query("Create the requested artifact.") == "Verified VALUE = 2."
    assert stages == [("strategy", False), ("changes", True), ("completion", True)]
    feedback = [json.loads(m.content.split(DATA_MARKER)[1]) for m in seen["work"][-1]
                if isinstance(m, HumanMessage) and is_feedback(m.content)]
    assert [f["checkpoint"] for f in feedback] == ["strategy", "changes"]
    assert client.agent._confirm_tool.call_count == 1
    assert read_session(client.agent.session_log.path)["turns"] == 1
    assert [r["checkpoint"] for r in client.reviewer.last["checkpoints"]] == ["strategy", "changes", "completion"]


def test_inline_document_writing_is_reviewed_without_file_tools(setup):
    client, _ = setup
    judge = Judge()
    client._area_model = lambda _: judge
    feed(client, [AIMessage(content="Dear team, here is the revised proposal.")], kind="document")
    assert client.query("Draft a proposal for the team.").startswith("Dear team")
    assert judge.calls == ["strategy", "completion"]
    client.agent._confirm_tool.assert_not_called()


@pytest.mark.parametrize("prompt", ["Hello", "thanks!", "Ciao", "Grazie"])
def test_casual_conversation_skips_both_strategy_and_reviewer(setup, prompt):
    client, _ = setup
    client._area_model = Mock(side_effect=AssertionError("No review for casual chat"))
    seen = feed(client, [AIMessage(content="Hello!")])
    assert client.query(prompt) == "Hello!"
    assert seen["strategy"] == [] and client.reviewer.last is None


def test_general_question_is_not_reviewed_or_given_a_draft_label(setup, capsys):
    client, _ = setup
    client._area_model = Mock(side_effect=AssertionError("No reviewer for general Q&A"))
    feed(client, [AIMessage(content="Paris.")], kind="none")
    assert client.query("What is the capital of France?") == "Paris."
    assert client.reviewer.last is None
    assert "review pending" not in capsys.readouterr().out


def test_related_followup_reaches_scope_selection_with_previous_artifact_task(setup):
    client, _ = setup
    client._area_model = lambda _: Judge()
    feed(client, [AIMessage(content="A draft.")], kind="document")
    client.query("Write a project proposal.")
    seen = feed(client, [AIMessage(content="A shorter draft.")], kind="document")
    client.query("Make it shorter.")
    assert seen["strategy"][0]["previous_artifact_task"] == "Write a project proposal."


def test_strategy_revisions_happen_before_any_write(setup):
    client, root = setup
    target = root / "value.py"
    strategy_calls = []
    def inspect(checkpoint, packet):
        if checkpoint == "strategy":
            assert not target.exists()
            strategy_calls.append(packet)
            return verdict("revise" if len(strategy_calls) == 1 else "pass")
        return verdict()
    client._area_model = lambda _: Judge(inspect)
    seen = feed(client, [write(target), AIMessage(content="Done.")], strategies=[
        {"kind": "code", "strategy": "Inspect and implement."},
        {"kind": "code", "strategy": "Inspect, implement, preserve constraints, run checks."},
    ])
    client.query("Implement the requested value.")
    assert len(seen["strategy"]) == 2 and len(strategy_calls) == 2
    assert "reviewer_feedback" in seen["strategy"][1]
    assert target.exists()


def test_outage_continues_work_warns_once_and_retries_on_later_task(setup, capsys):
    client, root = setup
    def unavailable(*args):
        raise RuntimeError("503 service unavailable")
    judge = Judge(unavailable)
    client._area_model = lambda _: judge
    for i in range(2):
        target = root / f"file{i}.py"
        feed(client, [write(target), AIMessage(content="The file was changed.")])
        assert client.query(f"Create {target.name}") == "The file was changed."
        assert target.exists()
        assert client.reviewer.last["unavailable"] and client.reviewer.last["verdict"] != "pass"
    output = capsys.readouterr().out
    assert output.count("Peer review unavailable; continuing without it.") == 1
    assert "review incomplete" not in output and "unreviewed" not in output
    assert judge.calls == ["strategy"] * 6, "no change/final retries after this task's outage"
    # A valid response re-arms the notice for a genuinely later outage.
    client._area_model = lambda _: Judge()
    feed(client, [AIMessage(content="A document.")], kind="document")
    client.query("Write a report.")
    capsys.readouterr()
    client._area_model = lambda _: Judge(unavailable)
    feed(client, [AIMessage(content="Another document.")], kind="document")
    client.query("Write another report.")
    assert capsys.readouterr().out.count("Peer review unavailable; continuing without it.") == 1


def test_strategy_cannot_execute_returned_tool_calls(setup):
    client, root = setup
    target = root / "never-created"
    actor = feed(client, [AIMessage(content="Normal answer.")])
    original = client.agent._stream_response
    def bad(messages, *args, **kwargs):
        if messages[0].content == config.require_prompt("WORK_STRATEGY_PROMPT"):
            return write(target), False
        return original(messages, *args, **kwargs)
    client.agent._stream_response = bad
    client.query("Implement the change.")
    assert not target.exists() and actor["work"]


def test_step_limit_cannot_present_a_strategy_pass_as_final_verification(setup):
    client, root = setup
    client.agent.recursion_limit = 3
    client._area_model = lambda _: Judge()
    feed(client, [write(root / "partial.py")])
    client.query("Implement the requested file.")
    assert client.reviewer.last["verdict"] == "inconclusive"
    assert client.reviewer.last["checkpoint"] == "completion"


def test_router_receives_the_user_request_not_strategy_feedback(setup):
    client, _ = setup
    classifier = Mock(return_value="simple_qa")
    client.agent.router = SimpleNamespace(classify=classifier)
    client.agent.orchestrator_enabled = False
    client.agent.graph = client.agent._build_graph()
    client._area_model = lambda _: Judge()
    feed(client, [AIMessage(content="Draft report.")], kind="document")
    client.query("Write a report.")
    assert classifier.call_args.args[0] == "Write a report."
    assert not is_feedback(classifier.call_args.args[0])


def test_overhead_budget_does_not_count_normal_implementation_time(setup, monkeypatch):
    client, _ = setup
    work = WorkReview(client, "Write code", [])
    work.remaining = 10
    clock = [100.0]
    monkeypatch.setattr("mnemoai.client.work_review.time", SimpleNamespace(monotonic=lambda: clock[0]))
    with work._charged():
        clock[0] += 2
    clock[0] += 1000  # normal work between checkpoints
    with work._charged():
        clock[0] += 3
    assert work.remaining == 5


def test_change_review_cap_does_not_prevent_subsequent_edits(setup, monkeypatch):
    client, root = setup
    monkeypatch.setitem(config._config_data, "REVIEW", {"MAX_CHANGE_REVIEWS": 1})
    judge = Judge()
    client._area_model = lambda _: judge
    paths = [root / f"file{i}" for i in range(3)]
    feed(client, [*(write(p) for p in paths), AIMessage(content="All files changed.")])
    client.query("Implement these changes.")
    assert all(path.exists() for path in paths)
    assert judge.calls == ["strategy", "changes", "completion"]


def test_cancellation_during_strategy_does_not_leave_a_running_badge(setup):
    client, root = setup
    client.agent._stream_response = Mock(side_effect=KeyboardInterrupt)
    assert client.query("Implement code.") == "Operation was cancelled."
    assert client.reviewer.view is None
    client.agent._confirm_tool.assert_not_called()
    assert read_session(client.agent.session_log.path)["turns"] == 1


def test_orchestrator_reviews_between_dependency_waves_and_preserves_original_query(setup):
    client, root = setup
    paths = [root / "first.py", root / "second.py"]
    judge = Judge(lambda phase, packet: verdict("revise" if phase == "changes" else "pass"))
    client._area_model = lambda _: judge
    feed(client, [])
    client.agent.router = SimpleNamespace(classify=lambda query, context: "full")
    client.agent.orchestrator_enabled = True
    client.agent._max_subagent_concurrency = 1
    task_text = "Implement the two requested changes in the project files."
    def decompose(query, *args, **kwargs):
        assert query == task_text
        return [{"description": "first", "category": "full", "depends_on": []},
                {"description": "second", "category": "full", "depends_on": [0]}]
    client.agent._decompose_task = decompose
    def worker(index, tasks, done, history=None):
        if index == 0:
            assert judge.calls == ["strategy"]
        else:
            assert judge.calls == ["strategy", "changes"]
            assert any(is_feedback(m.content) and '"checkpoint": "changes"' in m.content
                       for m in history if isinstance(m, HumanMessage))
        messages = [write(paths[index])]
        client.agent._run_tool_calls(messages[0].tool_calls, client.agent.tools, messages)
        return {"task": tasks[index]["description"], "result": "Done.", "messages": messages}
    client.agent._run_subtask = worker
    client.agent._aggregate_results = lambda *a: "Both changes are done."
    client.agent.graph = client.agent._build_graph()
    assert client.query(task_text) == "Both changes are done."
    assert judge.calls == ["strategy", "changes", "completion"]
    assert all(path.exists() for path in paths)


def test_changed_snapshot_feedback_is_not_applied_to_newer_file_contents(setup):
    client, root = setup
    target = root / "code.py"
    def changing(phase, packet):
        if phase == "changes":
            target.write_text("NEWER CONTENT")
            return verdict("revise")
        return verdict()
    client._area_model = lambda _: Judge(changing)
    seen = feed(client, [write(target), AIMessage(content="Done.")])
    client.query("Implement the code.")
    feedback = [m.content for m in seen["work"][-1] if isinstance(m, HumanMessage) and is_feedback(m.content)]
    assert not any('"checkpoint": "changes"' in item for item in feedback)
    assert client.reviewer.last["checkpoints"][1]["stale"]


def test_guidance_received_during_change_review_precedes_next_actor_call(setup):
    client, root = setup
    def review_and_steer(phase, packet):
        if phase == "changes":
            assert client.agent.accept_mid_turn("Also preserve the existing comment.")
        return verdict()
    client._area_model = lambda _: Judge(review_and_steer)
    seen = feed(client, [write(root / "code.py"), AIMessage(content="Done, preserving the comment.")])
    client.query("Implement this code.")
    assert any("Also preserve the existing comment." in str(message.content)
               for message in seen["work"][-1] if isinstance(message, HumanMessage))
    saved = read_session(client.agent.session_log.path)
    assert "Also preserve the existing comment." in str(saved["messages"])
    assert saved["turns"] == 1


def test_unavailable_final_answer_has_no_repeated_warning_or_false_pass():
    rendered = review_view.final_answer("Actor's answer", {"verdict": "inconclusive", "unavailable": True})
    assert "Actor's answer" in rendered
    assert "unreviewed" not in rendered and "review incomplete" not in rendered
    assert "review passed" not in rendered.lower()
