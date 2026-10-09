"""Optional memory maintenance must not prevent or replace a chat answer."""

import threading
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from mnemoai.client import client as client_mod
from mnemoai.client import review
from mnemoai.client.client import LangGraphClient
from mnemoai.client.managers.agent_conversation_manager import CompactionError


def _client():
    client = LangGraphClient.__new__(LangGraphClient)
    client.spinner = Mock()
    client.spinner_lock = threading.Lock()
    client.callback_handler = Mock()
    client.mcp_client = nullcontext()
    client.agent = Mock(return_value="the answer")
    client.agent.messages = []
    client.episodic_memory = None
    client.plan_mode_active = False
    client._steering_reminder = lambda: ""
    client._summary_model = lambda: object()
    client._profile_turn = lambda: None
    client._print_context_size = lambda: None

    async def manage(*args):
        return None

    client.conversation_manager = SimpleNamespace(manage_messages=manage)
    return client


def test_recall_outage_does_not_prevent_the_chat_call():
    client = _client()
    client.episodic_memory = object()
    client._inject_episodic_context = Mock(side_effect=RuntimeError("embedding outage"))
    assert client.query("question") == "the answer"
    client.agent.assert_called_once_with("question")


def test_first_turn_activates_external_tools_before_calling_the_agent(monkeypatch):
    client = _client()
    # This fixture's stand-in represents the real agent lifecycle in query().
    monkeypatch.setattr(client_mod, "LangGraphAgent", type(client.agent))
    steps = []
    client.refresh_tools = lambda **kwargs: steps.append(("tools", kwargs))
    client.agent.side_effect = lambda prompt: steps.append(("model", prompt)) or "answer"
    assert client.query("question") == "answer"
    assert steps == [("tools", {"wait": True}), ("model", "question")]
    client.agent._cancel_event.clear.assert_called_once()


def test_cancel_during_discovery_does_not_call_model_or_change_history(monkeypatch):
    client = _client()
    monkeypatch.setattr(client_mod, "LangGraphAgent", type(client.agent))
    client.refresh_tools = Mock(side_effect=KeyboardInterrupt)
    client.query("question")
    client.agent.assert_not_called()
    assert client.agent.messages == []


def test_incomplete_compaction_preserves_the_answer_already_produced():
    client = _client()

    async def fail(*args):
        raise CompactionError("summary batch unavailable")

    client.conversation_manager.manage_messages = fail
    assert client.query("question") == "the answer"
    client.spinner.stop.assert_called()


def test_connection_failure_cannot_reflect_on_the_previous_turn():
    client = _client()
    client.reflector = object()
    client.agent.messages = [SimpleNamespace(type="human", content="previous task")]

    @contextmanager
    def unavailable():
        raise OSError("MCP connection unavailable")
        yield

    client.mcp_client = unavailable()
    client.query("new task")
    client.agent.assert_not_called()
    assert client._reflection_messages == []


def test_agent_failure_before_new_history_does_not_reuse_old_evidence():
    client = _client()
    client.reflector = object()
    client.agent.messages = [SimpleNamespace(type="human", content="previous task")]
    client.agent.side_effect = RuntimeError("agent not ready")
    client.query("new task")
    assert client._reflection_messages == []


def test_reflection_evidence_is_captured_before_post_answer_compaction():
    client = _client()
    client.reflector = object()
    current = [SimpleNamespace(type="human", content="new task")]

    def answer(prompt):
        client.agent.messages.extend(current)
        return "answer"

    async def compact(*args):
        client.agent.messages = []

    client.agent.side_effect = answer
    client.conversation_manager.manage_messages = compact
    client.query("new task")
    assert client._reflection_messages == current


def test_playbook_lock_error_does_not_block_an_ordinary_query(tmp_path, monkeypatch):
    from mnemoai.client.memory.playbook_store import PlaybookStore

    client = _client()
    client.playbook = PlaybookStore(str(tmp_path))
    client.system_prompt = client.agent.system_prompt = "base"
    client.session_id = "test"
    client.agent.session_log = None
    def denied(*args):
        raise PermissionError("lock inaccessible")
    monkeypatch.setattr("mnemoai.client.memory.playbook_store.file_lock", denied)
    assert client.query("question") == "the answer"
    client.agent.assert_called_once_with("question")
    assert client.playbook.error


def test_playbook_directory_error_does_not_escape_optional_initialization():
    client = LangGraphClient.__new__(LangGraphClient)
    client._model_scoped_dir = Mock(side_effect=PermissionError("directory unavailable"))
    client._initialize_playbook()
    assert client.playbook is None and client.reflector is None
    assert client._playbook_error == "directory unavailable"


@pytest.mark.parametrize("actor_path", ["direct", "orchestrated", "single_worker"])
def test_every_parent_result_reaches_the_same_review_checkpoint(actor_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = _client()
    client.reviewer = review.Reviewer(enabled=True)
    client.llm_controller = SimpleNamespace(model_name="fixture", model_type="ollama")
    client.agent._cancel_event = threading.Event()
    client.agent.session_log = None
    client.agent.usage = None
    reviewer_model = Mock()
    reviewer_model.invoke.return_value = AIMessage(content='{"verdict":"pass","summary":"Checked","findings":[]}')
    client._area_model = Mock(return_value=reviewer_model)
    client.auto_approve_mode = "off"
    client.agent._trusted_confirm_categories = set()

    def answer(prompt):
        # Parent execution returns the same contract for each graph route.
        assert client.agent._review_capture is not None
        client.agent._review_capture.record(
            "execute_bash", {"command": "pytest"}, {"exit_code": 0, "stdout": actor_path}, "completed",
        )
        client.agent.messages.append(SimpleNamespace(type="ai", content=actor_path))
        return actor_path

    client.agent.side_effect = answer
    assert client.query("Run the tests") == actor_path
    reviewer_model.invoke.assert_called_once()
    assert client.reviewer.last["verdict"] == "pass"
    assert actor_path in reviewer_model.invoke.call_args.args[0][1].content
    assert client.auto_approve_mode == "off" and client.agent._trusted_confirm_categories == set()
    assert client.agent._review_capture is None


def test_review_off_leaves_execution_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = _client()
    client.reviewer = review.Reviewer(enabled=False)
    client._area_model = Mock(side_effect=AssertionError("review must remain inert"))
    monkeypatch.setattr(review, "Capture", Mock(side_effect=AssertionError("must not inspect files")))
    assert client.query("question") == "the answer"
    client._area_model.assert_not_called()


def test_unavailable_reviewer_preserves_answer_and_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = _client()
    client.reviewer = review.Reviewer(enabled=True)
    client.llm_controller = SimpleNamespace(model_name="fixture", model_type="ollama")
    client.agent._cancel_event = threading.Event()
    client.agent.session_log = None
    client.agent.usage = None
    client._area_model = Mock(return_value=None)
    assert client.query("question") == "the answer"
    assert client.reviewer.last["verdict"] == "inconclusive"


def test_actor_failure_does_not_review_an_older_answer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = _client()
    client.reviewer = review.Reviewer(enabled=True)
    client.agent.side_effect = RuntimeError("actor failed")
    client._area_model = Mock(side_effect=AssertionError("must not invoke reviewer"))
    assert "Something went wrong" in client.query("question")
    assert client.reviewer.last["verdict"] == "inconclusive"
    client._area_model.assert_not_called()


def test_replaced_conversation_invalidates_the_last_review():
    client = _client()
    client.reviewer = review.Reviewer(enabled=True)
    client.reviewer.incomplete("old task")
    client._forget_context_size()
    assert client.reviewer.enabled and client.reviewer.last is None
