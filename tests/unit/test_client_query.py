"""Optional memory maintenance must not prevent or replace a chat answer."""

import threading
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

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
