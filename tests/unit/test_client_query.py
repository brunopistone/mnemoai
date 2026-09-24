"""Optional memory maintenance must not prevent or replace a chat answer."""

import threading
from contextlib import nullcontext
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
