"""Clearing a conversation must separate its persisted history, not just its UI."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from mnemoai.client import session_log
from mnemoai.client.client import LangGraphClient
from mnemoai.client.managers.agent_conversation_manager import AgentConversationManager
from mnemoai.client.ui.chat_interface import ChatInterface
from mnemoai.main import _format_session_label
from mnemoai.utils import paths


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path / "app"))
    monkeypatch.setattr(paths, "_profile_name", lambda: "tester")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "mnemoai.client.client.config.get",
        lambda key, default=None: {
            "ENABLE_RAG": False,
            "MODEL_ID": {"NAME": "test-model"},
        }.get(key, default),
    )

    def create(record=True):
        client = LangGraphClient.__new__(LangGraphClient)
        messages = []
        client.agent = SimpleNamespace(
            messages=messages,
            clear_messages=Mock(side_effect=messages.clear),
            retire_background_work=Mock(),
            session_log=None,
            provenance=None,
            usage=Mock(),
            files=Mock(),
        )
        client.session_id = "original-session"
        client.system_prompt = "system prompt"
        client.current_conversation_path = "/old-saved-chat.json"
        client.conversation_manager = AgentConversationManager(max_tokens=1000)
        client._build_system_prompt = lambda: "fresh system prompt"
        client._system_prompt_with_playbook = lambda: "fresh system prompt"
        client._new_session_id = lambda: "new-session"
        client._flush_chunk_cache_store = Mock()
        client._flush_rag_store = Mock()
        client._initialize_chunk_cache = Mock()
        client._print_context_size = lambda: None
        client.refresh_steering = lambda **kw: None
        client._session_max_age_days = lambda: 30 if record else 0
        client._attach_session_log()
        return client

    return create


def _turn(client, prompt):
    messages = [HumanMessage(content=prompt), AIMessage(content="answer")]
    client.agent.messages.extend(messages)
    if client.agent.session_log is not None:
        client.agent.session_log.log_turn(messages)
    return messages


def test_clear_command_new_topic_resume_and_picker_are_independent(make_client):
    client = make_client()
    _turn(client, "What can I do in Las Vegas?")
    original = client.agent.session_log.path
    original_bytes = original.read_bytes()

    interface = ChatInterface.__new__(ChatInterface)
    interface.client = client
    interface._ChatInterface__clear_screen = lambda: None
    interface._ChatInterface__welcome_message = lambda: None
    interface._dispatch("/clear")

    current = client.agent.session_log.path
    assert current != original
    assert client.agent.messages == []
    assert client.current_conversation_path is None
    _turn(client, "Can you research Google Gemma 4?")
    assert original.read_bytes() == original_bytes

    rows = session_log.list_sessions()
    assert {row["preview"] for row in rows} == {
        "What can I do in Las Vegas?", "Can you research Google Gemma 4?",
    }
    row = next(row for row in rows if row["path"] == str(current))
    assert row["exchanges"] == row["turns"] == 1
    assert "Gemma" in _format_session_label(row)
    assert "Las Vegas" not in _format_session_label(row)
    assert not row["resumed_from"]
    assert session_log.last_live_turn(current)["n"] == 1

    resumed = make_client()
    assert resumed.resume_session(str(current))
    assert [m.content for m in resumed.agent.messages] == [
        "Can you research Google Gemma 4?", "answer",
    ]
    _turn(resumed, "How can I integrate it with Gmail?")
    again = make_client()
    assert again.resume_session(str(resumed.agent.session_log.path))
    assert "Las Vegas" not in str(again.agent.messages)
    assert "Gmail" in str(again.agent.messages)


def test_clear_preserves_named_and_compacted_old_conversation(make_client):
    client = make_client()
    old_messages = _turn(client, "Las Vegas")
    previous = client.agent.session_log
    previous.set_label("Travel plans")
    previous.log_compaction(summary="Earlier travel plans", kept=old_messages[-1:])
    client.conversation_manager.summary_text = "Earlier travel plans"
    client.conversation_manager.previous_summary = "Earlier travel plans"
    before = previous.path.read_bytes()

    client.clear_context()
    _turn(client, "Google Gemma")
    data = session_log.read_session(client.agent.session_log.path)
    assert data["label"] == data["summary"] == data["resumed_from"] == ""
    assert data["checkpoint"] is False
    assert client.conversation_manager.summary_text == ""
    assert client.conversation_manager.previous_summary is None
    assert previous.path.read_bytes() == before
    assert session_log.read_session(previous.path)["label"] == "Travel plans"


def test_clearing_a_resumed_conversation_does_not_hide_its_parent(make_client):
    source = make_client()
    _turn(source, "Las Vegas")
    original = source.agent.session_log.path
    resumed = make_client()
    assert resumed.resume_session(str(original))
    duplicate = resumed.agent.session_log.path

    resumed.clear_context()
    assert not duplicate.exists()  # unused restore adds nothing to its parent
    _turn(resumed, "Gemma")
    assert {row["preview"] for row in session_log.list_sessions()} == {"Las Vegas", "Gemma"}
    assert original.exists()


def test_clear_discards_empty_logs_and_exit_does_not_rotate(make_client):
    client = make_client()
    first = client.agent.session_log.path
    client.clear_context()
    second = client.agent.session_log.path
    assert first != second and not first.exists()
    client.clear_context()
    third = client.agent.session_log
    assert not second.exists()
    client.clear_context(start_new_session=False)
    assert client.agent.session_log is third
    assert list(third.path.parent.glob("session_*.jsonl")) == [third.path]
    assert third.discard_if_empty()
    assert session_log.list_sessions() == []


def test_exit_keeps_the_completed_transcript(make_client):
    client = make_client()
    _turn(client, "Gemma")
    log = client.agent.session_log
    before = log.path.read_bytes()
    client.clear_context(start_new_session=False)
    assert client.agent.session_log is log
    assert log.path.read_bytes() == before
    assert not log.discard_if_empty()


@pytest.mark.parametrize("ending", ["/quit", "/exit", "eof"])
def test_plain_loop_shutdown_does_not_start_another_transcript(make_client, monkeypatch, ending):
    client = make_client()
    _turn(client, "Gemma")
    log = client.agent.session_log
    interface = ChatInterface.__new__(ChatInterface)
    interface.client = client

    def read(prompt=""):
        if ending == "eof":
            raise EOFError
        return ending

    monkeypatch.setattr("builtins.input", read)
    interface._plain_loop()
    assert client.agent.session_log is log
    assert list(log.path.parent.glob("session_*.jsonl")) == [log.path]


def test_clear_keeps_persistence_disabled(make_client):
    client = make_client(record=False)
    client.clear_context()
    _turn(client, "Gemma")
    assert client.agent.session_log is client.agent.provenance is None
    assert not list(paths.app_home().rglob("session_*.jsonl"))


def test_new_transcript_failure_never_reuses_old_log(make_client, monkeypatch):
    client = make_client()
    _turn(client, "Las Vegas")
    old = client.agent.session_log.path
    before = old.read_bytes()

    def failed(**kwargs):
        raise OSError("test: transcript unavailable")

    monkeypatch.setattr("mnemoai.client.client.SessionLog", failed)
    client.clear_context()
    _turn(client, "Gemma")
    assert client.agent.session_log is None
    assert old.read_bytes() == before


def test_clear_preserves_the_launch_directory(make_client, tmp_path, monkeypatch):
    client = make_client()
    _turn(client, "Las Vegas")
    old = client.agent.session_log.path
    other = tmp_path / "other-project"
    other.mkdir()
    monkeypatch.chdir(other)
    client.clear_context()
    _turn(client, "Gemma")
    assert client.agent.session_log.path.parent == old.parent
    assert session_log.list_sessions(cwd=str(other)) == []
