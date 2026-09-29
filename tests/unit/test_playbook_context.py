"""Retractions reach the next model send without rewriting historical messages."""

import json
import threading

import pytest
from langchain_core.messages import (
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from mnemoai.client.agent.agent import LangGraphAgent, _StreamIdleTimeout
from mnemoai.client.memory.playbook_context import refresh_messages
from mnemoai.client.memory.playbook_records import PlaybookEntry
from mnemoai.client.memory.playbook_store import PlaybookStore


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = PlaybookStore(str(tmp_path / "playbook"))
    store.append(PlaybookEntry(
        "editing", "withdraw-this-fixture-note", "fixture", scope=str(tmp_path),
        source_refs=[{"session_id": "s", "turn": 1, "tool_call_id": "call"}],
    ))
    original, _ = store.prepare_prompt()
    entry = store.snapshot()[0]
    plan = store.preview_retraction(entry["id"], "retract", "Incorrect")
    store.apply_retraction(entry["id"], action="retract", reason="Incorrect", token=plan["token"])
    return store, original


@pytest.mark.parametrize("blocks", [False, True])
def test_fresh_read_filters_stale_system_messages_and_preserves_cache_metadata(setup, blocks):
    store, original = setup
    historical = f"<conversation_summary>\nAn old quotation:\n{original}\n</conversation_summary>"
    text = f"instructions\n\n{original}\n\n{historical}"
    content = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}] if blocks else text
    message = SystemMessage(content=content)
    tool = ToolMessage(content=original, tool_call_id="historical")
    refreshed = refresh_messages([message, tool], lambda: store.prepare_prompt()[0])
    value = refreshed[0].content[0]["text"] if blocks else refreshed[0].content
    assert value == "instructions\n\n" + historical
    assert message.content == content
    assert refreshed[1] is tool  # no claim to erase old tool results or transcripts
    if blocks:
        assert refreshed[0].content[0]["cache_control"] == {"type": "ephemeral"}


def test_custom_worker_prompts_without_a_playbook_block_are_untouched():
    original = [SystemMessage(content="custom agent scope"), HumanMessage(content="task")]
    def forbidden():
        pytest.fail("Do not inject the parent's memory into a custom prompt")
    assert refresh_messages(original, forbidden) is original


@pytest.mark.parametrize("blocks", [False, True])
def test_withdrawing_the_only_system_block_omits_the_empty_message(setup, blocks):
    store, original = setup
    content = [{"type": "text", "text": original}] if blocks else original
    system = SystemMessage(content=content)
    human = HumanMessage(content="task")
    assert refresh_messages([system, human], lambda: store.prepare_prompt()[0]) == [human]
    assert system.content == content


def test_removing_an_empty_generated_text_block_preserves_other_blocks(setup):
    store, original = setup
    untouched = {"type": "text", "text": "base", "cache_control": {"type": "ephemeral"}}
    content = [{"type": "text", "text": original}, untouched]
    system = SystemMessage(content=content)
    refreshed = refresh_messages([system], lambda: store.prepare_prompt()[0])
    assert refreshed[0].content == [untouched]
    assert system.content == content


@pytest.mark.parametrize("idle", [0, 1])
def test_actual_stream_boundary_checks_fresh_state_even_for_quiet_workers(setup, idle):
    store, original = setup
    agent = LangGraphAgent.__new__(LangGraphAgent)
    agent._stream_idle_timeout = idle
    agent._cancel_event = threading.Event()
    agent._playbook_context_provider = lambda: store.prepare_prompt()[0]
    observed = []
    class Model:
        def stream(self, messages, config=None):
            observed.append(messages)
            yield AIMessageChunk(content="ok")
    messages = [SystemMessage(content="base\n\n" + original), HumanMessage(content="task")]
    reply, _ = agent._stream_once(Model(), messages, {}, quiet=True)
    assert reply.content == "ok"
    assert "withdraw-this-fixture-note" not in json.dumps([m.content for m in observed[0]])
    assert "withdraw-this-fixture-note" in messages[0].content


def test_unavailable_lifecycle_state_removes_stale_injection(setup):
    _, original = setup
    def unavailable():
        raise OSError("store inaccessible")
    messages = [SystemMessage(content="base\n\n" + original)]
    assert refresh_messages(messages, unavailable)[0].content == "base"


@pytest.mark.parametrize("idle", [0, 1])
def test_cancellation_during_refresh_never_starts_the_model(setup, idle):
    _, original = setup
    agent = LangGraphAgent.__new__(LangGraphAgent)
    agent._stream_idle_timeout = idle
    agent._cancel_event = threading.Event()
    def refresh():
        agent._cancel_event.set()
        return ""
    agent._playbook_context_provider = refresh
    class Model:
        def stream(self, *args, **kwargs):
            pytest.fail("Cancelled during the pre-send check")
            yield
    with pytest.raises(KeyboardInterrupt):
        list(agent._iter_stream_with_idle_timeout(
            Model(), [SystemMessage(content="base\n\n" + original)], {},
        ))


def test_refresh_finishing_after_timeout_never_starts_an_abandoned_request(setup, monkeypatch):
    _, original = setup
    agent = LangGraphAgent.__new__(LangGraphAgent)
    agent._stream_idle_timeout = 0.02
    agent._cancel_event = threading.Event()
    entered, release = threading.Event(), threading.Event()
    readers = []
    real_thread = threading.Thread

    def thread(*args, **kwargs):
        result = real_thread(*args, **kwargs)
        readers.append(result)
        return result

    monkeypatch.setattr("mnemoai.client.agent.agent.threading.Thread", thread)

    def refresh():
        entered.set()
        assert release.wait(2)
        return ""

    agent._playbook_context_provider = refresh
    called = threading.Event()

    class Model:
        def stream(self, *args, **kwargs):
            called.set()
            yield AIMessageChunk(content="must not happen")

    try:
        with pytest.raises(_StreamIdleTimeout):
            list(agent._iter_stream_with_idle_timeout(
                Model(), [SystemMessage(content="base\n\n" + original)], {},
            ))
        assert entered.wait(1)
    finally:
        release.set()
        for reader in readers:
            reader.join(timeout=2)
            assert not reader.is_alive()
    assert not called.is_set()
