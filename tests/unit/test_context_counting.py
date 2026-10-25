"""The complete fallback estimate and cache invalidation across history restores."""

import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mnemoai.client import context_injection, context_report
from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.client import LangGraphClient
from mnemoai.client.managers.agent_conversation_manager import AgentConversationManager
from mnemoai.client.memory.steering_store import SteeringSnapshot
from mnemoai.client.session_log import SessionLog, read_session
from mnemoai.client.ui import chat_interface
from mnemoai.client.usage_tracker import UsageTracker
from mnemoai.utils.config import config


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path / "home"))
    monkeypatch.setitem(config._config_data, "MODEL_ID", {"TYPE": "bedrock"})
    monkeypatch.setitem(config._config_data, "LLM", {})
    monkeypatch.setattr(context_injection, "steering_snapshot", lambda client: SteeringSnapshot())

    def create(messages=()):
        client = LangGraphClient.__new__(LangGraphClient)
        client.system_prompt = "A short system prompt."
        client.agent = SimpleNamespace(
            messages=list(messages), system_prompt=client.system_prompt,
            _last_input_tokens=None, usage=UsageTracker(), session_log=None,
            _activity=SimpleNamespace(),
        )
        client.tools = []
        client.plan_mode_active = False
        client.conversation_manager = AgentConversationManager(max_tokens=1000)
        client.current_conversation_path = None
        client.callback_handler = SimpleNamespace()
        client._approve_plan = lambda *a, **kw: None
        client.spinner = Mock()
        client._summary_model = lambda: object()
        return client
    return create


@pytest.mark.parametrize("component", ["tools", "calls", "reasoning", "system", "steering", "plan"])
def test_fallback_counts_every_input_component(make_client, monkeypatch, component):
    client = make_client([HumanMessage(content="Hi"), AIMessage(content="OK")])
    before = client._count_context_tokens()
    large = "additional input " * 1000
    if component == "tools":
        client.tools = [SimpleNamespace(name="tool", description=large, args_schema={"type": "object"})]
    elif component == "calls":
        client.agent.messages[1] = AIMessage(content="OK", tool_calls=[
            {"id": "call", "name": "tool", "args": {"text": large}},
        ])
    elif component == "reasoning":
        client.agent.messages[1].additional_kwargs["reasoning_content"] = large
    elif component == "system":
        client.agent.system_prompt += large
    elif component == "steering":
        monkeypatch.setattr(context_injection, "steering_snapshot", lambda client: SteeringSnapshot(large))
    else:
        client.plan_mode_active = True
    after = client._count_context_tokens()
    assert after > before
    assert after == client._estimate_context_tokens() == sum(p.tokens for p in context_report.collect(client))


def test_native_and_normalized_tool_call_is_not_counted_twice():
    args = {"text": "payload " * 100}
    call = {"id": "call", "name": "write", "args": args}
    normal = AIMessage(content="", tool_calls=[call])
    native = AIMessage(content=[{"type": "tool_use", "id": "call", "name": "write", "input": args}],
                       tool_calls=[call])
    assert context_report._message_text(native) == context_report._message_text(normal)


@pytest.mark.parametrize("replacement", ["load", "resume"])
def test_pinned_footer_recounts_same_length_replaced_history(make_client, tmp_path, monkeypatch, replacement):
    client = make_client([HumanMessage(content="Hi"), AIMessage(content="OK")])
    client.clear_context = lambda: None  # only bypass cleanup when the fake UI exits
    large = [HumanMessage(content="large document " * 10000), AIMessage(content="OK")]
    if replacement == "load":
        other = make_client(large)
        path = tmp_path / "saved.json"
        other.save_conversation(path=str(path))
    else:
        log = SessionLog(cwd=str(tmp_path))
        log.log_turn(large)
        path = log.path

    class Reader:
        def __init__(self, **kwargs):
            self.count = inspect.getclosurevars(kwargs["footer_text"]).nonlocals["_context_tokens"]

        def __getattr__(self, name):
            return lambda *a, **kw: None

        def run(self):
            before, estimated = self.count()
            assert estimated
            client.agent._last_input_tokens = 123
            assert self.count() == (123, False)
            restore = client.load_conversation if replacement == "load" else client.resume_session
            assert restore(str(path))
            after, estimated = self.count()
            assert client.agent._last_input_tokens is None and estimated
            assert after == client._count_context_tokens() and after > before * 100
            assert client.agent.usage.totals()["calls"] == 0

    interface = chat_interface.ChatInterface.__new__(chat_interface.ChatInterface)
    interface.client = client
    interface._completion_commands = []
    interface.command_history = None
    monkeypatch.setattr(chat_interface, "PinnedPromptReader", Reader)
    interface._run_pinned_loop()


def test_pinned_footer_retains_measured_count_while_requests_run(make_client, monkeypatch):
    client = make_client([HumanMessage(content="Earlier work")])
    agent = LangGraphAgent.__new__(LangGraphAgent)
    agent.__dict__.update(vars(client.agent))
    agent._messages = client.agent.messages
    client.agent = agent
    client.clear_context = lambda: None
    client.conversation_manager.max_tokens = 1_000_000
    agent.callbacks = []
    agent._get_route_model = lambda state: object()
    agent._start_spinner = agent._stop_spinner = lambda *a: None
    agent._last_input_tokens = 505_000
    client._count_context_tokens = Mock(return_value=715_900)

    class Reader:
        def __init__(self, **kwargs):
            self.footer = kwargs["footer_text"]

        def __getattr__(self, name):
            return lambda *a, **kw: None

        def text(self):
            return "".join(text for _, text in self.footer(120))

        def run(self):
            for previous, measured in ((505_000, 508_000), (508_000, 510_300)):
                pending = []

                def stream(*args, **kwargs):
                    # Repaints before the first token and during streaming must
                    # keep the previous measurement, not jump to the safety estimate.
                    for _ in range(3):
                        pending.append(self.text())
                    return AIMessage(content="Done", usage_metadata={
                        "input_tokens": measured, "output_tokens": 10,
                        "total_tokens": measured + 10,
                    }), False

                agent._stream_response = stream
                agent._call_model({"messages": [
                    SystemMessage(content="test"), HumanMessage(content="Continue"),
                ]})
                for text in pending:
                    assert f"{previous / 1000:.1f}k" in text
                    assert f"~{previous / 1000:.1f}k" not in text
                    assert "72%" not in text
                assert f"{measured / 1000:.1f}k" in self.text()
            client._count_context_tokens.assert_not_called()
            assert agent.usage.totals()["calls"] == 2
            # Replacing the history must still discard the old measurement.
            client._forget_context_size()
            assert "~715.9k" in self.text()
            assert "72%" in self.text()

    interface = chat_interface.ChatInterface.__new__(chat_interface.ChatInterface)
    interface.client = client
    interface._completion_commands = []
    interface.command_history = None
    monkeypatch.setattr(chat_interface, "PinnedPromptReader", Reader)
    interface._run_pinned_loop()


def test_cache_key_detects_in_place_payload_changes_without_tokenizing(make_client, monkeypatch):
    client = make_client([AIMessage(content="old", tool_calls=[
        {"id": "call", "name": "tool", "args": {"value": "old"}},
    ])])
    monkeypatch.setattr(context_report, "count_tokens", Mock(side_effect=AssertionError("paint must not tokenize")))
    key = context_report.estimate_cache_key(client)
    assert context_report.estimate_cache_key(client) == key
    client.agent.messages[0].tool_calls[0]["args"]["value"] = "new"
    assert context_report.estimate_cache_key(client) != key
    key = context_report.estimate_cache_key(client)
    client.agent.system_prompt = "X" * len(client.agent.system_prompt)
    assert context_report.estimate_cache_key(client) != key


@pytest.mark.parametrize("component", ["system", "tools", "reasoning"])
def test_restore_preflight_includes_non_text_history_input(make_client, component):
    client = make_client([HumanMessage(content="Hi"), AIMessage(content="OK")])
    large = "size check " * 1000
    if component == "system":
        client.agent.system_prompt = large
    elif component == "tools":
        client.tools = [SimpleNamespace(name="tool", description=large, args_schema=None)]
    else:
        client.agent.messages[1].additional_kwargs["reasoning_content"] = large
    client.conversation_manager._compact = AsyncMock(return_value=True)
    assert client._compact_now()
    client.conversation_manager._compact.assert_awaited_once()


def test_eviction_cannot_hide_remaining_tool_schema_overhead(make_client, monkeypatch):
    client = make_client([ToolMessage(content="x" * 10000, tool_call_id="id")])
    client.tools = [SimpleNamespace(name="tool", description="schema " * 2000, args_schema=None)]
    monkeypatch.setitem(config._config_data, "LLM", {"TOOL_EVICTION_KEEP_RECENT": 0})
    client.conversation_manager._compact = AsyncMock(return_value=True)
    assert client._compact_now()
    assert len(client.agent.messages[0].content) < 10000
    client.conversation_manager._compact.assert_awaited_once()


def test_compacted_resume_and_save_load_counts_remain_stable(make_client, tmp_path):
    source = SessionLog(cwd=str(tmp_path))
    source.log_turn([HumanMessage(content="earlier " * 1000), AIMessage(content="earlier answer")])
    kept = [HumanMessage(content="latest"), AIMessage(content="latest answer")]
    source.log_turn(kept)
    source.log_compaction(summary="Earlier facts.", kept=kept)
    client = make_client()
    client.conversation_manager._session_blocks = lambda *a: []
    counts = []
    for i in range(10):
        destination = SessionLog(cwd=str(tmp_path))
        client.agent.session_log = destination
        assert client.resume_session(str(source.path))
        counts.append(client._count_context_tokens())
        saved = read_session(destination.path)
        assert len(saved["all_messages"]) == 4 and len(saved["messages"]) == 2
        assert client.conversation_manager.summary_text == "Earlier facts."
        source = destination
    assert len(set(counts)) == 1
    for i in range(10):
        path = tmp_path / f"save-{i}.json"
        client.save_conversation(path=str(path))
        assert client.load_conversation(str(path))
        assert client._count_context_tokens() == counts[-1]
        assert json.loads(path.read_text())["summary"] == "Earlier facts."
