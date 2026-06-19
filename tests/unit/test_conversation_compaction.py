"""Unit tests for conversation compaction (client/managers/agent_conversation_manager.py).

Covers the pure-logic pieces that don't need a live model:
- message -> dict conversion preserving tool calls / tool results
- rendering messages (incl. tools) to summary text
- the keep-recent-verbatim compaction behavior (with a fake async model)
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from personal_ai_assistant.client.managers.agent_conversation_manager import (
    AgentConversationManager,
    messages_to_dict_list,
)


class TestMessagesToDictList:
    def test_human_and_ai_roles(self):
        out = messages_to_dict_list([HumanMessage("hi"), AIMessage("hello")])
        assert out[0]["role"] == "user"
        assert out[1]["role"] == "assistant"

    def test_system_role_preserved(self):
        out = messages_to_dict_list([SystemMessage("sys")])
        assert out[0]["role"] == "system"

    def test_tool_message_is_tool_role_not_user(self):
        out = messages_to_dict_list(
            [ToolMessage(content="result", tool_call_id="t1", name="glob_search")]
        )
        assert out[0]["role"] == "tool"
        assert out[0]["tool_name"] == "glob_search"

    def test_ai_tool_calls_preserved(self):
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "execute_bash", "args": {"command": "ls"}, "id": "x"}],
        )
        out = messages_to_dict_list([ai])
        assert out[0]["tool_calls"][0]["name"] == "execute_bash"
        assert out[0]["tool_calls"][0]["args"] == {"command": "ls"}


class TestMessageTextForSummary:
    def test_tool_result_rendered_with_name(self):
        msg = {"role": "tool", "tool_name": "glob_search", "content": [{"text": "3 files"}]}
        text = AgentConversationManager._message_text_for_summary(msg)
        assert "glob_search" in text and "3 files" in text

    def test_assistant_tool_calls_rendered(self):
        msg = {
            "role": "assistant",
            "content": [{"text": "running it"}],
            "tool_calls": [{"name": "execute_bash", "args": {"command": "ls"}}],
        }
        text = AgentConversationManager._message_text_for_summary(msg)
        assert "execute_bash" in text
        assert "running it" in text

    def test_plain_user_text(self):
        msg = {"role": "user", "content": [{"text": "what is X"}]}
        assert AgentConversationManager._message_text_for_summary(msg) == "what is X"


class _FakeAsyncModel:
    """Stands in for a LangChain model: ainvoke returns a fixed summary."""

    def __init__(self):
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="SUMMARY OF OLDER MESSAGES")


class _FakeAgent:
    def __init__(self, messages):
        self.messages = messages
        self.system_prompt = ""


class _FakeSpinner:
    def start(self):
        pass

    def stop(self):
        pass


class _FakeClient:
    def __init__(self):
        self.spinner = _FakeSpinner()
        self.system_prompt = ""


def _run(coro):
    return asyncio.run(coro)


def _llm_config(**overrides):
    """Build a fake config.get that returns an LLM dict with a generous token
    budget by default (so message-count is the binding limit unless overridden).
    """
    llm = {"KEEP_RECENT_TOKEN_BUDGET": 10_000_000}
    llm.update(overrides)

    def _get(key, default=None):
        if key == "LLM":
            return llm
        if key == "SYSTEM_PROMPT":
            return None
        return default

    return _get


class TestCompactKeepsRecent:
    def test_manual_compact_keeps_recent_window_verbatim(self, monkeypatch):
        import personal_ai_assistant.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(
            mod.config, "get", _llm_config(MANUAL_COMPACT_KEEP_RECENT=3)
        )
        msgs = [HumanMessage(f"m{i}") if i % 2 == 0 else AIMessage(f"a{i}") for i in range(10)]
        agent = _FakeAgent(list(msgs))
        mgr = AgentConversationManager(max_tokens=100)

        did = _run(mgr.compact(_FakeClient(), _FakeAsyncModel(), agent))
        assert did is True
        assert len(agent.messages) == 3
        assert agent.messages == msgs[-3:]
        assert "SUMMARY OF OLDER MESSAGES" in agent.system_prompt

    def test_compact_noop_on_empty(self):
        mgr = AgentConversationManager(max_tokens=1)
        assert _run(mgr.compact(_FakeClient(), _FakeAsyncModel(), _FakeAgent([]))) is False

    def test_compact_returns_false_when_nothing_older(self, monkeypatch):
        import personal_ai_assistant.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(
            mod.config, "get", _llm_config(MANUAL_COMPACT_KEEP_RECENT=6)
        )
        msgs = [HumanMessage("a"), AIMessage("b")]
        agent = _FakeAgent(list(msgs))
        mgr = AgentConversationManager(max_tokens=100)
        did = _run(mgr.compact(_FakeClient(), _FakeAsyncModel(), agent))
        assert did is False
        assert agent.messages == msgs

    def test_internal_compact_keep_window(self, monkeypatch):
        import personal_ai_assistant.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config())
        msgs = [AIMessage(f"a{i}") for i in range(8)]
        agent = _FakeAgent(list(msgs))
        mgr = AgentConversationManager(max_tokens=100)
        did = _run(mgr._compact(_FakeClient(), _FakeAsyncModel(), agent, keep_recent=5))
        assert did is True
        assert len(agent.messages) == 5


class TestTokenAwareRetention:
    def test_oversized_recent_message_is_summarized_not_kept(self, monkeypatch):
        # The LAST message is a huge document. Even though the count window
        # (3) would keep it, the token budget must push it into 'older' so it
        # gets summarized rather than kept verbatim.
        import personal_ai_assistant.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(
            mod.config,
            "get",
            _llm_config(MANUAL_COMPACT_KEEP_RECENT=3, KEEP_RECENT_TOKEN_BUDGET=200),
        )
        huge = "X" * 50_000  # ~ tens of thousands of chars => over the budget
        msgs = [
            HumanMessage("small 1"),
            AIMessage("small 2"),
            HumanMessage(huge),  # most recent, oversized
        ]
        agent = _FakeAgent(list(msgs))
        mgr = AgentConversationManager(max_tokens=1000)

        did = _run(mgr.compact(_FakeClient(), _FakeAsyncModel(), agent))
        assert did is True
        # The huge message must NOT be among the kept-verbatim messages.
        kept_texts = [str(m.content) for m in agent.messages]
        assert huge not in kept_texts
        # Summary was produced (the huge doc folded into it).
        assert "SUMMARY OF OLDER MESSAGES" in agent.system_prompt

    def test_token_budget_caps_kept_window_below_count(self, monkeypatch):
        # Count window allows 6, but token budget only fits ~2 small messages.
        import personal_ai_assistant.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(
            mod.config,
            "get",
            _llm_config(MANUAL_COMPACT_KEEP_RECENT=6, KEEP_RECENT_TOKEN_BUDGET=30),
        )
        # Each message ~40 chars -> ~30 tokens with char/4; budget fits ~1.
        msgs = [AIMessage("word " * 8) for _ in range(6)]
        agent = _FakeAgent(list(msgs))
        mgr = AgentConversationManager(max_tokens=1000)

        did = _run(mgr.compact(_FakeClient(), _FakeAsyncModel(), agent))
        assert did is True
        # Token budget binds before the count of 6.
        assert len(agent.messages) < 6
