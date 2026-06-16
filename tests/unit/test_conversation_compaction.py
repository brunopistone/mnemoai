"""Unit tests for conversation compaction (Strands branch).

Strands messages are native dicts (``{"role", "content": [{"text"|"toolUse"|
"toolResult"}]}``), so these tests use that format. Covers token-aware
keep-recent retention and the manual compact() path with a fake async model.
"""

import asyncio

from client.managers.agent_conversation_manager import AgentConversationManager


def _msg(role, text):
    return {"role": role, "content": [{"text": text}]}


class _FakeAsyncModel:
    """Stands in for a Strands model: stream() yields one text delta."""

    async def stream(self, messages, system_prompt=None, think=False):
        yield {
            "contentBlockDelta": {"delta": {"text": "SUMMARY OF OLDER MESSAGES"}}
        }


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
    """Fake config.get with a generous token budget by default."""
    llm = {"KEEP_RECENT_TOKEN_BUDGET": 10_000_000}
    llm.update(overrides)

    def _get(key, default=None):
        if key == "LLM":
            return llm
        if key == "SYSTEM_PROMPT":
            return None
        if key == "MODEL_ID":
            return {"TYPE": "ollama"}
        return default

    return _get


class TestCompactKeepsRecent:
    def test_manual_compact_keeps_recent_window_verbatim(self, monkeypatch):
        import client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config(MANUAL_COMPACT_KEEP_RECENT=3))
        msgs = [_msg("user" if i % 2 == 0 else "assistant", f"m{i}") for i in range(10)]
        agent = _FakeAgent(list(msgs))
        mgr = AgentConversationManager(max_tokens=100)

        did = _run(mgr.compact(_FakeClient(), _FakeAsyncModel(), agent))
        assert did is True
        assert len(agent.messages) == 3
        assert agent.messages == msgs[-3:]
        assert "SUMMARY OF OLDER MESSAGES" in agent.system_prompt

    def test_compact_noop_on_empty(self, monkeypatch):
        import client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config())
        mgr = AgentConversationManager(max_tokens=1)
        assert _run(mgr.compact(_FakeClient(), _FakeAsyncModel(), _FakeAgent([]))) is False

    def test_compact_returns_false_when_nothing_older(self, monkeypatch):
        import client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config(MANUAL_COMPACT_KEEP_RECENT=6))
        msgs = [_msg("user", "a"), _msg("assistant", "b")]
        agent = _FakeAgent(list(msgs))
        mgr = AgentConversationManager(max_tokens=100)
        did = _run(mgr.compact(_FakeClient(), _FakeAsyncModel(), agent))
        assert did is False
        assert agent.messages == msgs


class TestTokenAwareRetention:
    def test_oversized_recent_message_is_summarized_not_kept(self, monkeypatch):
        import client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(
            mod.config,
            "get",
            _llm_config(MANUAL_COMPACT_KEEP_RECENT=3, KEEP_RECENT_TOKEN_BUDGET=200),
        )
        huge = "X" * 50_000
        msgs = [
            _msg("user", "small 1"),
            _msg("assistant", "small 2"),
            _msg("user", huge),  # most recent, oversized
        ]
        agent = _FakeAgent(list(msgs))
        mgr = AgentConversationManager(max_tokens=1000)

        did = _run(mgr.compact(_FakeClient(), _FakeAsyncModel(), agent))
        assert did is True
        kept_text = "".join(
            block.get("text", "")
            for m in agent.messages
            for block in m.get("content", [])
        )
        assert huge not in kept_text
        assert "SUMMARY OF OLDER MESSAGES" in agent.system_prompt

    def test_token_budget_caps_kept_window_below_count(self, monkeypatch):
        import client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(
            mod.config,
            "get",
            _llm_config(MANUAL_COMPACT_KEEP_RECENT=6, KEEP_RECENT_TOKEN_BUDGET=30),
        )
        msgs = [_msg("assistant", "word " * 8) for _ in range(6)]
        agent = _FakeAgent(list(msgs))
        mgr = AgentConversationManager(max_tokens=1000)

        did = _run(mgr.compact(_FakeClient(), _FakeAsyncModel(), agent))
        assert did is True
        assert len(agent.messages) < 6


class TestSplitKeepRecent:
    def test_split_index_counts_from_end(self, monkeypatch):
        import client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config())
        mgr = AgentConversationManager(max_tokens=1000)
        msgs = [_msg("user", f"m{i}") for i in range(8)]
        # keep_recent=5 -> split at 3 (summarize 3, keep 5)
        assert mgr._split_keep_recent(msgs, 5) == 3

    def test_keep_recent_zero_summarizes_all(self, monkeypatch):
        import client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config())
        mgr = AgentConversationManager(max_tokens=1000)
        msgs = [_msg("user", "a"), _msg("user", "b")]
        assert mgr._split_keep_recent(msgs, 0) == 2
