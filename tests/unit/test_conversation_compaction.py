"""Unit tests for conversation compaction (client/managers/agent_conversation_manager.py).

Covers the pure-logic pieces that don't need a live model:
- message -> dict conversion preserving tool calls / tool results
- rendering messages (incl. tools) to summary text
- the keep-recent-verbatim compaction behavior (with a fake async model)
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mnemoai.client.managers.agent_conversation_manager import (
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
    def start(self, label="Thinking"):
        pass

    def set_label(self, label):
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


class _PhaseRecordingSpinner:
    """Records the phase labels passed to the spinner during compaction."""

    def __init__(self):
        self.labels = []

    def start(self, label="Thinking"):
        self.labels.append(("start", label))

    def set_label(self, label):
        self.labels.append(("set", label))

    def stop(self):
        pass


class _PhaseClient(_FakeClient):
    def __init__(self):
        super().__init__()
        self.spinner = _PhaseRecordingSpinner()


class TestCompactProgressPhases:
    def test_compaction_sets_phase_labels(self, monkeypatch):
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(
            mod.config, "get", _llm_config(MANUAL_COMPACT_KEEP_RECENT=2)
        )
        msgs = [HumanMessage(f"m{i}") for i in range(6)]
        client = _PhaseClient()
        mgr = AgentConversationManager(max_tokens=100)
        _run(mgr.compact(client, _FakeAsyncModel(), _FakeAgent(list(msgs))))

        phases = client.spinner.labels
        # Phase 1: summarizing N older messages; Phase 2: applying.
        assert phases[0][0] == "start"
        assert "Summarizing" in phases[0][1] and "older messages" in phases[0][1]
        assert ("set", "Applying summary") in phases


class TestCompactKeepsRecent:
    def test_manual_compact_keeps_recent_window_verbatim(self, monkeypatch):
        import mnemoai.client.managers.agent_conversation_manager as mod

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
        import mnemoai.client.managers.agent_conversation_manager as mod

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
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config())
        msgs = [AIMessage(f"a{i}") for i in range(8)]
        agent = _FakeAgent(list(msgs))
        mgr = AgentConversationManager(max_tokens=100)
        did = _run(mgr._compact(_FakeClient(), _FakeAsyncModel(), agent, keep_recent=5))
        assert did is True
        assert len(agent.messages) == 5


class TestToolBoundarySafety:
    """The kept-verbatim window must never start with an orphaned tool result,
    nor end the summarized set on a tool-call turn whose results were kept.

    Regression for the OpenAI Responses 400: "No tool call found for function
    call output with call_id …" after compaction split a tool pair.
    """

    def _mgr(self):
        return AgentConversationManager(max_tokens=1000)

    def test_split_not_inside_tool_pair_keeps_call_with_result(self, monkeypatch):
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config())
        mgr = self._mgr()
        # 0: user, 1: assistant(tool_call), 2: tool result, 3: assistant answer
        msgs = [
            HumanMessage("do it"),
            AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "call_2"}]),
            ToolMessage(content="ok", tool_call_id="call_2", name="x"),
            AIMessage("done"),
        ]
        # A naive split of 2 would keep [tool result, answer] -> orphaned result.
        safe = mgr._safe_tool_boundary(msgs, 2)
        # Must move before the assistant tool-call turn (index 1).
        assert safe <= 1
        kept = msgs[safe:]
        assert not mgr._is_tool_message(kept[0])  # never starts on a tool result

    def test_split_orphaning_tool_result_is_pulled_back(self, monkeypatch):
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config())
        mgr = self._mgr()
        msgs = [
            HumanMessage("a"),
            AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "c1"}]),
            ToolMessage(content="r", tool_call_id="c1", name="x"),
        ]
        # Split of 2 would keep ONLY the tool result (its call summarized away)
        # -> must move back to 1, keeping the call+result pair together.
        assert mgr._safe_tool_boundary(msgs, 2) == 1
        # Split of 1 is already clean (kept window = call + its result).
        assert mgr._safe_tool_boundary(msgs, 1) == 1

    def test_clean_split_unchanged(self, monkeypatch):
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config())
        mgr = self._mgr()
        msgs = [HumanMessage("a"), AIMessage("b"), HumanMessage("c"), AIMessage("d")]
        # No tools involved -> split is already safe.
        assert mgr._safe_tool_boundary(msgs, 2) == 2

    def test_full_compact_never_orphans_tool_result(self, monkeypatch):
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(
            mod.config, "get", _llm_config(MANUAL_COMPACT_KEEP_RECENT=2)
        )
        mgr = self._mgr()
        msgs = [
            HumanMessage("q"),
            AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "c"}]),
            ToolMessage(content="res", tool_call_id="c", name="x"),
            AIMessage("final"),
        ]
        agent = _FakeAgent(list(msgs))
        _run(mgr.compact(_FakeClient(), _FakeAsyncModel(), agent))
        # Whatever was kept, it must not begin with a tool result.
        if agent.messages:
            assert getattr(agent.messages[0], "type", None) != "tool"

    def test_compact_sanitizes_orphaned_call_in_kept_window(self, monkeypatch):
        # An orphaned assistant tool-call (no matching result) inherited in the
        # kept window must be repaired so the next turn doesn't 400 with
        # "No tool output found for function call …".
        import mnemoai.client.managers.agent_conversation_manager as mod
        from mnemoai.client.agent.agent import LangGraphAgent

        monkeypatch.setattr(
            mod.config, "get", _llm_config(MANUAL_COMPACT_KEEP_RECENT=2)
        )
        mgr = self._mgr()
        orphan = AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "z"}])
        msgs = [HumanMessage("q1"), AIMessage("a1"), orphan, HumanMessage("q2")]

        # A fake agent that exposes the real sanitizer (as the live agent does).
        agent = _FakeAgent(list(msgs))
        agent._sanitize_tool_pairs = staticmethod(LangGraphAgent._sanitize_tool_pairs)
        _run(mgr.compact(_FakeClient(), _FakeAsyncModel(), agent))

        # No surviving assistant message may carry an unmatched tool call.
        for m in agent.messages:
            for c in getattr(m, "tool_calls", []) or []:
                assert c["id"] != "z", "orphaned tool call survived compaction"


class TestStripAnalysis:
    def test_strips_analysis_block(self):
        text = "<analysis>thinking hard</analysis>\n1. Primary: foo"
        out = AgentConversationManager._strip_analysis(text)
        assert "thinking hard" not in out
        assert "Primary: foo" in out

    def test_no_tags_unchanged(self):
        text = "1. Primary Request: bar"
        assert AgentConversationManager._strip_analysis(text).strip() == text

    def test_unbalanced_closing_tag_keeps_tail(self):
        text = "leftover analysis</analysis>\nThe summary."
        out = AgentConversationManager._strip_analysis(text)
        assert "The summary." in out and "leftover analysis" not in out


class TestTokenAwareRetention:
    def test_oversized_recent_message_is_summarized_not_kept(self, monkeypatch):
        # The LAST message is a huge document. Even though the count window
        # (3) would keep it, the token budget must push it into 'older' so it
        # gets summarized rather than kept verbatim.
        import mnemoai.client.managers.agent_conversation_manager as mod

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
        import mnemoai.client.managers.agent_conversation_manager as mod

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
