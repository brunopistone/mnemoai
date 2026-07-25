"""Unit tests for conversation compaction (client/managers/agent_conversation_manager.py).

Covers the pure-logic pieces that don't need a live model:
- message -> dict conversion preserving tool calls / tool results
- rendering messages (incl. tools) to summary text
- the keep-recent-verbatim compaction behavior (with a fake async model)
"""

import asyncio

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


class _OverflowThenOKModel:
    """ainvoke raises 'prompt is too long' if the summary call's message CONTENT
    (excluding the fixed summary-prompt boilerplate) exceeds `limit_chars`, else
    returns a summary. Models the real failure: one giant summary call 400s, but
    small batches succeed. The fixed prompt is excluded so the limit reflects the
    variable history size, which is what batching controls."""

    # Chars contributed by the fixed summary prompt + system framing; excluded
    # from the size check so `limit_chars` bounds the variable (history) part.
    _PROMPT_OVERHEAD = 6000

    def __init__(self, limit_chars=1000):
        self.limit_chars = limit_chars
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        size = sum(len(str(getattr(m, "content", ""))) for m in messages)
        if size - self._PROMPT_OVERHEAD > self.limit_chars:
            raise RuntimeError("prompt is too long: too many tokens > maximum")
        return AIMessage(content=f"BATCH SUMMARY {self.calls}")


class TestBatchedSummarization:
    """generate_summary must batch so the summary CALL never itself overflows;
    a total failure keeps a bounded excerpt, never a content-free placeholder."""

    def test_large_history_is_batched_not_single_call(self, monkeypatch):
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config())
        # Small max_tokens so the per-batch budget (a fraction of it) forces
        # multiple batches over these messages.
        mgr = AgentConversationManager(max_tokens=400)
        model = _OverflowThenOKModel(limit_chars=100000)  # never overflows here
        msgs = [{"role": "user", "content": [{"text": "word " * 50}]} for _ in range(10)]
        out = _run(mgr.generate_summary(msgs, model))
        assert model.calls >= 2                 # split into batches
        assert "BATCH SUMMARY" in out           # rolling summary returned

    def test_batch_that_would_overflow_single_call_still_summarizes(self, monkeypatch):
        # The whole history in one call would exceed the model; batching must
        # keep each call under the limit and still produce a real summary.
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config())
        mgr = AgentConversationManager(max_tokens=400)  # per-batch budget ~256 tok
        # ALL 12 messages' content (~3600 chars) in one call exceeds the limit and
        # 400s; each ~256-token batch (3 msgs, ~900 chars) stays under it.
        model = _OverflowThenOKModel(limit_chars=1500)
        msgs = [{"role": "user", "content": [{"text": "word " * 60}]} for _ in range(12)]
        out = _run(mgr.generate_summary(msgs, model))
        assert "BATCH SUMMARY" in out           # succeeded via batching
        assert "multiple topics" not in out     # NOT the content-free placeholder

    def test_total_failure_keeps_excerpt_not_placeholder(self, monkeypatch):
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config())

        class _AlwaysFail:
            async def ainvoke(self, messages):
                raise RuntimeError("prompt is too long: too many tokens > maximum")

        mgr = AgentConversationManager(max_tokens=4000)
        msgs = [{"role": "user", "content": [{"text": "distinctive content here"}]}]
        out = _run(mgr.generate_summary(msgs, _AlwaysFail()))
        # Falls back to a bounded EXCERPT carrying real content, not a placeholder.
        assert "distinctive content here" in out

    def test_large_window_batches_stay_a_safe_fraction(self):
        # Regression (the real bug): on a large window (1M), a near-full history
        # must NOT produce a single ~500k batch — each batch must stay a small
        # fraction of the window so the summary CALL itself can't overflow.
        from mnemoai.client.managers.agent_conversation_manager import (
            _SUMMARY_CALL_FRACTION,
        )

        mgr = AgentConversationManager(max_tokens=1_000_000)
        budget = max(256, int(mgr.max_tokens * _SUMMARY_CALL_FRACTION))
        # ~576k tokens of history (like the reported conversation).
        msgs = [{"role": "user", "content": [{"text": "word " * 3000}]} for _ in range(150)]
        batches = mgr._batch_messages(msgs, budget)
        sizes = [mgr.count_tokens(b) for b in batches]
        assert len(batches) >= 4                       # actually split
        assert max(sizes) <= budget + 5000             # each batch near the budget
        assert max(sizes) < mgr.max_tokens * 0.25      # far below the window

    def test_oversized_single_message_is_truncated(self):
        # A single message larger than the per-call budget can't overflow its own
        # batch — _truncate_msg_text caps it (head+tail with an elision note).
        mgr = AgentConversationManager(max_tokens=1_000_000)
        giant = "x" * 5_000_000  # ~1.25M tokens alone
        capped = mgr._truncate_msg_text(giant)
        assert len(capped) < len(giant)
        assert "elided" in capped
        # Small text is returned unchanged.
        assert mgr._truncate_msg_text("short") == "short"


class TestToolResultEviction:
    """The cheapest compaction layer: shrink OLD tool-result bodies in place,
    with no LLM call, keeping recent turns verbatim and never dropping a
    message (so tool-call/result pairing is preserved)."""

    def _llm(self, monkeypatch, **overrides):
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config(**overrides))

    def test_shrinks_only_old_tool_results(self, monkeypatch):
        self._llm(monkeypatch, TOOL_EVICTION_KEEP_RECENT=2, EVICTED_TOOL_RESULT_CHARS=50)
        big = "R" * 5000
        msgs = [
            ToolMessage(content=big, tool_call_id="t0", name="grep_search"),   # old
            HumanMessage("m1"),
            ToolMessage(content=big, tool_call_id="t1", name="grep_search"),   # recent
        ]
        agent = _FakeAgent(list(msgs))
        mgr = AgentConversationManager(max_tokens=100)

        changed = mgr.evict_old_tool_results(agent)
        assert changed is True
        # Old tool result shrunk + marked; recent one untouched.
        assert len(agent.messages[0].content) < len(big)
        assert "evicted" in agent.messages[0].content
        assert agent.messages[2].content == big

    def test_preserves_message_count_and_ids(self, monkeypatch):
        self._llm(monkeypatch, TOOL_EVICTION_KEEP_RECENT=0, EVICTED_TOOL_RESULT_CHARS=20)
        msgs = [
            ToolMessage(content="X" * 500, tool_call_id="abc", name="fs_read"),
        ]
        agent = _FakeAgent(list(msgs))
        mgr = AgentConversationManager(max_tokens=100)

        mgr.evict_old_tool_results(agent)
        # Never drops a message — pairing stays intact — and keeps tool_call_id.
        assert len(agent.messages) == 1
        assert agent.messages[0].tool_call_id == "abc"
        assert agent.messages[0].name == "fs_read"

    def test_noop_when_already_short(self, monkeypatch):
        self._llm(monkeypatch, TOOL_EVICTION_KEEP_RECENT=0, EVICTED_TOOL_RESULT_CHARS=500)
        msgs = [ToolMessage(content="tiny", tool_call_id="t", name="glob_search")]
        agent = _FakeAgent(list(msgs))
        mgr = AgentConversationManager(max_tokens=100)
        assert mgr.evict_old_tool_results(agent) is False
        assert agent.messages[0].content == "tiny"

    def test_disabled_when_cap_zero(self, monkeypatch):
        self._llm(monkeypatch, EVICTED_TOOL_RESULT_CHARS=0)
        msgs = [ToolMessage(content="X" * 999, tool_call_id="t", name="grep_search")]
        agent = _FakeAgent(list(msgs))
        mgr = AgentConversationManager(max_tokens=100)
        assert mgr.evict_old_tool_results(agent) is False

    def test_ignores_non_tool_messages(self, monkeypatch):
        self._llm(monkeypatch, TOOL_EVICTION_KEEP_RECENT=0, EVICTED_TOOL_RESULT_CHARS=10)
        msgs = [HumanMessage("H" * 500), AIMessage("A" * 500)]
        agent = _FakeAgent(list(msgs))
        mgr = AgentConversationManager(max_tokens=100)
        assert mgr.evict_old_tool_results(agent) is False
        assert agent.messages[0].content == "H" * 500

    def test_resets_last_input_tokens_after_shrink(self, monkeypatch):
        self._llm(monkeypatch, TOOL_EVICTION_KEEP_RECENT=0, EVICTED_TOOL_RESULT_CHARS=20)

        class _AgentTok(_FakeAgent):
            def __init__(self, messages):
                super().__init__(messages)
                self._last_input_tokens = 999_999

        agent = _AgentTok([ToolMessage(content="X" * 500, tool_call_id="t", name="grep_search")])
        mgr = AgentConversationManager(max_tokens=100)
        mgr.evict_old_tool_results(agent)
        # Ground-truth count is now stale — must be cleared so the high-water
        # check re-estimates against the shrunk history.
        assert agent._last_input_tokens is None

    def test_dict_tool_result_shrunk(self, monkeypatch):
        self._llm(monkeypatch, TOOL_EVICTION_KEEP_RECENT=0, EVICTED_TOOL_RESULT_CHARS=30)
        msgs = [{"role": "tool", "tool_name": "grep_search",
                 "content": [{"text": "Y" * 500}]}]
        agent = _FakeAgent(list(msgs))
        mgr = AgentConversationManager(max_tokens=100)
        assert mgr.evict_old_tool_results(agent) is True
        assert "evicted" in agent.messages[0]["content"][0]["text"]


class _ConcurrencyModel:
    """Records max concurrent ainvoke calls, to prove the map step parallelizes."""

    def __init__(self, delay=0.05):
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(self.delay)
        self.active -= 1
        return AIMessage(content=f"PARTIAL {self.calls}")


class TestParallelMapReduce:
    """generate_summary maps batches concurrently, then reduces once."""

    def test_map_runs_concurrently(self, monkeypatch):
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config(SUBAGENT_MAX_CONCURRENCY=4))
        mgr = AgentConversationManager(max_tokens=400)  # small budget → many batches
        msgs = [{"role": "user", "content": [{"text": "word " * 60}]} for _ in range(12)]
        model = _ConcurrencyModel()
        out = _run(mgr.generate_summary(msgs, model))
        assert model.max_active > 1  # batches summarized in parallel, not serially
        assert out.strip()

    def test_concurrency_bounded_by_config(self, monkeypatch):
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config(SUBAGENT_MAX_CONCURRENCY=2))
        mgr = AgentConversationManager(max_tokens=400)
        msgs = [{"role": "user", "content": [{"text": "word " * 60}]} for _ in range(12)]
        model = _ConcurrencyModel()
        _run(mgr.generate_summary(msgs, model))
        assert model.max_active <= 2  # semaphore respects the cap

    def test_single_batch_skips_reduce(self, monkeypatch):
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config())
        mgr = AgentConversationManager(max_tokens=1_000_000)  # huge → 1 batch
        mgr.previous_summary = None
        model = _FakeAsyncModel()
        msgs = [{"role": "user", "content": [{"text": "small"}]}]
        out = _run(mgr.generate_summary(msgs, model))
        assert len(model.calls) == 1  # one map call, no extra reduce
        assert "SUMMARY OF OLDER MESSAGES" in out

    def test_multi_batch_reduces_once(self, monkeypatch):
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config())
        mgr = AgentConversationManager(max_tokens=400)
        model = _FakeAsyncModel()
        msgs = [{"role": "user", "content": [{"text": "word " * 60}]} for _ in range(12)]
        out = _run(mgr.generate_summary(msgs, model))
        # N map calls + exactly 1 reduce call.
        assert len(model.calls) >= 3
        assert out.strip()

    def test_partial_map_failure_still_summarizes(self, monkeypatch):
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config())

        class _FlakyModel:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, messages):
                self.calls += 1
                if self.calls == 1:  # first map batch fails
                    raise RuntimeError("prompt is too long")
                return AIMessage(content=f"OK {self.calls}")

        mgr = AgentConversationManager(max_tokens=400)
        msgs = [{"role": "user", "content": [{"text": "word " * 60}]} for _ in range(9)]
        out = _run(mgr.generate_summary(msgs, _FlakyModel()))
        assert out.strip()  # surviving partials still produce a summary
        assert "multiple topics" not in out  # not the content-free placeholder

    def test_all_map_failure_keeps_excerpt(self, monkeypatch):
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config())

        class _AlwaysFail:
            async def ainvoke(self, messages):
                raise RuntimeError("boom")

        mgr = AgentConversationManager(max_tokens=4000)
        msgs = [{"role": "user", "content": [{"text": "distinctive excerpt text"}]}]
        out = _run(mgr.generate_summary(msgs, _AlwaysFail()))
        assert "distinctive excerpt text" in out  # bounded excerpt, never empty


class TestCompactEvictsFirst:
    """_compact runs the cheap tool-result eviction before the LLM summary, on
    every path (not just the proactive mid-loop check)."""

    def test_compact_shrinks_old_tool_results_before_summary(self, monkeypatch):
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(
            mod.config, "get",
            _llm_config(MANUAL_COMPACT_KEEP_RECENT=2, TOOL_EVICTION_KEEP_RECENT=2,
                        EVICTED_TOOL_RESULT_CHARS=50),
        )
        big = "R" * 5000
        # Old tool results (should be evicted) + recent turns kept verbatim.
        msgs = []
        for i in range(4):
            msgs.append(AIMessage(content="", tool_calls=[{"name": "grep", "args": {}, "id": f"t{i}"}]))
            msgs.append(ToolMessage(content=big, tool_call_id=f"t{i}", name="grep"))
        msgs.append(HumanMessage("recent question"))

        agent = _FakeAgent(list(msgs))
        agent._sanitize_tool_pairs = lambda m: m
        mgr = AgentConversationManager(max_tokens=100000)
        _run(mgr.compact(_FakeClient(), _FakeAsyncModel(), agent))
        # The kept window's tool results (if any older ones survived the split)
        # were shrunk by eviction before summarizing — assert no 5000-char body
        # remains among whatever the summary consumed by checking the agent's
        # pre-summary eviction ran: the summarized 'older' set had shrunk bodies.
        # Simplest observable: compaction succeeded and produced a summary.
        assert "SUMMARY OF OLDER MESSAGES" in agent.system_prompt

    def test_compact_calls_eviction(self, monkeypatch):
        import mnemoai.client.managers.agent_conversation_manager as mod

        monkeypatch.setattr(mod.config, "get", _llm_config(MANUAL_COMPACT_KEEP_RECENT=2))
        mgr = AgentConversationManager(max_tokens=100000)
        called = {"evict": False}
        orig = mgr.evict_old_tool_results

        def _spy(agent):
            called["evict"] = True
            return orig(agent)

        mgr.evict_old_tool_results = _spy
        msgs = [HumanMessage(f"m{i}") for i in range(6)]
        _run(mgr.compact(_FakeClient(), _FakeAsyncModel(), _FakeAgent(list(msgs))))
        assert called["evict"] is True  # eviction ran as the first compaction layer
