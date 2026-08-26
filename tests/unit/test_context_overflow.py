"""Unit tests for context-overflow handling (client/agent/agent.py).

When a request exceeds the model's context window, the agent must NOT loop on
the same oversized prompt. New behavior (1.0.2): `_stream_once` raises a typed
`_ContextOverflow`; `_call_model` catches it, force-compacts, and RE-INVOKES on
the shrunken prompt so the in-flight task continues. Only if compaction can't
shrink history (or it still overflows) does it return a terminal message.
"""

import inspect

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mnemoai.client.agent.agent import LangGraphAgent, _ContextOverflow


class TestOverflowClassifier:
    def test_matches_real_repro_error(self):
        e = Exception(
            "Error code: 400 - {'type': 'error', 'error': {'type': "
            "'invalid_request_error', 'message': 'prompt is too long: "
            "10597028 tokens > 1000000 maximum'}}"
        )
        assert LangGraphAgent._is_context_overflow_error(e)

    def test_matches_openai_context_length(self):
        e = Exception("This model's maximum context length is 128000 tokens")
        assert LangGraphAgent._is_context_overflow_error(e)

    def test_matches_bedrock_stop_reason(self):
        e = Exception("stop_reason: model_context_window_exceeded")
        assert LangGraphAgent._is_context_overflow_error(e)

    def test_rejects_unrelated_errors(self):
        assert not LangGraphAgent._is_context_overflow_error(Exception("Connection refused"))
        assert not LangGraphAgent._is_context_overflow_error(Exception("rate limit exceeded"))
        assert not LangGraphAgent._is_context_overflow_error(Exception("invalid api key"))


class _OverflowModel:
    """A model whose .stream()/.invoke() always raise a context-overflow error."""

    def stream(self, messages, config=None):
        raise RuntimeError("prompt is too long: 9999999 tokens > 1000000 maximum")

    def invoke(self, messages, config=None):
        raise RuntimeError("prompt is too long: 9999999 tokens > 1000000 maximum")


def _agent():
    a = LangGraphAgent.__new__(LangGraphAgent)
    a.verbose = False
    a.callbacks = []
    a.styled_turn_view = False
    a._start_spinner = lambda label="Thinking": None
    a._stop_spinner = lambda: None
    return a


class TestStreamOnceRaisesOverflow:
    def test_stream_once_raises_typed_overflow(self):
        # _stream_once no longer swallows overflow — it raises _ContextOverflow so
        # the caller can compact + retry (continue the task).
        a = _agent()
        import pytest

        with pytest.raises(_ContextOverflow):
            a._stream_once(_OverflowModel(), ["msg"], {})


class TestCallModelResumesAfterCompaction:
    """The heart of the fix: _call_model compacts and RE-INVOKES on the shrunken
    prompt so the in-flight task continues, instead of dead-ending."""

    def _agent_for_call_model(self, model_after_compact):
        a = _agent()
        a.system_prompt = "SYS"
        a._get_route_model = lambda state: state.get("_model")
        a._sanitize_tool_pairs = lambda msgs: list(msgs)
        # After compaction, self._messages is the shrunken history.
        a._messages = [HumanMessage("compacted history")]
        a._extract_thinking = lambda r: None
        a._extract_visible = lambda c: (c if isinstance(c, str) else "")
        a._was_truncated_by_tokens = lambda r: False
        return a

    def test_compacts_and_retries_on_shrunken_prompt(self):
        # First _stream_response overflows; after compaction the retry succeeds.
        a = self._agent_for_call_model(None)
        state_model = _StreamThenOK()

        # Compaction shrinks history (len drops), enabling the retry.
        def _compact(force=False):
            a._messages = [HumanMessage("small")]
            a.system_prompt = "SYS+SUMMARY"
            return True

        a._compact_provider = _compact
        # Pre-compaction history is larger so _compact_and_rebuild sees a shrink.
        a._messages = [HumanMessage("m")] * 10

        out = a._call_model({"messages": [HumanMessage("hi")], "_model": state_model})
        # The retry produced a real answer (task continued), not a terminal error.
        assert isinstance(out["messages"][0], AIMessage)
        assert "ANSWER" in out["messages"][0].content
        assert state_model.stream_calls == 2  # overflowed once, succeeded on retry

    def test_terminal_message_when_compaction_cannot_shrink(self):
        # If compaction can't shrink history, don't loop — return terminal msg.
        a = self._agent_for_call_model(None)
        a._messages = [HumanMessage("m")] * 5
        a._compact_provider = lambda force=False: False  # no-op, no shrink

        out = a._call_model({"messages": [HumanMessage("hi")], "_model": _OverflowModel()})
        assert isinstance(out["messages"][0], AIMessage)
        assert "context window" in out["messages"][0].content.lower()

    def test_no_provider_returns_terminal_message(self):
        a = self._agent_for_call_model(None)
        # no _compact_provider attribute at all
        out = a._call_model({"messages": [HumanMessage("hi")], "_model": _OverflowModel()})
        assert isinstance(out["messages"][0], AIMessage)
        assert "context window" in out["messages"][0].content.lower()


class _OverBudgetModel:
    """Overflows whenever the prompt exceeds ``limit`` messages.

    Stands in for the provider's own context window, so a test can assert that a
    LATER call in the same turn is sent a prompt that actually fits — the thing a
    "did it retry once" assertion can't see.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self.seen: list[int] = []

    def stream(self, messages, config=None):
        self.seen.append(len(messages))
        if len(messages) > self.limit:
            raise RuntimeError("prompt is too long: 3155357 tokens > 1000000 maximum")
        return iter(())

    def invoke(self, messages, config=None):
        if len(messages) > self.limit:
            raise RuntimeError("prompt is too long: 3155357 tokens > 1000000 maximum")
        return AIMessage(content="ANSWER")


class TestCompactionReachesTheRestOfTheTurn:
    """A mid-turn compaction must apply to EVERY later model call in that turn.

    The graph's ``messages`` channel is ``operator.add`` — append-only — so
    compacting ``self._messages`` is invisible to the running graph. Fixing only
    the retry prompt left the call after the next tool result re-sending the
    pre-compaction history, and by then compaction had nothing left to give, so
    the turn dead-ended on "couldn't compact it further" over an already-tiny
    history. Observed live as two overflows 2435 tokens apart: "compacted and
    retrying" followed by "could not compact".
    """

    def _turn_agent(self, seed_len: int):
        a = _agent()
        a.system_prompt = "SYS"
        a._sanitize_tool_pairs = lambda msgs: list(msgs)
        a._extract_thinking = lambda r: None
        a._extract_visible = lambda c: (c if isinstance(c, str) else "")
        a._was_truncated_by_tokens = lambda r: False
        a._turn_seed_len = seed_len
        a._mid_turn_compaction = None
        a._messages = [AIMessage(content=f"old{i}") for i in range(20)]

        def _compact(force=False):
            a._messages = [AIMessage(content="summarized away")]
            a.system_prompt = "SYS+SUMMARY"
            return True

        a._compact_provider = _compact
        return a

    def test_later_model_call_in_the_turn_gets_the_compacted_prompt(self):
        # Seeded with 6 messages of history; the provider fits 4.
        a = self._turn_agent(seed_len=6)
        model = _OverBudgetModel(limit=4)
        a._get_route_model = lambda state: model
        seeded = [SystemMessage(content="SYS")] + [
            AIMessage(content=f"old{i}") for i in range(5)
        ]

        # First call: overflows, compacts, retries on the shrunken prompt.
        first = a._call_model({"messages": list(seeded)})
        assert "context window" not in str(first["messages"][0].content).lower()
        assert a._mid_turn_compaction is not None

        # Second call — the state has GROWN by this turn's tool exchange, and
        # still carries all 6 seeded messages. Before the fix this re-overflowed
        # and the turn died; now the compacted history is substituted in.
        state2 = list(seeded) + [
            AIMessage(content="call"),
            ToolMessage(content="result", tool_call_id="1"),
        ]
        second = a._call_model({"messages": state2})
        assert isinstance(second["messages"][0], AIMessage)
        assert "context window" not in str(second["messages"][0].content).lower()
        assert max(model.seen) == 6, model.seen  # only the first, pre-fix prompt
        assert model.seen[-1] <= 4  # the last prompt actually fit

    def test_retry_keeps_the_tool_results_this_turn_already_produced(self):
        a = self._turn_agent(seed_len=6)
        model = _OverBudgetModel(limit=4)
        a._get_route_model = lambda state: model
        state = (
            [SystemMessage(content="SYS")]
            + [AIMessage(content=f"old{i}") for i in range(5)]
            + [ToolMessage(content="hard-won result", tool_call_id="1")]
        )

        a._call_model({"messages": state})
        # The rebuilt prompt is the compacted history PLUS this turn's tail — the
        # model must not have to redo work it already did.
        rebuilt = a._apply_mid_turn_compaction(state)
        assert [m.content for m in rebuilt][-1] == "hard-won result"
        assert "old0" not in [m.content for m in rebuilt]

    def test_substitution_is_a_no_op_before_any_compaction(self):
        a = _agent()  # no stash attribute at all (bare stub)
        msgs = [HumanMessage(content="a")]
        assert a._apply_mid_turn_compaction(msgs) is msgs

    def test_stash_does_not_leak_into_the_next_turn(self):
        # invoke() clears it; a previous turn's compacted prefix must never be
        # substituted into a fresh turn's prompt.
        a = self._turn_agent(seed_len=6)
        a._mid_turn_compaction = ([AIMessage(content="stale")], 6)
        a._turn_seed_len, a._mid_turn_compaction = 2, None
        msgs = [SystemMessage(content="SYS"), HumanMessage(content="new")]
        assert a._apply_mid_turn_compaction(msgs) is msgs


class TestCommitTurnDoesNotUndoCompaction:
    """_commit_turn must commit only what the turn PRODUCED.

    The append-only state still holds the pre-compaction history, and the
    ``m not in self._messages`` dedup stops recognizing it once compaction has
    replaced that list — so every summarized-away message was appended back and
    re-logged as this turn's work. Live signature: a compaction reporting "1209
    older messages" followed, one turn later, by "1165 older messages" — the same
    history minus the 44 ``HumanMessage``s this filter drops.
    """

    def _agent_after_compaction(self):
        a = _agent()
        a._messages = [AIMessage(content="kept")]  # all compaction left
        a._turn_seed_len = 6
        a._mid_turn_compaction = ([AIMessage(content="kept")], 6)
        a.logged = []
        a._log_turn = lambda msgs: a.logged.extend(msgs)
        return a

    def _state(self):
        seeded = [SystemMessage(content="SYS")] + [
            AIMessage(content=f"old{i}") for i in range(5)
        ]
        produced = [
            AIMessage(content="call"),
            ToolMessage(content="result", tool_call_id="1"),
            AIMessage(content="answer"),
        ]
        return {"messages": seeded + produced}

    def test_history_is_not_re_inflated(self):
        a = self._agent_after_compaction()
        added = a._commit_turn(self._state(), [])
        assert [m.content for m in added] == ["call", "result", "answer"]
        assert [m.content for m in a._messages] == ["kept", "call", "result", "answer"]

    def test_transcript_gets_only_this_turns_messages(self):
        a = self._agent_after_compaction()
        a._commit_turn(self._state(), [])
        assert [m.content for m in a.logged] == ["call", "result", "answer"]

    def test_normal_turn_is_unaffected(self):
        # No compaction: history already holds the seeded messages, and the turn's
        # own messages commit exactly as before.
        a = _agent()
        seeded = [AIMessage(content=f"old{i}") for i in range(3)]
        a._messages = list(seeded)
        a._turn_seed_len = 3
        a._mid_turn_compaction = None
        a.logged = []
        a._log_turn = lambda msgs: a.logged.extend(msgs)
        added = a._commit_turn({"messages": seeded + [AIMessage(content="new")]}, [])
        assert [m.content for m in added] == ["new"]
        assert len(a._messages) == 4


class TestStaleContextSizeAfterHistoryIsReplaced:
    """Replacing live history must invalidate the cached provider token count.

    ``_compact_now`` prefers the provider's exact ``input_tokens`` over its own
    (deliberately conservative) estimate. A resume, ``/load`` and ``/branch``
    rehydrate the append-only TRANSCRIPT, which still holds everything compaction
    had summarized away — so the stale count reads far too low for the history
    now in memory, the high-water check passes, and the turn goes straight to a
    provider-side overflow instead of compacting first.
    """

    def test_forget_context_size_clears_the_cached_count(self):
        from types import SimpleNamespace

        from mnemoai.client.client import LangGraphClient

        c = LangGraphClient.__new__(LangGraphClient)
        c.agent = SimpleNamespace(_last_input_tokens=647249)
        c._forget_context_size()
        assert c.agent._last_input_tokens is None

    def test_no_agent_is_harmless(self):
        from mnemoai.client.client import LangGraphClient

        c = LangGraphClient.__new__(LangGraphClient)
        c.agent = None
        c._forget_context_size()  # must not raise

    def test_every_history_replacing_path_invalidates_it(self):
        from mnemoai.client.client import LangGraphClient

        for name in ("resume_session", "load_conversation", "branch_conversation"):
            src = inspect.getsource(getattr(LangGraphClient, name))
            assert "_forget_context_size()" in src, name


class _StreamThenOK:
    """Overflows on the first _stream_response, succeeds on the second."""

    def __init__(self):
        self.stream_calls = 0

    def stream(self, messages, config=None):
        self.stream_calls += 1
        if self.stream_calls == 1:
            raise RuntimeError("prompt is too long: 9999999 tokens > 1000000 maximum")
        # Second call: yield nothing (empty stream) so _stream_once returns the
        # accumulated response; invoke() below provides the real content.
        return iter(())

    def invoke(self, messages, config=None):
        return AIMessage(content="ANSWER")
