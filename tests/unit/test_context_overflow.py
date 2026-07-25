"""Unit tests for context-overflow handling (client/agent/agent.py).

When a request exceeds the model's context window, the agent must NOT loop on
the same oversized prompt. New behavior (1.0.2): `_stream_once` raises a typed
`_ContextOverflow`; `_call_model` catches it, force-compacts, and RE-INVOKES on
the shrunken prompt so the in-flight task continues. Only if compaction can't
shrink history (or it still overflows) does it return a terminal message.
"""

from langchain_core.messages import AIMessage, HumanMessage

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
