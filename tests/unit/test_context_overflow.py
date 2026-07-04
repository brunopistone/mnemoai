"""Unit tests for the context-overflow backstop (client/agent/agent.py).

When a request exceeds the model's context window, the agent must NOT loop on
the same oversized prompt. It classifies the overflow error, force-compacts for
the next turn, and returns a terminal message so the graph ends cleanly.
"""

from langchain_core.messages import AIMessage

from mnemoai.client.agent.agent import LangGraphAgent


class TestOverflowClassifier:
    def test_matches_real_repro_error(self):
        # The exact phrasing from the reported loop.
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
        assert not LangGraphAgent._is_context_overflow_error(
            Exception("Connection refused")
        )
        assert not LangGraphAgent._is_context_overflow_error(
            Exception("rate limit exceeded")
        )
        assert not LangGraphAgent._is_context_overflow_error(
            Exception("invalid api key")
        )


class _OverflowModel:
    """A model whose .stream()/.invoke() always raise a context-overflow error."""

    def stream(self, messages, config=None):
        raise RuntimeError("prompt is too long: 9999999 tokens > 1000000 maximum")

    def invoke(self, messages, config=None):
        raise RuntimeError("prompt is too long: 9999999 tokens > 1000000 maximum")


def _overflow_agent():
    a = LangGraphAgent.__new__(LangGraphAgent)
    a.verbose = False
    a.callbacks = []
    a.styled_turn_view = False
    a._start_spinner = lambda label="Thinking": None
    a._stop_spinner = lambda: None
    return a


class TestOverflowBackstop:
    def test_returns_terminal_message_without_looping(self):
        # _stream_once must NOT re-invoke on overflow (that 400s again); it returns
        # a terminal AIMessage so the turn ends.
        a = _overflow_agent()
        calls = {"compact": 0}
        a._compact_provider = lambda force=False: calls.__setitem__(
            "compact", calls["compact"] + 1
        ) or True

        response, _ = a._stream_once(_OverflowModel(), ["msg"], {})
        assert isinstance(response, AIMessage)
        assert "context window" in response.content.lower()
        assert calls["compact"] == 1  # force-compacted exactly once, no retry loop

    def test_forced_compaction_called_with_force_true(self):
        a = _overflow_agent()
        seen = {}
        a._compact_provider = lambda force=False: seen.__setitem__("force", force) or True
        a._stream_once(_OverflowModel(), ["msg"], {})
        assert seen.get("force") is True

    def test_no_provider_still_terminates(self):
        # Without a compaction provider (bare object), it must still return the
        # terminal message rather than raise or loop.
        a = _overflow_agent()
        response, _ = a._stream_once(_OverflowModel(), ["msg"], {})
        assert isinstance(response, AIMessage)
        assert "context window" in response.content.lower()
