"""Unit tests for the no-silent-empty-turn guarantee in the agent.

Regression for: the agent ran a tool (e.g. bash), got an error/timeout result,
then the model ended on a totally-empty turn (no content, no reasoning). invoke()
used to return "" — a silent turn. It must instead salvage the last tool result
or fall back to a message, never an empty string.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mnemoai.client.agent.agent import LangGraphAgent


def _agent():
    a = LangGraphAgent.__new__(LangGraphAgent)
    # _extract_visible is a plain method; bind nothing else needed for these.
    return a


def test_last_tool_result_returns_most_recent():
    a = _agent()
    msgs = [
        HumanMessage(content="run it"),
        ToolMessage(content="first result", tool_call_id="1", name="x"),
        ToolMessage(content='{"error": true, "message": "timed out"}', tool_call_id="2", name="execute_bash"),
        AIMessage(content=""),
    ]
    assert "timed out" in a._last_tool_result(msgs)


def test_last_tool_result_empty_when_no_tool():
    a = _agent()
    msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
    assert a._last_tool_result(msgs) == ""


def test_last_tool_result_truncates():
    a = _agent()
    big = "x" * 1000
    msgs = [ToolMessage(content=big, tool_call_id="1", name="x")]
    assert len(a._last_tool_result(msgs)) <= 500


def test_stream_error_prefers_complete_nonstreaming_result():
    """On a streaming error, the truncated partial chunk must be discarded in
    favor of the complete non-streaming invoke() result (else a mid-stream
    parse failure surfaces as an empty/incomplete turn)."""
    class _FakeChunk:
        def __init__(self):
            self.content = ""
            self.tool_calls = []
        def __add__(self, other):
            return self

    class _FakeModel:
        def stream(self, messages, config=None):
            yield _FakeChunk()  # a partial chunk arrives...
            raise ValueError("simulated mid-stream parse error")
        def invoke(self, messages, config=None):
            return AIMessage(content="COMPLETE ANSWER")

    a = _agent()
    a.callbacks = []
    a.verbose = False
    a._code_formatter = type("F", (), {"process_chunk": lambda s, c: None})()
    a._stop_spinner = lambda: None
    a._extract_content = lambda chunk: (getattr(chunk, "content", ""), None)

    resp, _ = a._stream_response(["msg"], {}, model=_FakeModel())
    assert getattr(resp, "content", None) == "COMPLETE ANSWER"


def test_last_visible_from_skips_empty_ai_turns():
    a = _agent()
    msgs = [
        AIMessage(content="real answer"),
        ToolMessage(content="tool out", tool_call_id="1", name="x"),
        AIMessage(content=""),  # trailing empty turn
    ]
    # Should return the earlier visible answer, not the empty trailing one.
    assert a._last_visible_from(msgs) == "real answer"


def _msg(meta):
    m = AIMessage(content="")
    m.response_metadata = meta
    return m


def test_truncation_detected_responses_incomplete():
    # Reasoning model on the Responses API runs out of tokens mid-reasoning.
    assert LangGraphAgent._was_truncated_by_tokens(
        _msg({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}})
    )


def test_truncation_detected_chat_length_finish():
    assert LangGraphAgent._was_truncated_by_tokens(_msg({"finish_reason": "length"}))


def test_truncation_detected_bedrock_max_tokens():
    assert LangGraphAgent._was_truncated_by_tokens(_msg({"stop_reason": "max_tokens"}))


def test_truncation_not_detected_on_normal_completion():
    assert not LangGraphAgent._was_truncated_by_tokens(
        _msg({"status": "completed", "finish_reason": "stop"})
    )


def test_truncation_not_detected_without_metadata():
    assert not LangGraphAgent._was_truncated_by_tokens(AIMessage(content="hi"))


class _AgentStreamHarness:
    """Bind just enough of the agent for _stream_response/_is_empty_response."""

    @staticmethod
    def make(retries):
        a = _agent()
        a.callbacks = []
        a.verbose = False
        a._empty_response_retries = retries
        a._code_formatter = type("F", (), {"process_chunk": lambda s, c: None})()
        a._stop_spinner = lambda: None
        a._start_spinner = lambda: None
        a._extract_content = lambda chunk: (getattr(chunk, "content", ""), None)
        return a


class _Chunk:
    """Minimal streamed chunk that aggregates by replacement."""

    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.response_metadata = {}
        self.additional_kwargs = {}

    def __add__(self, other):
        return other


class _SeqModel:
    """Yields a queued response per stream() call (one chunk each)."""

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0

    def stream(self, messages, config=None):
        self.calls += 1
        content = self._contents.pop(0) if self._contents else ""
        yield _Chunk(content=content)


def test_is_empty_response_detects_blank():
    a = _agent()
    assert a._is_empty_response(None)
    assert a._is_empty_response(AIMessage(content=""))
    assert a._is_empty_response(AIMessage(content=[]))


def test_is_empty_response_false_with_text_or_tool():
    a = _agent()
    assert not a._is_empty_response(AIMessage(content="hello"))
    tc = AIMessage(content="")
    tc.tool_calls = [{"name": "x", "args": {}, "id": "1"}]
    assert not a._is_empty_response(tc)


def test_stream_retries_on_empty_then_succeeds():
    # First stream returns empty (transient), second returns real text.
    a = _AgentStreamHarness.make(retries=2)
    model = _SeqModel(["", "REAL ANSWER"])
    resp, _ = a._stream_response(["msg"], {}, model=model)
    assert resp.content == "REAL ANSWER"
    assert model.calls == 2


def test_stream_gives_up_after_retries():
    a = _AgentStreamHarness.make(retries=2)
    model = _SeqModel(["", "", ""])  # always empty
    resp, _ = a._stream_response(["msg"], {}, model=model)
    assert a._is_empty_response(resp)
    assert model.calls == 3  # 1 + 2 retries


def test_stream_no_retry_when_first_succeeds():
    a = _AgentStreamHarness.make(retries=2)
    model = _SeqModel(["GOOD"])
    resp, _ = a._stream_response(["msg"], {}, model=model)
    assert resp.content == "GOOD"
    assert model.calls == 1


def _harness_counting_marker(retries=0):
    a = _AgentStreamHarness.make(retries=retries)
    a._marker_calls = 0

    def _mark():
        a._marker_calls += 1

    a._print_answer_marker = _mark
    return a


def test_answer_marker_printed_when_marking_and_no_reasoning():
    # No reasoning shown + mark_answer=True -> exactly one marker before answer.
    a = _harness_counting_marker()
    model = _SeqModel(["the answer"])
    a._stream_response(["msg"], {}, model=model, mark_answer=True)
    assert a._marker_calls == 1


def test_answer_marker_not_printed_when_not_marking():
    # Worker streams (mark_answer=False) must NOT print the marker.
    a = _harness_counting_marker()
    model = _SeqModel(["the answer"])
    a._stream_response(["msg"], {}, model=model, mark_answer=False)
    assert a._marker_calls == 0


def test_answer_marker_printed_once_across_chunks():
    # A single answer streamed as several chunks gets exactly one marker.
    a = _harness_counting_marker()

    class _MultiChunk:
        def stream(self, messages, config=None):
            for piece in ("Hel", "lo ", "there"):
                yield _Chunk(content=piece)

    a._stream_response(["msg"], {}, model=_MultiChunk(), mark_answer=True)
    assert a._marker_calls == 1
