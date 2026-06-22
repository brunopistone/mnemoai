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
