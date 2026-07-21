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


def test_stream_error_reraises_no_blocking_fallback():
    """A mid-stream error must RE-RAISE (so the abortable retry wrapper handles
    it), NOT be swallowed by a blocking non-streaming invoke(). That blocking
    fallback couldn't be cancelled and wedged the turn on a 500 api_error; it was
    removed. invoke() must never be called from the stream error path."""
    import pytest

    class _FakeChunk:
        def __init__(self):
            self.content = ""
            self.tool_calls = []
        def __add__(self, other):
            return self

    class _FakeModel:
        invoked = False

        def stream(self, messages, config=None):
            yield _FakeChunk()  # a partial chunk arrives...
            raise ValueError("simulated mid-stream parse error")

        def invoke(self, messages, config=None):
            _FakeModel.invoked = True
            raise AssertionError("blocking non-streaming fallback must not run")

    a = _agent()
    a.callbacks = []
    a.verbose = False
    a.styled_turn_view = False
    a.reasoning_sink = None
    a._stream_idle_timeout = 0
    a._code_formatter = type("F", (), {"process_chunk": lambda s, c: None})()
    a._stop_spinner = lambda: None
    a._start_spinner = lambda *x, **k: None
    a._extract_content = lambda chunk: (getattr(chunk, "content", ""), None)

    # A non-transient ValueError isn't retried by the wrapper → it propagates.
    with pytest.raises(ValueError, match="mid-stream parse error"):
        a._stream_response(["msg"], {}, model=_FakeModel())
    assert _FakeModel.invoked is False


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
        return "«M»"  # sentinel prefix so tests can assert placement too

    a._answer_marker = _mark
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


def test_answer_marker_printed_after_reasoning():
    # Reasoning shown first, then the answer: the marker still fires exactly
    # once, before the first answer chunk (not before the reasoning).
    a = _harness_counting_marker()
    a.verbose = True
    a._extract_content = lambda ch: (
        getattr(ch, "content", ""),
        getattr(ch, "reasoning", ""),
    )

    class _ReasoningChunk(_Chunk):
        def __init__(self, content="", reasoning=""):
            super().__init__(content=content)
            self.reasoning = reasoning

    class _Model:
        def stream(self, messages, config=None):
            yield _ReasoningChunk(reasoning="pondering")
            yield _ReasoningChunk(content="The answer")

    a._stream_response(["msg"], {}, model=_Model(), mark_answer=True)
    assert a._marker_calls == 1


def test_marker_prepended_to_first_answer_bytes():
    # The marker must be the leading bytes of the answer's committed output — not
    # a separate write — so it can't be erased independently of the answer.
    import contextlib
    import io

    from mnemoai.utils.formatting.code_formatter import CodeFormatter

    a = _AgentStreamHarness.make(retries=0)
    a.callbacks = ["cb"]
    a._code_formatter = CodeFormatter()

    class _Multi:
        def stream(self, messages, config=None):
            for piece in ("Hello ", "there"):
                yield _Chunk(content=piece)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        a._stream_response(["msg"], {}, model=_Multi(), mark_answer=True)

    out = buf.getvalue()
    assert "●" in out
    # Marker comes before the answer text, on the same line (no newline between).
    assert out.index("●") < out.index("Hello")
    assert "\n" not in out[out.index("●"):out.index("Hello")]


def test_spinner_kept_running_through_hidden_reasoning():
    # Regression: a model that streams reasoning we DON'T display (redacted, or
    # non-verbose — e.g. Anthropic via Bedrock) must keep the spinner running
    # until the visible answer arrives, not stop on the first hidden-reasoning
    # chunk (which left a dead pause before any text printed).
    import contextlib
    import io

    a = _AgentStreamHarness.make(retries=0)
    a.verbose = False  # reasoning is NOT displayed
    a.callbacks = ["cb"]  # non-empty so the stop-condition is reachable

    # Record how much visible text had streamed at each _stop_spinner call.
    streamed = {"text": ""}
    stops = []
    a._stop_spinner = lambda: stops.append(streamed["text"])
    a._extract_content = lambda ch: (
        getattr(ch, "content", ""), getattr(ch, "reasoning", "")
    )

    class _RC(_Chunk):
        def __init__(self, content="", reasoning=""):
            super().__init__(content=content)
            self.reasoning = reasoning

    class _Model:
        def stream(self, messages, config=None):
            # Two hidden-reasoning chunks, THEN the visible answer.
            yield _RC(reasoning="secret 1")
            yield _RC(reasoning="secret 2")
            yield _RC(content="visible answer")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        a._stream_response(["msg"], {}, model=_Model(), mark_answer=True)

    # Spinner stopped exactly ONCE — and not during the two hidden-reasoning
    # chunks (it kept spinning until the visible answer arrived).
    assert len(stops) == 1
    # The visible answer did get printed.
    assert "visible answer" in buf.getvalue()


def test_spinner_kept_running_through_buffered_reasoning_styled():
    # Regression (pinned UI): styled mode buffers reasoning, so the spinner's one
    # stop must fire at the ANSWER, not on the first reasoning chunk (dead pause).
    import contextlib
    import io

    a = _AgentStreamHarness.make(retries=0)
    a.verbose = True  # verbose, but styled → reasoning is buffered, not streamed
    a.styled_turn_view = True
    a.callbacks = ["cb"]
    a._flush_reasoning_block = lambda parts, started: None

    # Track how many reasoning chunks had been consumed when the spinner stopped.
    seen = {"reasoning_chunks": 0, "answer": False}
    stop_state = []
    a._stop_spinner = lambda: stop_state.append(dict(seen))

    def _extract(ch):
        content = getattr(ch, "content", "")
        reasoning = getattr(ch, "reasoning", "")
        if reasoning:
            seen["reasoning_chunks"] += 1
        if content:
            seen["answer"] = True
        return content, reasoning

    a._extract_content = _extract

    class _RC(_Chunk):
        def __init__(self, content="", reasoning=""):
            super().__init__(content=content)
            self.reasoning = reasoning

    class _Model:
        def stream(self, messages, config=None):
            yield _RC(reasoning="thinking 1")
            yield _RC(reasoning="thinking 2")
            yield _RC(reasoning="thinking 3")
            yield _RC(content="the answer")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        a._stream_response(["msg"], {}, model=_Model(), mark_answer=True)

    # One stop, and only after the answer arrived (not during buffered reasoning).
    assert len(stop_state) == 1
    assert stop_state[0]["answer"] is True, (
        "spinner stopped during buffered reasoning (dead pause)"
    )
    assert "the answer" in buf.getvalue()


def test_reasoning_sink_fed_live_then_stopped():
    # styled mode with a live sink: reasoning chunks are appended as they stream,
    # and the sink is stopped (transient view cleared) when the answer commits.
    import contextlib
    import io

    from mnemoai.client.ui.turn_view import ReasoningStatus

    a = _AgentStreamHarness.make(retries=0)
    a.verbose = True
    a.styled_turn_view = True
    a.callbacks = ["cb"]
    a.reasoning_sink = ReasoningStatus()

    appended = []
    a.reasoning_sink.append = lambda t: appended.append(t)
    a._extract_content = lambda ch: (
        getattr(ch, "content", ""), getattr(ch, "reasoning", "")
    )

    class _RC(_Chunk):
        def __init__(self, content="", reasoning=""):
            super().__init__(content=content)
            self.reasoning = reasoning

    class _Model:
        def stream(self, messages, config=None):
            yield _RC(reasoning="step 1 ")
            yield _RC(reasoning="step 2")
            yield _RC(content="answer")

    with contextlib.redirect_stdout(io.StringIO()):
        a._stream_response(["msg"], {}, model=_Model(), mark_answer=True)

    # Both reasoning chunks were fed live, and the sink ended stopped (cleared).
    assert appended == ["step 1 ", "step 2"]
    assert a.reasoning_sink.active is False


# --- Worker-loop (orchestrator) empty-turn salvage ---------------------------
# Regression: a worker finishing with no tool calls and no visible content
# (reasoning-only / empty turn) returned "" with nothing streamed to screen, so
# the orchestrator surfaced a blank answer. _salvage_empty_worker_turn must
# recover a visible reply, mirroring _call_model's guarantee.


def _salvage_agent():
    a = _agent()
    a._start_spinner = lambda label="Thinking": None
    a._stop_spinner = lambda: None
    a._disable_reasoning = lambda: {}
    a._restore_reasoning = lambda saved: None
    return a


def test_worker_salvage_retries_and_recovers_visible_answer():
    a = _salvage_agent()
    # The retry stream yields a real answer.
    a._stream_response = lambda *args, **kw: (AIMessage(content="RECOVERED"), False)
    msgs = [HumanMessage(content="please do it")]
    out = a._salvage_empty_worker_turn(msgs, {}, object())
    assert out == "RECOVERED"
    # The recovered message is appended for saving.
    assert any(getattr(m, "content", "") == "RECOVERED" for m in msgs)


def test_worker_salvage_falls_back_when_retry_still_empty():
    import contextlib
    import io

    a = _salvage_agent()
    a._stream_response = lambda *args, **kw: (AIMessage(content=""), False)
    msgs = [HumanMessage(content="please do it")]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = a._salvage_empty_worker_turn(msgs, {}, object())
    # Never empty — a visible fallback is returned AND printed.
    assert out
    assert "wasn't able to produce" in out
    assert "wasn't able to produce" in buf.getvalue()


# --- Auto-continue on output-token truncation --------------------------------
# A turn cut off by MAX_TOKENS (reasoning + partial answer/tool call) must
# auto-continue — feed the partial back and resume — instead of dead-ending and
# forcing the user to type "continue".


def _truncated(content=""):
    m = AIMessage(content=content)
    m.response_metadata = {"stop_reason": "max_tokens"}
    return m


def _complete(content="", tool_calls=None):
    m = AIMessage(content=content)
    m.response_metadata = {"stop_reason": "end_turn"}
    if tool_calls:
        m.tool_calls = tool_calls
    return m


def _continue_agent(retries=3):
    a = _agent()
    a._start_spinner = lambda label="Thinking": None
    a._stop_spinner = lambda: None
    a._max_continue_retries = retries
    a._capture_input_tokens = lambda r: None
    a._extract_thinking = lambda r: None
    return a


def test_continue_resumes_and_returns_assembled_answer():
    # Partial visible text, then one continuation that finishes cleanly. Parts are
    # glued directly (each is _extract_visible-stripped): a truncated stream
    # resumes at the exact character, so a space would corrupt a split word.
    a = _continue_agent()
    a._stream_response = lambda *args, **kw: (_complete(content="world."), False)
    out = a._continue_truncated_turn(
        ["m"], _truncated("Hello "), "Hello", None, object(), {}
    )
    assert out is not None
    assert out["messages"][0].content == "Helloworld."


def test_continue_stops_early_on_tool_call():
    # A continuation that emits a tool call is returned so the graph runs it.
    a = _continue_agent()
    tc = [{"name": "fs_read", "args": {"path": "x"}, "id": "1"}]
    a._stream_response = lambda *args, **kw: (_complete(tool_calls=tc), False)
    out = a._continue_truncated_turn(
        ["m"], _truncated("partial"), "partial", None, object(), {}
    )
    assert out is not None
    assert out["messages"][0].tool_calls == tc
    # The partial text gathered so far is preserved on the tool-call turn.
    assert out["messages"][0].content == "partial"


def test_continue_loops_until_not_truncated():
    # Two truncated continuations, then a clean finish — assembled across all.
    a = _continue_agent(retries=5)
    seq = [_truncated("B"), _truncated("C"), _complete(content="D")]
    calls = {"n": 0}

    def _stream(*args, **kw):
        r = seq[calls["n"]]
        calls["n"] += 1
        return (r, False)

    a._stream_response = _stream
    out = a._continue_truncated_turn(["m"], _truncated("A"), "A", None, object(), {})
    assert out["messages"][0].content == "ABCD"
    assert calls["n"] == 3


def test_continue_gives_up_after_cap_keeps_partial():
    # Always truncated: after the cap, whatever was assembled is still returned
    # (never a dead-end), so the user sees partial progress, not a blank stop.
    a = _continue_agent(retries=2)
    a._stream_response = lambda *args, **kw: (_truncated("more"), False)
    out = a._continue_truncated_turn(["m"], _truncated("start"), "start", None, object(), {})
    assert out is not None
    assert out["messages"][0].content.startswith("startmore")


def test_continue_returns_none_when_nothing_usable():
    # No partial text and continuations yield nothing → None (caller falls back).
    a = _continue_agent(retries=2)
    a._stream_response = lambda *args, **kw: (_truncated(""), False)
    out = a._continue_truncated_turn(["m"], _truncated(""), "", None, object(), {})
    assert out is None


def test_continue_handles_context_overflow_gracefully():
    from mnemoai.client.agent.agent import _ContextOverflow

    a = _continue_agent(retries=3)

    def _boom(*args, **kw):
        raise _ContextOverflow("too big")

    a._stream_response = _boom
    # Partial text exists → returned despite the overflow on continuation.
    out = a._continue_truncated_turn(["m"], _truncated("kept"), "kept", None, object(), {})
    assert out is not None
    assert out["messages"][0].content == "kept"
