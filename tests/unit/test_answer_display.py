"""Unit tests for the answer-display safety net (no silent turns).

The app's display contract is "streaming prints the answer": a turn's answer is
normally shown live as it streams through ``_stream_once``/``CodeFormatter``. But
several paths produce the final answer text WITHOUT streaming — the orchestrator
single-subtask result, the aggregation-fallback concatenation, and the
context-overflow / stream-error / recursion terminal messages — and would
otherwise reach the user as a SILENT turn (only ``[Context: N]`` prints).

``invoke()`` tracks ``_answer_displayed`` (set True while streaming) and, at the
one convergence point where every path returns its final text, calls
``_emit_answer`` to display anything that wasn't streamed. These tests cover that
net directly (no LLM/graph needed) plus the end-to-end single-subtask case.
"""

import contextlib
import io

from langchain_core.messages import AIMessage

from mnemoai.client.agent.agent import LangGraphAgent


def _agent():
    a = LangGraphAgent.__new__(LangGraphAgent)
    a.styled_turn_view = False
    a.callbacks = []
    a._stop_spinner = lambda: None
    a._answer_displayed = False
    return a


def _capture(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


class TestEmitAnswerSafetyNet:
    def test_emits_when_not_streamed(self):
        a = _agent()
        out = _capture(lambda: a._emit_answer("The **answer** is 42."))
        assert "answer" in out and "42" in out
        assert "\033[36m●" in out          # cyan ● marker, like a streamed answer
        assert a._answer_displayed is True  # flips so it can't double-print

    def test_noop_when_already_streamed(self):
        # A normal streamed turn already set _answer_displayed — the net must NOT
        # re-print (which would duplicate the answer).
        a = _agent()
        a._answer_displayed = True
        assert _capture(lambda: a._emit_answer("do not print again")) == ""

    def test_noop_on_empty_text(self):
        a = _agent()
        assert _capture(lambda: a._emit_answer("")) == ""
        assert _capture(lambda: a._emit_answer(None)) == ""

    def test_styled_mode_commits_trailing_newline(self):
        a = _agent()
        a.styled_turn_view = True
        out = _capture(lambda: a._emit_answer("hi"))
        assert out.endswith("\n")  # final line committed to scrollback


class TestInvokeDisplaysNonStreamedAnswer:
    """End-to-end: a turn whose terminal AIMessage was produced WITHOUT streaming
    (the orchestrator single-subtask case) must be displayed by invoke()."""

    def _invoke_agent(self, terminal_text):
        a = _agent()
        a._messages = []
        a._cancel_event = None
        a.recursion_limit = 200
        a.system_prompt = None
        a._thinking = None
        a._strip_ephemeral = staticmethod(lambda t: t)
        a._extract_visible = lambda c: c if isinstance(c, str) else ""

        class _Graph:
            def invoke(self, state, config=None):
                # Mimic the orchestrator END state: a terminal AIMessage produced
                # quietly (never streamed), like the single-subtask branch.
                return {
                    "messages": state["messages"] + [AIMessage(content=terminal_text)],
                    "thinking": None,
                }

        a.graph = _Graph()
        return a

    def test_single_subtask_answer_is_displayed(self):
        a = self._invoke_agent("Hyperparameters: **lr=1e-4**, epochs=4.")
        out = _capture(lambda: setattr(self, "_ret", a.invoke("calc hyperparams")))
        assert "Hyperparameters" in out and "lr=1e-4" in out  # not a silent turn
        assert self._ret.startswith("Hyperparameters")
        assert a._answer_displayed is True

    def test_streamed_answer_not_double_printed(self):
        # If streaming already displayed the answer (flag True before invoke's
        # extraction), invoke must not print it again.
        a = self._invoke_agent("streamed answer body")

        # Simulate streaming having set the flag during graph.invoke.
        class _Graph:
            def invoke(_self, state, config=None):
                a._answer_displayed = True  # streaming happened
                return {
                    "messages": state["messages"]
                    + [AIMessage(content="streamed answer body")],
                    "thinking": None,
                }

        a.graph = _Graph()
        out = _capture(lambda: a.invoke("q"))
        assert "streamed answer body" not in out  # not re-printed by the net


class TestNoDoublePrintFromNonStreamedFallbacks:
    """The non-streamed fallback messages (_call_model truncated / reasoning-only,
    worker salvage) must NOT print themselves — they return the AIMessage and let
    the central net emit it exactly once. Prints at those sites + the net's
    re-emit would double the text. Guards against re-introducing an ad-hoc print."""

    def _invoke_with_terminal(self, text):
        a = _agent()
        a._messages = []
        a._cancel_event = None
        a.recursion_limit = 200
        a.system_prompt = None
        a._thinking = None
        a._strip_ephemeral = staticmethod(lambda t: t)
        a._extract_visible = lambda c: c if isinstance(c, str) else ""

        class _Graph:
            def invoke(_self, state, config=None):
                # A terminal AIMessage produced WITHOUT streaming (like the
                # truncated/reasoning-only/salvage fallbacks): flag stays False.
                return {
                    "messages": state["messages"] + [AIMessage(content=text)],
                    "thinking": None,
                }

        a.graph = _Graph()
        return a

    def test_fallback_text_printed_exactly_once(self):
        for text in (
            "My response was cut off by the output-token limit before I could finish.",
            "I wasn't able to produce a response for that.",
        ):
            a = self._invoke_with_terminal(text)
            out = _capture(lambda: a.invoke("q"))
            assert out.count(text[:30]) == 1, f"{text[:30]!r} not printed once: {out!r}"

    def test_call_model_fallbacks_do_not_print_directly(self):
        # The truncated / reasoning-only fallback branches must not carry a direct
        # print of the answer text (only _stream_once and _emit_answer display).
        import inspect

        from mnemoai.client.agent.agent import LangGraphAgent

        src = inspect.getsource(LangGraphAgent._call_model)
        # No print of a message .content in the fallback branches.
        assert "print(truncated.content" not in src
        assert "print(fallback.content" not in src
