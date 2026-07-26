"""Regression: cancelling a turn mid-flight must CLOSE the turn out, not delete it.
So the user message STAYS and an explicit ``INTERRUPTED_MARKER`` assistant
message is appended: the turn reads as terminated (no silent resume) while
"continue" still has the context. The UI cancels by injecting KeyboardInterrupt
into the worker thread, which surfaces inside graph.invoke(); invoke() appends
the marker and re-raises (client.query turns that into "cancelled").
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from mnemoai.client.agent.agent import INTERRUPTED_MARKER, LangGraphAgent


class _OkGraph:
    def invoke(self, state, config=None):
        return {
            "messages": list(state["messages"]) + [AIMessage(content="done")],
            "thinking": None,
        }


class _CancelGraph:
    """Simulates a user cancel mid-turn (KeyboardInterrupt from graph.invoke)."""

    def invoke(self, state, config=None):
        raise KeyboardInterrupt()


def _agent(graph):
    a = LangGraphAgent.__new__(LangGraphAgent)
    a._messages = []
    a.system_prompt = ""
    a.recursion_limit = 50
    a._thinking = None
    a._last_input_tokens = 123  # should be reset on cancel
    a.graph = graph
    a._stop_spinner = lambda: None
    a._extract_visible = lambda c: c if isinstance(c, str) else ""
    return a


class TestCancelKeepsContext:
    def test_cancelled_question_survives_in_history(self):
        # The whole point: the user can say "continue" afterwards.
        a = _agent(_CancelGraph())
        with pytest.raises(KeyboardInterrupt):
            a.invoke("calculate the DPO hyperparameters")
        humans = [m.content for m in a._messages if isinstance(m, HumanMessage)]
        assert humans == ["calculate the DPO hyperparameters"]

    def test_interrupted_marker_is_appended(self):
        a = _agent(_CancelGraph())
        with pytest.raises(KeyboardInterrupt):
            a.invoke("do the thing")
        assert a._messages[-1].content == INTERRUPTED_MARKER
        assert isinstance(a._messages[-1], AIMessage)

    def test_next_turn_sees_both_the_question_and_the_marker(self):
        # "sorry, continue" must arrive with the cancelled request still visible
        # AND an explicit signal that the attempt was interrupted.
        a = _agent(_CancelGraph())
        with pytest.raises(KeyboardInterrupt):
            a.invoke("calculate the hyperparameters")

        seen = {}
        a.graph = _OkGraph()
        orig = a.graph.invoke

        def _capture(state, config=None):
            seen["contents"] = [str(m.content) for m in state["messages"]]
            return orig(state, config)

        a.graph.invoke = _capture
        a.invoke("Sorry, continue")

        joined = "\n".join(seen["contents"])
        assert "calculate the hyperparameters" in joined  # context preserved
        assert INTERRUPTED_MARKER in joined  # turn explicitly terminated
        assert "Sorry, continue" in joined

    def test_prior_completed_turns_are_untouched(self):
        a = _agent(_OkGraph())
        a.invoke("first question")
        a.graph = _CancelGraph()
        with pytest.raises(KeyboardInterrupt):
            a.invoke("second question (cancelled)")
        humans = [m.content for m in a._messages if isinstance(m, HumanMessage)]
        assert humans == ["first question", "second question (cancelled)"]

    def test_last_input_tokens_reset_on_cancel(self):
        a = _agent(_CancelGraph())
        with pytest.raises(KeyboardInterrupt):
            a.invoke("x")
        assert a._last_input_tokens is None

    def test_successful_turn_has_no_marker(self):
        a = _agent(_OkGraph())
        a.invoke("normal question")
        assert not any(
            INTERRUPTED_MARKER in str(m.content) for m in a._messages
        )

    def test_two_cancels_in_a_row_each_get_their_own_marker(self):
        a = _agent(_CancelGraph())
        for q in ("first", "second"):
            with pytest.raises(KeyboardInterrupt):
                a.invoke(q)
        markers = [m for m in a._messages if str(m.content) == INTERRUPTED_MARKER]
        assert len(markers) == 2
        # …and the pairing is user-then-marker, so neither turn reads as answered.
        assert [str(m.content) for m in a._messages] == [
            "first", INTERRUPTED_MARKER, "second", INTERRUPTED_MARKER,
        ]


class TestMarkerIsNotAnAnswer:
    def test_marker_is_never_surfaced_as_salvaged_text(self):
        # _last_visible_from salvages a cut-short answer for the user; the marker
        # is bookkeeping and must never be shown as if it were the reply.
        a = _agent(_CancelGraph())
        with pytest.raises(KeyboardInterrupt):
            a.invoke("q")
        assert a._last_visible_from(a._messages) == ""

    def test_real_partial_answer_still_salvaged_over_marker(self):
        a = _agent(_CancelGraph())
        a._messages = [
            HumanMessage(content="q"),
            AIMessage(content="here is the partial answer"),
            AIMessage(content=INTERRUPTED_MARKER),
        ]
        assert a._last_visible_from(a._messages) == "here is the partial answer"
