"""Regression: cancelling a turn mid-flight must roll the whole turn out of
history — the user message AND any partial work — so the cancelled request
doesn't linger in agent.messages and get answered (out of context) next turn.

The UI cancels by injecting KeyboardInterrupt into the worker thread, which
surfaces inside graph.invoke(); invoke() catches it, truncates _messages back to
the pre-turn length, and re-raises (client.query turns that into "cancelled").
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from mnemoai.client.agent.agent import LangGraphAgent


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
    a._last_input_tokens = 123  # should be reset on rollback
    a.graph = graph
    a._stop_spinner = lambda: None
    a._extract_visible = lambda c: c if isinstance(c, str) else ""
    return a


class TestCancelRollback:
    def test_cancelled_turn_leaves_no_message_in_history(self):
        a = _agent(_CancelGraph())
        with pytest.raises(KeyboardInterrupt):
            a.invoke("what's the world cup result?")
        # The cancelled user turn was rolled back — history is empty.
        assert a._messages == []

    def test_next_turn_does_not_see_cancelled_message(self):
        # turn 1 cancelled, turn 2 succeeds — the model must see ONLY turn 2.
        a = _agent(_CancelGraph())
        with pytest.raises(KeyboardInterrupt):
            a.invoke("cancelled question")

        seen = {}
        a.graph = _OkGraph()
        # capture what the second turn's model call receives
        orig = a.graph.invoke

        def _capture(state, config=None):
            seen["humans"] = [
                m.content for m in state["messages"]
                if isinstance(m, HumanMessage)
            ]
            return orig(state, config)

        a.graph.invoke = _capture
        a.invoke("hello, who are you?")
        assert seen["humans"] == ["hello, who are you?"]
        assert not any("cancelled question" in h for h in seen["humans"])

    def test_rollback_preserves_prior_completed_turns(self):
        # A completed turn 1 stays; a cancelled turn 2 rolls back to just turn 1.
        a = _agent(_OkGraph())
        a.invoke("first question")  # completes → kept
        assert len([m for m in a._messages if isinstance(m, HumanMessage)]) == 1

        a.graph = _CancelGraph()
        with pytest.raises(KeyboardInterrupt):
            a.invoke("second question (cancelled)")
        # Only turn 1 remains; turn 2's user message is gone.
        humans = [m.content for m in a._messages if isinstance(m, HumanMessage)]
        assert humans == ["first question"]

    def test_last_input_tokens_reset_on_cancel(self):
        a = _agent(_CancelGraph())
        with pytest.raises(KeyboardInterrupt):
            a.invoke("x")
        assert a._last_input_tokens is None

    def test_successful_turn_still_persists(self):
        # Sanity: the rollback path doesn't affect a normal completed turn.
        a = _agent(_OkGraph())
        a.invoke("normal question")
        humans = [m.content for m in a._messages if isinstance(m, HumanMessage)]
        ais = [m for m in a._messages if isinstance(m, AIMessage)]
        assert humans == ["normal question"] and len(ais) == 1
