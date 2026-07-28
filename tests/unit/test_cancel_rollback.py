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


class TestRecursionLimitKeepsTheWork:
    """Hitting the safety step limit must PRESERVE the turn's work.

    `GraphRecursionError` carries no state and `graph.invoke()` returns nothing,
    so the turn used to be discarded entirely: every tool call and assistant
    message vanished, and the user's follow-up ("continue", "run the tests")
    arrived with no record that any of the work had happened. The graph is now
    STREAMED (`stream_mode="values"`), so the last snapshot before the limit is
    the work so far and gets committed exactly like a completed turn's.

    Only a turn that produced NOTHING falls back to the interrupted marker.
    """

    def _looping_agent(self, limit=4):
        import operator
        from typing import Annotated, TypedDict

        from langgraph.graph import StateGraph

        class _S(TypedDict):
            messages: Annotated[list, operator.add]
            thinking: object

        def _work(state):
            n = sum(1 for m in state["messages"] if isinstance(m, AIMessage))
            return {"messages": [AIMessage(content=f"work {n}")], "thinking": None}

        g = StateGraph(_S)
        g.add_node("a", _work)
        g.set_entry_point("a")
        g.add_conditional_edges("a", lambda s: "a", {"a": "a"})

        a = _agent(g.compile())
        a.recursion_limit = limit
        a._emit_answer = lambda m: None
        a._last_tool_result = lambda m: ""
        return a

    def test_work_done_before_the_limit_survives(self):
        a = self._looping_agent()
        a.invoke("big task")
        produced = [
            m.content for m in a._messages if isinstance(m, AIMessage)
        ]
        assert produced, "the turn's work was discarded"
        assert any("work 0" in c for c in produced)

    def test_user_message_survives(self):
        a = self._looping_agent()
        a.invoke("big task")
        humans = [m.content for m in a._messages if isinstance(m, HumanMessage)]
        assert humans == ["big task"]

    def test_next_turn_sees_the_work(self):
        # The whole point: "continue" must arrive with the pre-limit work visible.
        a = self._looping_agent()
        a.invoke("refactor everything")

        seen = {}
        a.graph = _OkGraph()
        orig = a.graph.invoke

        def _capture(state, config=None):
            seen["contents"] = [str(m.content) for m in state["messages"]]
            return orig(state, config)

        a.graph.invoke = _capture
        a.invoke("continue")
        joined = "\n".join(seen["contents"])
        assert "work 0" in joined and "refactor everything" in joined

    def test_reply_explains_the_limit_was_hit(self):
        a = self._looping_agent()
        out = a.invoke("big task")
        assert "step limit" in out and "RECURSION_LIMIT" in out

    def test_previous_turns_answer_is_never_reused(self):
        # With the work preserved this can't happen, but assert it directly:
        # _last_visible_from must be scoped to THIS turn, not the whole history.
        a = _agent(_OkGraph())
        a.invoke("earlier question")  # leaves "done" in history
        looping = self._looping_agent()
        a.graph = looping.graph
        a.recursion_limit = 4
        a._emit_answer = lambda m: None
        a._last_tool_result = lambda m: ""
        out = a.invoke("the big task")
        assert "done" not in out

    def test_turn_with_no_output_is_closed_with_the_marker(self):
        # A limit hit that produced nothing still must not leave a dangling
        # user message, or the next turn re-answers it out of context.
        from langgraph.errors import GraphRecursionError

        class _Boom:
            def invoke(self, state, config=None):
                raise GraphRecursionError("limit")

            def stream(self, state, config=None, stream_mode=None):
                raise GraphRecursionError("limit")

        a = _agent(_Boom())
        a._emit_answer = lambda m: None
        a.invoke("a task")
        assert a._messages[-1].content == INTERRUPTED_MARKER

    def test_stale_token_count_is_reset(self):
        a = self._looping_agent()
        a._last_input_tokens = 999
        a.invoke("x")
        assert a._last_input_tokens is None


class TestErroredTurnIsStillRecorded:
    """A turn killed by a mid-flight ERROR must still reach the session log.

    Whatever the turn produced stays in live `agent.messages` and the user keeps
    talking, so a transcript that skips it drifts out of sync with the
    conversation on screen. Observed in the wild: a dropped provider connection
    ~30 minutes in meant the log stopped at the last SUCCESSFUL turn while the
    chat ran on for another hour — `--resume` restored 64 of 408 messages, even
    though `/save` had all of them. Silent, because `_log_turn` swallows errors.
    """

    class _Log:
        def __init__(self):
            self.turns = []

        def log_turn(self, messages):
            self.turns.append([str(m.content) for m in messages])

    class _Boom:
        def stream(self, state, config=None, stream_mode=None):
            raise RuntimeError("bedrock connection drop")
            yield

    class _PartialBoom:
        """Streams one snapshot, then the connection dies."""

        def stream(self, state, config=None, stream_mode=None):
            yield {
                "messages": list(state["messages"]) + [AIMessage(content="partial work")],
                "thinking": None,
            }
            raise RuntimeError("drop mid-stream")

    def _agent_with_log(self, graph):
        a = _agent(graph)
        a._emit_answer = lambda m: None
        a.session_log = self._Log()
        return a

    def test_prompt_is_logged_when_the_turn_dies_immediately(self):
        a = self._agent_with_log(self._Boom())
        with pytest.raises(RuntimeError):
            a.invoke("please analyze this codebase")
        assert a.session_log.turns, "the errored turn was never recorded"
        assert "please analyze this codebase" in a.session_log.turns[0]

    def test_partial_work_is_logged_when_the_stream_dies(self):
        a = self._agent_with_log(self._PartialBoom())
        with pytest.raises(RuntimeError):
            a.invoke("do the big task")
        logged = a.session_log.turns[0]
        assert "do the big task" in logged
        assert "partial work" in logged

    def test_the_log_matches_live_history(self):
        # The invariant behind the bug: transcript and history must not diverge.
        a = self._agent_with_log(self._PartialBoom())
        with pytest.raises(RuntimeError):
            a.invoke("do the big task")
        assert a.session_log.turns[0] == [str(m.content) for m in a._messages]

    def test_the_error_still_propagates(self):
        # Logging must not swallow the failure the caller has to handle.
        a = self._agent_with_log(self._Boom())
        with pytest.raises(RuntimeError, match="bedrock connection drop"):
            a.invoke("q")

    def test_a_later_turn_still_logs_normally(self):
        a = self._agent_with_log(self._Boom())
        with pytest.raises(RuntimeError):
            a.invoke("the failed question")
        a.graph = _OkGraph()
        a.invoke("the next question")
        assert len(a.session_log.turns) == 2
        assert "the next question" in a.session_log.turns[1]

    def test_stale_token_count_is_reset(self):
        a = self._agent_with_log(self._Boom())
        a._last_input_tokens = 999
        with pytest.raises(RuntimeError):
            a.invoke("q")
        assert a._last_input_tokens is None
