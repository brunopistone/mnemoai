"""A turn that dies must leave a RECORD and name a way out.

Two defects are pinned here, each of which made a failed turn worse than the
failure itself:

* the user's prompt reached live history but not the transcript, because
  ``_commit_turn`` returned before logging when the turn produced nothing — so
  the two disagreed until the next ``--resume``;
* nothing marked the turn as failed, leaving history ending on a dangling user
  message that the provider adapters MERGE with the next prompt.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from mnemoai.client.agent import turn_failure
from mnemoai.client.agent.agent import LangGraphAgent


class _Log:
    def __init__(self):
        self.turns = []

    def log_turn(self, messages):
        self.turns.append([str(m.content) for m in messages])


class _OkGraph:
    def invoke(self, state, config=None):
        return {
            "messages": list(state["messages"]) + [AIMessage(content="done")],
            "thinking": None,
        }


class _SeededBoom:
    """The shape that hid the bug: `stream_mode="values"` yields the SEEDED state
    before any node runs, so a turn that dies on its first model call arrives at
    `_commit_turn` with a truthy result and no new messages."""

    def stream(self, state, config=None, stream_mode=None):
        yield {"messages": list(state["messages"]), "thinking": None}
        raise ValueError("No generation chunks were returned")


def _agent(graph=None):
    a = LangGraphAgent.__new__(LangGraphAgent)
    a._messages = []
    a.system_prompt = ""
    a.recursion_limit = 50
    a._thinking = None
    a._last_input_tokens = None
    a.graph = graph
    a.session_log = _Log()
    a._stop_spinner = lambda: None
    a._emit_answer = lambda m: None
    a._extract_visible = lambda c: c if isinstance(c, str) else ""
    return a


class TestFailureMarkerText:
    def test_marker_names_the_exception_class(self):
        class ValidationException(Exception):
            pass

        marker = turn_failure.failure_marker(ValidationException("a long body"))
        assert "ValidationException" in marker

    def test_marker_accepts_an_exception_or_a_resolved_name(self):
        # The diagnostic probe can name the failure better than the exception that
        # escaped, so a caller must be able to pass the better name directly.
        assert turn_failure.failure_marker(ValueError("x")) == turn_failure.failure_marker(
            "ValueError"
        )

    def test_marker_carries_no_provider_prose(self):
        # It IS sent to the model, so a provider's error body must stay out of it.
        body = "ValidationException: " + "x" * 5000
        marker = turn_failure.failure_marker(RuntimeError(body))
        assert "x" * 50 not in marker
        assert len(marker) < 120

    def test_a_nameless_failure_still_produces_a_marker(self):
        assert turn_failure.is_failure_marker(turn_failure.failure_marker(""))

    def test_recognizes_its_own_marker_and_nothing_else(self):
        assert turn_failure.is_failure_marker(turn_failure.failure_marker("Boom"))
        assert not turn_failure.is_failure_marker("Turn failed, sorry")
        assert not turn_failure.is_failure_marker(None)
        assert not turn_failure.is_failure_marker(12)


class TestFailedTurnReachesTheTranscript:
    def test_a_turn_that_produced_nothing_is_still_logged(self):
        # The defect, at its own level: `stream_mode="values"` yields the SEEDED
        # state before any node runs, so a turn that died on its first model call
        # reached _commit_turn with a truthy result and no new messages — and the
        # early return skipped _log_turn, leaving the prompt in live history and
        # absent from the transcript.
        a = _agent()
        a._messages = [HumanMessage(content="q")]
        a._turn_seed_len = 1
        seeded = {"messages": [HumanMessage(content="q")], "thinking": None}
        assert a._commit_turn(seeded, [HumanMessage(content="q")]) == []
        assert a.session_log.turns == [["q"]], "the turn never reached the transcript"

    def test_log_false_buffers_without_writing(self):
        # The failure path appends its marker AFTER committing, then writes one
        # record for the whole turn.
        a = _agent()
        turn_log = [HumanMessage(content="q")]
        result = {"messages": [HumanMessage(content="q"), AIMessage(content="partial")]}
        a._turn_seed_len = 1
        a._commit_turn(result, turn_log, log=False)
        assert a.session_log.turns == []
        assert [str(m.content) for m in turn_log] == ["q", "partial"]

    def test_prompt_is_logged_when_the_turn_dies_immediately(self):
        a = _agent(_SeededBoom())
        with pytest.raises(ValueError):
            a.invoke("summarize the customization scripts")
        assert a.session_log.turns, "the failed turn never reached the transcript"
        assert "summarize the customization scripts" in a.session_log.turns[0]

    def test_transcript_matches_live_history_exactly(self):
        a = _agent(_SeededBoom())
        with pytest.raises(ValueError):
            a.invoke("q")
        assert a.session_log.turns[0] == [str(m.content) for m in a._messages]

    def test_one_record_for_one_turn(self):
        # The marker rides in the SAME record: anything counting records reads two
        # as two turns (the picker's turn count, `discard_if_empty`).
        a = _agent(_SeededBoom())
        with pytest.raises(ValueError):
            a.invoke("q")
        assert len(a.session_log.turns) == 1

    def test_a_step_limit_hit_with_no_work_writes_one_record(self):
        # Same invariant on the neighbouring path: the limit branch appends its own
        # marker, so committing must NOT also log or the turn is recorded twice.
        from langgraph.errors import GraphRecursionError

        class _SeededLimit:
            def stream(self, state, config=None, stream_mode=None):
                yield {"messages": list(state["messages"]), "thinking": None}
                raise GraphRecursionError("limit")

        a = _agent(_SeededLimit())
        a._last_tool_result = lambda m: ""
        a.invoke("a big task")
        assert len(a.session_log.turns) == 1
        assert a.session_log.turns[0] == [str(m.content) for m in a._messages]

    def test_a_successful_turn_is_unaffected(self):
        a = _agent(_OkGraph())
        a.invoke("hello")
        assert a.session_log.turns == [["hello", "done"]]


class TestFailedTurnIsClosedOut:
    def test_marker_is_appended_to_history(self):
        a = _agent(_SeededBoom())
        with pytest.raises(ValueError):
            a.invoke("do the thing")
        last = a._messages[-1]
        assert isinstance(last, AIMessage)
        assert turn_failure.is_failure_marker(str(last.content))

    def test_the_prompt_survives(self):
        a = _agent(_SeededBoom())
        with pytest.raises(ValueError):
            a.invoke("the question that failed")
        humans = [m.content for m in a._messages if isinstance(m, HumanMessage)]
        assert humans == ["the question that failed"]

    def test_the_next_prompt_is_not_glued_to_the_failed_one(self):
        # The reason a marker is needed at all: the provider adapters MERGE
        # consecutive user messages (langchain-aws joins two HumanMessages into one
        # Bedrock `user` block), so without something between them the failed
        # prompt silently becomes a prefix of the next question.
        a = _agent(_SeededBoom())
        with pytest.raises(ValueError):
            a.invoke("first question (failed)")
        a.graph = _OkGraph()
        seen = {}

        def _capture(state, config=None):
            seen["messages"] = list(state["messages"])
            return _OkGraph().invoke(state, config)

        a.graph.invoke = _capture
        a.invoke("second question")

        kinds = [type(m).__name__ for m in seen["messages"]]
        first_human = kinds.index("HumanMessage")
        assert kinds[first_human + 1] != "HumanMessage", (
            f"two adjacent user messages would be merged by the provider: {kinds}"
        )

    def test_the_model_can_see_that_the_turn_failed(self):
        a = _agent(_SeededBoom())
        with pytest.raises(ValueError):
            a.invoke("q1")
        a.graph = _OkGraph()
        seen = {}
        a.graph.invoke = lambda state, config=None: (
            seen.setdefault("t", [str(m.content) for m in state["messages"]]),
            _OkGraph().invoke(state, config),
        )[1]
        a.invoke("q2")
        assert any(turn_failure.is_failure_marker(c) for c in seen["t"])

    def test_marker_is_never_surfaced_as_an_answer(self):
        a = _agent(_SeededBoom())
        with pytest.raises(ValueError):
            a.invoke("q")
        assert a._last_visible_from(a._messages) == ""

    def test_a_real_partial_answer_still_wins_over_the_marker(self):
        a = _agent()
        a._messages = [
            HumanMessage(content="q"),
            AIMessage(content="the partial answer"),
            AIMessage(content=turn_failure.failure_marker("Boom")),
        ]
        assert a._last_visible_from(a._messages) == "the partial answer"

    def test_two_failures_each_close_their_own_turn(self):
        a = _agent(_SeededBoom())
        for q in ("first", "second"):
            with pytest.raises(ValueError):
                a.invoke(q)
        kinds = [type(m).__name__ for m in a._messages]
        assert kinds == ["HumanMessage", "AIMessage"] * 2

    def test_stale_token_count_is_reset(self):
        a = _agent(_SeededBoom())
        a._last_input_tokens = 999
        with pytest.raises(ValueError):
            a.invoke("q")
        assert a._last_input_tokens is None

    def test_the_error_still_propagates(self):
        # Recording must not swallow the failure the caller has to report.
        a = _agent(_SeededBoom())
        with pytest.raises(ValueError, match="No generation chunks"):
            a.invoke("q")
