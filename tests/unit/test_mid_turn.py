"""Unit tests: a message the user sends while a turn is already RUNNING.

The default is FIFO — the submission runs as its own turn afterwards — but when
the message is a correction of the work in flight ("also check the other file",
"no, in Italian") waiting means the turn spends minutes finishing the thing the
user just redirected. So the running turn gets first refusal and folds it in at
its next **drain point**.

The first attempt at this was removed in 1.8.0 because it drained only between
tool rounds: a message typed during the turn's FINAL, tool-call-free model call
was never drained and surfaced inside an unrelated later turn. These tests pin the
structure that fixes it — two drain points, an acceptance window that closes
atomically at the last one, and a reclaim so a cancelled turn hands the text back.

Pure logic: a `__new__` agent stub, no LLM, no graph run, no prompt_toolkit.
"""

import threading

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from mnemoai.client.agent import mid_turn
from mnemoai.client.agent.agent import LangGraphAgent


def _agent():
    """A bare agent with only the mid-turn fields `__init__` would have set."""
    a = LangGraphAgent.__new__(LangGraphAgent)
    a._mid_turn_queue = []
    a._mid_turn_lock = threading.Lock()
    a._mid_turn_open = False
    a._on_mid_turn_delivered = None
    return a


class _Tool:
    def __init__(self, name):
        self.name = name

    def invoke(self, args):
        return f"{self.name}-out"


def _tool_running_agent():
    """A mid-turn agent wired just enough to run the _execute_tools chokepoint."""
    a = _agent()
    a.verbose = False
    a.tools = [_Tool("fs_read")]
    a.tools_by_route = None
    a._cancelled = lambda: False
    a._start_spinner = lambda *x, **k: None
    a._stop_spinner = lambda *x, **k: None
    a._effective_route = lambda state: None
    a._run_spawn_batch = lambda tool_calls: {}
    a._is_blocked_by_plan_mode = lambda *x: False
    a._confirm_tool = lambda *x: True
    a._invoke_tool = lambda tool, name, args, quiet=False: tool.invoke(args)
    return a


class TestAcceptanceWindow:
    def test_nothing_is_accepted_before_a_turn_opens_the_window(self):
        # With no turn running there is no drain point to ride on, so the caller
        # must keep the text and run it as a turn of its own.
        a = _agent()
        assert mid_turn.accept(a, "hello") is False
        assert mid_turn.has_pending(a) is False

    def test_a_running_turn_accepts(self):
        a = _agent()
        mid_turn.open_window(a)
        assert mid_turn.accept(a, "also check the other file") is True
        assert mid_turn.has_pending(a) is True

    def test_empty_text_is_never_accepted(self):
        a = _agent()
        mid_turn.open_window(a)
        assert mid_turn.accept(a, "   ") is False
        assert mid_turn.accept(a, None) is False
        assert mid_turn.has_pending(a) is False

    def test_a_closed_window_refuses_but_keeps_what_is_pending(self):
        # close() stops accepting; it must NOT discard a message that arrived
        # before it, which still belongs to the caller (reclaim).
        a = _agent()
        mid_turn.open_window(a)
        mid_turn.accept(a, "first")
        mid_turn.close(a)
        assert mid_turn.accept(a, "second") is False
        assert mid_turn.reclaim(a) == ["first"]

    def test_close_if_empty_closes_only_when_nothing_is_pending(self):
        # The one critical section that makes turn end safe: a message accepted
        # after this point would have no drain point left.
        a = _agent()
        mid_turn.open_window(a)
        assert mid_turn.close_if_empty(a) is True
        assert mid_turn.accept(a, "too late") is False

        mid_turn.open_window(a)
        mid_turn.accept(a, "pending")
        assert mid_turn.close_if_empty(a) is False
        # Still open, because the turn owes a delivery and will drain it.
        assert mid_turn.accept(a, "one more") is True

    def test_opening_a_turn_discards_a_previous_turn_s_leftovers(self):
        # Those were reclaimed by the caller and re-queued; answering them inside
        # THIS turn would ask them twice.
        a = _agent()
        mid_turn.open_window(a)
        mid_turn.accept(a, "from the old turn")
        mid_turn.open_window(a)
        assert mid_turn.has_pending(a) is False

    def test_a_bare_stub_with_no_lock_still_works(self):
        # The collaborators tolerate an agent that never ran __init__ (the unit
        # tests' __new__ stubs), like cancellation/tool_loop do.
        a = LangGraphAgent.__new__(LangGraphAgent)
        mid_turn.open_window(a)
        assert mid_turn.accept(a, "x") is True
        assert [m.content for m in mid_turn.drain(a)][0].endswith("x")


class TestDrain:
    def test_everything_pending_folds_into_one_message(self):
        # The provider adapters merge consecutive user messages, so two would be
        # indistinguishable from one anyway.
        a = _agent()
        mid_turn.open_window(a)
        mid_turn.accept(a, "first")
        mid_turn.accept(a, "second")
        out = mid_turn.drain(a)
        assert len(out) == 1 and isinstance(out[0], HumanMessage)
        assert "first" in out[0].content and "second" in out[0].content

    def test_the_delivered_message_is_framed_for_the_model(self):
        a = _agent()
        mid_turn.open_window(a)
        mid_turn.accept(a, "in Italian")
        content = mid_turn.drain(a)[0].content
        assert f"<{mid_turn.BLOCK_TAG}>" in content
        assert f"</{mid_turn.BLOCK_TAG}>" in content

    def test_draining_empties_the_queue(self):
        a = _agent()
        mid_turn.open_window(a)
        mid_turn.accept(a, "once")
        mid_turn.drain(a)
        assert mid_turn.has_pending(a) is False
        assert mid_turn.drain(a) == []

    def test_the_delivery_notice_is_cosmetic(self):
        # The UI hook moves the row out of the pinned block and echoes it; a hook
        # that raises must not cost the user their message.
        a = _agent()
        seen = []
        a._on_mid_turn_delivered = lambda texts: seen.append(list(texts))
        mid_turn.open_window(a)
        mid_turn.accept(a, "noted")
        assert mid_turn.drain(a)
        assert seen == [["noted"]]

        a._on_mid_turn_delivered = lambda texts: 1 / 0
        mid_turn.accept(a, "still delivered")
        assert mid_turn.drain(a)[0].content.endswith("still delivered")


class TestReclaim:
    def test_a_cancelled_turn_hands_the_text_back(self):
        a = _agent()
        mid_turn.open_window(a)
        mid_turn.accept(a, "and in Italian")
        assert mid_turn.reclaim(a) == ["and in Italian"]
        assert mid_turn.has_pending(a) is False

    def test_reclaiming_closes_the_window(self):
        # The turn is over; anything accepted after this would never be drained.
        a = _agent()
        mid_turn.open_window(a)
        mid_turn.reclaim(a)
        assert mid_turn.accept(a, "later") is False

    def test_reclaim_is_empty_when_everything_was_delivered(self):
        a = _agent()
        mid_turn.open_window(a)
        mid_turn.accept(a, "done")
        mid_turn.drain(a)
        assert mid_turn.reclaim(a) == []


class TestStoredText:
    def test_the_framing_is_stripped_back_off(self):
        # History must keep what the USER typed, not the instruction the model
        # needed to read it correctly.
        a = _agent()
        mid_turn.open_window(a)
        mid_turn.accept(a, "no, in Italian")
        assert mid_turn.stored_text(mid_turn.drain(a)[0]) == "no, in Italian"

    def test_an_ordinary_prompt_is_not_a_mid_turn_message(self):
        # This is how _commit_turn tells the two apart: the seeded prompt is
        # already stored, so re-adding it would duplicate the question.
        assert mid_turn.stored_text(HumanMessage(content="hello")) == ""
        assert mid_turn.stored_text(SystemMessage(content="be nice")) == ""
        assert mid_turn.stored_text(AIMessage(content="hi")) == ""

    def test_a_non_string_content_is_tolerated(self):
        # A multimodal prompt's content is a list of blocks.
        assert mid_turn.stored_text(HumanMessage(content=[{"text": "hi"}])) == ""


class TestAgentWiring:
    """The agent's own surface: the two drain points and the graph route."""

    def test_accept_and_reclaim_are_exposed_to_the_ui(self):
        a = _agent()
        mid_turn.open_window(a)
        assert a.accept_mid_turn("look at X too") is True
        assert a.reclaim_mid_turn() == ["look at X too"]

    def test_the_deliver_node_returns_the_message(self):
        a = _agent()
        mid_turn.open_window(a)
        a.accept_mid_turn("one more thing")
        out = a._deliver_mid_turn({"messages": []})["messages"]
        assert len(out) == 1 and "one more thing" in out[0].content

    def test_a_tool_round_drains_after_every_tool_message(self):
        # An unanswered tool_call_id makes the provider reject the next call, so
        # the mid-turn message must come AFTER all of them.
        a = _tool_running_agent()
        mid_turn.open_window(a)
        a.accept_mid_turn("also in Italian")
        ai = AIMessage(
            content="",
            tool_calls=[
                {"name": "fs_read", "args": {}, "id": "c1"},
                {"name": "fs_read", "args": {}, "id": "c2"},
            ],
        )
        out = a._execute_tools({"messages": [ai]})["messages"]
        assert isinstance(out[-1], HumanMessage)
        assert "also in Italian" in out[-1].content
        assert [type(m).__name__ for m in out[:-1]] == ["ToolMessage", "ToolMessage"]

    def test_a_tool_round_adds_nothing_when_no_one_typed(self):
        a = _tool_running_agent()
        mid_turn.open_window(a)
        ai = AIMessage(
            content="", tool_calls=[{"name": "fs_read", "args": {}, "id": "c1"}]
        )
        out = a._execute_tools({"messages": [ai]})["messages"]
        assert [type(m).__name__ for m in out] == ["ToolMessage"]

    def test_turn_end_routes_to_the_deliver_node(self):
        # The drain point the 1.8.0 version lacked: no tool calls means this was
        # the LAST model call, so it is the last chance to deliver.
        a = _agent()
        a._cancelled = lambda: False
        mid_turn.open_window(a)
        a.accept_mid_turn("wait")
        state = {"messages": [AIMessage(content="all done")]}
        assert a._should_continue(state) == "deliver"

    def test_turn_end_ends_when_nothing_is_pending(self):
        a = _agent()
        a._cancelled = lambda: False
        mid_turn.open_window(a)
        state = {"messages": [AIMessage(content="all done")]}
        assert a._should_continue(state) == "end"
        # And the window is shut, so a later submission can't strand itself.
        assert a.accept_mid_turn("too late") is False

    def test_a_delivered_message_cannot_loop_the_graph(self):
        # The deliver node re-enters the model, which asks _should_continue again:
        # the queue is empty by then, so the turn ends. At most one extra call.
        a = _agent()
        a._cancelled = lambda: False
        mid_turn.open_window(a)
        a.accept_mid_turn("wait")
        state = {"messages": [AIMessage(content="all done")]}
        assert a._should_continue(state) == "deliver"
        a._deliver_mid_turn(state)
        assert a._should_continue(state) == "end"

    def test_a_cancelled_turn_ends_without_delivering(self):
        # The UI reclaims instead: a message steered into a cancelled turn must
        # neither vanish nor surface inside an unrelated later one.
        a = _agent()
        a._cancelled = lambda: True
        mid_turn.open_window(a)
        a.accept_mid_turn("wait")
        assert a._should_continue({"messages": [AIMessage(content="x")]}) == "end"
        assert a.reclaim_mid_turn() == ["wait"]


class TestCommitTurn:
    """A mid-turn message exists only inside the turn — it must be stored."""

    def _committer(self):
        a = _agent()
        a._messages = [HumanMessage(content="the original question")]
        a._turn_seed_len = 1
        a.session_log = None
        return a

    def test_it_is_stored_as_the_user_s_own_text(self):
        a = self._committer()
        mid_turn.open_window(a)
        a.accept_mid_turn("no, in Italian")
        delivered = mid_turn.drain(a)[0]
        result = {
            "messages": [
                HumanMessage(content="the original question"),
                AIMessage(content="partial"),
                delivered,
                AIMessage(content="in Italian now"),
            ]
        }
        added = a._commit_turn(result, [], log=False)
        stored = [m.content for m in added if isinstance(m, HumanMessage)]
        assert stored == ["no, in Italian"]  # framing stripped
        # In chronological position: between the two answers it sits between.
        assert [type(m).__name__ for m in added] == [
            "AIMessage",
            "HumanMessage",
            "AIMessage",
        ]

    def test_the_seeded_prompt_is_still_skipped(self):
        # It was already stored as the clean prompt; the reminder-bearing copy the
        # model ran on must not be re-added.
        a = self._committer()
        result = {
            "messages": [
                HumanMessage(content="the original question"),
                HumanMessage(content="<steering>rules</steering>\nask again"),
                AIMessage(content="answer"),
            ]
        }
        added = a._commit_turn(result, [], log=False)
        assert [type(m).__name__ for m in added] == ["AIMessage"]

    def test_the_same_words_twice_in_one_turn_are_two_messages(self):
        # Deliberately not dedup'd: a user who repeats themselves mid-turn sent
        # two real messages, and the dedup below is for MODEL output.
        a = self._committer()
        mid_turn.open_window(a)
        a.accept_mid_turn("hurry up")
        first = mid_turn.drain(a)[0]
        a.accept_mid_turn("hurry up")
        second = mid_turn.drain(a)[0]
        result = {
            "messages": [
                HumanMessage(content="the original question"),
                first,
                AIMessage(content="ok"),
                second,
                AIMessage(content="ok ok"),
            ]
        }
        added = a._commit_turn(result, [], log=False)
        assert [m.content for m in added if isinstance(m, HumanMessage)] == [
            "hurry up",
            "hurry up",
        ]
