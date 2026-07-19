"""Unit tests for mid-turn steering (folding a message into the running turn).

Distinct from STEERING.md (user-authored always-on instructions, see
test_steering.py). This covers the runtime queue: the user types WHILE a turn
runs, the message is queued on the agent and drained at the next tool-round
boundary as a wrapped HumanMessage, so the model addresses it without the turn
ending.
"""

import threading

from langchain_core.messages import HumanMessage

from mnemoai.client.agent.agent import LangGraphAgent


def _agent():
    a = LangGraphAgent.__new__(LangGraphAgent)
    a._steer_queue = []
    a._steer_lock = threading.Lock()
    return a


class TestSteerEnqueue:
    def test_steer_enqueues_text(self):
        a = _agent()
        a.steer("look at file X too")
        assert a._steer_queue == ["look at file X too"]
        assert a._has_steering()

    def test_blank_is_ignored(self):
        a = _agent()
        a.steer("   ")
        a.steer("")
        assert a._steer_queue == []
        assert not a._has_steering()

    def test_multiple_preserve_order(self):
        a = _agent()
        a.steer("first")
        a.steer("second")
        assert a._steer_queue == ["first", "second"]

    def test_thread_safe_enqueue(self):
        a = _agent()

        def _push(n):
            for i in range(50):
                a.steer(f"{n}-{i}")

        threads = [threading.Thread(target=_push, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(a._steer_queue) == 200  # nothing lost under contention


class TestDrainSteering:
    def test_drain_returns_wrapped_human_messages(self):
        a = _agent()
        a.steer("also check the tests")
        drained = a._drain_steering()
        assert len(drained) == 1
        msg = drained[0]
        assert isinstance(msg, HumanMessage)
        assert "also check the tests" in msg.content
        # Wrapped with the "new message while you were working" framing.
        assert "while you were working" in msg.content.lower()
        assert "address" in msg.content.lower()

    def test_drain_empties_the_queue(self):
        a = _agent()
        a.steer("x")
        a._drain_steering()
        assert a._steer_queue == []
        assert a._drain_steering() == []  # second drain is empty

    def test_drain_empty_returns_empty_list(self):
        a = _agent()
        assert a._drain_steering() == []

    def test_drain_preserves_order(self):
        a = _agent()
        a.steer("one")
        a.steer("two")
        drained = a._drain_steering()
        assert "one" in drained[0].content
        assert "two" in drained[1].content

    def test_bare_agent_without_lock_degrades(self):
        # A bare object (no _steer_lock) still works — steer + drain fall back to
        # lock-free paths rather than crashing.
        a = LangGraphAgent.__new__(LangGraphAgent)
        a._steer_queue = []
        a.steer("no lock here")
        assert a._has_steering()
        drained = a._drain_steering()
        assert len(drained) == 1 and "no lock here" in drained[0].content


class TestClearSteering:
    """On cancel, steering queued into the aborted turn must be discarded — else
    it leaks into the next turn (the model answers a cancelled question)."""

    def test_clear_discards_pending(self):
        a = _agent()
        a.steer("what can you do?")
        assert a._has_steering()
        a.clear_steering()
        assert not a._has_steering()
        assert a._drain_steering() == []

    def test_clear_is_idempotent_and_safe_when_empty(self):
        a = _agent()
        a.clear_steering()  # nothing queued — must not raise
        assert a._drain_steering() == []

    def test_cleared_message_does_not_leak_to_next_drain(self):
        # Simulate: message steered into turn 1, turn 1 cancelled (clear), then
        # turn 2 drains — it must NOT see turn 1's message.
        a = _agent()
        a.steer("cancelled-turn question")
        a.clear_steering()  # turn 1 cancelled
        a.steer("next-turn question")  # turn 2's own steer
        drained = a._drain_steering()
        assert len(drained) == 1
        assert "next-turn question" in drained[0].content
        assert "cancelled-turn question" not in drained[0].content

    def test_clear_without_lock_degrades(self):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a._steer_queue = ["x"]
        a.clear_steering()  # no _steer_lock attr → lock-free path
        assert a._steer_queue == []


class TestExecuteToolsInjectsSteering:
    """_execute_tools appends drained steering AFTER the tool results, so the next
    model call sees it and the graph loops back to the agent (turn continues)."""

    def _agent(self):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a._steer_queue = []
        a._steer_lock = threading.Lock()
        a.verbose = False
        a.callbacks = []
        a.tools = []
        a._start_spinner = lambda *x, **k: None
        a._stop_spinner = lambda *x, **k: None
        a._get_route_tools = lambda state: []
        a._normalize_tool_args = lambda args: args
        a._print_tool_marker = lambda tc: None
        return a

    def test_no_tool_calls_returns_empty(self):
        from langchain_core.messages import AIMessage

        a = self._agent()
        state = {"messages": [AIMessage(content="done")]}  # no tool_calls
        out = a._execute_tools(state)
        assert out == {"messages": []}

    def test_steering_appended_after_tool_results(self):
        from langchain_core.messages import AIMessage

        a = self._agent()
        # A tool call that resolves to "tool not found" (no real tool needed).
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "nope", "id": "t1", "args": {}}],
        )
        a.steer("actually, also update the README")
        out = a._execute_tools({"messages": [ai]})
        msgs = out["messages"]
        # Last message is the steered HumanMessage, appended after the tool result.
        assert isinstance(msgs[-1], HumanMessage)
        assert "update the README" in msgs[-1].content
        # The tool result still precedes it (pairing preserved).
        assert msgs[0].__class__.__name__ == "ToolMessage"
        # Queue drained.
        assert a._steer_queue == []
