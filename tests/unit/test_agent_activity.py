"""Unit tests for the hidden-agent live activity store (client/agent/agent_activity.py).

Pure state + threading — no LLM. Covers: sink events land on the right run,
snapshots are immutable copies (writers keep appending safely), the events ring is
bounded, finished-run eviction never drops a running run, concurrent writers from
many threads don't corrupt the store, and runs are stamped with the turn that
spawned them so a LIVE panel can show this turn's agents while every retained run
stays browsable. The last class drives the real ``invoke`` on a ``__new__`` agent
stub — without it the turn stamp could be wired nowhere and every other test here
would still pass.
"""

import threading

from mnemoai.client.agent.agent_activity import (
    EVENTS_PER_RUN,
    ActivitySink,
    AgentActivityStore,
    glance_runs,
)


class TestActivityBasics:
    def test_open_run_returns_sink_and_shows_running(self):
        store = AgentActivityStore()
        sink = store.open_run("explore", "Find Ray script", "spawn")
        assert isinstance(sink, ActivitySink)
        rows = store.snapshot()
        assert len(rows) == 1
        r = rows[0]
        assert r.agent_type == "explore"
        assert r.description == "Find Ray script"
        assert r.origin == "spawn"
        assert r.status == "running"
        assert store.any_active() is True

    def test_events_recorded_in_order_with_counts(self):
        store = AgentActivityStore()
        sink = store.open_run("explore", "d", "spawn")
        sink.tool_call("grep_search", {"pattern": "fully_shard"})
        sink.tool_result("grep_search", "3 matches")
        sink.tool_error("glob_search", "boom")
        r = store.snapshot()[0]
        kinds = [e.kind for e in r.events]
        assert kinds == ["tool_call", "tool_result", "tool_error"]
        assert r.tool_call_count() == 1
        assert r.events[0].args == {"pattern": "fully_shard"}

    def test_finish_freezes_status_and_elapsed(self):
        store = AgentActivityStore()
        sink = store.open_run("explore", "d", "spawn")
        sink.finish("done")
        r = store.snapshot()[0]
        assert r.status == "done"
        assert r.end is not None
        assert store.any_active() is False
        # elapsed is frozen at end, not "now"
        e1 = r.elapsed(now=r.end + 100)
        e2 = r.elapsed(now=r.end + 200)
        assert e1 == e2  # end is set, so `now` is ignored

    def test_finish_ok_records_final_and_marks_done(self):
        # The shared success-path transition used by every hidden run: a run that
        # returned normally records its final answer and flips to "done".
        store = AgentActivityStore()
        sink = store.open_run("explore", "d", "spawn")
        sink.finish_ok("the final report")
        r = store.snapshot()[0]
        assert r.status == "done"
        assert r.end is not None
        assert any(e.kind == "final" and e.text == "the final report" for e in r.events)

    def test_finish_ok_marks_stopped_when_cancelled(self):
        # If the run was cancelled mid-flight (x / stop-all), finish_ok must mark
        # it "stopped" (NOT "done") and must NOT record a final-answer event.
        store = AgentActivityStore()
        sink = store.open_run("explore", "d", "spawn")
        store.request_stop(store.snapshot()[0].run_id)  # UI asked it to stop
        sink.finish_ok("partial work")
        r = store.snapshot()[0]
        assert r.status == "stopped"
        assert not any(e.kind == "final" for e in r.events)

    def test_finish_if_running_is_idempotent(self):
        # Backstop for abnormal exit (e.g. a KeyboardInterrupt cancel): marks a
        # still-running run stopped WITHOUT clobbering a real done/failed status,
        # and freezes the timer (end set).
        store = AgentActivityStore()
        # (a) still running -> gets marked, end frozen
        a = store.open_run("explore", "a", "spawn")
        a.finish_if_running("failed")
        ra = store.snapshot()[0]
        assert ra.status == "failed" and ra.end is not None
        # (b) already done -> backstop is a no-op (doesn't overwrite)
        b = store.open_run("explore", "b", "spawn")
        b.finish("done")
        b.finish_if_running("failed")
        rb = store.snapshot()[-1]
        assert rb.status == "done"

    def test_none_store_sink_is_noop(self):
        # The main foreground turn / plain loop pass no sink; ActivitySink(None,...)
        # must be a safe no-op so the quiet loop is untouched.
        sink = ActivitySink(None, "x")
        sink.tool_call("t", {})
        sink.tool_result("t", "r")
        sink.tool_error("t", "e")
        sink.final("answer")
        sink.finish("done")  # must not raise
        sink.finish_if_running("failed")  # must not raise
        assert sink.is_cancelled() is False  # None store → never cancelled


class TestPerRunStop:
    """x / stop-all set a per-run cancel Event the worker polls — so ANY agent
    (foreground, background, orchestrator) can be stopped individually, even
    across turns (unlike the global turn-cancel, which resets each turn)."""

    def test_request_stop_sets_only_that_run(self):
        store = AgentActivityStore()
        a = store.open_run("explore", "a", "spawn")
        b = store.open_run("explore", "b", "background")
        assert store.request_stop(a._run_id) is True
        assert a.is_cancelled() is True
        assert b.is_cancelled() is False

    def test_request_stop_on_finished_returns_false(self):
        store = AgentActivityStore()
        a = store.open_run("explore", "a", "spawn")
        a.finish("done")
        assert store.request_stop(a._run_id) is False

    def test_request_stop_all_sets_every_running(self):
        store = AgentActivityStore()
        a = store.open_run("explore", "a", "spawn")
        b = store.open_run("explore", "b", "background")
        c = store.open_run("explore", "c", "orchestrator")
        c.finish("done")  # already finished → not counted
        n = store.request_stop_all()
        assert n == 2
        assert a.is_cancelled() and b.is_cancelled()

    def test_stop_reaches_live_run_via_snapshot_id(self):
        # The UI holds a SNAPSHOT (copy); stopping by its run_id must reach the
        # LIVE run's cancel Event (shared object), which the worker's sink polls.
        store = AgentActivityStore()
        live = store.open_run("explore", "a", "spawn")
        snap = store.snapshot()[0]
        store.request_stop(snap.run_id)
        assert live.is_cancelled() is True  # live sink sees the snapshot's stop

    def test_is_cancelling_window(self):
        # is_cancelling is True only between the stop request and the worker
        # actually finishing — drives the live "cancelling…" label. Snapshot
        # copies share the cancel Event, so they report it too.
        store = AgentActivityStore()
        sink = store.open_run("explore", "a", "spawn")
        assert store.snapshot()[0].is_cancelling() is False  # running, not stopped
        store.request_stop(sink._run_id)
        assert store.snapshot()[0].is_cancelling() is True  # stop requested
        sink.finish("stopped")
        assert store.snapshot()[0].is_cancelling() is False  # done → not cancelling
        assert store.snapshot()[0].status == "stopped"


class TestImmutabilityAndBounds:
    def test_snapshot_is_a_copy_writer_keeps_appending(self):
        store = AgentActivityStore()
        sink = store.open_run("explore", "d", "spawn")
        sink.tool_call("a", {})
        snap = store.snapshot()[0]
        n_before = len(snap.events)
        # Appending after the snapshot must NOT change the already-taken copy.
        sink.tool_call("b", {})
        assert len(snap.events) == n_before
        assert len(store.snapshot()[0].events) == n_before + 1

    def test_events_ring_is_bounded(self):
        store = AgentActivityStore()
        sink = store.open_run("explore", "d", "spawn")
        for i in range(EVENTS_PER_RUN + 50):
            sink.tool_call(f"t{i}", {})
        r = store.snapshot()[0]
        assert len(r.events) == EVENTS_PER_RUN  # oldest dropped

    def test_result_text_is_capped_but_final_is_not(self):
        store = AgentActivityStore()
        sink = store.open_run("explore", "d", "spawn")
        big = "x" * 5000
        sink.tool_result("t", big)
        sink.final(big)
        events = list(store.snapshot()[0].events)
        result_evt = next(e for e in events if e.kind == "tool_result")
        final_evt = next(e for e in events if e.kind == "final")
        assert len(result_evt.text) < len(big)  # capped
        assert final_evt.text == big  # full answer kept

    def test_eviction_drops_oldest_finished_never_running(self):
        store = AgentActivityStore(max_runs=3)
        # Two finished, then keep opening running ones over the cap.
        s0 = store.open_run("a", "0", "spawn")
        s0.finish("done")
        s1 = store.open_run("a", "1", "spawn")
        s1.finish("done")
        r2 = store.open_run("a", "2", "spawn")  # running
        r3 = store.open_run("a", "3", "spawn")  # running -> over cap (4>3)
        ids = {r.run_id for r in store.snapshot()}
        # The oldest FINISHED run (s0's) was evicted; both running ones survive.
        assert len(store.snapshot()) == 3
        # Every running run is still present.
        running_ids = {r.run_id for r in store.snapshot() if r.status == "running"}
        assert len(running_ids) == 2

    def test_all_running_over_cap_are_not_evicted(self):
        store = AgentActivityStore(max_runs=2)
        store.open_run("a", "0", "spawn")
        store.open_run("a", "1", "spawn")
        store.open_run("a", "2", "spawn")  # over cap but all running -> keep all
        assert len(store.snapshot()) == 3

    def test_get_returns_frozen_copy_or_none(self):
        store = AgentActivityStore()
        sink = store.open_run("explore", "d", "spawn")
        rid = store.snapshot()[0].run_id
        got = store.get(rid)
        assert got is not None and got.run_id == rid
        n = len(got.events)
        sink.tool_call("later", {})
        assert len(got.events) == n  # frozen copy unaffected
        assert store.get("nonexistent") is None

    def test_clear_drops_all(self):
        store = AgentActivityStore()
        store.open_run("a", "0", "spawn")
        store.clear()
        assert store.snapshot() == []


class TestTurnScoping:
    """The store keeps the whole session (a finished agent's report must stay
    readable), so a LIVE panel needs to know which runs belong to the turn that is
    running now — otherwise three turns of ✓ rows push the agents actually working
    out of the viewport, which is the one thing a glance is for."""

    def test_runs_are_stamped_with_the_current_turn(self):
        store = AgentActivityStore()
        store.open_run("explore", "before", "spawn")
        assert store.begin_turn() == 1
        store.open_run("explore", "first", "spawn")
        store.begin_turn()
        store.open_run("explore", "second", "spawn")
        assert [(r.description, r.turn) for r in store.snapshot()] == [
            ("before", 0),
            ("first", 1),
            ("second", 2),
        ]
        assert store.current_turn == 2

    def test_an_earlier_turns_finished_run_leaves_the_glance(self):
        store = AgentActivityStore()
        store.begin_turn()
        store.open_run("explore", "old", "spawn").finish("done")
        store.begin_turn()
        store.open_run("code", "new", "spawn")
        glance = glance_runs(store.snapshot(), store.current_turn)
        assert [r.description for r in glance] == ["new"]
        # …but nothing was dropped: the report is still reachable via Ctrl+A.
        assert [r.description for r in store.snapshot()] == ["old", "new"]

    def test_an_earlier_turns_running_run_stays(self):
        # A background sub-agent outlives the turn that spawned it, and one still
        # working is exactly what the panel is for.
        store = AgentActivityStore()
        store.begin_turn()
        store.open_run("explore", "background", "background")
        store.begin_turn()
        store.open_run("code", "current", "spawn")
        glance = glance_runs(store.snapshot(), store.current_turn)
        assert [r.description for r in glance] == ["background", "current"]

    def test_this_turns_finished_run_stays(self):
        # Within a turn the finished ones must stay listed, so a fan-out reads as
        # "2 done, 1 still going".
        store = AgentActivityStore()
        store.begin_turn()
        store.open_run("explore", "done-now", "spawn").finish("done")
        store.open_run("code", "going", "spawn")
        glance = glance_runs(store.snapshot(), store.current_turn)
        assert [r.description for r in glance] == ["done-now", "going"]

    def test_without_begin_turn_nothing_is_scoped_out(self):
        # The plain off-TTY loop and every caller that doesn't count turns keep
        # today's behavior: turn 0 everywhere, so the glance is the whole store.
        store = AgentActivityStore()
        store.open_run("a", "0", "spawn").finish("done")
        store.open_run("a", "1", "spawn").finish("done")
        assert len(glance_runs(store.snapshot(), store.current_turn)) == 2

    def test_clear_resets_the_turn(self):
        store = AgentActivityStore()
        store.begin_turn()
        store.clear()
        assert store.current_turn == 0

    def test_glance_tolerates_a_run_without_a_turn(self):
        # Pure and tolerant: anything shaped like a run is simply shown, so an
        # older/foreign object can't blank the panel.
        class _Bare:
            pass

        assert glance_runs([_Bare()], 3) == []  # no turn, not running → earlier
        running = _Bare()
        running.status = "running"
        assert glance_runs([running], 3) == [running]


class TestInvokeStampsTheTurn:
    """The wiring: ``LangGraphAgent.invoke`` begins a turn, so the panel scopes
    itself without every spawn site having to pass a turn number."""

    def _agent(self, store):
        from langchain_core.messages import AIMessage

        from mnemoai.client.agent.agent import LangGraphAgent

        class _Graph:
            def invoke(self, state, config=None):
                return {
                    "messages": list(state["messages"]) + [AIMessage(content="done")],
                    "thinking": None,
                }

        a = LangGraphAgent.__new__(LangGraphAgent)
        a._messages = []
        a.system_prompt = ""
        a.recursion_limit = 50
        a._thinking = None
        a._last_input_tokens = None
        a.graph = _Graph()
        a.session_log = None
        a._stop_spinner = lambda: None
        a._emit_answer = lambda m: None
        a._extract_visible = lambda c: c if isinstance(c, str) else ""
        a._activity = store
        return a

    def test_each_turn_advances_the_stamp(self):
        store = AgentActivityStore()
        a = self._agent(store)
        a.invoke("first question")
        store.open_run("explore", "turn-1 agent", "spawn").finish("done")
        a.invoke("second question")
        store.open_run("code", "turn-2 agent", "spawn")
        assert store.current_turn == 2
        glance = glance_runs(store.snapshot(), store.current_turn)
        assert [r.description for r in glance] == ["turn-2 agent"]

    def test_a_turn_runs_with_no_store_at_all(self):
        # The bump is tolerant on purpose: an agent built without an activity
        # store (off-TTY paths, test stubs) must still take a turn.
        a = self._agent(None)
        del a._activity
        assert a.invoke("hello") == "done"


class TestConcurrency:
    def test_concurrent_writers_do_not_corrupt(self):
        store = AgentActivityStore(max_runs=64)
        sinks = [store.open_run("a", str(i), "spawn") for i in range(8)]

        def hammer(sink):
            for _ in range(200):
                sink.tool_call("t", {"k": "v"})

        threads = [threading.Thread(target=hammer, args=(s,)) for s in sinks]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        rows = store.snapshot()
        assert len(rows) == 8
        # Each run capped at the ring size; total is consistent, no crash/torn read.
        for r in rows:
            assert len(r.events) == EVENTS_PER_RUN

    def test_on_change_fires_outside_lock_and_survives_exception(self):
        store = AgentActivityStore()
        calls = {"n": 0}

        def cb():
            calls["n"] += 1
            # Re-entering the store from the callback must not deadlock (fired
            # outside the lock), and a raising callback must not break the writer.
            store.any_active()
            raise RuntimeError("callback boom")

        store.on_change = cb
        sink = store.open_run("a", "d", "spawn")  # fires on_change
        sink.tool_call("t", {})  # fires again
        assert calls["n"] >= 2  # did not deadlock, did not propagate
