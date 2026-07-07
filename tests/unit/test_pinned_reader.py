"""Unit tests for the pinned-input REPL plumbing (PinnedPromptReader).

The pinned reader runs a real prompt_toolkit Application, which needs a TTY — so
these tests exercise the parts that DON'T need one: the queue→worker→exit control
flow (driven directly) and the accept-handler enqueue. The actual bottom-pinned
rendering (input at the bottom, output scrolling above via patch_stdout) can only
be verified on a real terminal. These protect FIFO order, one-at-a-time dispatch
(never concurrent), exit on _ExitRepl, and busy/pending bookkeeping.
"""

import asyncio
import io
import sys

import pytest

from mnemoai.client.ui.spinner import Spinner, SpinnerStatus, spinner_toolbar_text
from mnemoai.client.ui.tui import PinnedPromptReader, _ExitRepl


@pytest.fixture(autouse=True)
def _stub_run_in_terminal(monkeypatch):
    """Stub run_in_terminal as a completed Future (no live app in these tests).

    The worker awaits run_in_terminal to echo each line above the pinned region;
    without a running Application the real one blocks forever. Return an
    already-done Future so it's awaitable (worker) AND safe to ignore un-awaited
    (the fire-and-forget call in _request_cancel), running the callable inline.
    """

    class _DoneAwaitable:
        """Awaitable that resolves immediately, with no event loop required —
        awaitable by the _worker coroutine AND safe to drop un-awaited by the
        fire-and-forget calls in the sync keybinding handlers."""

        def __await__(self):
            return iter(())

    def _fake(func, *a, **k):
        try:
            func()
        except Exception:
            pass
        return _DoneAwaitable()

    monkeypatch.setattr("mnemoai.client.ui.tui.run_in_terminal", _fake)


class _FakeApp:
    """Stand-in for the prompt_toolkit Application: records exit, no rendering."""

    def __init__(self):
        self.exited = False

    def invalidate(self):
        pass

    def exit(self):
        self.exited = True


def _reader(dispatch):
    r = PinnedPromptReader(
        prompt_text=lambda: "> ",
        commands=[],
        dispatch=dispatch,
        toolbar_text=lambda: "",
    )
    # Swap the real Application for a fake so the worker can call exit()/invalidate
    # without a TTY. The queue is created by _drive() below.
    r._app = _FakeApp()
    return r


def _drive(reader, lines):
    """Enqueue lines via the accept handler, then run the worker to drain them.

    Mirrors the real flow: _on_accept enqueues (event-loop thread), _worker drains
    one at a time on a thread. Runs until the worker exits on _ExitRepl.
    """

    async def main():
        reader._queue = asyncio.Queue()
        for line in lines:

            class _Buff:
                text = line

            reader._on_accept(_Buff())
        await asyncio.wait_for(reader._worker(), timeout=5)

    asyncio.run(main())


def test_dispatches_in_fifo_order_until_exit():
    seen = []

    def dispatch(line):
        seen.append(line)
        return _ExitRepl if line == "/quit" else None

    _drive(_reader(dispatch), ["a", "b", "/quit"])
    assert seen == ["a", "b", "/quit"]


def test_exits_on_sentinel_and_app_exit_called():
    seen = []

    def dispatch(line):
        seen.append(line)
        return _ExitRepl if line == "/quit" else None

    reader = _reader(dispatch)
    _drive(reader, ["/quit", "after"])
    # Worker stops at /quit; "after" is enqueued but never dispatched.
    assert seen == ["/quit"]
    assert reader._app.exited is True


def test_dispatch_runs_one_at_a_time_never_concurrent():
    active = {"now": 0, "max": 0}

    def dispatch(line):
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
        for _ in range(1000):
            pass
        active["now"] -= 1
        return _ExitRepl if line == "/quit" else None

    _drive(_reader(dispatch), ["a", "b", "c", "/quit"])
    assert active["max"] == 1


def test_busy_flag_reflects_in_flight_turn():
    states = []

    def dispatch(line):
        states.append(reader.busy)  # worker sets busy=True before dispatch
        return _ExitRepl if line == "/quit" else None

    reader = _reader(dispatch)
    _drive(reader, ["/quit"])
    assert states == [True]
    assert reader.busy is False


def test_accept_handler_enqueues_and_tracks_pending():
    reader = _reader(lambda line: None)

    async def main():
        reader._queue = asyncio.Queue()

        class _Buff:
            text = "hello"

        # Returns False (clears the input); enqueues; bumps pending; shows live.
        assert reader._on_accept(_Buff()) is False
        assert reader.pending == 1
        assert reader._queued_lines == ["hello"]
        assert reader._queue.get_nowait() == "hello"

    asyncio.run(main())


def test_queued_line_shown_while_pending_then_cleared_on_dispatch():
    # A message submitted during a busy turn is listed in _queued_lines (shown
    # live in the pinned region) and removed once the worker starts it.
    seen_queue_at_dispatch = []

    def dispatch(line):
        # When each line runs, it must no longer be in the pending list.
        seen_queue_at_dispatch.append(list(reader._queued_lines))
        return _ExitRepl if line == "/quit" else None

    reader = _reader(dispatch)
    _drive(reader, ["first", "second", "/quit"])
    # At each dispatch, the just-started line is already removed from the list.
    assert "first" not in seen_queue_at_dispatch[0]
    # All drained → nothing left pending.
    assert reader._queued_lines == []

    # The queued-text renderer produces a dim "> … (queued)" line per entry.
    reader._queued_lines = ["a", "b"]
    frags = reader._queued_text()
    rendered = "".join(t for _, t in frags)
    assert "> a  (queued)" in rendered and "> b  (queued)" in rendered


def test_accept_handler_ignores_blank_lines():
    reader = _reader(lambda line: None)

    async def main():
        reader._queue = asyncio.Queue()

        class _Blank:
            text = "   "

        reader._on_accept(_Blank())
        assert reader.pending == 0
        assert reader._queue.empty()

    asyncio.run(main())


def test_request_cancel_interrupts_blocking_dispatch(monkeypatch):
    # Esc must interrupt a dispatch that's blocked (e.g. waiting on the model).
    # _request_cancel injects KeyboardInterrupt into the worker's thread;
    # _dispatch_tracked swallows it and returns None so the REPL continues.
    # (run_in_terminal is stubbed by the autouse fixture.)
    import threading
    import time as _time

    started = threading.Event()
    interrupted = {"value": False}

    def dispatch(line):
        started.set()
        try:
            # Simulate a long blocking turn.
            for _ in range(100):
                _time.sleep(0.05)
        except KeyboardInterrupt:
            interrupted["value"] = True
            raise
        return None

    reader = _reader(dispatch)

    async def main():
        reader._queue = asyncio.Queue()

        class _Buff:
            text = "long task"

        reader._on_accept(_Buff())

        async def _cancel_soon():
            # Wait until the dispatch is actually running, then cancel.
            await asyncio.get_event_loop().run_in_executor(None, started.wait)
            await asyncio.sleep(0.05)
            reader._request_cancel()

        # Worker runs one line; the cancel task interrupts it. Worker returns
        # (dispatch was not _ExitRepl) — drain a sentinel to end it.
        cancel_task = asyncio.ensure_future(_cancel_soon())
        reader._queue.put_nowait("/quit")  # so the worker exits after cancel
        await asyncio.wait_for(reader._worker(), timeout=5)
        await cancel_task

    # Give dispatch a quit sentinel that exits; the first (long) line is cancelled.
    def dispatch2(line):
        if line == "/quit":
            return _ExitRepl
        return dispatch(line)

    reader._dispatch = dispatch2
    asyncio.run(main())
    assert interrupted["value"] is True
    assert reader.busy is False


class TestSinkSpinner:
    """In pinned mode the spinner is state-only — no thread, no stdout writes;
    the prompt's bottom_toolbar renders the frame from the shared status.
    """

    def test_sink_mode_writes_nothing_to_stdout(self):
        status = SpinnerStatus()
        sp = Spinner(sink=status)
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            sp.start("Working")
            assert sp.thread is None  # no animation thread in sink mode
            sp.stop()
        finally:
            sys.stdout = old
        assert buf.getvalue() == ""

    def test_sink_reflects_active_and_label(self):
        status = SpinnerStatus()
        sp = Spinner(sink=status)
        sp.start("Running command")
        active, label = status.snapshot()
        assert active is True and label == "Running command"
        sp.stop()
        assert status.snapshot()[0] is False

    def test_toolbar_text_active_vs_idle(self):
        status = SpinnerStatus()
        assert spinner_toolbar_text(status) == ""  # idle → nothing
        status.set(True, "Thinking")
        text = spinner_toolbar_text(status)
        assert "Thinking" in text and "esc to cancel" in text

    def test_toolbar_dots_cycle_over_time(self, monkeypatch):
        # The trailing dots animate 0→1→2→3 (like the classic stdout spinner),
        # driven by wall-clock time so the toolbar's refresh advances them.
        status = SpinnerStatus()
        status.set(True, "Thinking")
        import mnemoai.client.ui.spinner as spinner_mod

        seen = set()
        for t in (0.0, 0.3, 0.6, 0.9):  # tick = t*10 → 0,3,6,9 → 0,1,2,3 dots
            monkeypatch.setattr(spinner_mod.time, "time", lambda t=t: t)
            text = spinner_toolbar_text(status)
            # Count the dots between the label and the trailing " (esc..."
            dots = text.split("Thinking", 1)[1].split(" (esc")[0]
            seen.add(dots)
        assert {"", ".", "..", "..."} <= seen

    def test_default_spinner_unaffected(self):
        # Without a sink, the spinner keeps its stdout-animation API (thread-based).
        sp = Spinner()
        assert sp._sink is None


class _FakeBuffer:
    def __init__(self, text=""):
        self.text = text

    def reset(self):
        self.text = ""


class _FakeEvent:
    def __init__(self, buffer, app):
        self.current_buffer = buffer
        self.app = app


def _ctrl_c_handler(reader):
    """Extract the c-c key handler from the reader's bindings."""
    for b in reader._make_bindings().bindings:
        keys = tuple(getattr(k, "value", k) for k in b.keys)
        if keys == ("c-c",):
            return b.handler
    raise AssertionError("no c-c binding found")


class TestCtrlC:
    def test_clears_non_empty_input_when_idle(self):
        r = _reader(lambda line: None)
        r._busy = False
        app = _FakeApp()
        buf = _FakeBuffer("sdadssad")
        _ctrl_c_handler(r)(_FakeEvent(buf, app))
        assert buf.text == ""       # input cleared
        assert app.exited is False  # did NOT exit

    def test_empty_input_exits_when_idle(self):
        r = _reader(lambda line: None)
        r._busy = False

        class _ExitApp:
            exc = None

            def exit(self, exception=None):
                self.exc = exception

        app = _ExitApp()
        _ctrl_c_handler(r)(_FakeEvent(_FakeBuffer(""), app))
        assert app.exc is KeyboardInterrupt  # empty line → exit path

    def test_first_ctrl_c_cancels_and_arms_no_quit(self, monkeypatch):
        r = _reader(lambda line: None)
        r._busy = True
        r._ctrl_c_while_busy = False
        r._worker_tid = 12345
        requested = {"n": 0}
        monkeypatch.setattr(r, "_request_cancel", lambda: requested.__setitem__("n", 1))
        monkeypatch.setattr(r, "_force_quit", lambda: pytest.fail("must not quit on 1st"))
        buf = _FakeBuffer("typed while running")
        _ctrl_c_handler(r)(_FakeEvent(buf, _FakeApp()))
        # First Ctrl+C requests cancel, arms force-quit, keeps the input intact.
        assert requested["n"] == 1
        assert r._ctrl_c_while_busy is True
        assert buf.text == "typed while running"

    def test_second_ctrl_c_while_busy_force_quits(self, monkeypatch):
        # A SECOND Ctrl+C while still busy (cancel not landing — worker wedged in a
        # blocking call) must force-quit. Independent of how cancel was triggered
        # (Esc or the first Ctrl+C), so gated on _ctrl_c_while_busy, not _cancelled.
        r = _reader(lambda line: None)
        r._busy = True
        r._ctrl_c_while_busy = True  # first Ctrl+C already armed it
        called = {"force": False}
        monkeypatch.setattr(r, "_force_quit", lambda: called.__setitem__("force", True))
        _ctrl_c_handler(r)(_FakeEvent(_FakeBuffer(""), _FakeApp()))
        assert called["force"] is True


class TestPlanApprovalUI:
    def test_no_app_auto_approves(self):
        # Without a running app (non-TTY), plan approval must not block: it
        # returns ("approve", plan) so scripted/piped runs proceed.
        r = _reader(lambda line: None)
        r._app = None
        r._loop = None
        verdict, plan = r.plan_approval_ui("# Plan\n1. x")
        assert verdict == "approve"
        assert plan == "# Plan\n1. x"

    def test_edit_key_binding_present(self):
        # The e/E key is armed (only meaningful during the plan-approval prompt).
        r = _reader(lambda line: None)
        keysets = [
            tuple(getattr(k, "value", k) for k in b.keys)
            for b in r._make_bindings().bindings
        ]
        assert ("e",) in keysets and ("E",) in keysets


class TestConfirmPromptRendering:
    def test_question_and_keys_styled_separately(self):
        # The pinned confirm line splits into an accent question segment and a
        # dimmed keys segment (so the eye separates prompt from actionable keys).
        r = _reader(lambda line: None)
        r._confirm_prompt = "▶ Write to file?"
        r._confirm_keys = "[y = yes · n = no · a = allow all]"
        segments = r._status_text()
        styles = [s for s, _ in segments]
        texts = [t for _, t in segments]
        assert "class:pinned-confirm" in styles
        assert "class:pinned-confirm-keys" in styles
        assert any("Write to file?" in t for t in texts)
        assert any("y = yes" in t for t in texts)
        # The keys are in their OWN (dimmed) segment, not the accent question one.
        q = next(t for s, t in segments if s == "class:pinned-confirm")
        assert "y = yes" not in q

    def test_no_keys_renders_single_segment(self):
        r = _reader(lambda line: None)
        r._confirm_prompt = "▶ Approve this plan?"
        r._confirm_keys = None
        segments = r._status_text()
        assert len(segments) == 1
        assert segments[0][0] == "class:pinned-confirm"


def _escape_binding(reader):
    """The reader's own bare-Escape binding (its filter gates word-motion)."""
    for b in reader._make_bindings().bindings:
        keys = tuple(getattr(k, "value", k) for k in b.keys)
        if keys == ("escape",):
            return b
    raise AssertionError("no escape binding found")


class TestEscapeWordMotion:
    """Bare-Esc cancels only while busy, and is NOT eager — so macOS Option+←/→
    (ESC b / ESC f) reach backward/forward-word instead of the Esc firing on the
    prefix (which would cancel the turn / drop the sequence). Word-motion must
    work whether idle OR typing a queued message mid-turn."""

    def test_escape_not_eager(self):
        # The core fix: eager Esc would consume the ESC prefix of Option+arrow.
        assert _escape_binding(_reader(lambda line: None)).eager() is False

    def test_escape_inactive_when_idle(self):
        r = _reader(lambda line: None)
        r._busy = False
        assert _escape_binding(r).filter() is False

    def test_escape_active_when_busy(self):
        r = _reader(lambda line: None)
        r._busy = True
        assert _escape_binding(r).filter() is True
