"""Unit tests for the blocking confirm wait (_await_confirm) not wedging the app.

The confirm prompt is drawn by the pinned app while the WORKER thread blocks on
an Event. Two ways that used to hang the whole app forever:

  1. The scrollback echo was dispatched fire-and-forget
     (``call_soon_threadsafe(lambda: run_in_terminal(echo))``), so the returned
     awaitable was dropped: an echo that raised or never completed was silent,
     and the pinned prompt could fail to paint.
  2. ``done.wait()`` had no timeout, so if the prompt never painted the user saw
     a bare cursor with no ``Proceed?`` and Esc/Ctrl+C could not reach a thread
     parked in ``Event.wait()`` — an unrecoverable hang.

These drive ``_await_confirm`` on a stub reader (no real prompt_toolkit app) with
the echo dispatch faked, so they're pure and fast.
"""

import threading

import pytest

from mnemoai.client.ui.tui import PinnedPromptReader


class _FakeLoop:
    """Runs call_soon_threadsafe callbacks inline; create_task just calls once."""

    def __init__(self, echo_raises=False):
        self.echo_raises = echo_raises
        self.tasks = []

    def call_soon_threadsafe(self, fn, *a):
        fn(*a)

    def create_task(self, coro):
        # Drive the coroutine to completion synchronously; `await
        # run_in_terminal(...)` is stubbed to a already-finished awaitable.
        self.tasks.append(coro)
        try:
            coro.send(None)
        except StopIteration:
            pass
        return coro


class _FakeApp:
    def __init__(self):
        self.invalidated = 0

    def invalidate(self):
        self.invalidated += 1


def _reader(monkeypatch, echo_raises=False):
    r = PinnedPromptReader.__new__(PinnedPromptReader)
    r._app = _FakeApp()
    r._loop = _FakeLoop()
    r._confirm_prompt = None
    r._confirm_keys = None
    r._confirm_answer = None
    r._confirm_pending = False
    r._cancelled = False

    class _Done:
        def __await__(self):
            if False:
                yield
            return None

    def fake_rit(func):
        func()  # the echo itself still runs (scrollback write)
        return _Done()

    monkeypatch.setattr("mnemoai.client.ui.tui.run_in_terminal", fake_rit)
    return r


class TestConfirmWaitCannotHang:
    def test_cancel_breaks_out_instead_of_blocking_forever(self, monkeypatch):
        # Simulates the reported hang: the prompt never gets answered. A cancel
        # must release the worker rather than parking it forever.
        r = _reader(monkeypatch)

        def cancel_soon():
            r._cancelled = True

        threading.Timer(0.05, cancel_soon).start()
        # If the wait were unbounded this call would never return.
        assert r._await_confirm("▶ Run?", "[y/n]", lambda: None) == "no"

    def test_cancel_denies_even_when_default_is_approve(self, monkeypatch):
        # plan_approval_ui passes default="approve"; a prompt nobody answered
        # must NOT be treated as approval.
        r = _reader(monkeypatch)
        r._cancelled = True
        assert (
            r._await_confirm("▶ Approve?", "[y/n]", lambda: None, default="approve")
            == "no"
        )

    def test_answer_still_returns_normally(self, monkeypatch):
        r = _reader(monkeypatch)

        def answer_soon():
            r._confirm_answer("yes")

        threading.Timer(0.05, answer_soon).start()
        assert r._await_confirm("▶ Run?", "[y/n]", lambda: None) == "yes"

    def test_failing_echo_does_not_block_the_prompt(self, monkeypatch):
        # An echo that raises must not stop the prompt being armed/answerable.
        r = _reader(monkeypatch)

        def boom():
            raise RuntimeError("terminal write failed")

        threading.Timer(0.05, lambda: r._confirm_answer("all")).start()
        assert r._await_confirm("▶ Run?", "[y/n]", boom) == "all"

    def test_pending_state_is_torn_down(self, monkeypatch):
        r = _reader(monkeypatch)
        r._cancelled = True
        r._await_confirm("▶ Run?", "[y/n]", lambda: None)
        # A stale prompt left armed would keep the status line showing a
        # question that can no longer be answered.
        assert r._confirm_pending is False
        assert r._confirm_prompt is None
        assert r._confirm_keys is None
        assert r._confirm_answer is None


class TestEchoIsAwaited:
    def test_echo_runs_and_is_dispatched_as_a_task(self, monkeypatch):
        r = _reader(monkeypatch)
        wrote = []
        threading.Timer(0.05, lambda: r._confirm_answer("yes")).start()
        r._await_confirm("▶ Run?", "[y/n]", lambda: wrote.append(1))
        assert wrote == [1]  # the scrollback echo still happened
        assert r._loop.tasks  # via a real task (awaitable not dropped)


class TestNotice:
    """Scrollback notices ("(cancelling…)", the force-quit hint) are best-effort:
    a failed terminal write must never raise into a key handler, and the
    awaitable must not be dropped (a dropped one hides the failure and can chain-
    stall a later in-terminal write)."""

    def test_writes_the_text(self, monkeypatch):
        r = _reader(monkeypatch)
        seen = []
        monkeypatch.setattr(
            "mnemoai.client.ui.tui.run_in_terminal",
            lambda fn: (fn(), _Awaited())[1],
        )
        monkeypatch.setattr("builtins.print", lambda *a, **k: seen.append(a))
        r._notice("(cancelling…)")
        assert seen and "(cancelling…)" in seen[0][0]

    def test_failing_write_does_not_raise(self, monkeypatch):
        r = _reader(monkeypatch)

        def boom(fn):
            raise RuntimeError("terminal gone")

        monkeypatch.setattr("mnemoai.client.ui.tui.run_in_terminal", boom)
        r._notice("(cancelling…)")  # must not propagate

    def test_no_loop_is_a_noop(self):
        r = PinnedPromptReader.__new__(PinnedPromptReader)
        r._loop = None
        r._notice("x")  # must not raise

    def test_closing_loop_is_a_noop(self, monkeypatch):
        r = _reader(monkeypatch)

        class _Closing:
            def create_task(self, coro):
                coro.close()
                raise RuntimeError("Event loop is closed")

        r._loop = _Closing()
        r._notice("x")  # must not raise


class _Awaited:
    def __await__(self):
        if False:
            yield
        return None


class TestCancelRequested:
    def test_missing_attribute_is_not_cancelled(self):
        r = PinnedPromptReader.__new__(PinnedPromptReader)
        assert r._cancel_requested() is False

    @pytest.mark.parametrize("flag,expected", [(False, False), (True, True)])
    def test_reflects_cancel_flag(self, flag, expected):
        r = PinnedPromptReader.__new__(PinnedPromptReader)
        r._cancelled = flag
        assert r._cancel_requested() is expected
