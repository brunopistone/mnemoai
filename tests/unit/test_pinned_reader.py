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


def test_mid_turn_message_is_queued_not_folded_into_running_turn():
    # A message submitted WHILE a turn runs must be QUEUED to run as its own turn
    # after, never folded into the running turn. Mid-turn steering (which did the
    # folding) was removed in 1.8.0 because draining only at tool-round boundaries
    # stranded a message typed during the final tool-call-free model call into the
    # next turn; this test pins the queuing behavior that replaced it.
    reader = _reader(lambda line: None)

    async def main():
        reader._queue = asyncio.Queue()
        reader._busy = True  # a turn is running

        class _Buff:
            text = "look at file X too"

        assert reader._on_accept(_Buff()) is False
        assert reader._queued_lines == ["look at file X too"]
        assert reader._queue.get_nowait() == "look at file X too"

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

    class _Doc:
        def __init__(self, text):
            self.text_before_cursor = text

    @property
    def document(self):
        # Cursor is treated as at end-of-text for these tests.
        return self._Doc(self.text)

    def delete_before_cursor(self, n):
        if n:
            self.text = self.text[:-n]
        return n


class _FakeEvent:
    def __init__(self, buffer, app=None):
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


class TestFooterLine:
    """The persistent footer under the input: provided by the caller, sized by the
    app, and never able to take the UI down with it."""

    def test_no_provider_means_no_footer(self):
        assert _reader(lambda line: None)._footer_text() == []

    def test_provider_receives_a_width_and_its_segments_are_returned(self):
        seen = {}

        def _footer(width):
            seen["width"] = width
            return [("class:pinned-footer", "claude-opus-5")]

        r = PinnedPromptReader(
            prompt_text=lambda: "> ", commands=[], dispatch=lambda line: None,
            footer_text=_footer,
        )
        segments = r._footer_text()
        assert segments == [("class:pinned-footer", "claude-opus-5")]
        assert seen["width"] >= 1  # falls back to the terminal size off-TTY

    def test_a_raising_provider_hides_the_footer_instead_of_crashing(self):
        def _boom(width):
            raise RuntimeError("no")

        r = PinnedPromptReader(
            prompt_text=lambda: "> ", commands=[], dispatch=lambda line: None,
            footer_text=_boom,
        )
        assert r._footer_text() == []

    def test_footer_styles_are_registered(self):
        from mnemoai.client.ui.tui import _TUI_STYLE

        names = {rule[0] for rule in _TUI_STYLE.style_rules}
        for cls in ("pinned-footer", "pinned-footer-model",
                    "pinned-footer-warn", "pinned-footer-crit"):
            assert cls in names


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


def _paste_binding(reader):
    """The reader's BracketedPaste key handler."""
    from mnemoai.client.ui.tui import Keys

    for b in reader._make_bindings().bindings:
        keys = tuple(getattr(k, "value", k) for k in b.keys)
        if keys == (Keys.BracketedPaste.value,):
            return b.handler
    raise AssertionError("no bracketed-paste binding found")


class _PasteBuf:
    """Minimal buffer capturing inserted text for the paste handler."""

    def __init__(self):
        self.text = ""

    def insert_text(self, t):
        self.text += t


class _PasteEvent:
    def __init__(self, data, buf):
        self.data = data
        self.current_buffer = buf


class TestPasteNormalization:
    """Pasted content is normalized (CRLF/CR → LF, ANSI stripped, tabs expanded,
    control chars dropped) at the paste boundary — so a paste with `\\r` line
    endings (e.g. a table copied from a UI) can't overwrite earlier text via
    carriage returns and garble the echo. This is the real fix for the reported
    'weight_decaylit_0.01oxc6...' single-line collapse."""

    def test_normalize_folds_cr_and_crlf_to_lf(self):
        from mnemoai.client.ui.tui import _normalize_paste
        assert _normalize_paste("a\r\nb\rc\nd") == "a\nb\nc\nd"
        assert "\r" not in _normalize_paste("x\r\ny\rz")

    def test_normalize_strips_ansi_and_expands_tabs(self):
        from mnemoai.client.ui.tui import _normalize_paste
        assert _normalize_paste("\033[31mred\033[0m") == "red"
        assert "\t" not in _normalize_paste("k\tv")
        assert _normalize_paste("k\tv").startswith("k")

    def test_normalize_drops_other_control_chars(self):
        from mnemoai.client.ui.tui import _normalize_paste
        # a stray NUL / bell must not survive, but newline is kept
        assert _normalize_paste("a\x00b\x07c\nd") == "abc\nd"

    def test_cr_table_paste_stores_clean_lf_text(self):
        # The reported case: a table copied with CR/CRLF line endings. After the
        # handler runs, the stored paste must be clean LF text (no \r), so its
        # echo can't collapse rows onto one line.
        rows = ["Key\tValue", "learning_rate\t0.0001",
                "model_name_or_path\tnvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                "name\texample-name-xc6gd", "weight_decay\t0.01"]
        r = _reader(lambda line: None)
        r._pasted = {}
        r._paste_counter = 0
        buf = _PasteBuf()
        _paste_binding(r)(_PasteEvent("\r".join(rows), buf))  # CR-only paste
        # collapsed to a placeholder (it's multi-line)
        assert buf.text.startswith("[Pasted text #1")
        stored = r._pasted[1]
        assert "\r" not in stored              # normalized to LF
        assert stored.count("\n") == len(rows) - 1   # rows preserved as lines
        assert "example-name-xc6gd" in stored and "weight_decay" in stored

    def test_short_cr_paste_inserted_verbatim_but_normalized(self):
        # A short paste isn't collapsed, but its \r is still folded to \n.
        r = _reader(lambda line: None)
        r._pasted = {}
        r._paste_counter = 0
        buf = _PasteBuf()
        _paste_binding(r)(_PasteEvent("a\rb", buf))  # short, 1 line-break
        assert buf.text == "a\nb"  # inserted inline, CR normalized


class TestPasteCollapse:
    """A long paste is collapsed to a compact `[Pasted text #N +M lines]`
    placeholder in the input (readable) and stored full; on submit the
    placeholder is EXPANDED back to the real text for BOTH the model and the
    scrollback echo (the placeholder was only the composing-time view). Short
    pastes insert verbatim."""

    def test_num_lines_counts_linebreaks(self):
        from mnemoai.client.ui.tui import _paste_num_lines
        assert _paste_num_lines("a\nb\nc") == 2      # breaks, not visual lines
        assert _paste_num_lines("none") == 0
        assert _paste_num_lines("x\r\ny\rz\n") == 3

    def test_format_ref(self):
        from mnemoai.client.ui.tui import _format_paste_ref
        assert _format_paste_ref(8, 524) == "[Pasted text #8 +524 lines]"
        assert _format_paste_ref(3, 0) == "[Pasted text #3]"  # no suffix when 0

    def test_expand_replaces_placeholder_with_full_text(self):
        r = _reader(lambda line: None)
        r._pasted = {1: "L1\nL2", 2: "SECOND"}
        out = r._expand_pastes("a [Pasted text #1 +1 lines] b [Pasted text #2] c")
        assert out == "a L1\nL2 b SECOND c"

    def test_expand_leaves_unknown_ids(self):
        r = _reader(lambda line: None)
        r._pasted = {}
        assert r._expand_pastes("hi [Pasted text #9]") == "hi [Pasted text #9]"

    def test_expand_does_not_reexpand_placeholder_inside_content(self):
        # A placeholder-looking string INSIDE one paste's content must not be
        # treated as a ref (reverse-offset splice guarantees this).
        r = _reader(lambda line: None)
        r._pasted = {1: "contains [Pasted text #2] literally"}
        assert r._expand_pastes("[Pasted text #1]") == "contains [Pasted text #2] literally"

    def test_submit_dispatches_expanded_text(self):
        # End-to-end: the collapsed placeholder is what's queued, but dispatch
        # (the model) receives the FULL expanded paste.
        seen = []

        def dispatch(line):
            seen.append(line)
            return _ExitRepl if line == "/quit" else None

        r = _reader(dispatch)
        r._pasted = {1: "line one\nline two\nline three"}
        _drive(r, ["Review this: [Pasted text #1 +2 lines]", "/quit"])
        assert seen[0] == "Review this: line one\nline two\nline three"

    def test_submit_echoes_expanded_text_to_scrollback(self, capsys):
        # A SMALL submitted paste is echoed to scrollback expanded + gray-dimmed
        # (per line), not as the placeholder. The run_in_terminal stub prints
        # inline, so capsys captures it (CRLF line endings under raw mode).
        def dispatch(line):
            return _ExitRepl if line == "/quit" else None

        r = _reader(dispatch)
        r._pasted = {1: "FULL PASTED BODY\nsecond line"}
        _drive(r, ["Look: [Pasted text #1 +1 lines]", "/quit"])
        out = capsys.readouterr().out
        assert "FULL PASTED BODY" in out and "second line" in out
        assert "[Pasted text #1" not in out       # placeholder is not what's echoed
        assert "\033[90mFULL PASTED BODY\033[0m" in out  # dimmed per line

    def test_echo_paste_body_expands_only_pasted_portion(self):
        # echo=True dims ONLY the pasted content (per line), not typed text; the
        # model path (echo=False) is plain, verbatim, no ANSI.
        r = _reader(lambda line: None)
        r._pasted = {1: "BODY"}
        assert r._expand_pastes("typed [Pasted text #1]", echo=True) == (
            "typed \033[90mBODY\033[0m"
        )
        assert r._expand_pastes("typed [Pasted text #1]") == "typed BODY"

    def test_large_paste_echo_is_capped_but_model_gets_full(self):
        # A large paste: the echo is capped to head+tail with a "… +N lines …"
        # marker (so it can't flood scrollback / garble the pinned repaint), while
        # the model still receives the FULL untruncated text.
        r = _reader(lambda line: None)
        body = "\n".join(f"line {i}" for i in range(100))
        r._pasted = {1: body}
        ref = "See: [Pasted text #1 +100 lines]"

        model = r._expand_pastes(ref)                    # echo=False
        assert model == f"See: {body}"                   # full, verbatim, no ANSI
        assert "… +" not in model and "\033[90m" not in model

        echo = r._expand_pastes(ref, echo=True)
        assert "… +82 lines …" in echo                   # 100 - 12 head - 6 tail
        assert "line 0" in echo and "line 99" in echo    # head + tail shown
        assert "line 50" not in echo                      # middle hidden


def _backspace_binding(reader):
    # prompt_toolkit normalizes "backspace" to Keys.ControlH (value "c-h").
    for b in reader._make_bindings().bindings:
        keys = tuple(getattr(k, "value", k) for k in b.keys)
        if keys in (("backspace",), ("c-h",)):
            return b
    raise AssertionError("no backspace binding found")


class TestPasteAtomicDelete:
    """Backspace deletes a `[Pasted text …]` placeholder as one token, 
    and forgets its stored content."""

    def test_backspace_deletes_whole_placeholder(self):
        r = _reader(lambda line: None)
        r._pasted = {3: "big body"}
        buff = _FakeBuffer("Review: [Pasted text #3 +157 lines]")
        _backspace_binding(r).handler(_FakeEvent(buff))
        assert buff.text == "Review: "        # whole token gone in one keystroke
        assert 3 not in r._pasted             # stored content forgotten

    def test_backspace_deletes_suffixless_placeholder(self):
        r = _reader(lambda line: None)
        r._pasted = {7: "x"}
        buff = _FakeBuffer("a [Pasted text #7]")
        _backspace_binding(r).handler(_FakeEvent(buff))
        assert buff.text == "a "

    def test_backspace_normal_char_when_no_placeholder(self):
        r = _reader(lambda line: None)
        buff = _FakeBuffer("hello")
        _backspace_binding(r).handler(_FakeEvent(buff))
        assert buff.text == "hell"            # ordinary single-char delete
