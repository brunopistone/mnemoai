"""Pinned-input prompt_toolkit UI + dialogs for the chat loop.

``PinnedPromptReader`` is the interactive TTY UI: a persistent (non-full-screen)
prompt_toolkit ``Application`` that keeps the ``>`` input pinned at the bottom of
the terminal while each query runs on a worker thread and its output streams
*above* it (via ``patch_stdout``) into native scrollback — so wrapping, copy/paste
and scrollback are preserved (the Claude-Code / Kiro layout). It handles slash
completion, history, an animated status/spinner line, Esc-to-cancel, an input
queue, in-app y/N/a confirmations, and exit-then-relaunch for full-screen dialog
commands (``/load``, ``/config``, …).

Also provides the full-screen dialogs used by ``/load`` and the configurator
(``select_from_list``, ``confirm_inline``): Enter confirms, Esc cancels.

Non-TTY sessions (pipes / CI / tests) never use this — the chat loop degrades to
plain ``input()``.
"""

import sys
from typing import Any, Callable, Iterable, List, Optional

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import History, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    Float,
    FloatContainer,
    HSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import confirm
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Button, Dialog, Label, RadioList

# The default ``bottom-toolbar`` class is reverse-video (a solid light bar). The
# pinned status/queue lines should read like dim console text, not a highlighted
# status bar, so override to plain gray on the default background.
_TUI_STYLE = Style(
    [
        ("bottom-toolbar", "noreverse bg:default fg:#888888"),
        ("bottom-toolbar.text", "noreverse bg:default fg:#888888"),
        # Pinned-input status line (spinner): dim gray on the default background,
        # so it reads as a subtle status, not a highlighted bar.
        ("pinned-status", "noreverse bg:default fg:#888888"),
        # Confirmation prompt shown in the status region: yellow, to stand out.
        ("pinned-confirm", "noreverse bg:default fg:ansiyellow bold"),
        # Queued (not-yet-started) messages shown live above the input: dim.
        ("pinned-queued", "noreverse bg:default fg:#888888"),
    ]
)


class SlashCommandCompleter(Completer):
    """Suggest slash commands, but only when the line starts with '/'.

    Ported from the former ``ChatInterface._SlashCommandCompleter`` so behavior
    is identical: a message that doesn't begin with '/' yields no completions,
    and only a single leading token (no space) is completed.
    """

    def __init__(self, commands: List[tuple]) -> None:
        self._commands = commands

    def get_completions(self, document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for cmd, desc in self._commands:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                    display_meta=desc,
                )


# Sentinel returned by a dispatch callback to end the pinned REPL.
_ExitRepl = object()

# App-exit result meaning "a dialog command wants to run; relaunch me after".
_RESTART = object()


def _dialog_is_tty() -> bool:
    """True when a full-screen dialog can render (a real interactive terminal)."""
    return (
        hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
        and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    )


class PinnedPromptReader:
    """Pinned-input REPL with the ``>`` input fixed at the BOTTOM of the screen.

    Runs a custom (non-full-screen) prompt_toolkit ``Application`` whose layout is
    a small bottom region — a status line (spinner, shown only while busy) *above*
    the ``>`` input line. ``patch_stdout(raw=True)`` routes all ordinary
    ``print()`` output **above** that region, so messages/answers/tool markers
    scroll up in native terminal scrollback while the input stays pinned at the
    very bottom (the Claude-Code / Kiro layout). Native wrapping + copy/paste are
    preserved (not a full-screen app).

    Submitting a line enqueues it (``accept_handler`` runs on the app's event-loop
    thread); a background worker coroutine drains the queue **one at a time** and
    runs each line via ``asyncio.to_thread`` — on a worker thread because
    ``client.query()`` calls ``asyncio.run()`` internally, which raises on a thread
    that already owns a running loop. So the input stays live and responsive while
    a turn generates; a second Enter queues (FIFO), never launching a concurrent
    query. This is the default interactive UI on a TTY.
    """

    def __init__(
        self,
        *,
        prompt_text: Callable[[], Any],
        commands: List[tuple],
        dispatch: Callable[[str], Any],
        history: Optional[History] = None,
        toolbar_text: Optional[Callable[[], Any]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
    ) -> None:
        """Build the pinned app.

        Args:
            prompt_text: Returns the input prefix each render (e.g. ``> `` / HTML
                with a plan-mode tag) — re-evaluated so the tag updates live.
            commands: Slash-command ``(cmd, desc)`` pairs for completion.
            dispatch: Called on a worker thread with the submitted line; returns
                :data:`_ExitRepl` to end the REPL, else ``None``.
            history: Optional shared prompt history.
            toolbar_text: Returns the status-line content (animated spinner). An
                empty string hides the status line (region shrinks to just input).
            on_cancel: Called (UI thread) on Esc during a running turn — requests
                cancellation of the in-flight query (wired in a later stage).
        """
        self._prompt_text = prompt_text
        self._dispatch = dispatch
        self._toolbar_text = toolbar_text or (lambda: "")
        self._on_cancel = on_cancel
        self._busy = False
        self._pending = 0  # queued-but-not-started lines (for the status line)
        self._queued_lines = []  # queued text, shown live in the pinned region
        self._worker_tid = None  # OS thread id of the running dispatch, for Esc
        self._cancelled = False  # guards a double Esc for the same turn
        self._loop = None  # app event loop, set in _run_async (for UI bridging)
        # In-app confirmation state (worker requests, UI captures a keypress).
        self._confirm_pending = False
        self._confirm_answer = None
        self._confirm_prompt = None
        # Set when a dialog command asks to exit-run-relaunch the app.
        self._pending_dialog = None
        # Keep constructor args so the app can be rebuilt after a dialog exit.
        self._history = history or InMemoryHistory()
        self._commands = commands
        self._queue = None  # asyncio.Queue, created in _run_async
        self._app = None

        self._buffer = Buffer(
            multiline=False,
            history=self._history,
            completer=SlashCommandCompleter(commands),
            # Live slash-command completions. NOTE: intentionally NOT combined
            # with enable_history_search — prompt_toolkit warns the two conflict
            # (both react to text changes), which made the popup appear only
            # intermittently. ↑/↓ still navigate history (plain, not prefix-match).
            complete_while_typing=True,
            accept_handler=self._on_accept,
        )
        self._app = self._build_app()

    # --- layout / app construction -------------------------------------------

    def _status_text(self):
        """Formatted text for the status line: confirm prompt, else the spinner."""
        if self._confirm_prompt:
            return [("class:pinned-confirm", self._confirm_prompt)]
        text = self._toolbar_text() or ""
        return [("class:pinned-status", text)] if text else []

    def _queued_text(self):
        """Formatted text listing messages queued (submitted, not yet started).

        Shown live in the pinned region so a message typed during a busy turn is
        visibly acknowledged — dim ``> …`` lines that clear once each is
        dequeued and echoed for real (in :meth:`_worker`).
        """
        lines = []
        for i, q in enumerate(self._queued_lines):
            prefix = "\n" if i else ""
            lines.append(("class:pinned-queued", f"{prefix}> {q}  (queued)"))
        return lines

    # Rows reserved below the input for the completion menu when it's expected.
    _MENU_RESERVE = 8

    def _input_height(self) -> Dimension:
        """Height of the input window: 1 line, growing to reserve menu space.

        When a completion menu is active (``complete_state`` populated — which,
        with ``complete_while_typing`` on, happens as soon as a matching prefix is
        typed), reserve ``_MENU_RESERVE`` rows below the input so the menu Float
        always has room to render — even with the input pinned at the terminal
        bottom (scrollback scrolls up to make the room). Collapses back to one
        line when there are no completions, so there's no permanent gap.
        """
        if not get_app().is_done and self._buffer.complete_state is not None:
            return Dimension(min=self._MENU_RESERVE)
        return Dimension()

    def _build_app(self) -> Application:
        has_status = Condition(
            lambda: bool(
                self._toolbar_text() or self._pending or self._confirm_prompt
            )
        )

        input_window = Window(
            BufferControl(
                buffer=self._buffer,
                input_processors=[BeforeInput(self._prompt_text)],
            ),
            # Dynamic height: normally the input hugs one line, but when a
            # completion menu is (about to be) shown, grow to reserve rows for it
            # so the menu always has somewhere to render — even with the input
            # pinned at the terminal bottom (scrollback scrolls up to make room).
            # Mirrors PromptSession._get_default_buffer_control_height.
            height=self._input_height,
            wrap_lines=True,
        )
        status_window = ConditionalContainer(
            Window(
                FormattedTextControl(self._status_text),
                height=1,
                style="class:pinned-status",
            ),
            filter=has_status,
        )
        # Queued messages (dim) shown above the status line while they wait.
        queued_window = ConditionalContainer(
            Window(
                FormattedTextControl(self._queued_text),
                dont_extend_height=True,
                style="class:pinned-queued",
            ),
            filter=Condition(lambda: bool(self._queued_lines)),
        )

        root = FloatContainer(
            content=HSplit([queued_window, status_window, input_window]),
            floats=[
                # Slash-command completions pop up just above the input.
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=8, scroll_offset=1),
                ),
            ],
        )

        return Application(
            layout=Layout(root, focused_element=input_window),
            key_bindings=merge_key_bindings(
                [load_key_bindings(), self._make_bindings()]
            ),
            style=_TUI_STYLE,
            # Non-full-screen: the layout is pinned at the bottom, prints scroll
            # above it (via patch_stdout). ~10 Hz refresh animates the spinner.
            full_screen=False,
            refresh_interval=0.1,
            erase_when_done=True,
        )

    def _make_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("c-j")
        def _(event) -> None:
            """Ctrl+J inserts a newline (Enter submits)."""
            event.current_buffer.insert_text("\n")

        @kb.add("escape", eager=True)
        def _(event) -> None:
            """Esc cancels the in-flight turn (interrupts the worker thread)."""
            if self._busy:
                self._request_cancel()
                if self._on_cancel is not None:
                    self._on_cancel()

        @kb.add("c-c")
        def _(event) -> None:
            """Ctrl+C cancels an in-flight turn; when idle, aborts (exit path).

            Mirrors Esc during a turn — interrupt the worker thread (client.query
            turns the KeyboardInterrupt into "Operation was cancelled."). Only
            when nothing is running does Ctrl+C fall through to the caller's
            double-press-to-exit handling.
            """
            if self._busy:
                self._request_cancel()
            else:
                event.app.exit(exception=KeyboardInterrupt)

        @kb.add("c-d")
        def _(event) -> None:
            """Ctrl+D on an empty line ends the session (EOF)."""
            if not event.current_buffer.text:
                event.app.exit(exception=EOFError)

        # In-app confirmation: while a Proceed? prompt is pending, y/n/a answer
        # it (and are swallowed so they don't reach the input buffer). Eager so
        # they win over the default self-insert binding.
        confirming = Condition(lambda: self._confirm_pending)

        @kb.add("y", filter=confirming, eager=True)
        @kb.add("Y", filter=confirming, eager=True)
        def _(event) -> None:
            if self._confirm_answer:
                self._confirm_answer("yes")

        @kb.add("n", filter=confirming, eager=True)
        @kb.add("N", filter=confirming, eager=True)
        @kb.add("enter", filter=confirming, eager=True)
        @kb.add("escape", filter=confirming, eager=True)
        def _(event) -> None:
            if self._confirm_answer:
                self._confirm_answer("no")

        @kb.add("a", filter=confirming, eager=True)
        @kb.add("A", filter=confirming, eager=True)
        def _(event) -> None:
            if self._confirm_answer:
                self._confirm_answer("all")

        return kb

    # --- accept / run / worker -----------------------------------------------

    def _on_accept(self, buff: Buffer) -> bool:
        """Enqueue the submitted line (runs on the app's event-loop thread).

        The line is NOT committed to scrollback here — only when the worker
        starts it (see :meth:`_worker`), so each ``>`` echo sits directly above
        its own answer. While it waits, it's shown live in the pinned region
        (:meth:`_queued_text`) so a message typed during a busy turn is visibly
        acknowledged. Returns False so prompt_toolkit clears the input.
        """
        text = buff.text
        if text.strip() and self._queue is not None:
            self._pending += 1
            self._queued_lines.append(text)
            self._queue.put_nowait(text)
        return False

    def run(self) -> None:
        """Run the pinned REPL until dispatch exits or Ctrl+C/Ctrl+D.

        Synchronous entry point. Wraps the session in ``patch_stdout`` so worker
        prints render above the pinned region, and drives the async app + worker.
        Re-raises ``KeyboardInterrupt`` / ``EOFError`` so the caller applies its
        double-press-to-exit handling.
        """
        import asyncio

        with patch_stdout(raw=True):
            asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        import asyncio

        self._queue = asyncio.Queue()
        # The app's event loop — needed to marshal worker-thread requests
        # (confirmations, dialogs) back onto the UI thread.
        self._loop = asyncio.get_event_loop()
        worker = asyncio.ensure_future(self._worker())
        try:
            while True:
                result = await self._app.run_async()
                if result is _RESTART and self._pending_dialog is not None:
                    # A dialog command asked to run outside the app. The app has
                    # now fully stopped (terminal back to cooked mode), so run the
                    # dialog here, hand back its result, then rebuild + relaunch
                    # the pinned app to continue the session.
                    func, box, done = self._pending_dialog
                    self._pending_dialog = None
                    try:
                        box["value"] = await asyncio.to_thread(func)
                    except BaseException as exc:  # surfaced to the worker
                        box["error"] = exc
                    finally:
                        done.set()
                    self._app = self._build_app()
                    continue
                return
        finally:
            worker.cancel()

    def confirm_ui(self, header: str, detail: str, category: str) -> str:
        """In-app y/N/a confirmation (called from the worker thread).

        A query can't exit the app (it's mid-turn), and ``input()`` can't read
        while the app owns stdin in raw mode. So this captures a single keypress
        THROUGH the running app: it shows the prompt in the pinned status region
        and installs temporary ``y``/``n``/``a`` bindings, blocking this worker
        thread on an ``Event`` until one is pressed. Returns ``"yes"|"no"|"all"``.
        """
        import threading

        if self._app is None or self._loop is None:
            return "no"

        done = threading.Event()
        result = {"value": "no"}

        # Shown in the status region while we wait for the keypress.
        self._confirm_prompt = (
            f"{header}  {detail}   [y = yes · n = no · a = allow all]"
        )

        def _answer(value: str) -> None:
            result["value"] = value
            self._confirm_prompt = None
            done.set()
            self._app.invalidate()

        # Bindings are consulted via _confirm_pending in the key handlers below.
        self._confirm_answer = _answer
        self._confirm_pending = True
        # Echo the prompt above the region too, so it's visible in scrollback.
        self._loop.call_soon_threadsafe(
            lambda: run_in_terminal(
                lambda: print(
                    f"\n\033[93m{header}\033[0m\n  \033[1m{detail}\033[0m\n"
                    "  \033[90m[y = yes · n = no · a = allow all this session]"
                    "\033[0m"
                )
            )
        )
        self._loop.call_soon_threadsafe(self._app.invalidate)

        done.wait()
        self._confirm_pending = False
        self._confirm_answer = None
        return result["value"]

    def run_dialog(self, func):
        """Run a blocking full-screen dialog command by EXITING the app first.

        A nested full-screen prompt_toolkit ``Application`` can't run inside the
        running one, and ``input()`` can't read while the app owns stdin. So this
        (called from the worker thread) asks the app to exit, waits for the loop
        to fully stop, runs ``func`` in the now-cooked terminal — where the normal
        full-screen dialogs and ``input()`` work — then signals the run loop to
        relaunch the pinned app. Blocks the worker for ``func``'s return value.
        """
        import threading

        if self._app is None or self._loop is None:
            return func()

        done = threading.Event()
        box = {}

        def _stop_app() -> None:
            # Ask the run loop to: stop the app, run func, restart the app.
            self._pending_dialog = (func, box, done)
            self._app.exit(result=_RESTART)

        self._loop.call_soon_threadsafe(_stop_app)
        done.wait()
        if "error" in box:
            raise box["error"]
        return box.get("value")

    async def _worker(self) -> None:
        """Drain the input queue one line at a time, dispatching on a thread."""
        import asyncio

        while True:
            line = await self._queue.get()
            self._pending = max(0, self._pending - 1)
            # Remove from the live "queued" list — it's about to be committed to
            # scrollback for real, so it shouldn't also show as pending.
            if self._queued_lines and self._queued_lines[0] == line:
                self._queued_lines.pop(0)
            else:
                try:
                    self._queued_lines.remove(line)
                except ValueError:
                    pass
            self._busy = True
            self._cancelled = False
            # Echo the line NOW (at dispatch), not at submit — so a queued line's
            # `>` prints directly above its own answer, keeping each input paired
            # with its output even when several were queued during a long turn.
            await run_in_terminal(lambda line=line: print(f"\033[36m>\033[0m {line}"))
            if self._app is not None:
                self._app.invalidate()
            try:
                result = await asyncio.to_thread(self._dispatch_tracked, line)
            finally:
                self._busy = False
                self._worker_tid = None
                if self._app is not None:
                    self._app.invalidate()
            self._queue.task_done()
            if result is _ExitRepl and self._app is not None:
                self._app.exit()
                return

    def _dispatch_tracked(self, line: str):
        """Run ``dispatch`` on the pool thread, recording its id for cancellation.

        Esc injects ``KeyboardInterrupt`` into this exact thread (see
        :meth:`_request_cancel`); ``client.query()`` already turns that into a
        clean "Operation was cancelled." so no work is corrupted.
        """
        import threading

        self._worker_tid = threading.get_ident()
        try:
            return self._dispatch(line)
        except KeyboardInterrupt:
            # Cancellation landed here rather than inside query() (e.g. between
            # steps). Swallow so the REPL continues; the turn is abandoned.
            return None

    def _request_cancel(self) -> None:
        """Inject KeyboardInterrupt into the busy worker thread (Esc handler).

        Uses ``PyThreadState_SetAsyncExc`` — the only way to interrupt a blocking
        call on another thread. Best-effort and idempotent (guarded by
        ``_cancelled``); a no-op if no turn is running.
        """
        import ctypes

        tid = self._worker_tid
        if not self._busy or tid is None or self._cancelled:
            return
        self._cancelled = True
        # Print a hint above the pinned region so the user knows it registered.
        run_in_terminal(lambda: print("\033[90m(cancelling…)\033[0m"))
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(tid), ctypes.py_object(KeyboardInterrupt)
        )

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def pending(self) -> int:
        """Number of lines submitted but not yet started."""
        return self._pending


def confirm_inline(message: str) -> bool:
    """Ask a yes/no question inline (no screen clear), default No.

    Uses prompt_toolkit's ``confirm`` on a TTY (styled y/n that stays in the
    normal terminal flow); falls back to plain ``input()`` otherwise. Ctrl+C /
    Ctrl+D / EOF are treated as No.
    """
    if _dialog_is_tty():
        try:
            return confirm(message)
        except (EOFError, KeyboardInterrupt):
            print()
            return False

    try:
        answer = input(f"{message} (y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def select_from_list(
    title: str,
    options: List[tuple],
    *,
    cancel_text: str = "Cancel",
) -> Optional[Any]:
    """Let the user pick one value from ``options`` via a full-screen menu.

    ``options`` is ``[(value, label), …]``; returns the chosen ``value`` or
    ``None`` if cancelled/empty. On a TTY this is a centered dialog where
    **↑/↓ move, Enter confirms the highlighted item, Esc cancels** — no
    Tab-to-button step (the stock ``radiolist_dialog`` requires Tab to reach
    Ok/Cancel, which is unintuitive for a pick-one menu). When stdin isn't a TTY
    it degrades to a numbered ``input()`` prompt so pipes / tests never block on
    a modal that can't render.
    """
    if not options:
        return None

    if _dialog_is_tty():
        result = _radio_pick(title, options)
        return None if result is _CANCEL else result

    # Non-TTY fallback: numbered list + input().
    print(f"\n{title}")
    for i, (_, label) in enumerate(options, 1):
        print(f"  {i}) {label}")
    try:
        answer = input("  Select [1, or Enter to cancel]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not answer:
        return None
    try:
        idx = int(answer)
    except ValueError:
        return None
    if not (1 <= idx <= len(options)):
        return None
    return options[idx - 1][0]


# Sentinel: user cancelled the pick dialog (distinct from a valid None value).
_CANCEL = object()


def _radio_pick(title: str, options: List[tuple]):
    """Full-screen single-pick menu: ↑/↓ move, Enter confirms, Esc cancels.

    Returns the chosen value, or ``_CANCEL``. Enter is bound on the RadioList's
    own control so it confirms the highlighted row directly — no Tab-to-button
    (the stock ``radiolist_dialog`` requires Tab, which is unintuitive here).
    """
    radio = RadioList(values=options)

    def _ok() -> None:
        get_app().exit(result=radio.current_value)

    def _cancel() -> None:
        get_app().exit(result=_CANCEL)

    radio.control.key_bindings.add("enter")(lambda event: _ok())

    dialog = Dialog(
        title=title,
        body=HSplit(
            [Label(text="↑/↓ to move · Enter to confirm · Esc to cancel"), radio],
            padding=1,
        ),
        buttons=[
            Button(text="OK", handler=_ok),
            Button(text="Cancel", handler=_cancel),
        ],
        with_background=True,
    )

    kb = KeyBindings()

    @kb.add("escape")
    @kb.add("c-c")
    def _(event) -> None:
        _cancel()

    app = Application(
        layout=Layout(dialog),
        key_bindings=kb,
        mouse_support=False,
        full_screen=True,
    )
    return app.run()
