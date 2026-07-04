"""Pinned-input prompt_toolkit UI + dialogs for the chat loop.

``PinnedPromptReader`` is the interactive TTY UI: a non-full-screen ``Application`` 
keeps the ``>`` input pinned at the bottom while queries run on a worker thread and 
output streams above it via ``patch_stdout``. Also provides the full-screen dialogs 
(``select_from_list``, ``confirm_inline``) used by ``/load`` and the configurator. 
Non-TTY sessions degrade to plain ``input()`` and never use this.
"""

import sys
from typing import Any, Callable, Iterable, List, Optional

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI
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

# Override the default reverse-video bottom-toolbar so the pinned status/queue
# lines read as dim console text, not a highlighted bar.
_TUI_STYLE = Style(
    [
        ("bottom-toolbar", "noreverse bg:default fg:#888888"),
        ("bottom-toolbar.text", "noreverse bg:default fg:#888888"),
        ("pinned-status", "noreverse bg:default fg:#888888"),
        ("pinned-confirm", "noreverse bg:default fg:ansiyellow bold"),
        ("pinned-queued", "noreverse bg:default fg:#888888"),
    ]
)


class SlashCommandCompleter(Completer):
    """Suggest slash commands, only when the line is a single leading '/' token."""

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

    A non-full-screen prompt_toolkit ``Application`` keeps a status line + ``>``
    input pinned at the bottom while ``patch_stdout(raw=True)`` routes output
    above it into native scrollback. Submitting enqueues a line; a worker coroutine drains
    the queue one at a time via ``asyncio.to_thread`` — a worker thread is
    required because ``client.query()`` calls ``asyncio.run()``, which raises on a
    thread already owning a loop. A second Enter queues (FIFO), never running a
    concurrent query. Default interactive UI on a TTY.
    """

    def __init__(
        self,
        *,
        prompt_text: Callable[[], Any],
        commands: List[tuple],
        dispatch: Callable[[str], Any],
        history: Optional[History] = None,
        toolbar_text: Optional[Callable[[], Any]] = None,
        reasoning_text: Optional[Callable[[], str]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
    ) -> None:
        """Build the pinned app.

        Args:
            prompt_text: Returns the input prefix each render (re-evaluated so a
                plan-mode tag updates live).
            commands: Slash-command ``(cmd, desc)`` pairs for completion.
            dispatch: Called on a worker thread; returns :data:`_ExitRepl` to end
                the REPL, else ``None``.
            history: Optional shared prompt history.
            toolbar_text: Returns the status-line content; empty hides the line.
            reasoning_text: Returns the live-reasoning ANSI block; empty hides it.
            on_cancel: Called (UI thread) on Esc during a turn to request cancel.
        """
        self._prompt_text = prompt_text
        self._dispatch = dispatch
        self._toolbar_text = toolbar_text or (lambda: "")
        self._reasoning_text = reasoning_text or (lambda: "")
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
            # NOT combined with enable_history_search — ptk warns they conflict
            # (both react to text changes), making the popup appear only sometimes.
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
        """Dim ``> … (queued)`` lines for submitted-but-not-started messages,
        acknowledging them live until :meth:`_worker` dequeues and echoes each."""
        lines = []
        for i, q in enumerate(self._queued_lines):
            prefix = "\n" if i else ""
            lines.append(("class:pinned-queued", f"{prefix}> {q}  (queued)"))
        return lines

    # Rows reserved below the input for the completion menu when it's expected.
    _MENU_RESERVE = 8

    def _input_height(self) -> Dimension:
        """Input height: 1 line, growing to reserve ``_MENU_RESERVE`` rows when a
        completion menu is active so it has room even with the input at the
        terminal bottom. Collapses back when there are no completions."""
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
            height=self._input_height,  # reserves menu space (see _input_height)
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
        queued_window = ConditionalContainer(
            Window(
                FormattedTextControl(self._queued_text),
                dont_extend_height=True,
                style="class:pinned-queued",
            ),
            filter=Condition(lambda: bool(self._queued_lines)),
        )
        # Live reasoning (styled block) shown transiently above the status line;
        # the final block commits to scrollback when the answer starts.
        reasoning_window = ConditionalContainer(
            Window(
                FormattedTextControl(lambda: ANSI(self._reasoning_text())),
                dont_extend_height=True,
                wrap_lines=True,  # else long reasoning runs off the right edge
            ),
            filter=Condition(lambda: bool(self._reasoning_text())),
        )

        root = FloatContainer(
            content=HSplit(
                [queued_window, reasoning_window, status_window, input_window]
            ),
            floats=[
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
            full_screen=False,  # pinned at bottom; prints scroll above via patch_stdout
            refresh_interval=0.1,  # ~10 Hz, animates the spinner
            erase_when_done=True,
        )

    def _make_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("c-j")
        def _(event) -> None:
            """Ctrl+J inserts a newline (Enter submits)."""
            event.current_buffer.insert_text("\n")

        # Bare-Esc cancels the in-flight turn — but NOT eager: an eager Esc fires
        # on the ``ESC`` prefix of macOS Option+←/→ (``ESC b`` / ``ESC f``), so
        # word-motion while typing (even a queued message mid-turn) would cancel
        # instead. Non-eager lets prompt_toolkit buffer the prefix, so ``ESC b`` /
        # ``ESC f`` reach the default backward/forward-word bindings and a lone
        # Esc still cancels.
        @kb.add("escape", filter=Condition(lambda: self._busy))
        def _(event) -> None:
            """Esc cancels the in-flight turn (interrupts the worker thread)."""
            self._request_cancel()
            if self._on_cancel is not None:
                self._on_cancel()

        @kb.add("c-c")
        def _(event) -> None:
            """Ctrl+C: cancel an in-flight turn; else clear a non-empty input;
            else (empty input) exit via the double-press path."""
            if self._busy:
                self._request_cancel()
            elif event.current_buffer.text:
                event.current_buffer.reset()  # clear the line, don't exit
            else:
                event.app.exit(exception=KeyboardInterrupt)

        @kb.add("c-d")
        def _(event) -> None:
            """Ctrl+D on an empty line ends the session (EOF)."""
            if not event.current_buffer.text:
                event.app.exit(exception=EOFError)

        # While a Proceed? prompt is pending, y/n/a answer it (eager, so they win
        # over self-insert and never reach the input buffer).
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
        """Enqueue the submitted line (on the event-loop thread).

        Not echoed to scrollback here — only when :meth:`_worker` starts it, so
        each ``>`` sits directly above its own answer; meanwhile it shows live via
        :meth:`_queued_text`. Returns False so prompt_toolkit clears the input.
        """
        text = buff.text
        if text.strip() and self._queue is not None:
            self._pending += 1
            self._queued_lines.append(text)
            self._queue.put_nowait(text)
        return False

    def run(self) -> None:
        """Run the pinned REPL (sync entry) under ``patch_stdout`` until dispatch
        exits or Ctrl+C/Ctrl+D; re-raises KeyboardInterrupt/EOFError to the caller."""
        import asyncio

        with patch_stdout(raw=True):
            asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        import asyncio

        self._queue = asyncio.Queue()
        # Needed to marshal worker-thread requests (confirmations, dialogs) back
        # onto the UI thread.
        self._loop = asyncio.get_event_loop()
        worker = asyncio.ensure_future(self._worker())
        try:
            while True:
                result = await self._app.run_async()
                if result is _RESTART and self._pending_dialog is not None:
                    # App has fully stopped (terminal back to cooked mode): run the
                    # dialog, hand back its result, then rebuild + relaunch.
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
        """In-app y/N/a confirmation (from the worker thread).

        ``input()`` can't read while the app owns stdin in raw mode, so this
        shows the prompt in the status region, arms the y/n/a bindings, and blocks
        the worker on an ``Event`` until a key is pressed. Returns yes|no|all.
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

        self._confirm_answer = _answer
        self._confirm_pending = True
        # Echo the prompt into scrollback too.
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

        A nested full-screen app can't run inside the running one. So (from the
        worker thread) exit the app, run ``func`` in the now-cooked terminal, then
        relaunch the pinned app. Blocks the worker for ``func``'s return value.
        """
        import threading

        if self._app is None or self._loop is None:
            return func()

        done = threading.Event()
        box = {}

        def _stop_app() -> None:
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
            # Echo NOW (at dispatch, not submit) so a queued line's `>` prints
            # directly above its own answer.
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
        """Run ``dispatch`` on the pool thread, recording its id so Esc can inject
        KeyboardInterrupt into it (client.query turns that into a clean cancel)."""
        import threading

        self._worker_tid = threading.get_ident()
        try:
            return self._dispatch(line)
        except KeyboardInterrupt:
            # Cancellation landed between steps rather than inside query(); swallow
            # so the REPL continues (turn abandoned).
            return None

    def _request_cancel(self) -> None:
        """Inject KeyboardInterrupt into the busy worker thread (Esc handler).

        ``PyThreadState_SetAsyncExc`` is the only way to interrupt a blocking call
        on another thread. Best-effort, idempotent, no-op when idle.
        """
        import ctypes

        tid = self._worker_tid
        if not self._busy or tid is None or self._cancelled:
            return
        self._cancelled = True
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
    """Inline yes/no question, default No (ptk ``confirm`` on a TTY, else
    ``input()``); Ctrl+C/Ctrl+D/EOF are treated as No."""
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
    """Pick one value from ``options`` (``[(value, label), …]``) via a full-screen
    menu; returns the chosen value or None. TTY: ↑/↓ move, Enter confirms, Esc
    cancels. Non-TTY: a numbered ``input()`` prompt so pipes/tests never block.
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
    """Full-screen single-pick menu (↑/↓ move, Enter confirms, Esc cancels);
    returns the chosen value or ``_CANCEL``. Enter is bound on the RadioList so it
    confirms the highlighted row directly (no Tab-to-button step)."""
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
