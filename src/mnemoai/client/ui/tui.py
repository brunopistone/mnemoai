"""Pinned-input prompt_toolkit UI + dialogs for the chat loop.

``PinnedPromptReader`` is the interactive TTY UI: a non-full-screen ``Application`` 
keeps the ``>`` input pinned at the bottom while queries run on a worker thread and 
output streams above it via ``patch_stdout``. Also provides the full-screen dialogs 
(``select_from_list``, ``confirm_inline``) used by ``/load`` and the configurator. 
Non-TTY sessions degrade to plain ``input()`` and never use this.
"""

import re
import shutil
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
from prompt_toolkit.keys import Keys
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

from mnemoai.client.ui import turn_view

# Override the default reverse-video bottom-toolbar so the pinned status/queue
# lines read as dim console text, not a highlighted bar.
_TUI_STYLE = Style(
    [
        ("bottom-toolbar", "noreverse bg:default fg:#888888"),
        ("bottom-toolbar.text", "noreverse bg:default fg:#888888"),
        ("pinned-status", "noreverse bg:default fg:#888888"),
        ("pinned-confirm", "noreverse bg:default fg:ansiyellow bold"),
        ("pinned-confirm-keys", "noreverse bg:default fg:#888888"),
        ("pinned-queued", "noreverse bg:default fg:#888888"),
    ]
)

# A paste is collapsed into a `[Pasted text #N +M lines]` placeholder (rather than
# inserted verbatim) when it's long by EITHER measure: > this many chars, OR 
# more than this many line breaks.
_PASTE_CHAR_THRESHOLD = 800
_PASTE_LINE_THRESHOLD = 2
# The scrollback ECHO of a submitted paste is expanded but capped to a head+tail
# preview (with a "… +N lines …" middle marker) when it's large — so a huge paste
# can't flood scrollback (or overrun the pinned-app repaint and garble). The MODEL
# always gets the full untruncated text; only the on-screen echo is capped.
_ECHO_PASTE_HEAD_LINES = 12
_ECHO_PASTE_TAIL_LINES = 6
# Matches a placeholder for expansion on submit; the `+M lines` suffix is optional
# (a paste with no newlines collapses to `[Pasted text #N]`).
_PASTE_REF_RE = re.compile(r"\[Pasted text #(\d+)(?: \+\d+ lines)?\]")
# Same placeholder anchored at the END of the text before the cursor — used to
# delete a placeholder as one token on backspace.
_PASTE_REF_AT_END_RE = re.compile(r"\[Pasted text #(\d+)(?: \+\d+ lines)?\]$")


def _paste_num_lines(text: str) -> int:
    """Number of line breaks in ``text`` (``\\r\\n``/``\\r``/``\\n``).

    This is line breaks, not visual lines — "a\\nb\\nc" → 2 — matching Claude
    Code's ``+M lines`` count (visual lines minus one)."""
    return len(re.findall(r"\r\n|\r|\n", text))


def _format_paste_ref(paste_id: int, num_lines: int) -> str:
    """The compact input placeholder for a collapsed paste."""
    if num_lines == 0:
        return f"[Pasted text #{paste_id}]"
    return f"[Pasted text #{paste_id} +{num_lines} lines]"


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
        steer: Optional[Callable[[str], bool]] = None,
        clear_steering: Optional[Callable[[], None]] = None,
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
            on_cancel: Called (UI thread) on Esc/Ctrl+C during a turn — fires the
                cooperative cancel so blocking stream/backoff waits wake at once
                (the async KeyboardInterrupt alone can't preempt them).
            steer: **Currently unused** (dormant). A hook that would fold a
                mid-turn message into the running turn; retired as the default in
                favor of pure FIFO queuing (a message submitted mid-turn runs as
                its own turn after see :meth:`_on_accept`).
                Kept wired so re-enabling steering is a small change, not a rebuild.
            clear_steering: Called on cancel to purge any pending steer messages;
                a harmless no-op now that nothing is steered.
        """
        self._prompt_text = prompt_text
        self._dispatch = dispatch
        self._toolbar_text = toolbar_text or (lambda: "")
        self._reasoning_text = reasoning_text or (lambda: "")
        self._on_cancel = on_cancel
        self._steer = steer
        self._clear_steering = clear_steering
        self._busy = False
        self._pending = 0  # queued-but-not-started lines (for the status line)
        self._queued_lines = []  # queued text, shown live in the pinned region
        self._worker_tid = None  # OS thread id of the running dispatch, for Esc
        self._cancelled = False  # guards a double Esc for the same turn
        self._ctrl_c_while_busy = False  # first Ctrl+C armed force-quit this turn
        self._loop = None  # app event loop, set in _run_async (for UI bridging)
        # In-app confirmation state (worker requests, UI captures a keypress).
        self._confirm_pending = False
        self._confirm_answer = None
        self._confirm_prompt = None
        self._confirm_keys = None  # dimmed [y · n · a] segment of the pinned prompt
        # Large-paste collapse: a big paste is shown in the input as a compact
        # `[Pasted text #N +M lines]` placeholder and stored full here; on submit
        # the placeholder is expanded back to the real text for the model. Keeps
        # the input readable when pasting a long transcript/file.
        self._pasted: dict = {}       # id -> full pasted text
        self._paste_counter = 0       # per-session incrementing id
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
            # Question in the accent color, the [y · n · a] options dimmed — so the
            # eye separates the prompt from the actionable keys.
            segments = [("class:pinned-confirm", self._confirm_prompt)]
            if self._confirm_keys:
                segments.append(("class:pinned-confirm-keys", f"   {self._confirm_keys}"))
            return segments
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

        @kb.add(Keys.BracketedPaste)
        def _(event) -> None:
            """Collapse a LONG paste into a compact placeholder in the input.

            A big paste (a transcript, a file) would otherwise flood the input
            box. If it's long by char or line count, we stash the full text and
            insert a `[Pasted text #N +M lines]` placeholder instead; on submit
            the placeholder is expanded back to the real text for the model
            (:meth:`_expand_pastes`). Short pastes insert verbatim as usual."""
            data = event.data or ""
            num_lines = _paste_num_lines(data)
            if len(data) > _PASTE_CHAR_THRESHOLD or num_lines > _PASTE_LINE_THRESHOLD:
                self._paste_counter += 1
                pid = self._paste_counter
                self._pasted[pid] = data
                event.current_buffer.insert_text(_format_paste_ref(pid, num_lines))
            else:
                event.current_buffer.insert_text(data)

        @kb.add("backspace")
        def _(event) -> None:
            """Backspace deletes a paste placeholder as ONE token, not char by char.

            When the text right before the cursor is a `[Pasted text #N …]`
            placeholder, remove the whole thing (and forget its stored content) in
            one keystroke. Otherwise fall back to the normal single-character backspace."""
            buff = event.current_buffer
            before = buff.document.text_before_cursor
            m = _PASTE_REF_AT_END_RE.search(before)
            if m:
                self._pasted.pop(int(m.group(1)), None)
                buff.delete_before_cursor(len(m.group(0)))
            else:
                buff.delete_before_cursor(1)

        # Bare-Esc cancels the in-flight turn — but NOT eager: an eager Esc fires
        # on the ``ESC`` prefix of macOS Option+←/→ (``ESC b`` / ``ESC f``), so
        # word-motion while typing (even a queued message mid-turn) would cancel
        # instead. Non-eager lets prompt_toolkit buffer the prefix, so ``ESC b`` /
        # ``ESC f`` reach the default backward/forward-word bindings and a lone
        # Esc still cancels.
        @kb.add("escape", filter=Condition(lambda: self._busy))
        def _(event) -> None:
            """Esc cancels the in-flight turn (interrupts the worker thread).

            ``_request_cancel`` also fires ``on_cancel`` (the cooperative signal),
            so both Esc and Ctrl+C wake blocking waits — no separate call here."""
            self._request_cancel()

        @kb.add("c-c")
        def _(event) -> None:
            """Ctrl+C: cancel an in-flight turn (first press), force-quit on a
            SECOND press while still busy — the worker may be wedged in a blocking
            tool call the injected KeyboardInterrupt can't reach, and a clean exit
            would hang joining that thread. Else clear a non-empty input; else
            (empty) exit."""
            if self._busy:
                if self._ctrl_c_while_busy:
                    self._force_quit()
                else:
                    # First Ctrl+C on this turn: request cancel and arm force-quit.
                    self._ctrl_c_while_busy = True
                    self._request_cancel()
                    run_in_terminal(
                        lambda: print(
                            "\033[90m(press Ctrl+C again to force-quit)\033[0m"
                        )
                    )
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

        @kb.add("e", filter=confirming, eager=True)
        @kb.add("E", filter=confirming, eager=True)
        def _(event) -> None:
            # Only the plan-approval prompt offers "edit"; the y/n/a confirm
            # ignores it (its handler maps unknown answers itself).
            if self._confirm_answer:
                self._confirm_answer("edit")

        @kb.add("a", filter=confirming, eager=True)
        @kb.add("A", filter=confirming, eager=True)
        def _(event) -> None:
            if self._confirm_answer:
                self._confirm_answer("all")

        return kb

    # --- accept / run / worker -----------------------------------------------

    def notify_background_complete(self) -> None:
        """Called (from a background sub-agent's daemon thread) when one finishes.

        Marshals to the event-loop thread and enqueues a delivery-only turn (an
        empty line) so the finished report is auto-surfaced while the user is
        idle. If a turn is already running or one is already queued, does nothing
        — the agent drains pending completions at the start of every turn, so the
        in-flight/queued turn will pick it up; we never pile up redundant empty
        turns."""
        loop = self._loop
        if loop is None or self._queue is None:
            return

        def _enqueue() -> None:
            if self._busy or self._queued_lines or self._pending:
                return  # a running/queued turn will drain the completion itself
            self._pending += 1
            self._queue.put_nowait("")  # empty line → delivery-only turn

        loop.call_soon_threadsafe(_enqueue)

    def _expand_pastes(self, text: str, echo: bool = False) -> str:
        """Replace each `[Pasted text #N …]` placeholder with its stored text.

        Splices by match offset in REVERSE so a placeholder-looking string inside
        one paste's content can't be re-expanded, and later offsets stay valid.
        Unknown ids (e.g. a placeholder the user typed by hand) are left as-is.

        ``echo=False`` (the MODEL path) inserts the FULL text verbatim. ``echo=True``
        (the scrollback path) inserts a **capped, gray-dimmed** rendering of the
        paste (:meth:`_echo_paste_body`): a big paste is truncated to head+tail so
        it can't flood scrollback / overrun the pinned repaint — the model still
        receives the full text via the ``echo=False`` call."""
        if not self._pasted:
            return text
        matches = list(_PASTE_REF_RE.finditer(text))
        for m in reversed(matches):
            full = self._pasted.get(int(m.group(1)))
            if full is not None:
                repl = self._echo_paste_body(full) if echo else full
                text = text[: m.start()] + repl + text[m.end():]
        return text

    @staticmethod
    def _echo_paste_body(body: str) -> str:
        """Render a pasted body for the scrollback echo: capped to head+tail with a
        ``… +N lines …`` marker when large, and dimmed **per line** (so a torn
        write can't strand the gray SGR across the block)."""
        lines = body.split("\n")
        head, tail = _ECHO_PASTE_HEAD_LINES, _ECHO_PASTE_TAIL_LINES
        if len(lines) > head + tail + 1:
            hidden = len(lines) - head - tail
            shown = lines[:head] + [f"… +{hidden} lines …"] + lines[-tail:]
        else:
            shown = lines
        # Dim each line independently (reset at each newline) so an interleaved
        # repaint can't leave color state bleeding into later scrollback.
        return "\n".join(f"\033[90m{ln}\033[0m" for ln in shown)

    @staticmethod
    def _print_echo_block(text: str) -> None:
        """Print a (possibly multi-line) scrollback echo safely under the pinned
        UI. ``patch_stdout(raw=True)`` leaves the tty in raw mode (ONLCR off), so a
        bare ``\\n`` line-feeds without a carriage return — a multi-line block then
        staircases and desyncs the pinned renderer's row/column model. Emit CRLF
        line endings and a single trailing newline so each line starts at column 0
        and the block scrolls cleanly as one unit."""
        sys.stdout.write(text.replace("\n", "\r\n") + "\n")
        sys.stdout.flush()

    def _on_accept(self, buff: Buffer) -> bool:
        """Enqueue the submitted line (on the event-loop thread).

        A message submitted WHILE a turn is running is QUEUED (FIFO) and runs as
        its OWN separate turn after the current one fully ends — matching Claude
        Code. It shows live as a dim ``> … (queued)`` line via
        :meth:`_queued_text` and is echoed to scrollback only when :meth:`_worker`
        dequeues it (so each ``>`` sits above its own answer). Returns False so
        prompt_toolkit clears the input.

        (Mid-turn *steering* — folding the message into the running turn — was
        retired here as the default: it could strand a message steered during the
        final, tool-call-free model call [it was never drained at turn end] into
        the NEXT turn. The agent-side steering machinery [``agent.steer`` /
        ``_drain_steering``] is left intact but unused by this UI.)
        """
        text = buff.text
        if not text.strip() or self._queue is None:
            return False

        self._pending += 1
        self._queued_lines.append(text)
        self._queue.put_nowait(text)
        return False

    def _echo_steered(self, text: str) -> None:
        """Echo a steered message to scrollback (dim), acknowledging it was folded
        into the running turn rather than queued as a separate turn."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                lambda: run_in_terminal(
                    lambda: print(f"\033[90m> {text}  (steering →)\033[0m")
                )
            )

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

        self._confirm_prompt = header
        self._confirm_keys = "[y = yes · n = no · a = allow all]"

        def _answer(value: str) -> None:
            result["value"] = value
            self._confirm_prompt = None
            self._confirm_keys = None
            done.set()
            self._app.invalidate()

        self._confirm_answer = _answer
        self._confirm_pending = True
        # Echo only the header + detail (the command/content) to scrollback; the
        # options hint lives in the pinned line, so don't duplicate it here.
        self._loop.call_soon_threadsafe(
            lambda: run_in_terminal(
                lambda: print(f"\n\033[93m{header}\033[0m\n  \033[1m{detail}\033[0m")
            )
        )
        self._loop.call_soon_threadsafe(self._app.invalidate)

        done.wait()
        self._confirm_pending = False
        self._confirm_answer = None
        return result["value"]

    def plan_approval_ui(self, plan: str) -> tuple:
        """Present a finished plan and capture the user's decision (worker thread).

        Renders the plan to scrollback, shows a pinned prompt, and blocks on an
        ``Event`` until the user presses y (approve & run) / e (edit in $EDITOR) /
        n (keep planning) — reusing the same confirm-key machinery as
        :meth:`confirm_ui`. On 'e' it drops the app, opens the plan in $EDITOR
        (via :meth:`run_dialog`), then re-prompts with the edited text. Returns
        ``(verdict, plan)`` where verdict is ``"approve"|"keep_planning"`` and
        plan is the (possibly edited) text to use."""
        import threading

        if self._app is None or self._loop is None:
            return ("approve", plan)

        while True:
            done = threading.Event()
            result = {"value": "approve"}

            self._confirm_prompt = "▶ Approve this plan?"
            self._confirm_keys = "[y = approve & run · e = edit · n = keep planning]"

            def _answer(value: str) -> None:
                result["value"] = value
                self._confirm_prompt = None
                self._confirm_keys = None
                done.set()
                self._app.invalidate()

            self._confirm_answer = _answer
            self._confirm_pending = True
            try:
                cols = max(40, min(100, shutil.get_terminal_size().columns - 4))
            except Exception:
                cols = 80
            # Echo only the plan block to scrollback; the actionable prompt +
            # options live in the pinned line (no duplicate hint).
            block = turn_view.render_plan(plan, width=cols)
            self._loop.call_soon_threadsafe(
                lambda b=block: run_in_terminal(lambda: print(f"\n{b}\n"))
            )
            self._loop.call_soon_threadsafe(self._app.invalidate)

            done.wait()
            self._confirm_pending = False
            self._confirm_answer = None
            verdict = result["value"]

            if verdict == "edit":
                # Drop the app, edit the plan in $EDITOR, then re-prompt.
                plan = self.run_dialog(lambda p=plan: _edit_plan_in_editor(p))
                continue
            # The shared "n"/Esc key maps to "no"; here that means keep planning.
            decision = "keep_planning" if verdict in ("keep_planning", "no") else "approve"
            return (decision, plan)

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
            self._ctrl_c_while_busy = False
            # The live queue kept the COLLAPSED line (compact `[Pasted text …]`).
            # The MODEL gets the FULL expanded paste (echo=False); the SCROLLBACK
            # echo gets a capped, gray, head+tail rendering (echo=True) so a huge
            # paste can't flood scrollback or overrun the pinned-app repaint.
            dispatched = self._expand_pastes(line)
            echoed = self._expand_pastes(line, echo=True)
            # Echo NOW (at dispatch, not submit) so a queued line's `>` prints
            # directly above its own answer. An empty line is a delivery-only
            # turn (a background sub-agent finished) — show a marker, not a bare `>`.
            if line.strip():
                await run_in_terminal(
                    lambda t=echoed: self._print_echo_block(f"\033[36m>\033[0m {t}")
                )
            else:
                await run_in_terminal(
                    lambda: print(
                        "\033[90m[↳ a background sub-agent finished]\033[0m"
                    )
                )
            if self._app is not None:
                self._app.invalidate()
            try:
                result = await asyncio.to_thread(self._dispatch_tracked, dispatched)
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
        # Discard anything steered into the turn being cancelled — otherwise a
        # mid-turn message tied to this (aborted) turn would leak into the next
        # one and get answered out of context.
        if self._clear_steering is not None:
            try:
                self._clear_steering()
            except Exception:
                pass
        # Cooperative cancel FIRST: wake any blocking stream/backoff wait instantly
        # (the async KeyboardInterrupt below can't preempt those C-level waits, so
        # on its own a stalled-stream cancel stalls for the whole idle/backoff).
        # Covers both Esc and Ctrl+C, which both route through here.
        if self._on_cancel is not None:
            try:
                self._on_cancel()
            except Exception:
                pass
        run_in_terminal(lambda: print("\033[90m(cancelling…)\033[0m"))
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(tid), ctypes.py_object(KeyboardInterrupt)
        )

    def _force_quit(self) -> None:
        """Hard-exit when a cancel is stuck (worker wedged in a blocking call).

        A clean ``app.exit`` would hang: ``asyncio.run`` joins the
        ``asyncio.to_thread`` executor thread on shutdown, and that thread is
        blocked in native code the injected KeyboardInterrupt can't reach.
        ``os._exit`` skips the join and terminates immediately — but it also skips
        prompt_toolkit's terminal teardown, so we must restore the tty to cooked
        mode with echo ourselves, or the shell is left invisible/unresponsive."""
        import os

        self._restore_terminal()
        print("\n\033[90mForce-quit (worker was unresponsive).\033[0m", flush=True)
        os._exit(130)  # 128 + SIGINT

    @staticmethod
    def _restore_terminal() -> None:
        """Put the tty back into a sane cooked+echo state after a hard exit.

        prompt_toolkit's raw-mode teardown normally runs on a clean exit; ``os._exit``
        bypasses it, leaving the terminal in cbreak/no-echo so typed input is
        invisible. ``stty sane`` (or a termios ECHO|ICANON restore) fixes it."""
        try:
            import termios

            fd = sys.stdin.fileno()
            attrs = termios.tcgetattr(fd)
            attrs[3] |= termios.ECHO | termios.ICANON | termios.ISIG  # lflags
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except Exception:
            # Fallback: shell out to `stty sane` (covers non-termios edge cases).
            try:
                import subprocess

                subprocess.run(["stty", "sane"], check=False)
            except Exception:
                pass

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def pending(self) -> int:
        """Number of lines submitted but not yet started."""
        return self._pending


def _edit_plan_in_editor(plan: str) -> str:
    """Open ``plan`` in $EDITOR (fallback vi/nano) and return the edited text.

    Run in cooked mode (via ``run_dialog``), so it can drive a full-screen
    editor. On any failure the original plan is returned unchanged."""
    import os
    import subprocess
    import tempfile

    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(plan)
            path = f.name
        try:
            subprocess.run([*editor.split(), path], check=False)
            with open(path, encoding="utf-8") as f:
                edited = f.read().strip()
            return edited or plan
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    except Exception:
        return plan


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
