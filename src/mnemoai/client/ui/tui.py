"""prompt_toolkit input reader + dialogs for the chat loop.

``PromptReader`` reads one line at a time on a TTY: a rich prompt with
slash-command completion, history, a plan-mode tag, and a ``ctrl+o`` panel that
toggles the last turn's tool calls. prompt_toolkit runs only while reading
input; the query then streams to the terminal with ordinary ``print()``, so
native wrapping, scrollback, and copy/paste are preserved.

Also provides the full-screen dialogs used by ``/load`` and the configurator
(``select_from_list``, ``confirm_inline``): Enter confirms, Esc cancels.

Non-TTY sessions (pipes / CI / tests) never use this — callers degrade to plain
``input()``.
"""

import sys
from typing import Any, Callable, Iterable, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import History, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.shortcuts import confirm
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Button, Dialog, Label, RadioList

# The default ``bottom-toolbar`` class is reverse-video (a solid light bar). The
# ctrl+o panel should read like dim console text, not a highlighted status bar,
# so override to plain gray on the default background.
_TUI_STYLE = Style(
    [
        ("bottom-toolbar", "noreverse bg:default fg:#888888"),
        ("bottom-toolbar.text", "noreverse bg:default fg:#888888"),
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


class PromptReader:
    """Reads one line of input at a time via a fresh prompt_toolkit prompt.

    The app is live only during :meth:`read` (blocking on the user's line);
    while a query runs afterwards the terminal is plain, so streamed output
    prints normally.

    ``ctrl+o`` toggles an ephemeral panel — drawn as the prompt's live
    ``bottom_toolbar`` — showing the last turn's tool calls with full, un-elided
    arguments. Press again to hide it. Because it's rendered by the running app
    (not printed), it appears/disappears in place and leaves nothing in the
    terminal scrollback (Claude-Code-style), unlike a plain print.
    """

    def __init__(
        self,
        *,
        prompt_text: Callable[[], Any],
        commands: List[tuple],
        history: Optional[History] = None,
        tool_calls_provider: Optional[Callable[[], list]] = None,
    ) -> None:
        self._prompt_text = prompt_text
        self._tool_calls_provider = tool_calls_provider or (lambda: [])
        # Whether the ctrl+o panel is currently shown (toggled per keypress).
        self._show_tools = False

        bindings = KeyBindings()

        @bindings.add("c-j")
        def _(event) -> None:
            """Ctrl+J inserts a newline (Enter submits)."""
            event.current_buffer.insert_text("\n")

        @bindings.add("c-o")
        def _(event) -> None:
            """Ctrl+O toggles the last-turn tool-calls panel (show/hide)."""
            self._show_tools = not self._show_tools
            self._sync_toolbar()
            event.app.invalidate()

        self._session: PromptSession = PromptSession(
            history=history or InMemoryHistory(),
            key_bindings=bindings,
            multiline=False,
            completer=SlashCommandCompleter(commands),
            complete_while_typing=True,
            style=_TUI_STYLE,
            # bottom_toolbar stays None (truly hidden) until ctrl+o toggles it on.
            # prompt_toolkit gates toolbar visibility on `is not None`, so a
            # callable that returns None would still show an (empty) bar — hence
            # we flip the attribute itself in _sync_toolbar.
            bottom_toolbar=None,
        )

    def read(self) -> str:
        """Prompt for and return one submitted line.

        The tool panel starts hidden each prompt (a fresh line shouldn't inherit
        a panel toggled open for a previous turn). Raises ``KeyboardInterrupt`` /
        ``EOFError`` on Ctrl+C / Ctrl+D so the caller can apply its
        double-press-to-exit logic (matching the previous behavior).
        """
        self._show_tools = False
        self._sync_toolbar()
        return self._session.prompt(self._prompt_text())

    def _sync_toolbar(self) -> None:
        """Point the session's bottom_toolbar at panel text, or None to hide it.

        Setting the attribute to ``None`` (rather than returning ``None`` from a
        callable) is what actually removes the bar — prompt_toolkit only shows a
        toolbar when ``bottom_toolbar is not None``.
        """
        if self._show_tools:
            self._session.bottom_toolbar = self._format_tool_calls(
                self._tool_calls_provider() or []
            )
        else:
            self._session.bottom_toolbar = None

    @staticmethod
    def _format_tool_calls(calls: list) -> str:
        """Build the panel text: each call's name + full, un-elided arguments.

        The marker line shown during a turn middle-elides each argument (limit
        72 chars) so a long command/path/payload doesn't flood the screen; this
        panel shows the SAME calls in full, newlines preserved. A turn with
        several parallel calls (e.g. many ``execute_bash``) lists each in order.
        """
        if not calls:
            return "ctrl+o · no tool calls in the last turn (press ctrl+o to hide)"
        lines = [
            f"ctrl+o · {len(calls)} tool call(s) last turn "
            "(press ctrl+o to hide):"
        ]
        for i, call in enumerate(calls, 1):
            lines.append(f"[{i}] {call.get('name', 'tool')}")
            args = call.get("args") or {}
            if not args:
                lines.append("    (no arguments)")
                continue
            for key, value in args.items():
                first, *rest = str(value).split("\n")
                lines.append(f"    {key} = {first}")
                for line in rest:
                    lines.append(f"    {' ' * len(key)}   {line}")
        return "\n".join(lines)


def confirm_inline(message: str) -> bool:
    """Ask a yes/no question inline (no screen clear), default No.

    Uses prompt_toolkit's ``confirm`` on a TTY (styled y/n that stays in the
    normal terminal flow); falls back to plain ``input()`` otherwise. Ctrl+C /
    Ctrl+D / EOF are treated as No.
    """
    is_tty = (
        hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
        and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    )
    if is_tty:
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

    is_tty = (
        hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
        and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    )
    if is_tty:
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
