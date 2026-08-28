"""Chat interface handling for the application."""

import datetime as _dt
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory

from mnemoai import app_version
from mnemoai.client import file_ledger, file_mentions
from mnemoai.client.memory.episodic_memory import (
    extract_tools_from_messages,
    is_task_successful,
)
from mnemoai.client.memory.memory_store import MemoryStore
from mnemoai.client.memory.reflector import current_turn_messages
from mnemoai.client.memory.skill_store import SkillStore
from mnemoai.client.ui import notify, status_bar, turn_view
from mnemoai.client.ui.tui import (
    _DELETE,
    PinnedPromptReader,
    _dialog_is_tty,
    _ExitRepl,
    confirm_dialog,
    confirm_inline,
    select_from_list,
)
from mnemoai.client.user_commands import UserCommandStore
from mnemoai.utils.config import config
from mnemoai.utils.configurator import (
    run_features_override,
    run_model_override,
    run_params_override,
    run_reconfigure,
)
from mnemoai.utils.console import print_error
from mnemoai.utils.logger import logger, one_line
from mnemoai.utils.paths import (
    mcp_config_path,
    memory_file_path,
    skills_dir,
)


class ChatInterface:
    """Handles chat interface and user interaction."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.command_history = InMemoryHistory()
        # User-defined slash commands, built on first use (see user_commands).
        self._user_commands = None
        # Decided HERE, not in _run_pinned_loop: on a TTY the footer will carry the
        # context size, so nothing else should print it — and `--resume` replays
        # (and would print it) BEFORE the loop starts, from main._resume_session.
        client.status_footer_active = _dialog_is_tty()

    @property
    def user_commands(self) -> UserCommandStore:
        """The user's slash-command store (``~/.mnemoai/commands/*.md``).

        Built on first use, not in ``__init__``: resolving the root CREATES the
        directory, and nothing should touch the app home just because a
        ChatInterface was constructed. The store reserves the built-in names by
        default (``user_commands.BUILTIN_COMMANDS``, pinned to :attr:`_COMMANDS` by
        a test) — ``_dispatch`` matches those first, so a colliding file would
        never fire, and it is rejected with a reason instead of looking broken.
        """
        # getattr, not self._user_commands: the unit tests build this class with
        # __new__ (no __init__), and a report or a dispatch must not depend on that.
        if getattr(self, "_user_commands", None) is None:
            self._user_commands = UserCommandStore()
        return self._user_commands

    def _completion_commands(self) -> list:
        """Built-in + user-defined ``(cmd, desc)`` pairs for the ``/`` menu.

        Re-read per keystroke (the reader takes this as a callable) so a command
        file added mid-session appears in the menu, matching the fact that it
        already dispatches. Best-effort: a broken commands dir must not stop the
        built-ins from completing.
        """
        try:
            return list(self._COMMANDS) + self.user_commands.completions()
        except Exception:
            return list(self._COMMANDS)

    def _prompt_html(self) -> str:
        """Build the input-prompt line, with a 🔒 plan tag while plan mode is on.

        The (potentially long) model name is intentionally omitted — use /model.
        """
        if getattr(self.client, "plan_mode_active", False):
            return "<ansiyellow>🔒 plan</ansiyellow> <ansiblue>></ansiblue> "
        return "<ansiblue>></ansiblue> "

    # ASCII wordmark shown on launch (ANSI "Shadow" style), rendered in indigo.
    _BANNER = [
        "███╗   ███╗███╗   ██╗███████╗███╗   ███╗ ██████╗      █████╗ ██╗",
        "████╗ ████║████╗  ██║██╔════╝████╗ ████║██╔═══██╗    ██╔══██╗██║",
        "██╔████╔██║██╔██╗ ██║█████╗  ██╔████╔██║██║   ██║    ███████║██║",
        "██║╚██╔╝██║██║╚██╗██║██╔══╝  ██║╚██╔╝██║██║   ██║    ██╔══██║██║",
        "██║ ╚═╝ ██║██║ ╚████║███████╗██║ ╚═╝ ██║╚██████╔╝    ██║  ██║██║",
        "╚═╝     ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝     ╚═╝ ╚═════╝     ╚═╝  ╚═╝╚═╝",
    ]

    # Command groups for the welcome box: (heading, [(command, description)]).
    # Grouped by the QUESTION you'd have, not by implementation area — "Conversation"
    # had grown to nine entries doing three unrelated jobs (trimming the live
    # context, managing session files, steering the assistant), which is what made
    # the banner read as a wall. Keep any new command in the group whose heading
    # answers why you'd reach for it; split the group before letting one pass ~5.
    _COMMAND_GROUPS = [
        ("Context", [
            ("/context", "What's using the context window"),
            ("/clear", "Clear conversation context"),
            ("/compact [focus]", "Summarize & shrink context"),
            ("/rewind", "Undo your last prompt (files untouched)"),
            ("/usage", "Token usage for this session (per model)"),
        ]),
        ("Sessions", [
            ("/save [path]", "Save conversation (optional file/dir path)"),
            ("/load [path]", "Load a saved conversation (lists if no path)"),
            ("/export [md|txt]", "Write a shareable transcript here"),
            ("/branch [turn]", "Fork this session and continue there"),
            ("/rename [title]", "Name this session in the --resume picker"),
        ]),
        ("Workspace", [
            ("/files", "Files this session read, changed or attached"),
            ("/diff [path]", "Uncommitted changes (this session's marked)"),
            ("/copy [code|N]", "Copy the last answer or its code to clipboard"),
        ]),
        ("Assistant", [
            ("/plan", "Toggle read-only plan mode (blocks edits/bash)"),
            ("/memory [clear]", "View (or clear) persistent memory"),
            ("/skills [name]", "List installed skills (or preview one)"),
            ("/mcp", "List configured MCP servers & tools"),
            ("/hooks", "List the tool hooks this session runs"),
        ]),
        ("Configure", [
            ("/config", "Reconfigure config.yaml (overwrites it)"),
            ("/model", "Override one model (LLM/vision/embeddings)"),
            ("/params", "Tune inference params (temp, top_p, …)"),
            ("/features", "Enable/disable features (RAG, memory, web, …)"),
        ]),
        ("Help", [
            ("/doctor", "Check this install for problems"),
            ("/help", "This command reference + keyboard keys"),
        ]),
        ("Exit", [
            ("/exit, /quit", "Exit the application"),
        ]),
    ]

    # Slash commands for autocomplete — the actual insertable tokens (the
    # welcome-box labels carry arg hints / alternates instead).
    _COMMANDS = [
        ("/config", "Reconfigure config.yaml (overwrites it)"),
        ("/model", "Override one model (LLM/vision/embeddings)"),
        ("/params", "Tune model inference params (temperature, top_p, …)"),
        ("/features", "Enable/disable features (RAG, memory, web search, …)"),
        ("/mcp", "List configured MCP servers & their tools"),
        ("/hooks", "List the tool hooks this session runs"),
        ("/skills", "List installed skills (/skills <name> to preview)"),
        ("/clear", "Clear conversation context"),
        ("/compact", "Summarize & shrink context (optional focus)"),
        ("/rewind", "Take back your last prompt and the turn it ran"),
        ("/memory", "View persistent memory (/memory clear to wipe)"),
        ("/plan", "Toggle read-only plan mode (blocks edits & shell)"),
        ("/save", "Save conversation (/save [path])"),
        ("/load", "Load a saved conversation (/load lists saved)"),
        ("/usage", "Show token usage for this session"),
        ("/context", "Show what's using the context window"),
        ("/files", "Files this session read, changed or attached with @"),
        ("/diff", "Show uncommitted changes (/diff <path> for one file)"),
        ("/copy", "Copy the last answer (/copy code for its last code block)"),
        ("/export", "Export a shareable transcript (/export [md|txt] [path])"),
        ("/branch", "Fork this session at a turn and continue there"),
        ("/rename", "Name this session for the --resume picker"),
        ("/doctor", "Check this install for problems"),
        ("/help", "Show the command reference and keyboard keys"),
        ("/exit", "Exit the application"),
        ("/quit", "Exit the application"),
    ]

    # ANSI color codes
    _C = {
        "border": "\033[90m",   # grey
        "head": "\033[95m",     # magenta (group headings)
        "cmd": "\033[92m",      # green (commands)
        "text": "\033[97m",     # white
        "dim": "\033[90m",      # dim
        "reset": "\033[0m",
    }

    _ANSI_RE = re.compile(r"\033\[[0-9;]*m")

    def __clear_screen(self) -> None:
        """Clear screen + scrollback, cursor home (skipped when not a TTY)."""
        if not (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
            return
        # 3J scrollback, H home, 2J visible screen.
        print("\033[3J\033[H\033[2J", end="", flush=True)

    @classmethod
    def _vlen(cls, s: str) -> int:
        """Visible length (ANSI escapes don't occupy columns)."""
        return len(cls._ANSI_RE.sub("", s))

    # Rows the "Yours" group shows before collapsing the rest into a count: the box
    # is already the tallest thing on screen, and a user with 30 commands must not
    # push the built-in reference off it.
    _MAX_USER_COMMAND_ROWS = 6

    @staticmethod
    def _clip(text: str, width: int) -> str:
        """``text`` trimmed to ``width`` visible chars, ending in … when cut."""
        text = " ".join(str(text or "").split())
        return text if len(text) <= width else text[: width - 1].rstrip() + "…"

    @classmethod
    def _user_command_group(cls, commands: list) -> list:
        """The extra ``[("Yours", rows)]`` group for user-defined commands (or []).

        **A user row must never widen the box.** The built-in labels and
        descriptions are hand-fitted to keep the frame near 80 columns, and both
        columns are padded to their widest member — so one verbose user
        description (or a long ``argument_hint``) would push the entire reference
        past the terminal. Both are therefore clipped to the built-ins' own widths,
        and a label that only fits without its hint keeps the NAME (that's the part
        you type; the hint is still in the ``/`` menu and in the file).
        """
        if not commands:
            return []
        cmd_w = max(cls._vlen(c) for _, items in cls._COMMAND_GROUPS for c, _ in items)
        desc_w = max(cls._vlen(d) for _, items in cls._COMMAND_GROUPS for _, d in items)
        rows = []
        for c in commands[: cls._MAX_USER_COMMAND_ROWS]:
            label = c.label if cls._vlen(c.label) <= cmd_w else f"/{c.name}"
            rows.append((cls._clip(label, cmd_w), cls._clip(c.description, desc_w)))
        extra = len(commands) - len(rows)
        if extra:
            rows.append((f"… +{extra} more", "in ~/.mnemoai/commands/"))
        return [("Yours", rows)]

    @classmethod
    def _command_box(cls, footer: list, extra_groups: list = None) -> str:
        """The framed, grouped command list, with ``footer`` rows under a separator.

        The box IS the command reference, and ``/help`` is the only thing that
        prints it — the launch banner deliberately doesn't, so the reference is
        somewhere you go rather than a screenful you scroll past every start.
        ``extra_groups`` appends dynamic groups (the user's own commands) in the
        same shape as :attr:`_COMMAND_GROUPS`, so they are sized and framed by the
        same code as the built-ins.
        """
        C = cls._C
        vlen = cls._vlen
        groups = list(cls._COMMAND_GROUPS) + list(extra_groups or [])

        # Inner width: wordmark width (64), widened to fit the longest row. A
        # command row is "<heading gutter>  <command>  <description>", so the
        # gutter that holds the inlined group heading counts toward the width too;
        # the footer rows count as well, or a long hint breaks the frame.
        cmd_w = max(vlen(c) for _, items in groups for c, _ in items)
        gutter_w = max(vlen(h) for h, _ in groups)
        widest = max(
            gutter_w + 2 + cmd_w + 2 + vlen(desc)
            for _, items in groups for _, desc in items
        )
        W = max(64, widest, *(vlen(f) for f in footer)) if footer else max(64, widest)

        def row(content: str = "") -> str:
            """One framed row, padded to the visible inner width."""
            pad = " " * max(0, W - vlen(content))
            return f"{C['border']}│{C['reset']} {content}{pad} {C['border']}│{C['reset']}"

        lines = [f"{C['border']}╭{'─' * (W + 2)}╮{C['reset']}"]

        # Group headings sit on the SAME row as their first command rather than on
        # their own line. Five groups × (heading + spacer) is ten lines of pure
        # chrome in a box that's already the tallest thing on screen at launch;
        # inlining the heading buys all of it back and still reads as grouped,
        # because the headings are the only text in the left column.
        for heading, items in groups:
            for idx, (cmd, desc) in enumerate(items):
                label = heading if idx == 0 else ""
                gutter = f"{C['head']}{label}{C['reset']}" + " " * (
                    gutter_w - vlen(label)
                )
                padded_cmd = cmd + " " * (cmd_w - vlen(cmd))
                lines.append(
                    row(
                        f"{gutter}  {C['cmd']}{padded_cmd}{C['reset']}  "
                        f"{C['text']}{desc}{C['reset']}"
                    )
                )

        if footer:
            lines.append(f"{C['border']}├{'─' * (W + 2)}┤{C['reset']}")
            lines.extend(row(f) for f in footer)
        lines.append(f"{C['border']}╰{'─' * (W + 2)}╯{C['reset']}")
        return "\n".join(lines)

    def __welcome_message(self) -> None:
        """Display the launch banner: wordmark, version, and the way in.

        Deliberately NOT the command reference. The box was 31 of the 41 lines
        this printed — a screenful of chrome before the first prompt, for a list
        that `/help` and the `/` menu each put one keystroke away (the box's own
        last row said exactly that). What's left is what a launch has to answer:
        which app, which version, and how to find everything else. The pinned
        footer already carries the model, provider and directory, so the version
        is the one fact nothing else surfaces.
        """
        C = self._C
        # --- Wordmark (indigo ≈ #5f5fff via 256-color 63) ---
        # Flush left, like the prompt and the footer under it: with the box gone
        # there is nothing left to center the block over.
        banner_w = max(self._vlen(line) for line in self._BANNER)
        print()
        for line in self._BANNER:
            print(f"\033[38;5;63m{line}\033[0m")
        # Tagline and version centered on the WORDMARK's own width, so they sit
        # under the letters rather than under the whole terminal.
        for sub in ("local agentic AI assistant · learns & remembers", self._version()):
            if sub:
                print(f"{C['dim']}{sub.center(banner_w)}{C['reset']}")
        print()
        # The three ways in, and the only record of them at launch: /help brings
        # the full reference back, / searches it, and @ is invisible otherwise.
        print(
            f"{C['dim']} · ".join(
                f"{C['cmd']}{key}{C['dim']} {what}"
                for key, what in (
                    ("/help", "for commands"),
                    ("/", "to search"),
                    ("@", "to attach a file"),
                )
            )
            + C["reset"]
        )
        print()

    @staticmethod
    def _version() -> str:
        """``vX.Y.Z`` for the banner, or "" when running from a checkout."""
        v = app_version()
        return f"v{v}" if v else ""

    def _help_text(self) -> str:
        """The ``/help`` screen: the command box plus the keys that aren't commands."""
        C = self._C
        # Keys, not commands: nothing else surfaces them once the banner is gone,
        # and Esc-to-interrupt in particular is the one a user needs mid-turn.
        keys = [
            ("Enter", "submit"),
            ("Ctrl+J", "new line"),
            ("↑ / ↓", "history"),
            ("/", "command menu"),
            ("@", "attach a file or directory (completes as you type)"),
            ("Esc", "interrupt the current turn"),
            ("Ctrl+A", "agents panel (↑↓ move · Enter view · x stop · Esc leave)"),
            ("Ctrl+X Ctrl+K", "stop every running agent"),
            ("Ctrl+C / Ctrl+D", "exit (twice)"),
        ]
        key_w = max(self._vlen(k) for k, _ in keys)
        footer = [f"{C['dim']}Keys{C['reset']}"] + [
            f"  {C['cmd']}{k.ljust(key_w)}{C['reset']}  {C['text']}{what}{C['reset']}"
            for k, what in keys
        ]
        return self._command_box(footer, self._own_command_group())

    def _own_command_group(self) -> list:
        """The user's own commands as a box group; [] on any failure.

        Best-effort by design: the command reference is what a stuck user reaches
        for, so a malformed commands dir must never be the reason it won't print.
        """
        try:
            return self._user_command_group(self.user_commands.list_commands())
        except Exception:
            return []

    def _store_success_episode(self, task: str, tools_used: list) -> None:
        """Persist a successful episode and record the profiling outcome.

        The shared tail of both storage paths (legacy-delayed + immediate): store
        the episode, then, when profiling is on, classify the task's intent and
        record the tool outcome. ``task`` is the same value used for both.
        """
        self.client.episodic_memory.store_episode(
            task=task, tools_used=tools_used, outcome="success"
        )
        logger.debug("✓ Episode stored successfully")
        if config.get("PROFILE", {}).get("USE_PROFILING", False):
            intent = self.client.profile_manager.classify_intent(task)
            self.client.profile_manager.record_tool_outcome(intent, tools_used, True)

    def __store_episode_in_episodic_memory(self, query: str) -> None:
        """Store the PREVIOUS interaction in episodic memory if successful (legacy
        delayed mode — evaluated when the next query arrives)."""
        logger.debug("Episodic memory is enabled")
        if (
            self.client.previous_query
            and self.client.previous_response
            and self.client.previous_messages
        ):
            logger.debug("Evaluating previous interaction for episodic storage")
            logger.debug(f"Previous query: {self.client.previous_query[:100]}...")
            logger.debug(f"Current query: {query[:100]}...")

            # Scoped for the same reason as the immediate path below: this snapshot
            # is the whole session, and the episode describes one interaction.
            tools_used = extract_tools_from_messages(
                current_turn_messages(self.client.previous_messages)
            )

            # Only store if actual work was done (tools used or substantial response).
            if not tools_used and len(self.client.previous_response) < 300:
                logger.debug(
                    "✗ Skipping storage - no tools used and response too short (likely greeting/simple response)"
                )
            elif is_task_successful(
                self.client.previous_response,
                self.client.previous_messages,
                query,
            ):
                logger.debug(
                    "✓ Previous task marked as successful - storing in episodic memory"
                )
                logger.debug(f"Tools used: {[t.get('name') for t in tools_used]}")

                # First user message in the conversation.
                initial_query = self.client.previous_query
                for msg in self.client.previous_messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and "text" in item:
                                    initial_query = item["text"]
                                    break
                        break

                logger.debug(f"Initial query extracted: {initial_query[:100]}...")
                logger.debug(
                    f"Conversation length: {len(self.client.previous_messages)} messages"
                )

                self._store_success_episode(initial_query, tools_used)
            else:
                logger.debug(
                    "✗ Previous task not marked as successful - skipping storage"
                )
        else:
            logger.debug("No previous interaction to evaluate")

    def __store_current_episode_immediately(self, query: str, response: str) -> None:
        """Store the CURRENT interaction in episodic memory right after the
        response (immediate mode — doesn't wait for the next query)."""
        if not self.client.agent or not self.client.agent.messages:
            logger.debug("No agent messages to evaluate")
            return

        if not response or not response.strip():
            logger.debug("✗ Skipping storage - empty response")
            return

        # THIS turn's tools, not the whole session: both consumers describe the
        # interaction just completed. Unscoped, `record_tool_outcome` re-counted
        # every earlier tool call on every turn, so `tool_patterns` totals grew
        # quadratically (the same bug class as `interaction_count`), and an episode
        # was labelled with tools it never used.
        messages = current_turn_messages(self.client.agent.messages)
        tools_used = extract_tools_from_messages(messages)
        min_length = config.get("EPISODIC_MEMORY", {}).get("MIN_TOOLS_OR_LENGTH", 300)

        # Quality filter: skip if no tools and response too short.
        if not tools_used and len(response) < min_length:
            logger.debug(
                f"✗ Skipping storage - no tools used and response too short "
                f"({len(response)} < {min_length} chars)"
            )
            return

        if is_task_successful(response, messages, next_user_message=None):
            logger.debug("✓ Task marked as successful - storing immediately")
            logger.debug(f"Tools used: {[t.get('name') for t in tools_used]}")

            self._store_success_episode(query, tools_used)
        else:
            logger.debug("✗ Task not marked as successful - skipping storage")

    def _print_mcp_status(self) -> None:
        """Show configured MCP servers (built-in + external) and tool counts.

        Collided external tools appear namespaced as ``server__tool``.
        """
        members = getattr(self.client.mcp_client, "_members", [])
        tools = self.client.tools or []
        print("\nMCP servers:")
        if members:
            for name, _ in members:
                prefix = f"{name}__"
                count = sum(
                    1 for t in tools if t.name.startswith(prefix)
                ) if name != "builtin" else None
                label = "built-in" if name == "builtin" else "external"
                if count is None:
                    print(f"  • {name} ({label}, connected)")
                else:
                    print(f"  • {name} ({label}, connected) — {count} namespaced tool(s)")
        else:
            print("  (none connected)")
        print(f"\n  Total tools available: {len(tools)}")
        print(f"\n  Declare more servers in:\n    {mcp_config_path()}")
        print('  Format: {"mcpServers": {"name": {"command": ..., "args": [...], "env": {...}}}}\n')

    def _select_saved_conversation(self):
        """List saved conversations (newest first) and let the user pick one via
        :func:`tui.select_from_list`; returns the chosen path or None. Used by
        ``/load`` with no argument.

        The picker offers a **Delete** button: pressing it asks "Are you sure?
        Yes/No", deletes the highlighted conversation on Yes, then reopens the
        (refreshed) picker — so the user can prune saved chats without leaving the
        dialog."""
        now = _dt.datetime.now().timestamp()

        def _ago(ts: float) -> str:
            s = max(0, int(now - ts))
            if s < 60:
                return f"{s}s ago"
            if s < 3600:
                return f"{s // 60}m ago"
            if s < 86400:
                return f"{s // 3600}h ago"
            return f"{s // 86400}d ago"

        while True:
            files = self.client.list_saved_conversations()
            if not files:
                print(
                    "No saved conversations found. Use /save first, or "
                    "/load <path> to load from a specific file."
                )
                return None

            shown = files[:20]  # cap the menu; older ones load via /load <path>
            # Label with the auto-derived title (first user message); fall back to
            # the filename when a conversation has no readable user text.
            options = [
                (
                    str(p),
                    f"{self.client.conversation_title(p) or p.name}  "
                    f"({_ago(p.stat().st_mtime)})",
                )
                for p in shown
            ]
            title = "Load conversation"
            if len(files) > len(shown):
                title += (
                    f"  (showing {len(shown)} of {len(files)}; /load <path> for older)"
                )
            choice = select_from_list(title, options, allow_delete=True)

            # Delete flow: (_DELETE, path) → confirm → delete → reopen the picker.
            if isinstance(choice, tuple) and len(choice) == 2 and choice[0] is _DELETE:
                path = choice[1]
                name = self.client.conversation_title(path) or Path(path).name
                short = name if len(name) <= 60 else name[:57] + "…"
                if confirm_dialog(f"Delete this conversation?\n\n{short}"):
                    if self.client.delete_conversation(path):
                        print(f"Deleted: {short}")
                    else:
                        print("Could not delete that conversation.")
                continue  # reopen the picker with the refreshed list

            return choice  # a chosen path, or None (cancelled)

    def _handle_export_command(self, arg: str) -> None:
        """Handle ``/export [md|txt] [path]`` — a shareable, one-way transcript.

        Distinct from ``/save``: that writes re-importable JSON into the profile;
        this writes readable Markdown/text into the CURRENT directory, for pasting
        into a bug report or PR. ``reasoning`` opts the thinking blocks in (off by
        default — they usually dwarf the conversation).
        """
        parts = arg.split()
        fmt = None
        include_reasoning = False
        rest = []
        for token in parts:
            low = token.lower()
            if low in ("md", "markdown", "txt", "text") and fmt is None:
                fmt = "txt" if low in ("txt", "text") else "md"
            elif low in ("reasoning", "--reasoning", "thinking"):
                include_reasoning = True
            else:
                rest.append(token)
        path = " ".join(rest) or None

        written = self.client.export_transcript(
            path=path, fmt=fmt, include_reasoning=include_reasoning
        )
        if written:
            print(f"Exported transcript to {written}")
        else:
            print(
                "Nothing to export yet — the conversation has no messages, or the "
                "path could not be written."
            )

    def _handle_branch_command(self, arg: str) -> None:
        """Handle ``/branch [turn]`` — fork this session and continue in the copy.

        With no argument it shows a turn picker (branch *after* the chosen turn);
        with a number it branches directly. The original session is copied, never
        modified, so it stays resumable exactly as it was.
        """
        turns = self.client.session_turns()
        if not turns:
            print(
                "Nothing to branch yet — no turns have been recorded in this "
                "session (session recording is off if SESSION_MAX_AGE_DAYS is 0)."
            )
            return

        through = None
        raw = arg.strip()
        if raw:
            try:
                through = int(raw)
            except ValueError:
                print(f"Not a turn number: {raw}. Use /branch or /branch <n>.")
                return
            if not 1 <= through <= len(turns):
                print(f"Pick a turn between 1 and {len(turns)}.")
                return
        else:
            options = [
                (
                    t["n"],
                    f"{t['n']}. {t['preview'][:70]}"
                    + ("  (latest)" if t["n"] == len(turns) else ""),
                )
                for t in turns
            ]
            through = select_from_list(
                "Branch after which turn?  (the branch keeps turns 1..n)", options
            )
            if through is None:
                return  # cancelled

        new_path = self.client.branch_conversation(through)
        if not new_path:
            print("Could not create the branch.")
            return
        dropped = len(turns) - through
        note = f" ({dropped} later turn{'s' if dropped != 1 else ''} left behind)" if dropped else ""
        print(
            f"Branched after turn {through}{note}. You're now in the branch — the "
            "original session is untouched and still resumable with --resume."
        )

    def _handle_rename_command(self, arg: str) -> None:
        """Handle ``/rename [title]`` — name this session in the ``--resume`` picker.

        No argument shows the current name; ``clear`` removes it (the picker falls
        back to the first prompt).
        """
        raw = arg.strip()
        current = self.client.session_label()

        if not raw:
            if current:
                print(f'This session is named "{current}". /rename <title> to change it.')
            else:
                print(
                    "This session has no name — the --resume picker labels it with its "
                    "first prompt. /rename <title> to name it."
                )
            return

        if raw.lower() == "clear":
            if not current:
                print("This session has no name to clear.")
                return
            if self.client.rename_session(""):
                print("Name cleared — the picker will show the first prompt again.")
            else:
                print("Could not clear the name (session recording is off).")
            return

        if self.client.rename_session(raw):
            print(f'Session named "{" ".join(raw.split())[:80]}".')
        else:
            print(
                "Could not name this session — nothing is being recorded "
                "(SESSION_MAX_AGE_DAYS is 0)."
            )

    def _handle_memory_command(self, arg: str) -> None:
        """Handle ``/memory`` (view) and ``/memory clear`` over ``MemoryStore``."""
        store = MemoryStore()
        sub = arg.strip().lower()

        if sub == "clear":
            if not store.read().strip():
                print("Memory is already empty.")
                return
            if confirm_inline("Clear ALL persistent memory?"):
                store.clear()
                print("Persistent memory cleared.")
            else:
                print("Cancelled.")
            return

        if sub:
            print(f"Unknown /memory subcommand '{sub}'. Use /memory or /memory clear.")
            return

        contents = store.read().strip()
        print(f"\nPersistent memory ({memory_file_path()}):")
        if contents:
            for line in contents.splitlines():
                print(f"  {line}")
        else:
            print("  (empty — the agent saves facts here as you work)")
        print()

    def _handle_skills_command(self, arg: str) -> None:
        """Handle ``/skills`` (list) and ``/skills <name>`` (preview) over
        ``SkillStore``."""
        store = SkillStore()
        name = arg.strip()

        if name:
            skill = store.load_body(name)
            if skill is None:
                available = ", ".join(n for n, _ in store.list_metadata()) or "(none)"
                print(f"\nNo skill named '{name}'. Installed: {available}.\n")
                return
            print(f"\nSkill '{skill.name}' ({skill.path}):\n")
            print(skill.body)
            print()
            return

        skills, issues = store._scan()
        print(f"\nInstalled skills ({skills_dir()}):")
        if skills:
            for s in skills:
                print(f"  • {s.name} — {s.description}")
            print("\n  Preview one with /skills <name>.")
        else:
            print("  (none — add one as <name>/SKILL.md here)")
        # Surface rejected skills so a malformed one isn't silently invisible.
        if issues:
            print("\n  Skipped (fix and they'll load):")
            for issue in issues:
                print(f"  ✗ {issue.name} — {issue.reason}")
        print()

    def _restart_in_place(self) -> None:
        """Re-exec the process (``os.execv``) so reloaded config takes full effect.

        The only way to apply *every* setting — the MCP subprocess fixes its tool
        set at boot and the model/memory wire at startup. In-memory conversation
        is intentionally dropped. The MCP subprocess is shut down first since
        ``os.execv`` doesn't reap children.
        """
        print("\nRestarting to apply the new configuration...\n")
        # os.execv REPLACES this process: no atexit, no finally, so main()'s
        # end-of-run cleanup never runs and a turn-less session file (always the
        # case when the restart follows a `--resume` the user hadn't typed into
        # yet) would linger on disk until it aged out. Discard it here instead.
        try:
            log = getattr(getattr(self.client, "agent", None), "session_log", None)
            if log is not None:
                log.discard_if_empty()
        except Exception as e:  # cleanup must never block the restart
            logger.debug(f"Session cleanup before restart failed: {e}")
        try:
            self.client.mcp_client.shutdown()
        except Exception as e:
            logger.debug(f"MCP shutdown before restart failed: {e}")
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        # Re-exec with the original interpreter + argv (preserves --no-verbose).
        os.execv(sys.executable, [sys.executable] + sys.argv)

    # Sentinel returned by _dispatch to signal the loop should exit.
    _EXIT = object()

    def show_welcome(self) -> None:
        """Print the launch banner (public so `--resume` can show it BEFORE
        replaying a transcript — the banner belongs at the top)."""
        self.__welcome_message()

    def run_chat_loop(self, welcome: bool = True) -> None:
        """Run the main chat loop: pinned-input UI on a TTY
        (:meth:`_run_pinned_loop`), else a plain ``input()`` loop
        (:meth:`_plain_loop`); both dispatch via :meth:`_dispatch`.

        ``welcome=False`` skips the banner for a caller that already printed it
        (``--resume``, which must show the banner before the restored transcript
        so the conversation reads bottom-most, nearest the prompt)."""
        if welcome:
            self.__welcome_message()

        # Same "real interactive terminal" predicate the dialogs use.
        if _dialog_is_tty():
            self._run_pinned_loop()
        else:
            self._plain_loop()

    def _plain_loop(self) -> None:
        """Plain ``input()`` REPL for non-TTY use; Ctrl+C/Ctrl+D twice exits."""
        interrupt_count = 0
        last_interrupt_time = 0

        while True:
            try:
                query = input("> ")
                interrupt_count = 0
            except (KeyboardInterrupt, EOFError):
                interrupt_count, last_interrupt_time, should_exit = (
                    self._note_interrupt(interrupt_count, last_interrupt_time)
                )
                if not should_exit:
                    continue
                print("\nExiting...")
                try:
                    self.client.clear_context()
                except KeyboardInterrupt:
                    pass
                break

            if self._dispatch(query) is self._EXIT:
                try:
                    self.client.clear_context()
                except KeyboardInterrupt:
                    pass
                break

    def _run_pinned_loop(self) -> None:
        """Drive the pinned-input REPL (default TTY UI).

        A spinner *sink* is attached so spinner control flips toolbar state
        instead of writing ``\\r`` (which would fight the pinned redraw).
        Ctrl+C / Ctrl+D twice exits.
        """
        from mnemoai.client.ui.spinner import (
            Spinner,
            SpinnerStatus,
            spinner_toolbar_text,
        )

        # Route spinner control to a shared status the toolbar reads.
        status = SpinnerStatus()
        self.client.spinner = Spinner(sink=status)
        self.client.callback_handler.spinner = self.client.spinner
        # Live-reasoning sink: the agent appends chunks, the reader renders them.
        reasoning = turn_view.ReasoningStatus()
        # Live step checklist: the wave scheduler marks steps running/done and the
        # reader re-renders the SAME rows, so they tick in place instead of the
        # plan growing a line per completion (a printed block is frozen).
        steps = turn_view.StepStatus()
        if getattr(self.client, "agent", None) is not None:
            self.client.agent.callbacks = [self.client.callback_handler]
            self.client.agent.styled_turn_view = True
            self.client.agent.reasoning_sink = reasoning
            self.client.agent.steps_sink = steps

        # Commands that open a full-screen dialog (or execv): a nested full-screen
        # app can't run inside the pinned app, so they go through
        # reader.run_dialog (exit → run → relaunch). Others run inline.
        # /branch is here only for its bare form (it opens a turn picker);
        # `/branch 3` still routes through the same handler, which skips the dialog.
        dialog_cmds = (
            "/load", "/config", "/model", "/params", "/features", "/memory", "/branch",
        )

        def _dispatch(line: str):
            first = line.strip().split(maxsplit=1)[0].lower() if line.strip() else ""
            if first in dialog_cmds:
                result = self._pinned_reader.run_dialog(lambda: self._dispatch(line))
            else:
                result = self._dispatch(line)
            return _ExitRepl if result is self._EXIT else None

        def _on_cancel() -> None:
            """Cooperatively signal the agent to abort NOW. The reader also injects
            a KeyboardInterrupt, but that can't preempt a worker parked in a
            blocking stream/backoff wait — this event wakes those waits instantly
            so a stalled-stream cancel isn't stuck for the whole idle/backoff."""
            agent = getattr(self.client, "agent", None)
            req = getattr(agent, "request_cancel", None) if agent else None
            if req is not None:
                req()

        def _agents_snapshot():
            agent = getattr(self.client, "agent", None)
            store = getattr(agent, "_activity", None) if agent else None
            return store.snapshot() if store is not None else []

        def _agents_get(run_id):
            agent = getattr(self.client, "agent", None)
            store = getattr(agent, "_activity", None) if agent else None
            return store.get(run_id) if store is not None else None

        def _agents_stop(run_id):
            agent = getattr(self.client, "agent", None)
            store = getattr(agent, "_activity", None) if agent else None
            return store.request_stop(run_id) if store is not None else False

        def _agents_stop_all():
            agent = getattr(self.client, "agent", None)
            store = getattr(agent, "_activity", None) if agent else None
            return store.request_stop_all() if store is not None else 0

        # Persistent footer: model · provider, launch dir, context meter. Model and
        # directory are read once — /model and /config re-exec the process, so they
        # can't change under a live footer.
        model_cfg = config.get("MODEL_ID", {}) or {}
        footer_model = str(model_cfg.get("NAME", "") or "")
        footer_provider = str(model_cfg.get("TYPE", "") or "")
        footer_cwd = os.getcwd()
        token_cache = {"key": None, "tokens": 0}

        def _context_tokens():
            """``(tokens, estimated)`` for the footer meter.

            Exact once a turn has reported its ``input_tokens``; until then a
            CACHED estimate — the footer repaints ~10x/second while the estimate
            walks the whole history, so it is recomputed only when the history
            (or the system prompt) actually changes.
            """
            agent = getattr(self.client, "agent", None)
            exact = int(getattr(agent, "_last_input_tokens", 0) or 0) if agent else 0
            if exact:
                return exact, False
            msgs = (getattr(agent, "messages", None) or []) if agent else []
            key = (len(msgs), len(getattr(self.client, "system_prompt", "") or ""))
            if token_cache["key"] != key:
                try:
                    token_cache["tokens"] = self.client._count_context_tokens()
                except Exception:
                    token_cache["tokens"] = 0
                token_cache["key"] = key
            return token_cache["tokens"], True

        def _footer(width: int):
            manager = getattr(self.client, "conversation_manager", None)
            tokens, estimated = _context_tokens()
            return status_bar.segments(
                model=footer_model,
                provider=footer_provider,
                cwd=footer_cwd,
                tokens=tokens,
                window=int(getattr(manager, "max_tokens", 0) or 0),
                estimated=estimated,
                width=width,
            )

        reader = PinnedPromptReader(
            prompt_text=lambda: HTML(self._prompt_html()),
            commands=self._completion_commands,
            history=self.command_history,
            dispatch=_dispatch,
            toolbar_text=lambda: spinner_toolbar_text(status),
            reasoning_text=lambda: reasoning.render(time.monotonic()),
            steps_text=steps.render,
            footer_text=_footer,
            on_cancel=_on_cancel,
            agents_provider=_agents_snapshot,
            agents_get=_agents_get,
            agents_stop=_agents_stop,
            agents_stop_all=_agents_stop_all,
        )

        # Route the worker-thread confirmation gate through the app (a plain
        # input() would fight the live app for stdin).
        if getattr(self.client, "agent", None) is not None:
            self.client.agent._confirm_ui = reader.confirm_ui
            # Plan-mode approval: the exit_plan_mode tool shows the plan in-app
            # and, on approval, flips plan mode off + persists the plan.
            self.client.agent._plan_approval_ui = reader.plan_approval_ui
            self.client.agent._exit_plan_mode_provider = self.client._approve_plan
            # ask_user_question: a model-initiated multiple-choice picker. Only
            # wired here (TTY) — off-TTY the tool reports itself unavailable
            # rather than blocking a scripted run on a prompt nobody sees.
            self.client.agent._question_ui = reader.question_ui
            # Background sub-agent completion: auto-trigger a delivery-only turn
            # while idle so the finished report surfaces without the user typing.
            self.client.agent._on_background_complete = (
                lambda agent_id: reader.notify_background_complete()
            )
            # Repaint the live agents panel immediately when a sub-agent records
            # activity (else it only updates on the 10Hz tick). TTY-only.
            self.client.agent._activity.on_change = reader.request_repaint
        # Exposed so _dispatch can route dialog commands through the reader.
        self._pinned_reader = reader

        interrupt_count = 0
        last_interrupt_time = 0.0
        while True:
            try:
                reader.run()
                break  # dispatch returned _ExitRepl
            except (KeyboardInterrupt, EOFError):
                interrupt_count, last_interrupt_time, should_exit = (
                    self._note_interrupt(interrupt_count, last_interrupt_time)
                )
                if not should_exit:
                    continue
                print("\nExiting...")
                break
        try:
            self.client.clear_context()
        except KeyboardInterrupt:
            pass

    @staticmethod
    def _turn_end_line(started: float, stopped: bool = False) -> str:
        """The dim end-of-turn marker for a turn that began at ``started``
        (a ``time.monotonic()`` reading)."""
        return turn_view.render_turn_end(
            time.monotonic() - started, time.time(), stopped=stopped
        )

    @staticmethod
    def _note_interrupt(count: int, last_time: float) -> tuple:
        """Advance the double-tap exit counter on a Ctrl+C/Ctrl+D.

        Returns ``(new_count, now, should_exit)``: the first press within the
        2-second window prints the hint and does NOT exit; a second press does.
        Shared by both REPL loops so the hint text lives once.
        """
        now = time.time()
        if now - last_time > 2:
            count = 0
        count += 1
        if count == 1:
            print(
                "\n\033[97m(To exit, press Ctrl+C or Ctrl+D again or type "
                "\033[92m/quit\033[97m)\033[0m"
            )
            return count, now, False
        return count, now, True

    def _expand_user_command(self, query: str) -> Any:
        """Expand ``/name args`` from the user's commands dir, or return None.

        Prints one dim line naming what ran: the prompt the model answers is the
        file's body, not what the user typed, so without it the transcript shows a
        turn whose question is nowhere on screen. Best-effort — a failure here
        leaves the line to be sent as typed.
        """
        if not query.strip().startswith("/"):
            return None
        try:
            match = self.user_commands.expand(query)
        except Exception as e:
            logger.warning(f"User command expansion failed: {one_line(e)}")
            return None
        if match is None:
            return None
        cmd, prompt = match
        print(turn_view.render_command_expansion(cmd.name, cmd.path))
        return prompt

    def _expand_mentions(self, query: str) -> str:
        """Attach every ``@path`` in the line to the prompt, announcing each.

        One dim line per mention, because the attachment is otherwise invisible:
        a path that resolved and one that was typo'd read identically on screen,
        and the second leaves the model guessing about a file it never saw.
        Best-effort — a failure here sends the line exactly as typed.
        """
        if "@" not in query:
            return query
        try:
            expanded, mentions = file_mentions.expand(query)
        except Exception as e:
            logger.warning(f"File mention expansion failed: {one_line(e)}")
            return query
        ledger = getattr(getattr(self.client, "agent", None), "files", None)
        for mention in mentions:
            print(turn_view.render_mention_notice(mention.label))
            # An attached FILE is a touch like any other, so /files can answer
            # "what did I hand it?" — a directory listing names no single file.
            if ledger is not None and mention.kind == "file" and mention.path:
                ledger.record(str(mention.path), file_ledger.ATTACHED)
        return expanded

    def _dispatch(self, query: str):
        """Handle one submitted line (slash command or query); returns
        :data:`_EXIT` to end the loop, else ``None``. Shared by both loops."""
        if query.lower() in ["/exit", "/quit"]:
            return self._EXIT

        if query.lower() == "/clear":
            self.client.clear_context()
            if config.get("ENABLE_RAG", False):
                self.client._initialize_rag_session()
            self.client._initialize_chunk_cache()
            # Wipe screen + scrollback so /clear is a true fresh start.
            self.__clear_screen()
            self.__welcome_message()
            print("Context cleared!")
            return None

        # Save to conversations/ by default, or to an optional path.
        if query.lower() == "/save" or query.lower().startswith("/save "):
            timestamp = self.client.session_id.split("_", 1)[1]
            save_path = query[len("/save"):].strip() or None
            self.client.save_conversation(timestamp, path=save_path)
            return None

        if query.lower() == "/help":
            print("\n" + self._help_text() + "\n")
            return None

        if query.lower() == "/usage":
            print("\n" + self.client.usage_report() + "\n")
            return None

        if query.lower() == "/context":
            print("\n" + self.client.context_report() + "\n")
            return None

        # /rewind — the conversation only: no file on disk is rolled back.
        if query.lower() == "/rewind":
            print("\n" + self.client.rewind_turn() + "\n")
            return None

        if query.lower() == "/files":
            print("\n" + self.client.files_report() + "\n")
            return None

        # /diff [path] — read-only: it never stages, stashes or checks anything out.
        if query.lower() == "/diff" or query.lower().startswith("/diff "):
            print("\n" + self.client.diff_report(query[len("/diff"):].strip()) + "\n")
            return None

        if query.lower() == "/copy" or query.lower().startswith("/copy "):
            print("\n" + self.client.copy_last(query[len("/copy"):].strip()) + "\n")
            return None

        # /export [md|txt] [path] — a shareable transcript, not a reloadable file.
        if query.lower() == "/export" or query.lower().startswith("/export "):
            self._handle_export_command(query[len("/export"):].strip())
            return None

        # /branch [turn] — fork this session and continue in the copy.
        if query.lower() == "/branch" or query.lower().startswith("/branch "):
            self._handle_branch_command(query[len("/branch"):].strip())
            return None

        # /config, /model, /params rewrite config.yaml then restart in place so
        # every setting (incl. boot-time MCP toggles) takes effect.
        if query.lower() == "/config":
            if run_reconfigure() is not None:
                self._restart_in_place()
            return None

        if query.lower() == "/model":
            if run_model_override() is not None:
                self._restart_in_place()
            return None

        # /params only edits inference knobs (temperature, top_p, …) — nothing the
        # MCP subprocess fixed at boot — so it reloads in place and KEEPS the
        # conversation. A restart here used to discard the chat (and, after a
        # `--resume`, left the restored history only in an abandoned file).
        if query.lower() == "/params":
            if run_params_override() is not None:
                if self.client.reload_inference_params():
                    print(
                        "\n\033[92mNew inference parameters applied.\033[0m "
                        "This conversation continues.\n"
                    )
                else:
                    # Rebuilding failed: the old model is still live and correct,
                    # so fall back to the restart rather than run on a half-applied
                    # config.
                    self._restart_in_place()
            return None

        if query.lower() == "/features":
            if run_features_override() is not None:
                self._restart_in_place()
            return None

        if query.lower() == "/mcp":
            self._print_mcp_status()
            return None

        if query.lower() == "/hooks":
            print("\n" + self.client.hooks_report() + "\n")
            return None

        if query.lower() == "/doctor":
            print("\n" + self.client.doctor_report() + "\n")
            return None

        # /rename [title] — label this session in the --resume picker.
        if query.lower() == "/rename" or query.lower().startswith("/rename "):
            self._handle_rename_command(query[len("/rename"):].strip())
            return None

        if query.lower() == "/skills" or query.lower().startswith("/skills "):
            self._handle_skills_command(query[len("/skills"):].strip())
            return None

        if query.lower() == "/memory" or query.lower().startswith("/memory "):
            self._handle_memory_command(query[len("/memory"):].strip())
            return None

        # Toggle enforced, read-only plan mode (mutating/exec tools blocked).
        if query.lower() == "/plan":
            self.client.plan_mode_active = not self.client.plan_mode_active
            if self.client.plan_mode_active:
                # Re-entering plan mode drops any bash pre-approvals AND the
                # full-toolset route pin from a prior approved plan — a fresh plan
                # re-declares what it needs and re-plans read-only.
                agent = getattr(self.client, "agent", None)
                if agent is not None:
                    agent._preapproved_bash = []
                    agent._execute_plan_route = False
                print(
                    "\n\033[93m🔒 Plan mode ON\033[0m — read-only. I'll research "
                    "and present a plan for your approval; approving turns plan "
                    "mode off and I execute it. Read-only shell commands (ls, "
                    "cat, grep, git status/log/diff) still run; file edits and "
                    "mutating commands are blocked. Type /plan again to exit.\n"
                )
            else:
                print(
                    "\n\033[92m🔓 Plan mode OFF\033[0m — changes allowed again.\n"
                )
            return None

        # /compact [focus instructions]
        if query.lower() == "/compact" or query.lower().startswith("/compact "):
            focus = query[len("/compact"):].strip()
            did = self.client.compact_conversation(focus)
            print(
                "Conversation compacted."
                if did
                else "Nothing to compact yet."
            )
            return None

        # /load: no path → pick from saved list; with a path → load directly.
        if query.lower() == "/load" or query.lower().startswith("/load "):
            file_path = query[len("/load"):].strip()
            if not file_path:
                file_path = self._select_saved_conversation()
                if not file_path:
                    return None  # nothing to load, or user cancelled
            if self.client.load_conversation(file_path):
                print("Conversation loaded successfully!")
            else:
                print_error("Failed to load conversation. Check the file path.")
            return None

        # A user-defined command (~/.mnemoai/commands/<name>.md) expands INTO the
        # prompt and then runs as an ordinary turn — checked after every built-in,
        # so a file can never shadow one, and an unknown /thing still falls through
        # to being sent as prose.
        expanded = self._expand_user_command(query)
        if expanded is not None:
            query = expanded

        # `@path` mentions attach the file itself. After the command expansion, so
        # a command's own body can mention files; after every built-in, so a path
        # typed as an argument to one (`/export @dir`) keeps its literal meaning.
        query = self._expand_mentions(query)

        if not query.strip():
            # An empty line is normally rejected — EXCEPT a delivery-only turn
            # auto-enqueued when a background sub-agent finished (surfaces its
            # report without the user typing). Run it only when a completion is
            # actually undelivered; otherwise it's a stray blank line.
            agent = getattr(self.client, "agent", None)
            has_undelivered = getattr(agent, "has_undelivered_background", None)
            if has_undelivered is not None and has_undelivered():
                started = time.monotonic()
                self.client.query("")  # delivery-only turn
                # Marked like any other turn — this one appeared without the user
                # typing, so where it ends is even less obvious.
                print("\n" + self._turn_end_line(started))
                notify.notify_turn_end(time.monotonic() - started)
            else:
                print("Input cannot be empty. Please try again.")
            return None

        use_immediate_storage = config.get("EPISODIC_MEMORY", {}).get(
            "IMMEDIATE_STORAGE", True
        )

        if self.client.episodic_memory and not use_immediate_storage:
            # Legacy mode: store the previous interaction before this query.
            self.__store_episode_in_episodic_memory(query)
        elif not self.client.episodic_memory:
            logger.debug("Episodic memory is disabled")

        started = time.monotonic()
        try:
            response = self.client.query(query)

            # Post-answer learning side effects are BEST-EFFORT: the user already
            # has their answer, so a failure here (e.g. a moved/locked ChromaDB —
            # SQLite code 1032 "readonly database moved", a backup/sync racing the
            # store dir) must be logged quietly, NOT surfaced as a turn error.
            # Each is guarded independently so one failing doesn't skip the others.
            if self.client.episodic_memory and use_immediate_storage:
                try:
                    self.__store_current_episode_immediately(query, response)
                except Exception as e:
                    logger.warning(f"Episodic storage failed (non-fatal): {e}")

            if self.client.reflector:
                try:
                    self.client.reflect_and_learn(query)
                except Exception as e:
                    logger.warning(f"Reflection failed (non-fatal): {e}")

            # Auto-distill durable facts into MEMORY.md (opt-in; runs in the
            # background so it never blocks the turn). getattr-guarded so a
            # minimal/stub client without the method still works.
            extract = getattr(self.client, "auto_extract_memory", None)
            if callable(extract):
                try:
                    extract(query, response)
                except Exception as e:
                    logger.warning(f"Memory auto-extraction failed (non-fatal): {e}")

            # Close the turn with a marker line: the answer above it is complete.
            # Timed to HERE, not to where query() returned — the learning steps
            # above still hold the prompt, so they are part of the wait.
            if response == "Operation was cancelled.":
                # Resolve the transient "(cancelling…)" line to a final state.
                print(self._turn_end_line(started, stopped=True))
            else:
                print("\n" + self._turn_end_line(started))
                # …and, if the turn ran long enough that the user has plausibly
                # looked away, ring the terminal. Deliberately NOT on the
                # cancelled path above: their hand is already on the keyboard.
                notify.notify_turn_end(time.monotonic() - started)
        except KeyboardInterrupt:
            # The cancel landed in this frame instead of being turned into a
            # "cancelled" response (Esc between steps, Ctrl+C in the plain loop).
            # The UI has already written "(cancelling…)", so it needs the same
            # resolution the graceful path gets — otherwise this is the one turn
            # that ends with nothing at all.
            print(self._turn_end_line(started, stopped=True))
            return None
        except Exception as e:
            # Full traceback to the log FILE only (console=False): the red line
            # below is the one report the user should see for this failure.
            logger.error(
                f"Error processing query: {str(e)}",
                exc_info=True,
                extra={"console": False},
            )
            print_error(f"{type(e).__name__}: {one_line(e)}")
        return None
