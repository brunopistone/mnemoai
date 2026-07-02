"""Chat interface handling for the application."""

import datetime as _dt
import os
import re
import sys
import time
from typing import Any

from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory

from mnemoai.client.memory.episodic_memory import (
    extract_tools_from_messages,
    is_task_successful,
)
from mnemoai.client.memory.memory_store import MemoryStore
from mnemoai.client.memory.skill_store import SkillStore
from mnemoai.client.ui.tui import (
    PinnedPromptReader,
    _ExitRepl,
    confirm_inline,
    select_from_list,
)
from mnemoai.utils.config import config
from mnemoai.utils.configurator import (
    run_model_override,
    run_params_override,
    run_reconfigure,
)
from mnemoai.utils.console import print_error
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import (
    mcp_config_path,
    memory_file_path,
    skills_dir,
)


class ChatInterface:
    """Handles chat interface and user interaction."""

    def __init__(self, client: Any) -> None:
        """Initialize chat interface.

        Args:
            client: LangGraphClient instance
        """
        self.client = client

        # Persistent command history for arrow key navigation
        self.command_history = InMemoryHistory()

    def _prompt_html(self) -> str:
        """Build the input-prompt line.

        Shows a compact 🔒 plan tag while plan mode is active so it's clear that
        mutations are blocked. The model name is intentionally NOT shown — it can
        be long (e.g. ``brnpistone/Qwen3.5-4B-AgentCoder-q6-k:latest``) and would
        crowd the input line; use ``/model`` to see/change it.
        """
        if getattr(self.client, "plan_mode_active", False):
            return "<ansiyellow>🔒 plan</ansiyellow> <ansiblue>></ansiblue> "
        return "<ansiblue>></ansiblue> "

    # ASCII wordmark shown on launch (ANSI "Shadow" style). Rendered in the
    # brand indigo. Kept as data so the banner is easy to restyle/replace.
    _BANNER = [
        "███╗   ███╗███╗   ██╗███████╗███╗   ███╗ ██████╗      █████╗ ██╗",
        "████╗ ████║████╗  ██║██╔════╝████╗ ████║██╔═══██╗    ██╔══██╗██║",
        "██╔████╔██║██╔██╗ ██║█████╗  ██╔████╔██║██║   ██║    ███████║██║",
        "██║╚██╔╝██║██║╚██╗██║██╔══╝  ██║╚██╔╝██║██║   ██║    ██╔══██║██║",
        "██║ ╚═╝ ██║██║ ╚████║███████╗██║ ╚═╝ ██║╚██████╔╝    ██║  ██║██║",
        "╚═╝     ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝     ╚═╝ ╚═════╝     ╚═╝  ╚═╝╚═╝",
    ]

    # Command groups for the welcome box: (heading, [(command, description)]).
    _COMMAND_GROUPS = [
        ("Configure", [
            ("/config", "Reconfigure config.yaml (overwrites it)"),
            ("/model", "Override one model (LLM/vision/embeddings)"),
            ("/params", "Tune model inference params (temp, top_p, …)"),
            ("/mcp", "List configured MCP servers & tools"),
            ("/skills [name]", "List installed skills (or preview one)"),
        ]),
        ("Conversation", [
            ("/clear", "Clear conversation context"),
            ("/compact [focus]", "Summarize & shrink context"),
            ("/memory [clear]", "View (or clear) persistent memory"),
            ("/plan", "Toggle read-only plan mode (blocks edits/bash)"),
            ("/save [path]", "Save conversation (optional file/dir path)"),
            ("/load [path]", "Load a saved conversation (lists saved if no path)"),
        ]),
        ("Exit", [
            ("/exit, /quit", "Exit the application"),
        ]),
    ]

    # Slash commands available for autocomplete: (command, description),
    # derived from the welcome-box groups so the two never drift. The display
    # labels there carry arg hints / alternates (e.g. "/compact [focus]",
    # "/exit, /quit"); here we list the actual insertable command tokens.
    _COMMANDS = [
        ("/config", "Reconfigure config.yaml (overwrites it)"),
        ("/model", "Override one model (LLM/vision/embeddings)"),
        ("/params", "Tune model inference params (temperature, top_p, …)"),
        ("/mcp", "List configured MCP servers & their tools"),
        ("/skills", "List installed skills (/skills <name> to preview)"),
        ("/clear", "Clear conversation context"),
        ("/compact", "Summarize & shrink context (optional focus)"),
        ("/memory", "View persistent memory (/memory clear to wipe)"),
        ("/plan", "Toggle read-only plan mode (blocks edits & shell)"),
        ("/save", "Save conversation (/save [path])"),
        ("/load", "Load a saved conversation (/load lists saved)"),
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

    # Slash-command autocompletion lives in mnemoai.client.ui.tui
    # (SlashCommandCompleter), shared by the inline TUI input field.

    _ANSI_RE = re.compile(r"\033\[[0-9;]*m")

    def __clear_screen(self) -> None:
        """Clear the terminal screen and scrollback, cursor to home.

        Skipped when stdout isn't a TTY (piped/redirected) so logs stay clean.
        """
        if not (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
            return
        # \033[3J clears scrollback, \033[H homes the cursor, \033[2J clears
        # the visible screen.
        print("\033[3J\033[H\033[2J", end="", flush=True)

    def __welcome_message(self) -> None:
        """Display the launch banner + a framed, grouped command list."""
        C = self._C

        def vlen(s: str) -> int:
            """Visible length (ANSI escapes don't occupy columns)."""
            return len(self._ANSI_RE.sub("", s))

        # Inner width: at least the wordmark banner width (64), but widen to fit
        # the longest command row ("  " + padded cmd + "  " + desc) so no row
        # overflows the box border.
        cmd_w = max(vlen(c) for _, items in self._COMMAND_GROUPS for c, _ in items)
        longest_row = max(
            2 + cmd_w + 2 + vlen(desc)
            for _, items in self._COMMAND_GROUPS for _, desc in items
        )
        W = max(64, longest_row)

        def row(content: str = "") -> None:
            """Print one framed row, padding to the visible inner width."""
            pad = " " * max(0, W - vlen(content))
            print(f"{C['border']}│{C['reset']} {content}{pad} {C['border']}│{C['reset']}")

        top = f"{C['border']}╭{'─' * (W + 2)}╮{C['reset']}"
        sep = f"{C['border']}├{'─' * (W + 2)}┤{C['reset']}"
        bot = f"{C['border']}╰{'─' * (W + 2)}╯{C['reset']}"

        # --- Wordmark banner (indigo ≈ #5f5fff via 256-color 63) ---
        print()
        for line in self._BANNER:
            print(f"\033[38;5;63m{line}\033[0m")
        print(f"{C['dim']}" + "local agentic AI assistant · learns & remembers".center(W + 4) + C["reset"])
        print()

        # --- Framed command list ---
        print(top)

        for gi, (heading, items) in enumerate(self._COMMAND_GROUPS):
            if gi:
                row()  # blank spacer between groups
            row(f"{C['head']}{heading}{C['reset']}")
            for cmd, desc in items:
                padded_cmd = cmd + " " * (cmd_w - vlen(cmd))
                row(f"  {C['cmd']}{padded_cmd}{C['reset']}  {C['text']}{desc}{C['reset']}")

        print(sep)
        row(f"{C['dim']}Ctrl+J{C['reset']} for new lines · {C['dim']}Enter{C['reset']} to submit")
        print(bot + "\n")

    def __store_episode_in_episodic_memory(self, query: str) -> None:
        """Evaluate and store previous interaction in episodic memory if successful.
        Args:
            query: Current user query
        """
        logger.debug("Episodic memory is enabled")
        if (
            self.client.previous_query
            and self.client.previous_response
            and self.client.previous_messages
        ):
            logger.debug("Evaluating previous interaction for episodic storage")
            logger.debug(f"Previous query: {self.client.previous_query[:100]}...")
            logger.debug(f"Current query: {query[:100]}...")

            # Extract tools used
            tools_used = extract_tools_from_messages(self.client.previous_messages)

            # Only store if there was actual work done (tools used or substantial response)
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

                # Find the initial user query (first user message in conversation)
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

                # Store with full conversation (agent.messages format)
                self.client.episodic_memory.store_episode(
                    task=initial_query,
                    tools_used=tools_used,
                    outcome="success",
                )
                logger.debug("✓ Episode stored successfully")

                # Record tool outcome for profile learning
                if config.get("PROFILE", {}).get("USE_PROFILING", False):
                    intent = self.client.profile_manager.classify_intent(initial_query)
                    self.client.profile_manager.record_tool_outcome(
                        intent, tools_used, True
                    )
            else:
                logger.debug(
                    "✗ Previous task not marked as successful - skipping storage"
                )
        else:
            logger.debug("No previous interaction to evaluate")

    def __store_current_episode_immediately(self, query: str, response: str) -> None:
        """Store CURRENT interaction in episodic memory immediately after response.

        This is the new immediate storage mode that doesn't wait for the next query.

        Args:
            query: Current user query
            response: Agent's response
        """
        if not self.client.agent or not self.client.agent.messages:
            logger.debug("No agent messages to evaluate")
            return

        if not response or not response.strip():
            logger.debug("✗ Skipping storage - empty response")
            return

        messages = self.client.agent.messages.copy()

        # Extract tools used
        tools_used = extract_tools_from_messages(messages)

        # Get minimum length threshold from config
        min_length = config.get("EPISODIC_MEMORY", {}).get("MIN_TOOLS_OR_LENGTH", 300)

        # Quality filter: skip if no tools and response too short
        if not tools_used and len(response) < min_length:
            logger.debug(
                f"✗ Skipping storage - no tools used and response too short "
                f"({len(response)} < {min_length} chars)"
            )
            return

        # Check success (no next_user_message since this is immediate)
        if is_task_successful(response, messages, next_user_message=None):
            logger.debug("✓ Task marked as successful - storing immediately")
            logger.debug(f"Tools used: {[t.get('name') for t in tools_used]}")

            # Use the query as-is (no need to extract from messages)
            self.client.episodic_memory.store_episode(
                task=query, tools_used=tools_used, outcome="success"
            )
            logger.debug("✓ Episode stored successfully (immediate mode)")

            # Record tool outcome for profile learning
            if config.get("PROFILE", {}).get("USE_PROFILING", False):
                intent = self.client.profile_manager.classify_intent(query)
                self.client.profile_manager.record_tool_outcome(
                    intent, tools_used, True
                )
        else:
            logger.debug("✗ Task not marked as successful - skipping storage")

    def _print_mcp_status(self) -> None:
        """Show the configured MCP servers (built-in + external) and tool counts.

        Reads the live ``MultiMCPClient`` members for connection status and the
        loaded tool list, plus ``mcp.json`` so the user sees where to declare
        more servers. External tools may appear namespaced as ``server__tool``
        when their name collides with a built-in one.
        """
        members = getattr(self.client.mcp_client, "_members", [])
        tools = self.client.tools or []
        print("\nMCP servers:")
        if members:
            for name, _ in members:
                # External tools that collided are exposed as "name__tool".
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
        """List saved conversations (newest first) and let the user pick one.

        Returns the chosen file path (str), or None if there are none or the
        user cancels. Used by ``/load`` with no argument. On a TTY this is a
        centered radiolist (arrow-key selection); non-TTY falls back to a
        numbered ``input()`` prompt — both via :func:`tui.select_from_list`.
        """
        files = self.client.list_saved_conversations()
        if not files:
            print(
                "No saved conversations found. Use /save first, or "
                "/load <path> to load from a specific file."
            )
            return None

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

        shown = files[:20]  # cap the menu; older ones load via /load <path>
        options = [
            (str(p), f"{p.name}  ({_ago(p.stat().st_mtime)})") for p in shown
        ]
        title = "Load conversation"
        if len(files) > len(shown):
            title += f"  (showing {len(shown)} of {len(files)}; /load <path> for older)"
        return select_from_list(title, options)

    def _handle_memory_command(self, arg: str) -> None:
        """Handle ``/memory`` (view) and ``/memory clear``.

        The agent normally curates MEMORY.md itself via the memory tool; this
        command lets the user inspect it, or wipe it. Reuses ``MemoryStore``.
        """
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
        """Handle ``/skills`` (list) and ``/skills <name>`` (preview a body).

        Skills are authored ``SKILL.md`` instruction packs the model loads on
        demand via the ``use_skill`` tool; this command lets the user see what's
        installed and preview one. Reuses ``SkillStore``.
        """
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
        """Restart the current process so reloaded config takes full effect.

        Replaces the running process image with a fresh one via ``os.execv``
        (same command, same terminal — no new window, nothing to re-type).
        This is the only way to reliably apply *every* setting, since the MCP
        server subprocess decides its tool set at boot and the model/memory
        are wired at startup. The in-memory conversation is intentionally
        dropped (a model switch shouldn't carry old history forward).

        ``os.execv`` does not reap child processes, so the MCP subprocess is
        shut down explicitly first to avoid orphaning it.
        """
        print("\nRestarting to apply the new configuration...\n")
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

    def run_chat_loop(self) -> None:
        """Run the main chat loop.

        On a TTY this is the pinned-input UI (:meth:`_run_pinned_loop`): the ``>``
        prompt stays fixed at the bottom while each query runs on a worker thread
        and its output — answer, styled reasoning/tool blocks — streams above it in
        native scrollback. A non-TTY session (pipes / CI / tests) uses a plain
        ``input()`` loop (:meth:`_plain_loop`). Either way the submitted line goes
        through :meth:`_dispatch`.
        """
        self.__welcome_message()

        is_tty = (
            hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
            and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        )
        if is_tty:
            self._run_pinned_loop()
        else:
            self._plain_loop()

    def _plain_loop(self) -> None:
        """Plain ``input()`` REPL for non-TTY use (pipes / CI / tests).

        No prompt_toolkit app — reads a line, dispatches it, repeats.
        Ctrl+C / Ctrl+D twice exits.
        """
        interrupt_count = 0
        last_interrupt_time = 0

        while True:
            try:
                query = input("> ")
                interrupt_count = 0
            except (KeyboardInterrupt, EOFError):
                current_time = time.time()
                if current_time - last_interrupt_time > 2:
                    interrupt_count = 0
                interrupt_count += 1
                last_interrupt_time = current_time
                if interrupt_count == 1:
                    print(
                        "\n\033[97m(To exit, press Ctrl+C or Ctrl+D again or type "
                        "\033[92m/quit\033[97m)\033[0m"
                    )
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
        """Drive the pinned-input REPL (the default TTY UI).

        The `>` prompt + an animated status toolbar stay pinned while each query
        runs on a worker thread and streams output above them. A spinner *sink*
        is attached so the spinner-control code (in the agent/callback) flips
        toolbar state instead of writing `\\r` (which would fight the pinned
        prompt's redraw). Ctrl+C / Ctrl+D twice exits.
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
        if getattr(self.client, "agent", None) is not None:
            self.client.agent.callbacks = [self.client.callback_handler]
            # Pinned UI: collapsed "Thought for Ns…" block + styled tool blocks.
            self.client.agent.styled_turn_view = True

        # Slash commands that open a full-screen dialog (or restart via execv).
        # A nested full-screen app can't run inside the pinned app, so these are
        # run via reader.run_dialog: it EXITS the app (terminal returns to cooked
        # mode), runs the command with the normal dialogs, then relaunches the
        # pinned app. Query turns and other commands run inline as usual.
        dialog_cmds = ("/load", "/config", "/model", "/params", "/memory")

        def _dispatch(line: str):
            # Reuse the shared slash/query handler; map its exit sentinel to the
            # REPL's. Ctrl+C inside a turn is swallowed by _dispatch already.
            first = line.strip().split(maxsplit=1)[0].lower() if line.strip() else ""
            if first in dialog_cmds:
                result = self._pinned_reader.run_dialog(lambda: self._dispatch(line))
            else:
                result = self._dispatch(line)
            return _ExitRepl if result is self._EXIT else None

        reader = PinnedPromptReader(
            prompt_text=lambda: HTML(self._prompt_html()),
            commands=self._COMMANDS,
            history=self.command_history,
            dispatch=_dispatch,
            toolbar_text=lambda: spinner_toolbar_text(status),
            on_cancel=lambda: None,  # Esc interrupt is handled inside the reader
        )

        # Route the worker-thread confirmation gate through the app (a plain
        # input() would fight the live app for stdin). The reader suspends the
        # app, prompts above it, and returns yes/no/all.
        if getattr(self.client, "agent", None) is not None:
            self.client.agent._confirm_ui = reader.confirm_ui
        # Dialogs (/load, /config, …) also run on the worker thread; expose the
        # reader so _dispatch can route them through the app (see _in_pinned_app).
        self._pinned_reader = reader

        interrupt_count = 0
        last_interrupt_time = 0.0
        while True:
            try:
                reader.run()
                break  # dispatch returned _ExitRepl
            except (KeyboardInterrupt, EOFError):
                current_time = time.time()
                if current_time - last_interrupt_time > 2:
                    interrupt_count = 0
                interrupt_count += 1
                last_interrupt_time = current_time
                if interrupt_count == 1:
                    print(
                        "\n\033[97m(To exit, press Ctrl+C or Ctrl+D again or type "
                        "\033[92m/quit\033[97m)\033[0m"
                    )
                    continue
                print("\nExiting...")
                break
        try:
            self.client.clear_context()
        except KeyboardInterrupt:
            pass

    def _dispatch(self, query: str):
        """Handle one submitted line: slash command or query.

        Returns :data:`_EXIT` to request loop termination, else ``None``.
        Shared by the TUI and plain loops.
        """
        # Handle special commands
        if query.lower() in ["/exit", "/quit"]:
            return self._EXIT

        if query.lower() == "/clear":
            self.client.clear_context()
            if config.get("ENABLE_RAG", False):
                self.client._initialize_rag_session()
            self.client._initialize_chunk_cache()
            # Wipe the screen + scrollback and re-show the welcome banner so
            # /clear is a true fresh start, not "Context cleared!" appended
            # below the old conversation.
            self.__clear_screen()
            self.__welcome_message()
            print("Context cleared!")
            return None

        # /save [path] — save to conversations/ by default, or to an
        # optional file/directory path.
        if query.lower() == "/save" or query.lower().startswith("/save "):
            timestamp = self.client.session_id.split("_", 1)[1]
            save_path = query[len("/save"):].strip() or None
            self.client.save_conversation(timestamp, path=save_path)
            return None

        # Reconfigure: rewrite config.yaml via the interactive configurator,
        # then restart the process in place so every setting (including MCP
        # tool toggles, which are decided at subprocess boot) takes effect.
        if query.lower() == "/config":
            if run_reconfigure() is not None:
                self._restart_in_place()
            return None

        # Override just one model section (LLM / vision / embeddings),
        # leaving the rest of config.yaml untouched, then restart in place.
        if query.lower() == "/model":
            if run_model_override() is not None:
                self._restart_in_place()
            return None

        # Tune a model's inference parameters (temperature, top_p, penalties,
        # reasoning, stop, stream, …) in place, then restart so the new
        # generation settings take effect.
        if query.lower() == "/params":
            if run_params_override() is not None:
                self._restart_in_place()
            return None

        # List configured MCP servers (built-in + external from mcp.json).
        if query.lower() == "/mcp":
            self._print_mcp_status()
            return None

        # List installed skills, or preview one: /skills [name].
        if query.lower() == "/skills" or query.lower().startswith("/skills "):
            self._handle_skills_command(query[len("/skills"):].strip())
            return None

        # View or clear the curated persistent memory (MEMORY.md).
        if query.lower() == "/memory" or query.lower().startswith("/memory "):
            self._handle_memory_command(query[len("/memory"):].strip())
            return None

        # Toggle enforced, read-only plan mode (mutating/exec tools blocked).
        if query.lower() == "/plan":
            self.client.plan_mode_active = not self.client.plan_mode_active
            if self.client.plan_mode_active:
                print(
                    "\n\033[93m🔒 Plan mode ON\033[0m — read-only. I'll research "
                    "and present a plan. Read-only shell commands (ls, cat, "
                    "grep, git status/log/diff) still run; file edits and "
                    "mutating commands are blocked. Type /plan again to exit "
                    "and allow changes.\n"
                )
            else:
                print(
                    "\n\033[92m🔓 Plan mode OFF\033[0m — changes allowed again.\n"
                )
            return None

        # Manually compact the conversation: /compact [focus instructions]
        if query.lower() == "/compact" or query.lower().startswith("/compact "):
            focus = query[len("/compact"):].strip()
            did = self.client.compact_conversation(focus)
            print(
                "Conversation compacted."
                if did
                else "Nothing to compact yet."
            )
            return None

        # Handle /load command. With no path, list saved conversations and
        # let the user pick one; with a path, load it directly.
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

        if not query.strip():
            print("Input cannot be empty. Please try again.")
            return None

        # Store previous interaction if using delayed mode (legacy)
        use_immediate_storage = config.get("EPISODIC_MEMORY", {}).get(
            "IMMEDIATE_STORAGE", True
        )

        if self.client.episodic_memory and not use_immediate_storage:
            # Legacy mode: store previous interaction before current query
            self.__store_episode_in_episodic_memory(query)
        elif not self.client.episodic_memory:
            logger.debug("Episodic memory is disabled")

        try:
            response = self.client.query(query)

            # Store CURRENT interaction immediately after response (new mode)
            if self.client.episodic_memory and use_immediate_storage:
                self.__store_current_episode_immediately(query, response)

            # ACE Reflection: learn from this interaction
            if self.client.reflector:
                self.client.reflect_and_learn(query)

            if response != "Operation was cancelled.":
                print("\n")
        except KeyboardInterrupt:
            return None
        except Exception as e:
            # Full traceback to the logger (stderr, LOG_LEVEL=DEBUG to see);
            # the user gets a concise red line with the actual cause.
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            print_error(f"Error: {e}")
        return None
