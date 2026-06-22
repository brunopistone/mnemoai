"""Chat interface handling for the application."""

import asyncio
import os
import re
import sys
import time
from typing import Any, Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings

from mnemoai.client.memory.episodic_memory import (
    extract_tools_from_messages,
    is_task_successful,
)
from mnemoai.utils.config import config
from mnemoai.utils.console import print_error
from mnemoai.utils.logger import logger


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

    def get_multiline_input(self) -> str:
        """Get input with Ctrl+J for new lines, Enter to submit.

        Returns:
            User input string
        """
        bindings = KeyBindings()

        @bindings.add("c-j")
        def _(event: Any) -> None:
            """Handle Ctrl+J to insert newline.

            Args:
                event: Key binding event
            """
            event.app.current_buffer.insert_text("\n")

        session = PromptSession(
            history=self.command_history,
            key_bindings=bindings,
            multiline=False,
            completer=self._SlashCommandCompleter(self._COMMANDS),
            complete_while_typing=True,
        )

        try:
            return session.prompt(HTML("<ansiblue>></ansiblue> "))
        except KeyboardInterrupt:
            raise

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
        ]),
        ("Conversation", [
            ("/clear", "Clear conversation context"),
            ("/compact [focus]", "Summarize & shrink context"),
            ("/memory [clear]", "View (or clear) persistent memory"),
            ("/plan", "Toggle read-only plan mode (blocks edits/bash)"),
            ("/save", "Save current conversation"),
            ("/load <path>", "Load a saved conversation"),
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
        ("/clear", "Clear conversation context"),
        ("/compact", "Summarize & shrink context (optional focus)"),
        ("/memory", "View persistent memory (/memory clear to wipe)"),
        ("/plan", "Toggle read-only plan mode (blocks edits & shell)"),
        ("/save", "Save current conversation"),
        ("/load", "Load a saved conversation (/load <path>)"),
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

    class _SlashCommandCompleter(Completer):
        """Suggest slash commands, but only when the line starts with '/'.

        Keeps autocomplete out of the way of normal chat: a message that
        doesn't begin with '/' yields no completions. Matches the typed prefix
        against the command list and shows each command's description as meta.
        """

        def __init__(self, commands):
            self._commands = commands

        def get_completions(self, document, complete_event) -> Iterable[Completion]:
            text = document.text_before_cursor
            # Only complete a single leading token that starts with '/'
            # (don't fire mid-sentence or after a space).
            if not text.startswith("/") or " " in text:
                return
            for cmd, desc in self._commands:
                if cmd.startswith(text):
                    yield Completion(
                        cmd,
                        start_position=-len(text),  # replace the typed prefix
                        display=cmd,
                        display_meta=desc,
                    )

    _ANSI_RE = re.compile(r"\033\[[0-9;]*m")

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
            logger.debug(f"Evaluating previous interaction for episodic storage")
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
        from mnemoai.utils.paths import mcp_config_path

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

    def _handle_memory_command(self, arg: str) -> None:
        """Handle ``/memory`` (view) and ``/memory clear``.

        The agent normally curates MEMORY.md itself via the memory tool; this
        command lets the user inspect it, or wipe it. Reuses ``MemoryStore``.
        """
        from mnemoai.client.memory.memory_store import MemoryStore
        from mnemoai.utils.paths import memory_file_path

        store = MemoryStore()
        sub = arg.strip().lower()

        if sub == "clear":
            if not store.read().strip():
                print("Memory is already empty.")
                return
            try:
                answer = input("  Clear ALL persistent memory? (y/N): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                answer = ""
            if answer in ("y", "yes"):
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

    def run_chat_loop(self) -> None:
        """Run the main chat loop.

        Returns:
            None
        """
        self.__welcome_message()

        interrupt_count = 0
        last_interrupt_time = 0

        while True:
            try:
                query = self.get_multiline_input()
                interrupt_count = 0
            except (KeyboardInterrupt, EOFError):
                current_time = time.time()

                if current_time - last_interrupt_time > 2:
                    interrupt_count = 0

                interrupt_count += 1
                last_interrupt_time = current_time

                if interrupt_count == 1:
                    print("\n> ")
                    print(
                        "\033[97m(To exit the CLI, press Ctrl+C or Ctrl+D again or type \033[92m/quit\033[97m)"
                    )
                    try:
                        loop = asyncio.get_event_loop()
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                    except:
                        pass
                    continue
                else:
                    print("\nExiting...")
                    try:
                        self.client.clear_context()
                    except KeyboardInterrupt:
                        pass  # User interrupted cleanup, just exit
                    break

            # Handle special commands
            if query.lower() in ["/exit", "/quit"]:
                try:
                    self.client.clear_context()  # This will flush RAG
                except KeyboardInterrupt:
                    pass  # User interrupted cleanup, just exit
                break

            if query.lower() == "/clear":
                self.client.clear_context()
                if config.get("ENABLE_RAG", False):
                    self.client._initialize_rag_session()
                self.client._initialize_chunk_cache()
                print("Context cleared!")
                continue

            if query.lower() == "/save":
                timestamp = self.client.session_id.split("_", 1)[1]
                self.client.save_conversation(timestamp)
                continue

            # Reconfigure: rewrite config.yaml via the interactive configurator,
            # then restart the process in place so every setting (including MCP
            # tool toggles, which are decided at subprocess boot) takes effect.
            if query.lower() == "/config":
                from mnemoai.utils.configurator import run_reconfigure

                if run_reconfigure() is not None:
                    self._restart_in_place()
                continue

            # Override just one model section (LLM / vision / embeddings),
            # leaving the rest of config.yaml untouched, then restart in place.
            if query.lower() == "/model":
                from mnemoai.utils.configurator import run_model_override

                if run_model_override() is not None:
                    self._restart_in_place()
                continue

            # Tune a model's inference parameters (temperature, top_p, penalties,
            # reasoning, stop, stream, …) in place, then restart so the new
            # generation settings take effect.
            if query.lower() == "/params":
                from mnemoai.utils.configurator import run_params_override

                if run_params_override() is not None:
                    self._restart_in_place()
                continue

            # List configured MCP servers (built-in + external from mcp.json).
            if query.lower() == "/mcp":
                self._print_mcp_status()
                continue

            # View or clear the curated persistent memory (MEMORY.md).
            if query.lower() == "/memory" or query.lower().startswith("/memory "):
                self._handle_memory_command(query[len("/memory"):].strip())
                continue

            # Toggle enforced, read-only plan mode (mutating/exec tools blocked).
            if query.lower() == "/plan":
                self.client.plan_mode_active = not self.client.plan_mode_active
                if self.client.plan_mode_active:
                    print(
                        "\n\033[93m🔒 Plan mode ON\033[0m — read-only. I'll research "
                        "and present a plan; file edits and shell commands are "
                        "blocked. Type /plan again to exit and allow changes.\n"
                    )
                else:
                    print(
                        "\n\033[92m🔓 Plan mode OFF\033[0m — changes allowed again.\n"
                    )
                continue

            # Manually compact the conversation: /compact [focus instructions]
            if query.lower() == "/compact" or query.lower().startswith("/compact "):
                focus = query[len("/compact"):].strip()
                did = self.client.compact_conversation(focus)
                print(
                    "Conversation compacted."
                    if did
                    else "Nothing to compact yet."
                )
                continue

            # Handle /load command
            if query.lower().startswith("/load"):
                if query.strip() == "/load":
                    print("Usage: /load <path>")
                    continue
                file_path = query[6:].strip()  # Remove "/load " prefix
                if self.client.load_conversation(file_path):
                    print("Conversation loaded successfully!")
                else:
                    print_error("Failed to load conversation. Check the file path.")
                continue

            if not query.strip():
                print("Input cannot be empty. Please try again.")
                continue

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
                continue
            except Exception as e:
                # Full traceback to the logger (stderr, LOG_LEVEL=DEBUG to see);
                # the user gets a concise red line with the actual cause.
                logger.error(f"Error processing query: {str(e)}", exc_info=True)
                print_error(f"Error: {e}")
