"""Chat interface handling for the application."""

import asyncio
from mnemoai.client.memory.episodic_memory import (
    is_task_successful,
    extract_tools_from_messages,
)
from mnemoai.utils.config import config
from mnemoai.utils.logger import logger
import os
import re
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
import time
from typing import Any


class ChatInterface:
    """Handles chat interface and user interaction."""

    def __init__(self, client: Any) -> None:
        """Initialize chat interface.

        Args:
            client: LangGraphClient instance
        """
        self.client = client

        self.interaction_quality = []  # Track quality per interaction
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
        ]),
        ("Conversation", [
            ("/clear", "Clear conversation context"),
            ("/compact [focus]", "Summarize & shrink context"),
            ("/save", "Save current conversation"),
            ("/load <path>", "Load a saved conversation"),
        ]),
        ("Data & exit", [
            ("/good", "Mark last response as good (training data)"),
            ("/exit, /quit", "Exit the application"),
        ]),
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

    def __welcome_message(self) -> None:
        """Display the launch banner + a framed, grouped command list."""
        C = self._C
        W = 64  # inner width; matches the wordmark banner width

        def vlen(s: str) -> int:
            """Visible length (ANSI escapes don't occupy columns)."""
            return len(self._ANSI_RE.sub("", s))

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

        cmd_w = max(vlen(c) for _, items in self._COMMAND_GROUPS for c, _ in items)
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
                self.client.save_conversation_with_quality(
                    timestamp, self.interaction_quality
                )
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
                    print("Failed to load conversation. Check the file path.")
                continue

            # Training data commands
            if query.lower() == "/good":
                interaction_idx = len(self.interaction_quality) - 1
                if interaction_idx >= 0:
                    self.interaction_quality[interaction_idx] = "good"
                    print("✓ Marked as good")
                else:
                    print("No interaction to mark")
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

                # Track this interaction (unlabeled by default)
                self.interaction_quality.append("unlabeled")

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
                logger.error(f"Error processing query: {str(e)}", exc_info=True)
                print(f"Error: Unable to process your request. Please try again.")
