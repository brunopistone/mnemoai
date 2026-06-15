"""Chat interface handling for the application."""

import asyncio
from client.memory.episodic_memory import (
    is_task_successful,
    extract_tools_from_messages,
)
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
import time
from typing import Any
from utils.config import config
from utils.logger import logger


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

    def __welcome_message(self) -> None:
        """Display welcome message with available commands."""
        print("\n\033[90m┌" + "─" * 58 + "┐\033[0m")
        print(
            "\033[90m│\033[96m"
            + "Welcome to Personal AI Assistant Application!".center(58)
            + "\033[90m│\033[0m"
        )
        print("\033[90m├" + "─" * 58 + "┤\033[0m")
        print(
            "\033[90m│\033[97m Available commands:                                      \033[90m│\033[0m"
        )
        print(
            "\033[90m│\033[97m   \033[92m/clear\033[97m - Clear conversation context                    \033[90m│\033[0m"
        )
        print(
            "\033[90m│\033[97m   \033[92m/load <path>\033[97m - Load a saved conversation               \033[90m│\033[0m"
        )
        print(
            "\033[90m│\033[97m   \033[92m/exit\033[97m or \033[92m/quit\033[97m - Exit the application                  \033[90m│\033[0m"
        )
        print(
            "\033[90m│\033[97m   \033[92m/save\033[97m - Save current conversation                      \033[90m│\033[0m"
        )
        print(
            "\033[90m│\033[97m   \033[92m/good\033[97m - Mark last response as good (training data)     \033[90m│\033[0m"
        )
        print(
            "\033[90m│\033[97m   \033[92m/compact [focus]\033[97m - Summarize & shrink context        \033[90m│\033[0m"
        )
        print("\033[90m├" + "─" * 58 + "┤\033[0m")
        print(
            "\033[90m│\033[97m Use \033[92mCtrl+J\033[97m for new lines, Enter to submit                \033[90m│\033[0m"
        )
        print("\033[90m└" + "─" * 58 + "┘\033[0m\n")

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
