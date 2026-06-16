"""Strands client implementation."""

import asyncio
from datetime import date, datetime
from client.managers.agent_conversation_manager import AgentConversationManager
from client.managers.user_profile_manager import UserProfileManager
from client.memory.episodic_memory import EpisodicMemoryManager
from client.memory.reflector import Reflector
from client.memory.playbook_store import PlaybookStore
from client.ui.spinner import Spinner
import os
import json
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from models.llm_controller import LLMController
import os
import re
from server.tools import count_tokens
import shutil
import sqlite3
from strands import Agent
from strands.tools.mcp import MCPClient
import sys
import threading
import traceback
from utils.formatting.code_formatter import CodeFormatter
from utils.config import config
from utils.logger import logger


class StrandsClient:

    def __init__(
        self,
        server_path: str = "server/server.py",
        verbose: bool = False,
    ) -> None:
        """
        Initialize the Strands client with a server configuration.

        Args:
            messages: Initial conversation messages
            server_path: Path to the server.py file to run the MCP server
        """
        self.verbose_mode = verbose  # Track verbose mode
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_path],
            env=os.environ.copy(),
        )

        # Initialize profile manager first
        self.profile_manager = UserProfileManager()

        self.system_prompt = config.system_prompt

        if self.system_prompt:
            current_date = date.today().strftime("%Y-%m-%d")
            self.system_prompt = self.system_prompt.format(current_date=current_date)
        else:
            self.system_prompt = ""

        if config.get("PROFILE", {}).get("USE_PROFILING", False):
            # Initialize profile manager first
            self.profile_manager = UserProfileManager()

            # Add user profile to system prompt
            profile_summary = self.profile_manager.get_profile_summary()
            if profile_summary:
                self.system_prompt = f"{self.system_prompt}\n\n{profile_summary}"

        # Initialize MCP client
        self.mcp_client = MCPClient(lambda: stdio_client(self.server_params))

        # Initialize session ID (used by RAG and chat interface)
        profile_name = config.get("PROFILE", {}).get("NAME", "default")
        self.session_id = f"{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.agent = None
        self.tools = None
        self.llm_controller = LLMController(verbose=self.verbose_mode)
        self.llm_controller.initialize_model()
        self.model = None

        self.conversation_manager = AgentConversationManager(
            max_tokens=config.get("MAX_CONVERSATION_TOKENS", 1024 * 4)
        )
        self.spinner = Spinner()
        self.spinner_lock = threading.Lock()  # Thread safety for spinner operations
        self.first_token_received = False
        self.visible_content = None

        # Query routing
        self.router = None

        # Initialize episodic memory if enabled
        self.episodic_memory = None
        if config.get("ENABLE_EPISODIC_MEMORY", False):
            logger.debug("Initializing episodic memory...")

            # Get embeddings configuration
            embed_model_config = config.get("EMBED_MODEL_ID")
            if not embed_model_config:
                raise ValueError(
                    "EMBED_MODEL_ID must be configured in config.yaml to use episodic memory"
                )

            user_home = os.path.expanduser("~")
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            episodic_path = os.path.join(
                user_home, "agent-conversations", profile_name, "episodic_memory"
            )
            os.makedirs(episodic_path, exist_ok=True)

            store_type = config.get("EPISODIC_MEMORY_STORE", "chromadb").lower()
            logger.debug(f"Using {store_type} store for episodic memory")
            logger.debug(f"Episodic memory path: {episodic_path}")

            # Initialize embeddings controller
            from models.embeddings_controller import EmbeddingsController

            embeddings_controller = EmbeddingsController(embed_model_config)

            self.episodic_memory = EpisodicMemoryManager(
                persist_path=episodic_path,
                store_type=store_type,
                embeddings_controller=embeddings_controller,
            )

            # Run automatic cleanup on startup
            self.episodic_memory.cleanup(max_episodes=1000, max_age_days=90)

            logger.debug(f"✓ {store_type.upper()} episodic memory initialized")
        else:
            logger.debug("Episodic memory is disabled")

        # ACE components (Reflector + Playbook)
        self.reflector = None
        self.playbook = None
        if config.get("ENABLE_PLAYBOOK", False):
            self._initialize_playbook()

        # Track previous interaction for episodic memory
        self.previous_query = None
        self.previous_response = None
        self.previous_messages = None

    def _initialize_playbook(self) -> None:
        """Initialize ACE Reflector and Playbook store."""
        logger.debug("Initializing ACE Playbook...")

        user_home = os.path.expanduser("~")
        profile_name = config.get("PROFILE", {}).get("NAME", "default")
        playbook_path = os.path.join(
            user_home, "agent-conversations", profile_name, "playbook"
        )
        os.makedirs(playbook_path, exist_ok=True)

        # Get embeddings for semantic deduplication if available
        embeddings = None
        if config.get("EMBED_MODEL_ID"):
            try:
                from models.embeddings_controller import EmbeddingsController

                embeddings = EmbeddingsController()
            except Exception as e:
                logger.warning(f"Could not initialize embeddings for playbook: {e}")

        self.reflector = Reflector(persist_path=playbook_path)
        self.playbook = PlaybookStore(
            persist_path=playbook_path,
            embeddings_controller=embeddings,
            max_entries=config.get("PLAYBOOK", {}).get("MAX_ENTRIES", 500),
            similarity_threshold=config.get("PLAYBOOK", {}).get(
                "SIMILARITY_THRESHOLD", 0.85
            ),
        )

        stats = self.playbook.get_stats()
        logger.debug(f"Playbook initialized ({stats['total_entries']} entries)")

    def compact_conversation(self, focus_instructions: str = "") -> bool:
        """Manually compact the conversation (the /compact command).

        Summarizes older messages and keeps recent turns verbatim.

        Args:
            focus_instructions: Optional guidance on what the summary should
                emphasize.

        Returns:
            True if compaction ran, False if there was nothing to compact.
        """
        if not self.agent:
            return False
        return asyncio.run(
            self.conversation_manager.compact(
                self, self.model, self.agent, focus_instructions
            )
        )

    def reflect_and_learn(self, task: str) -> None:
        """Run reflection on the last interaction and update playbook.

        Args:
            task: The original user task
        """
        if not self.reflector or not self.playbook:
            return

        if (
            not self.agent
            or not hasattr(self.agent, "messages")
            or not self.agent.messages
        ):
            return

        try:
            entries = self.reflector.reflect_on_trajectory(
                messages=self.agent.messages,
                task=task,
            )

            if entries:
                self.playbook.append_batch(entries)
                logger.debug(f"Reflector: learned {len(entries)} strategies")
        except Exception as e:
            logger.error(f"Reflection failed: {e}")

    def _get_playbook_context(self) -> str:
        """Get formatted playbook context for system prompt.

        Returns:
            Formatted playbook strategies or empty string
        """
        if not self.playbook:
            return ""

        # Get all entries for system prompt (not task-specific)
        entries = self.playbook.get_relevant_entries(
            task="",  # Empty task gets general strategies
            top_k=config.get("PLAYBOOK", {}).get("MAX_INJECT", 10),
            include_failures=True,
        )

        return self.playbook.format_for_prompt(entries) if entries else ""

    def _extract_visible(self, text: str) -> str:
        """Extract visible content by stripping thinking tags.

        Args:
            text: Response text that may contain thinking tags

        Returns:
            Visible text with thinking tags removed
        """
        return re.sub(
            r"<think(?:ing)?>.*?</think(?:ing)?>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

    def _flush_formatters(self) -> None:
        """Flush any remaining buffered content from code formatters."""
        if hasattr(self, "_code_formatter_minimal"):
            self._code_formatter_minimal.flush()
        if hasattr(self, "_code_formatter_verbose"):
            self._code_formatter_verbose.flush()

    def _start_spinner(self) -> None:
        """Restart the spinner and reset first token flag."""
        with self.spinner_lock:
            self.first_token_received = False
            self.spinner.start()

    def _disable_reasoning(self) -> dict:
        """Temporarily disable reasoning/thinking on the model.

        Returns:
            Saved state to pass to _restore_reasoning()
        """
        saved = {}
        return_thinking = getattr(self.model, "return_thinking", None)
        if return_thinking is not None:
            saved["return_thinking"] = return_thinking
            self.model.return_thinking = False
        return saved

    def _restore_reasoning(self, saved: dict) -> None:
        """Restore reasoning/thinking settings on the model.

        Args:
            saved: State from _disable_reasoning()
        """
        if "return_thinking" in saved:
            self.model.return_thinking = saved["return_thinking"]

    def _process_thinking_buffer(self, buffer: str, in_tag: bool) -> tuple:
        """Process buffered content, handling thinking tags that may span chunks.

        Args:
            buffer: Accumulated content buffer
            in_tag: Whether currently inside thinking tags

        Returns:
            Tuple of (content_to_emit, remaining_buffer, new_in_tag_state)
        """
        open_pattern = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
        close_pattern = re.compile(r"</think(?:ing)?>", re.IGNORECASE)

        result = ""
        remaining = buffer
        current_in_tag = in_tag

        while remaining:
            if current_in_tag:
                # Looking for closing tag
                match = close_pattern.search(remaining)
                if match:
                    # Found closing tag - discard thinking content, keep after
                    remaining = remaining[match.end() :]
                    current_in_tag = False
                else:
                    # No complete closing tag - check for partial
                    last_lt = remaining.rfind("<")
                    if last_lt >= 0 and last_lt > len(remaining) - 12:
                        remaining = remaining[last_lt:]
                    else:
                        remaining = ""
                    break
            else:
                # Looking for opening tag
                match = open_pattern.search(remaining)
                if match:
                    # Found opening tag - emit content before
                    result += remaining[: match.start()]
                    remaining = remaining[match.end() :]
                    current_in_tag = True
                else:
                    # No complete opening tag - check for partial at end
                    last_lt = remaining.rfind("<")
                    if last_lt >= 0 and last_lt > len(remaining) - 11:
                        result += remaining[:last_lt]
                        remaining = remaining[last_lt:]
                    else:
                        result += remaining
                        remaining = ""
                    break

        return result, remaining, current_in_tag

    def _process_thinking_buffer_verbose(self, buffer: str, in_tag: bool) -> tuple:
        """Process buffered content for verbose mode, extracting thinking content separately.

        Args:
            buffer: Accumulated content buffer
            in_tag: Whether currently inside thinking tags

        Returns:
            Tuple of (regular_content, thinking_content, remaining_buffer, new_in_tag_state)
        """
        open_pattern = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
        close_pattern = re.compile(r"</think(?:ing)?>", re.IGNORECASE)

        regular = ""
        thinking = ""
        remaining = buffer
        current_in_tag = in_tag

        while remaining:
            if current_in_tag:
                match = close_pattern.search(remaining)
                if match:
                    thinking += remaining[: match.start()]
                    remaining = remaining[match.end() :]
                    current_in_tag = False
                else:
                    last_lt = remaining.rfind("<")
                    if last_lt >= 0 and last_lt > len(remaining) - 12:
                        thinking += remaining[:last_lt]
                        remaining = remaining[last_lt:]
                    else:
                        thinking += remaining
                        remaining = ""
                    break
            else:
                match = open_pattern.search(remaining)
                if match:
                    regular += remaining[: match.start()]
                    remaining = remaining[match.end() :]
                    current_in_tag = True
                else:
                    last_lt = remaining.rfind("<")
                    if last_lt >= 0 and last_lt > len(remaining) - 11:
                        regular += remaining[:last_lt]
                        remaining = remaining[last_lt:]
                    else:
                        regular += remaining
                        remaining = ""
                    break

        return regular, thinking, remaining, current_in_tag

    # Custom callback handler to control verbosity
    def __minimal_callback_handler(self, **kwargs) -> None:
        """Handle streaming events without showing thinking content.

        Args:
            **kwargs: Event data from streaming response
        """
        # Reset state on new content block (important after tool execution)
        if "event" in kwargs and (
            "contentBlockStart" in kwargs["event"] or "messageStart" in kwargs["event"]
        ):
            self._code_formatter_minimal = CodeFormatter()
            self._tag_buffer_minimal = ""
            self._in_thinking_minimal = False

        # Stop spinner only when first actual data arrives (thread-safe)
        if not self.first_token_received:
            if "data" in kwargs and kwargs["data"]:
                with self.spinner_lock:
                    if not self.first_token_received:  # Double-check inside lock
                        self.spinner.stop()
                        self.first_token_received = True

        # Stop spinner when tool call starts (thread-safe)
        if "message" in kwargs and kwargs["message"].get("role") == "assistant":
            content = kwargs["message"].get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("toolUse"):
                        with self.spinner_lock:
                            self.spinner.stop()
                        break

        # Restart spinner after tool execution completes (thread-safe)
        if (
            "message" in kwargs
            and kwargs["message"].get("role") == "user"
            and "toolResult" in str(kwargs["message"].get("content", ""))
        ):
            # Tool result received, restart spinner for final response
            with self.spinner_lock:
                self.first_token_received = False
                self.spinner.start()

        if "data" in kwargs:
            data = kwargs["data"]

            # Initialize state if not exists
            if not hasattr(self, "_code_formatter_minimal"):
                self._code_formatter_minimal = CodeFormatter()
                self._tag_buffer_minimal = ""
                self._in_thinking_minimal = False

            # Buffer for handling tags split across chunks
            self._tag_buffer_minimal += data

            # Check for potential partial tags at end (any < within 12 chars of end)
            last_lt = self._tag_buffer_minimal.rfind("<")
            if last_lt >= 0 and last_lt > len(self._tag_buffer_minimal) - 12:
                # Potential partial tag - process up to it, keep rest in buffer
                to_process = self._tag_buffer_minimal[:last_lt]
                self._tag_buffer_minimal = self._tag_buffer_minimal[last_lt:]
            else:
                # No partial tag - process all
                to_process = self._tag_buffer_minimal
                self._tag_buffer_minimal = ""

            if to_process:
                # Strip thinking tags AND content inside them (minimal hides thinking)
                # First, remove complete <thinking>...</thinking> blocks
                cleaned = re.sub(
                    r"<think(?:ing)?>(.*?)</think(?:ing)?>",
                    "",
                    to_process,
                    flags=re.IGNORECASE | re.DOTALL,
                )

                # Handle orphan closing tags (content before </think> without opening tag)
                # This happens when model outputs thinking then </think> without <think>
                if not self._in_thinking_minimal and re.search(
                    r"</think(?:ing)?>", cleaned, re.IGNORECASE
                ):
                    # Found closing tag while not in thinking - strip everything before it
                    parts = re.split(r"</think(?:ing)?>", cleaned, flags=re.IGNORECASE)
                    cleaned = parts[-1]  # Keep only content after closing tag

                # Handle state for incomplete tags (opening without closing)
                elif re.search(
                    r"<think(?:ing)?>(?!.*</think(?:ing)?>)",
                    cleaned,
                    re.IGNORECASE | re.DOTALL,
                ):
                    # Found opening tag without closing - we're entering thinking
                    parts = re.split(r"<think(?:ing)?>", cleaned, flags=re.IGNORECASE)
                    cleaned = parts[0]  # Keep only content before opening tag
                    self._in_thinking_minimal = True
                elif self._in_thinking_minimal:
                    # We're inside thinking, look for closing tag
                    if re.search(r"</think(?:ing)?>", cleaned, re.IGNORECASE):
                        # Found closing tag - extract content after it
                        parts = re.split(
                            r"</think(?:ing)?>", cleaned, flags=re.IGNORECASE
                        )
                        cleaned = parts[-1]  # Keep only content after closing tag
                        self._in_thinking_minimal = False
                    else:
                        # Still inside thinking - discard all
                        cleaned = ""

                if cleaned:
                    self._code_formatter_minimal.process_chunk(cleaned)

    # Custom callback handler that shows all content including reasoning
    def __verbose_callback_handler(self, **kwargs) -> None:
        """Handle streaming events showing all content including thinking.

        Args:
            **kwargs: Event data from streaming response
        """
        # Reset state on new content block (important after tool execution)
        if "event" in kwargs and (
            "contentBlockStart" in kwargs["event"] or "messageStart" in kwargs["event"]
        ):
            self._code_formatter_verbose = CodeFormatter()
            self._tag_buffer_verbose = ""
            self._in_thinking_verbose = False

        # Stop spinner only when first actual data arrives (thread-safe)
        if not self.first_token_received:
            if "data" in kwargs and kwargs["data"]:
                with self.spinner_lock:
                    if not self.first_token_received:  # Double-check inside lock
                        self.spinner.stop()
                        self.first_token_received = True

        # Stop spinner when tool call starts (thread-safe)
        if "message" in kwargs and kwargs["message"].get("role") == "assistant":
            content = kwargs["message"].get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("toolUse"):
                        with self.spinner_lock:
                            self.spinner.stop()
                        break

        # Restart spinner after tool execution completes (thread-safe)
        if (
            "message" in kwargs
            and kwargs["message"].get("role") == "user"
            and "toolResult" in str(kwargs["message"].get("content", ""))
        ):
            # Tool result received, restart spinner for final response
            with self.spinner_lock:
                self.first_token_received = False
                self.spinner.start()

        if "data" in kwargs:
            data = kwargs["data"]

            # Initialize state if not exists
            if not hasattr(self, "_code_formatter_verbose"):
                self._code_formatter_verbose = CodeFormatter()
                self._tag_buffer_verbose = ""
                self._in_thinking_verbose = False

            # Buffer for handling tags split across chunks
            self._tag_buffer_verbose += data

            # Check for potential partial tags at end (any < within 12 chars of end)
            last_lt = self._tag_buffer_verbose.rfind("<")
            if last_lt >= 0 and last_lt > len(self._tag_buffer_verbose) - 12:
                # Potential partial tag - process up to it, keep rest in buffer
                to_process = self._tag_buffer_verbose[:last_lt]
                self._tag_buffer_verbose = self._tag_buffer_verbose[last_lt:]
            else:
                # No partial tag - process all
                to_process = self._tag_buffer_verbose
                self._tag_buffer_verbose = ""

            if to_process:
                # Process content, showing thinking in gray and regular content normally
                remaining = to_process

                while remaining:
                    if self._in_thinking_verbose:
                        # Inside thinking - look for closing tag
                        match = re.search(r"</think(?:ing)?>", remaining, re.IGNORECASE)
                        if match:
                            # Print thinking content in gray (before closing tag)
                            thinking_content = remaining[: match.start()]
                            if thinking_content:
                                print(
                                    f"\033[90m{thinking_content}\033[0m",
                                    end="",
                                    flush=True,
                                )
                            remaining = remaining[match.end() :]
                            self._in_thinking_verbose = False
                            # Add newline to separate reasoning from answer
                            print("\n", end="", flush=True)
                        else:
                            # No closing tag - all is thinking content
                            print(f"\033[90m{remaining}\033[0m", end="", flush=True)
                            remaining = ""
                    else:
                        # Outside thinking - look for opening tag
                        match = re.search(r"<think(?:ing)?>", remaining, re.IGNORECASE)
                        if match:
                            # Print regular content normally (before opening tag)
                            regular_content = remaining[: match.start()]
                            if regular_content:
                                self._code_formatter_verbose.process_chunk(
                                    regular_content
                                )
                            remaining = remaining[match.end() :]
                            self._in_thinking_verbose = True
                        else:
                            # No opening tag - all is regular content
                            self._code_formatter_verbose.process_chunk(remaining)
                            remaining = ""

    def __count_context_tokens(self) -> int:
        """Count total tokens in the current conversation context.

        Returns:
            Total token count
        """
        total_tokens = 0

        # Count system prompt tokens
        if self.system_prompt:
            total_tokens += count_tokens(self.system_prompt)

        # Count conversation messages tokens by converting to JSON string
        if self.agent and hasattr(self.agent, "messages"):
            total_tokens += count_tokens(json.dumps(self.agent.messages, default=str))

        return total_tokens

    def clear_context(self) -> None:
        """Clear conversation history but keep system prompt."""
        system_msg = config.get("SYSTEM_PROMPT")
        profile_name = config.get("PROFILE", {}).get("NAME", "default")

        self.agent.messages.clear()
        self.session_id = f"{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if system_msg:
            current_date = date.today().strftime("%Y-%m-%d")
            system_msg = system_msg.format(current_date=current_date)
            self.system_prompt = system_msg
            self.agent.system_prompt = system_msg

        # Flush RAG database when clearing context
        if config.get("ENABLE_RAG", False):
            self._flush_rag_store()

        self._flush_chunk_cache_store()

    def _initialize_rag_session(self) -> None:
        """Initialize RAG session at application startup."""
        try:
            user_home = os.path.expanduser("~")
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            rag_dir = os.path.join(user_home, "agent-conversations", profile_name)
            os.makedirs(rag_dir, exist_ok=True)

            session_file = os.path.join(rag_dir, "rag_session_id.txt")
            with open(session_file, "w") as f:
                f.write(self.session_id)

            logger.debug(f"RAG session initialized: {self.session_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize RAG session: {e}")

    def _initialize_chunk_cache(self) -> None:
        """Initialize chunk cache DB at application startup."""
        try:
            user_home = os.path.expanduser("~")
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            rag_dir = os.path.join(user_home, "agent-conversations", profile_name)
            os.makedirs(rag_dir, exist_ok=True)

            # Write session_id to file for MCP subprocess to read (same as RAG)
            session_file = os.path.join(rag_dir, "chunk_session_id.txt")
            with open(session_file, "w") as f:
                f.write(self.session_id)

            # Create session-specific DB
            db_path = os.path.join(rag_dir, f"chunk_cache_{self.session_id}.db")
            conn = sqlite3.connect(db_path)
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                CREATE TABLE IF NOT EXISTS chunk_cache (
                    key TEXT PRIMARY KEY,
                    summary TEXT,
                    updated_at TEXT
                )
                """
                )
                conn.commit()
                logger.debug(f"Chunk cache initialized: {os.path.basename(db_path)}")
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"Failed to initialize chunk cache: {e}")

    def _flush_chunk_cache_store(self) -> None:
        """Flush the RAG database and session-specific chunk cache."""
        try:
            from server.tools.readers.chunking_helper import reset_session_chunk_cache

            reset_session_chunk_cache()

            # Delete persisted session files
            user_home = os.path.expanduser("~")
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            rag_dir = os.path.join(user_home, "agent-conversations", profile_name)

            # Remove session-specific files (rag_store and chunk_cache with session_id)
            if os.path.exists(rag_dir):
                for file in os.listdir(rag_dir):
                    # Delete RAG store files and session-specific chunk cache
                    if file.startswith("chunk_cache_"):
                        file_path = os.path.join(rag_dir, file)
                        try:
                            os.remove(file_path)
                            logger.debug(f"Deleted session file: {file}")
                        except Exception as e:
                            logger.debug(f"Failed to delete {file}: {e}")

            logger.debug("Session reset - Chunk cache store and chunk cache cleared")
        except Exception as e:
            logger.warning(f"Failed to reset session: {e}")

    def _flush_rag_store(self) -> None:
        """Flush the RAG database and session-specific chunk cache."""
        try:
            if config.get("ENABLE_RAG", False):
                from server.tools.rag import reset_session_rag

                reset_session_rag()

            # Delete persisted session files
            user_home = os.path.expanduser("~")
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            rag_dir = os.path.join(user_home, "agent-conversations", profile_name)

            # Remove session-specific files (rag_store and chunk_cache with session_id)
            if os.path.exists(rag_dir):
                for file in os.listdir(rag_dir):
                    # Delete RAG store files/directories and session-specific chunk cache
                    if file.startswith("rag_store_"):
                        file_path = os.path.join(rag_dir, file)
                        try:
                            if os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                                logger.debug(f"Deleted session directory: {file}")
                            else:
                                os.remove(file_path)
                                logger.debug(f"Deleted session file: {file}")
                        except Exception as e:
                            logger.debug(f"Failed to delete {file}: {e}")

            logger.debug("Session reset - RAG store and chunk cache cleared")
        except Exception as e:
            logger.warning(f"Failed to reset session: {e}")

    def save_conversation(self, timestamp: str = None) -> None:
        """Save conversation to file.

        Args:
            timestamp: Optional timestamp for filename (default: current time)
        """
        if self.agent and self.agent.messages:
            # Use profile-based path with conversations subdirectory
            user_home = os.path.expanduser("~")
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            save_dir = os.path.join(
                user_home, "agent-conversations", profile_name, "conversations"
            )
            os.makedirs(save_dir, exist_ok=True)

            if not timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"conversation_{timestamp}.json"
            filepath = os.path.join(save_dir, filename)

            try:
                # Create conversation data with metadata
                conversation_data = {"messages": [], "tools": []}

                # Add system prompt as first message if it exists
                if self.system_prompt:
                    system_message = {
                        "role": "system",
                        "content": [{"text": self.system_prompt}],
                    }
                    conversation_data["messages"].append(system_message)

                # Add all conversation messages
                conversation_data["messages"].extend(self.agent.messages)

                # Add tools information if available
                if self.tools:
                    for tool in self.tools:
                        tool_info = {}

                        # Try different possible attribute names
                        if hasattr(tool, "name"):
                            tool_info["name"] = tool.name
                        elif hasattr(tool, "tool_name"):
                            tool_info["name"] = tool.tool_name
                        elif hasattr(tool, "__name__"):
                            tool_info["name"] = tool.__name__
                        else:
                            tool_info["name"] = str(tool)

                        # Try to get description
                        if hasattr(tool, "description"):
                            tool_info["description"] = tool.description
                        elif hasattr(tool, "__doc__"):
                            tool_info["description"] = tool.__doc__

                        # Try to get arguments/parameters
                        if hasattr(tool, "input_schema"):
                            tool_info["arguments"] = tool.input_schema
                        elif hasattr(tool, "parameters"):
                            tool_info["arguments"] = tool.parameters
                        elif hasattr(tool, "args"):
                            tool_info["arguments"] = tool.args
                        elif hasattr(tool, "schema"):
                            tool_info["arguments"] = tool.schema

                        conversation_data["tools"].append(tool_info)

                with open(filepath, "w") as f:
                    json.dump(conversation_data, f, indent=2, default=str)
                print(f"Conversation saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")

    def save_conversation_with_quality(
        self, timestamp: str = None, quality_markers: list = None
    ) -> None:
        """Save conversation with quality markers for training data.

        Args:
            timestamp: Optional timestamp for filename (default: current time)
            quality_markers: List of quality labels for each message (e.g., ['good', 'unlabeled'])
        """
        if self.agent and self.agent.messages:
            # Use profile-based path with conversations subdirectory
            user_home = os.path.expanduser("~")
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            save_dir = os.path.join(
                user_home, "agent-conversations", profile_name, "conversations"
            )
            os.makedirs(save_dir, exist_ok=True)

            if not timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"conversation_{timestamp}.json"
            filepath = os.path.join(save_dir, filename)

            try:
                conversation_data = {
                    "messages": [],
                    "tools": [],
                    "quality_markers": quality_markers or [],
                }

                if self.system_prompt:
                    system_message = {
                        "role": "system",
                        "content": [{" text": self.system_prompt}],
                    }
                    conversation_data["messages"].append(system_message)

                conversation_data["messages"].extend(self.agent.messages)

                # Get tools metadata from self.tools
                if self.tools:
                    for tool in self.tools:
                        try:
                            # Access the underlying MCP tool
                            mcp_tool = (
                                tool.mcp_tool if hasattr(tool, "mcp_tool") else tool
                            )

                            # Get parameters with better formatting
                            parameters = {}
                            if hasattr(mcp_tool, "inputSchema"):
                                schema = mcp_tool.inputSchema
                                properties = schema.get("properties", {})

                                # Clean up parameter descriptions
                                for param_name, param_info in properties.items():
                                    # Use description from schema if available, otherwise use title
                                    desc = param_info.get("description", "")
                                    if desc.startswith("Property "):
                                        # Generic description, try to get from title or just use param name
                                        desc = f"{param_name}: {param_info.get('type', 'any')}"

                                    parameters[param_name] = {
                                        "type": param_info.get("type", "string"),
                                        "description": desc,
                                        "default": (
                                            param_info.get("default")
                                            if "default" in param_info
                                            else None
                                        ),
                                    }

                            tool_info = {
                                "name": (
                                    mcp_tool.name
                                    if hasattr(mcp_tool, "name")
                                    else tool.tool_name
                                ),
                                "description": (
                                    mcp_tool.description
                                    if hasattr(mcp_tool, "description")
                                    else ""
                                ),
                                "parameters": parameters,
                            }
                            conversation_data["tools"].append(tool_info)
                        except Exception as e:
                            logger.error(f"Error extracting tool info: {e}")
                            continue

                with open(filepath, "w") as f:
                    json.dump(conversation_data, f, indent=2, default=str)
                print(f"Conversation saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")

    def load_conversation(self, file_path: str) -> bool:
        """Load conversation from file, excluding system prompt and tools.

        Args:
            file_path: Path to the conversation JSON file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Expand user path and check if file exists
            normalized_path = os.path.expanduser(file_path.strip())
            if not os.path.exists(normalized_path):
                logger.error(f"File not found: {normalized_path}")
                return False

            # Load conversation data
            with open(normalized_path, "r") as f:
                conversation_data = json.load(f)

            # Handle both old format (list) and new format (dict)
            if isinstance(conversation_data, list):
                messages = conversation_data
            else:
                messages = conversation_data.get("messages", [])

            # Filter out system messages and load only user/assistant messages
            conversation_messages = []
            for message in messages:
                if message.get("role") != "system":
                    conversation_messages.append(message)

            # Clear current conversation and load the saved one
            if self.agent:
                self.agent.messages.clear()
                self.agent.messages.extend(conversation_messages)
                logger.info(
                    f"Loaded {len(conversation_messages)} messages from {normalized_path}"
                )

                token_count = self.__count_context_tokens()
                print(f"\n\033[90m[Context: {token_count} tokens]\033[0m")

                return True
            else:
                logger.error("Agent not initialized")
                return False

        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return False

    def start(self, verbose: bool = False) -> None:
        """Start the client and initialize the agent.

        Args:
            verbose: Enable verbose mode to show thinking process
        """
        try:
            self.verbose_mode = verbose  # Store verbose mode

            with self.mcp_client:
                # Get tools from MCP server
                self.tools = self.mcp_client.list_tools_sync()

                # Initialize RAG session after MCP server is ready (if enabled)
                if config.get("ENABLE_RAG", False):
                    self._initialize_rag_session()

                # Initialize chunk cache DB
                self._initialize_chunk_cache()

                self.model = self.llm_controller.get_model()

                if not verbose:
                    additional_params = {
                        "callback_handler": self.__minimal_callback_handler
                    }
                else:
                    additional_params = {
                        "callback_handler": self.__verbose_callback_handler
                    }

                # Build system prompt with playbook context
                system_prompt_with_context = self.system_prompt
                if self.playbook:
                    playbook_context = self._get_playbook_context()
                    if playbook_context:
                        system_prompt_with_context = (
                            f"{self.system_prompt}\n\n{playbook_context}"
                        )

                # Initialize query router if enabled
                if config.get("ENABLE_ROUTING", False):
                    from client.router import QueryRouter

                    self.router = QueryRouter(self.model)
                    logger.debug("Query routing enabled")

                # Create agent with tools
                self.agent = Agent(
                    model=self.model,
                    tools=self.tools,
                    system_prompt=system_prompt_with_context,
                    **additional_params,
                )
        except Exception as e:
            stacktrace = traceback.format_exc()
            logger.error(stacktrace)

            raise e

    def _format_episodic_context(self, episodes: list) -> str:
        """Format episodic memory episodes as compact context.

        Args:
            episodes: List of episode dictionaries with task, tools, similarity

        Returns:
            Formatted context string
        """
        context = "[Episodic Memory - Similar Past Tasks]\n"
        for i, ep in enumerate(episodes, 1):
            task = ep.get("task", "Unknown task")
            # Truncate at word boundary, not mid-word
            if len(task) > 70:
                task = task[:70].rsplit(" ", 1)[0] + "..."

            tools = ep.get("tools", "")
            # Parse tools string back to list
            if isinstance(tools, str):
                import ast

                try:
                    tools_list = ast.literal_eval(tools)
                    tool_names = [
                        t.get("name", "") for t in tools_list if isinstance(t, dict)
                    ]
                except:
                    tool_names = []
            else:
                tool_names = []

            tools_str = ", ".join(tool_names) if tool_names else "no tools"
            similarity = ep.get("similarity", 0)
            context += f'{i}. "{task}" → {tools_str} → success (similarity: {similarity:.2f})\n'

        logger.debug("Formatted episodic memory context for prompt:")
        logger.debug(context)
        return context

    def _handle_simple_qa(self, prompt: str) -> str:
        """Handle simple Q&A queries by calling the model directly without tools.

        This provides faster responses for conversational queries that don't
        need any tool invocation.

        Args:
            prompt: User's query

        Returns:
            Model's response text
        """
        logger.debug("Handling query via simple_qa route (no tools)")

        # Add user message to agent's conversation history
        user_msg = {"role": "user", "content": [{"text": prompt}]}
        self.agent.messages.append(user_msg)

        # Build messages for the model call (include conversation history)
        messages = []
        for m in self.agent.messages:
            role = m.get("role")
            # Skip system messages and tool-related messages for simple QA
            if role in ("user", "assistant"):
                content = m.get("content", [])
                # Skip messages with tool use/results
                has_tool = False
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and (
                            "toolUse" in block or "toolResult" in block
                        ):
                            has_tool = True
                            break
                if not has_tool:
                    messages.append(m)

        # Stream response from model directly
        response_text = ""
        callback = (
            self._StrandsClient__verbose_callback_handler
            if self.verbose_mode
            else self._StrandsClient__minimal_callback_handler
        )

        think_param = config.get("LLM", {}).get("ENABLE_THINKING", False)
        if not self.verbose_mode:
            think_param = False

        async def _stream():
            nonlocal response_text
            async for event in self.model.stream(
                messages, system_prompt=self.system_prompt, think=think_param
            ):
                # Forward events to the callback handler for spinner/formatting
                if (
                    "contentBlockDelta" in event
                    and "delta" in event["contentBlockDelta"]
                    and "text" in event["contentBlockDelta"]["delta"]
                ):
                    text = event["contentBlockDelta"]["delta"]["text"]
                    response_text += text
                    callback(data=text)
                elif "contentBlockStart" in event or "messageStart" in event:
                    callback(event=event)

        asyncio.run(_stream())
        self._flush_formatters()

        # Add assistant response to conversation history
        visible = self._extract_visible(response_text)
        assistant_msg = {
            "role": "assistant",
            "content": [{"text": visible or response_text}],
        }
        self.agent.messages.append(assistant_msg)

        asyncio.run(
            self.conversation_manager.manage_messages(self, self.model, self.agent)
        )

        if config.get("PROFILE", {}).get("USE_PROFILING", False):
            self.profile_manager.analyze_conversation(self.agent.messages)

        return visible or response_text

    def _handle_agent_query(self, prompt: str) -> str:
        """Handle queries using the full Strands agent with tools.

        Args:
            prompt: User's query

        Returns:
            Agent's response text
        """
        response = self.agent(prompt)

        self._flush_formatters()

        asyncio.run(
            self.conversation_manager.manage_messages(self, self.model, self.agent)
        )

        if config.get("PROFILE", {}).get("USE_PROFILING", False):
            self.profile_manager.analyze_conversation(self.agent.messages)

        # Check if response is only thinking tags (no visible content)
        response_text = str(response)

        if not self._extract_visible(response_text):
            # Model produced only reasoning — retry with thinking disabled
            logger.debug("Model produced only reasoning, retrying without thinking")

            print("", flush=True)
            self._start_spinner()

            saved = self._disable_reasoning()
            try:
                retry_response = self.agent(
                    "You provided reasoning but no visible response. "
                    "Please provide your answer."
                )
            finally:
                self._restore_reasoning(saved)

            self._flush_formatters()

            retry_text = str(retry_response)
            if self._extract_visible(retry_text):
                response_text = retry_text

        return response_text

    def query(self, prompt: str) -> str:
        """
        Send a query to the Strands agent.

        Args:
            prompt: User's query

        Returns:
            Agent's response
        """
        if not self.agent:
            raise RuntimeError(
                "Client not started. Call start() or use with-statement first."
            )

        # Reset first token flag and start spinner (thread-safe)
        with self.spinner_lock:
            self.first_token_received = False
            self.spinner.start()

        try:
            # Retrieve similar episodes from episodic memory
            if self.episodic_memory:
                # Skip episodic injection for short follow-up queries when there's
                # an active conversation. Short queries like "can you search?",
                # "yes", "tell me more" are follow-ups that only make sense in
                # the current conversation context.
                has_conversation = (
                    self.agent
                    and hasattr(self.agent, "messages")
                    and len(self.agent.messages) > 0
                )
                query_words = prompt.strip().split()
                short_query_threshold = config.get("EPISODIC_MEMORY", {}).get(
                    "SHORT_QUERY_WORDS", 8
                )
                skip_episodic = (
                    has_conversation and len(query_words) <= short_query_threshold
                )

                if skip_episodic:
                    logger.debug(
                        f"Skipping episodic injection: short follow-up query "
                        f"({len(query_words)} words <= {short_query_threshold})"
                    )
                else:
                    logger.debug("Retrieving similar episodes from episodic memory...")
                    similar_episodes = self.episodic_memory.retrieve_similar_episodes(
                        prompt, top_k=3
                    )
                    logger.debug(f"Found {len(similar_episodes)} similar episodes")
                    logger.debug(f"Similar episodes: {similar_episodes}")

                    # Filter by configurable similarity threshold
                    retrieval_threshold = config.get("EPISODIC_MEMORY", {}).get(
                        "RETRIEVAL_THRESHOLD", 0.7
                    )
                    relevant_episodes = [
                        ep
                        for ep in similar_episodes
                        if ep.get("similarity", 0) > retrieval_threshold
                    ]

                    if relevant_episodes:
                        logger.debug(
                            f"Found {len(relevant_episodes)} relevant episodes"
                        )
                        context = self._format_episodic_context(relevant_episodes)
                        prompt = f"{context}\n\n{prompt}"
                    else:
                        logger.debug(
                            f"No relevant episodes found (similarity < {retrieval_threshold})"
                        )

            # Classify query if routing is enabled
            route = None
            if self.router:
                # Build conversation context from recent messages
                context = ""
                if (
                    self.agent
                    and hasattr(self.agent, "messages")
                    and len(self.agent.messages) > 1
                ):
                    recent = self.agent.messages[-min(4, len(self.agent.messages)) :]
                    context_parts = []
                    for m in recent:
                        text = ""
                        content = m.get("content", [])
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and "text" in block:
                                    text += block["text"][:200]
                        elif isinstance(content, str):
                            text = content[:200]
                        if text:
                            context_parts.append(text)
                    context = "\n".join(context_parts)

                route = self.router.classify(prompt, context)

            with self.mcp_client:
                # For simple_qa route, call model directly without tools
                if route == "simple_qa":
                    response_text = self._handle_simple_qa(prompt)
                else:
                    response_text = self._handle_agent_query(prompt)

                # Print token count in a clean format
                token_count = self.__count_context_tokens()
                print(f"\n\033[90m[Context: {token_count} tokens]\033[0m")

                # Store full conversation for episodic memory evaluation
                if self.episodic_memory:
                    logger.debug("Storing interaction for episodic memory evaluation")
                    self.previous_query = prompt
                    self.previous_response = response_text
                    self.previous_messages = self.agent.messages.copy()
                    logger.debug(
                        f"Stored {len(self.previous_messages)} messages for evaluation"
                    )

                return response_text
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully - cancel all pending tasks
            with self.spinner_lock:
                self.first_token_received = False
                self.spinner.stop()

            try:
                # Cancel all async tasks
                loop = asyncio.get_event_loop()
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()

                # Force close MCP client connection
                if hasattr(self.mcp_client, "_client") and self.mcp_client._client:
                    try:
                        asyncio.run(self.mcp_client._client.__aexit__(None, None, None))
                    except:
                        pass
            except:
                pass

            # Reset MCP client to clean state
            try:
                self.mcp_client = MCPClient(lambda: stdio_client(self.server_params))
            except:
                pass

            return "Operation was cancelled."
        except Exception as e:
            # Handle other exceptions - MCP client will be closed by context manager
            with self.spinner_lock:
                self.first_token_received = False
                self.spinner.stop()
            raise e
        finally:
            # Ensure spinner is stopped (thread-safe)
            with self.spinner_lock:
                self.first_token_received = False
                self.spinner.stop()
