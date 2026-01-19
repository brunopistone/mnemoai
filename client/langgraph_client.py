"""LangGraph-based client implementation."""

import asyncio
from datetime import date, datetime
from client.managers.agent_conversation_manager import AgentConversationManager
from client.managers.user_profile_manager import UserProfileManager
from client.managers.dpo_collector import DPOCollector
from client.memory.episodic_memory import EpisodicMemoryManager
from client.ui.spinner import Spinner
from client.langgraph_agent import (
    LangGraphAgent,
    convert_strands_messages_to_langchain,
    convert_langchain_messages_to_strands,
)
from client.mcp_tool_wrapper import MCPClientWrapper
import os
import json
from mcp import StdioServerParameters
import re
from server.tools import count_tokens
import shutil
import sqlite3
import sys
import threading
import traceback
from typing import Any, Dict, List, Optional
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessageChunk,
)
from langchain_core.callbacks import BaseCallbackHandler
from utils.formatting.code_formatter import CodeFormatter
from utils.config import config
from utils.logger import logger


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(
        self,
        verbose: bool = False,
        spinner: Optional[Spinner] = None,
        spinner_lock: Optional[threading.Lock] = None,
    ):
        """Initialize the streaming callback handler.

        Args:
            verbose: Show thinking content
            spinner: Spinner instance for UI feedback
            spinner_lock: Thread lock for spinner operations
        """
        self.verbose = verbose
        self.spinner = spinner
        self.spinner_lock = spinner_lock or threading.Lock()
        self.first_token_received = False
        self.code_formatter = CodeFormatter()
        self._in_thinking = False
        self._tag_buffer = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new tokens from the LLM.

        Args:
            token: The new token
            **kwargs: Additional arguments
        """
        # Stop spinner on first token
        if not self.first_token_received and self.spinner:
            with self.spinner_lock:
                if not self.first_token_received:
                    self.spinner.stop()
                    self.first_token_received = True

        if self.verbose:
            self._handle_verbose_token(token)
        else:
            self._handle_minimal_token(token)

    def _handle_verbose_token(self, token: str) -> None:
        """Handle token in verbose mode (show thinking).

        Args:
            token: The token to process
        """
        # Buffer for handling tags split across chunks
        self._tag_buffer += token

        # Check for potential partial tags at end
        last_lt = self._tag_buffer.rfind("<")
        if last_lt >= 0 and last_lt > len(self._tag_buffer) - 12:
            to_process = self._tag_buffer[:last_lt]
            self._tag_buffer = self._tag_buffer[last_lt:]
        else:
            to_process = self._tag_buffer
            self._tag_buffer = ""

        if to_process:
            remaining = to_process

            while remaining:
                if self._in_thinking:
                    # Inside thinking - look for closing tag
                    match = re.search(r"</think(?:ing)?>", remaining, re.IGNORECASE)
                    if match:
                        thinking_content = remaining[: match.start()]
                        if thinking_content:
                            print(f"\033[90m{thinking_content}\033[0m", end="", flush=True)
                        remaining = remaining[match.end():]
                        self._in_thinking = False
                        print("\n", end="", flush=True)
                    else:
                        print(f"\033[90m{remaining}\033[0m", end="", flush=True)
                        remaining = ""
                else:
                    # Outside thinking - look for opening tag
                    match = re.search(r"<think(?:ing)?>", remaining, re.IGNORECASE)
                    if match:
                        regular_content = remaining[: match.start()]
                        if regular_content:
                            self.code_formatter.process_chunk(regular_content)
                        remaining = remaining[match.end():]
                        self._in_thinking = True
                    else:
                        self.code_formatter.process_chunk(remaining)
                        remaining = ""

    def _handle_minimal_token(self, token: str) -> None:
        """Handle token in minimal mode (hide thinking).

        Args:
            token: The token to process
        """
        # Buffer for handling tags split across chunks
        self._tag_buffer += token

        # Check for potential partial tags at end
        last_lt = self._tag_buffer.rfind("<")
        if last_lt >= 0 and last_lt > len(self._tag_buffer) - 12:
            to_process = self._tag_buffer[:last_lt]
            self._tag_buffer = self._tag_buffer[last_lt:]
        else:
            to_process = self._tag_buffer
            self._tag_buffer = ""

        if to_process:
            # Remove complete thinking blocks
            cleaned = re.sub(
                r"<think(?:ing)?>.*?</think(?:ing)?>",
                "",
                to_process,
                flags=re.DOTALL | re.IGNORECASE,
            )

            # Handle orphan closing tags
            cleaned = re.sub(r"</think(?:ing)?>", "", cleaned, flags=re.IGNORECASE)

            # Handle orphan opening tags (start of thinking)
            if re.search(r"<think(?:ing)?>", cleaned, re.IGNORECASE):
                self._in_thinking = True
                cleaned = re.sub(r"<think(?:ing)?>.*$", "", cleaned, flags=re.IGNORECASE)

            # If we're inside a thinking block, skip the content
            if self._in_thinking:
                close_match = re.search(r"</think(?:ing)?>", to_process, re.IGNORECASE)
                if close_match:
                    self._in_thinking = False
                return

            if cleaned:
                self.code_formatter.process_chunk(cleaned)

    def on_llm_end(self, response, **kwargs) -> None:
        """Handle LLM completion.

        Args:
            response: The LLM response
            **kwargs: Additional arguments
        """
        # Flush any remaining buffer
        if self._tag_buffer:
            if not self._in_thinking:
                self.code_formatter.process_chunk(self._tag_buffer)
            self._tag_buffer = ""

        self.code_formatter.flush()

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        """Handle tool execution start.

        Args:
            serialized: Serialized tool info
            input_str: Tool input
            **kwargs: Additional arguments
        """
        if self.spinner:
            with self.spinner_lock:
                self.spinner.stop()

    def on_tool_end(self, output, **kwargs) -> None:
        """Handle tool execution end.

        Args:
            output: Tool output
            **kwargs: Additional arguments
        """
        if self.spinner:
            with self.spinner_lock:
                self.first_token_received = False
                self.spinner.start()

    def reset(self) -> None:
        """Reset the callback handler state."""
        self.first_token_received = False
        self.code_formatter = CodeFormatter()
        self._in_thinking = False
        self._tag_buffer = ""


class LangGraphClient:
    """LangGraph-based client that replaces StrandsClient."""

    def __init__(
        self,
        server_path: str = "server/server.py",
        verbose: bool = False,
    ) -> None:
        """Initialize the LangGraph client.

        Args:
            server_path: Path to the MCP server script
            verbose: Enable verbose mode to show thinking process
        """
        self.verbose_mode = verbose
        self.server_path = server_path

        # Initialize MCP client
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_path],
            env=None,
        )
        self.mcp_client = MCPClientWrapper(self.server_params)

        # Initialize profile manager
        self.profile_manager = UserProfileManager()

        # Build system prompt
        self.system_prompt = config.system_prompt or ""
        if self.system_prompt:
            current_date = date.today().strftime("%Y-%m-%d")
            self.system_prompt = self.system_prompt.format(current_date=current_date)

        if config.get("PROFILE", {}).get("USE_PROFILING", False):
            profile_summary = self.profile_manager.get_profile_summary()
            if profile_summary:
                self.system_prompt = f"{self.system_prompt}\n\n{profile_summary}"

        # Initialize session ID
        profile_name = config.get("PROFILE", {}).get("NAME", "default")
        self.session_id = f"{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Components
        self.agent: Optional[LangGraphAgent] = None
        self.tools = None
        self.model = None

        # Initialize LLM controller
        from models.langchain_llm_controller import LangChainLLMController
        self.llm_controller = LangChainLLMController(verbose=self.verbose_mode)

        # Managers
        self.conversation_manager = AgentConversationManager(
            max_tokens=config.get("MAX_CONVERSATION_TOKENS", 1024 * 4)
        )
        self.dpo_collector = DPOCollector()
        self.dpo_mode = False

        # UI
        self.spinner = Spinner()
        self.spinner_lock = threading.Lock()
        self.first_token_received = False
        self.visible_content = None

        # Streaming callback
        self.callback_handler = StreamingCallbackHandler(
            verbose=self.verbose_mode,
            spinner=self.spinner,
            spinner_lock=self.spinner_lock,
        )

        # Episodic memory
        self.episodic_memory = None
        if config.get("ENABLE_EPISODIC_MEMORY", False):
            self._initialize_episodic_memory()

        # Track previous interaction for episodic memory
        self.previous_query = None
        self.previous_response = None
        self.previous_messages = None

    def _initialize_episodic_memory(self) -> None:
        """Initialize episodic memory if enabled."""
        logger.debug("Initializing episodic memory...")

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

        from models.embeddings_controller import EmbeddingsController
        embeddings_controller = EmbeddingsController(embed_model_config)

        self.episodic_memory = EpisodicMemoryManager(
            persist_path=episodic_path,
            store_type=store_type,
            embeddings_controller=embeddings_controller,
        )

        self.episodic_memory.cleanup(max_episodes=1000, max_age_days=90)
        logger.debug(f"✓ {store_type.upper()} episodic memory initialized")

    def start(self, verbose: bool = False) -> None:
        """Start the client and initialize the agent.

        Args:
            verbose: Enable verbose mode to show thinking process
        """
        try:
            self.verbose_mode = verbose
            self.callback_handler.verbose = verbose

            # Connect to MCP server and get tools
            with self.mcp_client:
                self.tools = self.mcp_client.list_tools_sync()
                logger.info(f"Loaded {len(self.tools)} tools from MCP server")

                # Initialize RAG session if enabled
                if config.get("ENABLE_RAG", False):
                    self._initialize_rag_session()

                # Initialize chunk cache
                self._initialize_chunk_cache()

                # Initialize model with callbacks
                self.llm_controller.initialize_model(callbacks=[self.callback_handler])
                self.model = self.llm_controller.get_model()

                # Create LangGraph agent
                self.agent = LangGraphAgent(
                    model=self.model,
                    tools=self.tools,
                    system_prompt=self.system_prompt,
                    verbose=self.verbose_mode,
                )

                logger.info("LangGraph agent initialized successfully")

        except Exception as e:
            stacktrace = traceback.format_exc()
            logger.error(stacktrace)
            raise e

    def query(self, prompt: str) -> str:
        """Send a query to the agent.

        Args:
            prompt: User's query

        Returns:
            Agent's response
        """
        if not self.agent:
            raise RuntimeError(
                "Client not started. Call start() or use with-statement first."
            )

        # Reset callback state and start spinner
        self.callback_handler.reset()
        with self.spinner_lock:
            self.first_token_received = False
            self.spinner.start()

        try:
            # Retrieve similar episodes from episodic memory
            if self.episodic_memory:
                prompt = self._inject_episodic_context(prompt)

            with self.mcp_client:
                # Call agent with prompt
                response = self.agent(prompt)

                # Flush code formatter
                self.callback_handler.code_formatter.flush()

                # Manage conversation length
                asyncio.run(
                    self.conversation_manager.manage_messages(
                        self, self.model, self.agent
                    )
                )

                # Update user profile
                if config.get("PROFILE", {}).get("USE_PROFILING", False):
                    messages_for_profile = convert_langchain_messages_to_strands(
                        self.agent.messages
                    )
                    self.profile_manager.analyze_conversation(messages_for_profile)

                # Check for empty response
                response_text = str(response)
                visible_content = re.sub(
                    r"<think(?:ing)?>.*?</think(?:ing)?>",
                    "",
                    response_text,
                    flags=re.DOTALL | re.IGNORECASE,
                ).strip()

                if not visible_content:
                    response_text += "\n\nI apologize, but I need to provide a visible response. Could you please rephrase your request?"
                    print(
                        "\n\033[91m⚠️  Model provided only thinking without visible response\033[0m"
                    )
                    print(
                        "I apologize, but I need to provide a visible response. Could you please rephrase your request?"
                    )

                # Print token count
                token_count = self._count_context_tokens()
                print(f"\n\033[90m[Context: {token_count} tokens]\033[0m")

                # Store for episodic memory
                self.previous_query = prompt
                self.previous_response = response_text
                self.previous_messages = self.agent.messages.copy()

                return response_text

        except Exception as e:
            with self.spinner_lock:
                self.spinner.stop()
            raise e
        finally:
            with self.spinner_lock:
                self.spinner.stop()

    def _inject_episodic_context(self, prompt: str) -> str:
        """Inject episodic memory context into the prompt.

        Args:
            prompt: Original prompt

        Returns:
            Prompt with episodic context prepended
        """
        logger.debug("Retrieving similar episodes from episodic memory...")
        similar_episodes = self.episodic_memory.retrieve_similar_episodes(
            prompt, top_k=3
        )
        logger.debug(f"Found {len(similar_episodes)} similar episodes")

        retrieval_threshold = config.get("EPISODIC_MEMORY", {}).get(
            "RETRIEVAL_THRESHOLD", 0.7
        )
        relevant_episodes = [
            ep
            for ep in similar_episodes
            if ep.get("similarity", 0) > retrieval_threshold
        ]

        if relevant_episodes:
            logger.debug(f"Found {len(relevant_episodes)} relevant episodes")
            context = self._format_episodic_context(relevant_episodes)
            return f"{context}\n\n{prompt}"

        return prompt

    def _format_episodic_context(self, episodes: list) -> str:
        """Format episodic memory episodes as compact context.

        Args:
            episodes: List of episode dictionaries

        Returns:
            Formatted context string
        """
        context = "[Episodic Memory - Similar Past Tasks]\n"
        for i, ep in enumerate(episodes, 1):
            task = ep.get("task", "Unknown task")
            if len(task) > 70:
                task = task[:70].rsplit(" ", 1)[0] + "..."

            tools = ep.get("tools", "")
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

        return context

    def _count_context_tokens(self) -> int:
        """Count total tokens in the current conversation context.

        Returns:
            Total token count
        """
        total_tokens = 0

        if self.system_prompt:
            total_tokens += count_tokens(self.system_prompt)

        if self.agent and self.agent.messages:
            messages_str = json.dumps(
                [{"content": str(m.content)} for m in self.agent.messages],
                default=str
            )
            total_tokens += count_tokens(messages_str)

        return total_tokens

    def clear_context(self) -> None:
        """Clear conversation history but keep system prompt."""
        if self.agent:
            self.agent.clear_messages()

        profile_name = config.get("PROFILE", {}).get("NAME", "default")
        self.session_id = f"{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        system_msg = config.get("SYSTEM_PROMPT")
        if system_msg:
            current_date = date.today().strftime("%Y-%m-%d")
            self.system_prompt = system_msg.format(current_date=current_date)
            if self.agent:
                self.agent.system_prompt = self.system_prompt

        # Flush RAG database if enabled
        if config.get("ENABLE_RAG", False):
            self._flush_rag_store()

        self._flush_chunk_cache_store()

    def save_conversation(self, timestamp: str = None) -> None:
        """Save conversation to file.

        Args:
            timestamp: Optional timestamp for filename
        """
        if not self.agent:
            logger.error("Agent not initialized")
            return

        try:
            user_home = os.path.expanduser("~")
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            conversations_dir = os.path.join(
                user_home, "agent-conversations", profile_name
            )
            os.makedirs(conversations_dir, exist_ok=True)

            if timestamp is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filepath = os.path.join(conversations_dir, f"conversation_{timestamp}.json")

            # Convert LangChain messages to Strands format for compatibility
            strands_messages = convert_langchain_messages_to_strands(
                self.agent.messages
            )

            conversation_data = {
                "messages": [
                    {"role": "system", "content": [{"text": self.system_prompt}]}
                ] + strands_messages,
                "tools": [{"name": t.name, "description": t.description} for t in self.tools] if self.tools else [],
            }

            with open(filepath, "w") as f:
                json.dump(conversation_data, f, indent=2, default=str)

            logger.info(f"Conversation saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")

    def load_conversation(self, file_path: str) -> bool:
        """Load conversation from file.

        Args:
            file_path: Path to the conversation file

        Returns:
            True if successful, False otherwise
        """
        try:
            normalized_path = os.path.expanduser(file_path)
            if not os.path.exists(normalized_path):
                logger.error(f"File not found: {normalized_path}")
                return False

            with open(normalized_path, "r") as f:
                conversation_data = json.load(f)

            if isinstance(conversation_data, list):
                messages = conversation_data
            else:
                messages = conversation_data.get("messages", [])

            # Filter out system messages
            conversation_messages = [
                m for m in messages if m.get("role") != "system"
            ]

            # Convert to LangChain messages
            langchain_messages = convert_strands_messages_to_langchain(
                conversation_messages
            )

            if self.agent:
                self.agent.messages.clear()
                self.agent.messages.extend(langchain_messages)
                logger.info(
                    f"Loaded {len(langchain_messages)} messages from {normalized_path}"
                )

                token_count = self._count_context_tokens()
                print(f"\n\033[90m[Context: {token_count} tokens]\033[0m")
                return True
            else:
                logger.error("Agent not initialized")
                return False

        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return False

    def _initialize_rag_session(self) -> None:
        """Initialize RAG session."""
        # Implementation depends on your RAG setup
        pass

    def _initialize_chunk_cache(self) -> None:
        """Initialize chunk cache database."""
        # Implementation depends on your cache setup
        pass

    def _flush_rag_store(self) -> None:
        """Flush the RAG store."""
        # Implementation depends on your RAG setup
        pass

    def _flush_chunk_cache_store(self) -> None:
        """Flush the chunk cache store."""
        # Implementation depends on your cache setup
        pass

    def __enter__(self):
        """Context manager entry."""
        self.start(verbose=self.verbose_mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass


# Alias for backward compatibility
StrandsClient = LangGraphClient
