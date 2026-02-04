"""LangGraph-based client implementation."""

import ast
import asyncio
from client.agent import (
    LangGraphAgent,
    convert_strands_messages_to_langchain,
    convert_langchain_messages_to_strands,
)
from client.managers.agent_conversation_manager import AgentConversationManager
from client.managers.user_profile_manager import UserProfileManager
from client.mcp_tool_wrapper import MCPClientWrapper
from client.memory.episodic_memory import EpisodicMemoryManager
from client.memory.reflector import Reflector
from client.memory.playbook_store import PlaybookStore
from client.ui.spinner import Spinner
from datetime import date, datetime
import json
from langchain_core.callbacks import BaseCallbackHandler
from mcp import StdioServerParameters
from models.llm_controller import LangChainLLMController
import numpy as np
import os
from server.tools import count_tokens
import shutil
import sqlite3
import sys
import threading
import traceback
from typing import Optional
from utils.config import config
from utils.logger import logger


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for spinner control during streaming."""

    def __init__(
        self,
        spinner: Optional[Spinner] = None,
        spinner_lock: Optional[threading.Lock] = None,
    ) -> None:
        """Initialize the streaming callback handler.

        Args:
            spinner: Spinner instance for UI feedback
            spinner_lock: Thread lock for spinner operations
        """
        self.spinner = spinner
        self.spinner_lock = spinner_lock or threading.Lock()
        self.first_token_received = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new tokens from the LLM.

        Args:
            token: The new token
            **kwargs: Additional arguments
        """
        if not self.first_token_received and self.spinner:
            with self.spinner_lock:
                if not self.first_token_received:
                    self.spinner.stop()
                    self.first_token_received = True

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


class LangGraphClient:
    """LangGraph-based client for AI assistant."""

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

        # MCP client
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_path],
            env=os.environ.copy(),
        )
        self.mcp_client = MCPClientWrapper(self.server_params)

        # System prompt
        self.profile_manager = UserProfileManager()
        self.system_prompt = self._build_system_prompt()

        # Session
        profile_name = config.get("PROFILE", {}).get("NAME", "default")
        self.session_id = f"{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Components
        self.agent: Optional[LangGraphAgent] = None
        self.tools = None
        self.model = None

        # LLM controller
        self.llm_controller = LangChainLLMController(verbose=self.verbose_mode)

        # Managers
        self.conversation_manager = AgentConversationManager(
            max_tokens=config.get("MAX_CONVERSATION_TOKENS", 1024 * 4)
        )

        # UI
        self.spinner = Spinner()
        self.spinner_lock = threading.Lock()
        self.callback_handler = StreamingCallbackHandler(
            spinner=self.spinner,
            spinner_lock=self.spinner_lock,
        )

        # Episodic memory
        self.episodic_memory = None
        if config.get("ENABLE_EPISODIC_MEMORY", False):
            self._initialize_episodic_memory()

        # ACE components (Reflector + Playbook)
        self.reflector = None
        self.playbook = None
        if config.get("ENABLE_PLAYBOOK", False):
            self._initialize_playbook()

        # Previous interaction tracking
        self.previous_query = None
        self.previous_response = None
        self.previous_messages = None

    def _build_system_prompt(self) -> str:
        """Build the system prompt with profile information.

        Returns:
            Complete system prompt string
        """
        system_prompt = config.system_prompt or ""
        if system_prompt:
            current_date = date.today().strftime("%Y-%m-%d")
            system_prompt = system_prompt.format(current_date=current_date)

        if config.get("PROFILE", {}).get("USE_PROFILING", False):
            profile_summary = self.profile_manager.get_profile_summary()
            if profile_summary:
                system_prompt = f"{system_prompt}\n\n{profile_summary}"

        return system_prompt

    def _initialize_episodic_memory(self) -> None:
        """Initialize episodic memory if enabled."""
        logger.debug("Initializing episodic memory...")

        embed_model_config = config.get("EMBED_MODEL_ID")
        if not embed_model_config:
            raise ValueError("EMBED_MODEL_ID must be configured for episodic memory")

        user_home = os.path.expanduser("~")
        profile_name = config.get("PROFILE", {}).get("NAME", "default")
        episodic_path = os.path.join(
            user_home, "agent-conversations", profile_name, "episodic_memory"
        )
        os.makedirs(episodic_path, exist_ok=True)

        store_type = config.get("EPISODIC_MEMORY_STORE", "chromadb").lower()

        from models.embeddings_controller import EmbeddingsController

        embeddings_controller = EmbeddingsController(embed_model_config)

        self.episodic_memory = EpisodicMemoryManager(
            persist_path=episodic_path,
            store_type=store_type,
            embeddings_controller=embeddings_controller,
        )
        self.episodic_memory.cleanup(max_episodes=1000, max_age_days=90)
        logger.debug(f"✓ {store_type.upper()} episodic memory initialized")

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
        logger.debug(f"✓ Playbook initialized ({stats['total_entries']} entries)")

    def start(self, verbose: bool = False) -> None:
        """Start the client and initialize the agent.

        Args:
            verbose: Enable verbose mode to show thinking process
        """
        try:
            self.verbose_mode = verbose

            with self.mcp_client:
                self.tools = self.mcp_client.list_tools_sync()
                logger.info(f"Loaded {len(self.tools)} tools from MCP server")

                # Initialize RAG session if enabled
                if config.get("ENABLE_RAG", False):
                    self._initialize_rag_session()

                # Initialize chunk cache
                self._initialize_chunk_cache()

                self.llm_controller.initialize_model(callbacks=[self.callback_handler])
                self.model = self.llm_controller.get_model()

                # Build system prompt with playbook context
                system_prompt_with_context = self.system_prompt
                if self.playbook:
                    playbook_context = self._get_playbook_context()
                    if playbook_context:
                        system_prompt_with_context = (
                            f"{self.system_prompt}\n\n{playbook_context}"
                        )

                self.agent = LangGraphAgent(
                    model=self.model,
                    tools=self.tools,
                    system_prompt=system_prompt_with_context,
                    verbose=self.verbose_mode,
                    callbacks=[self.callback_handler],
                )

        except Exception as e:
            logger.error(traceback.format_exc())
            raise e

    def query(self, prompt: str) -> str:
        """Send a query to the agent.

        Args:
            prompt: User's query

        Returns:
            Agent's response
        """
        if not self.agent:
            raise RuntimeError("Client not started. Call start() first.")

        self.callback_handler.reset()
        with self.spinner_lock:
            self.spinner.start()

        try:
            if self.episodic_memory:
                prompt = self._inject_episodic_context(prompt)

            with self.mcp_client:
                response = self.agent(prompt)

                # Flush any remaining buffered code
                if hasattr(self.agent, "_code_formatter"):
                    self.agent._code_formatter.flush()

                asyncio.run(
                    self.conversation_manager.manage_messages(
                        self, self.model, self.agent
                    )
                )

                if config.get("PROFILE", {}).get("USE_PROFILING", False):
                    messages_for_profile = convert_langchain_messages_to_strands(
                        self.agent.messages
                    )
                    self.profile_manager.analyze_conversation(messages_for_profile)

                token_count = self._count_context_tokens()
                print(f"\n\033[90m[Context: {token_count} tokens]\033[0m")

                # Store for episodic memory evaluation
                if self.episodic_memory:
                    self.previous_query = prompt
                    self.previous_response = response
                    self.previous_messages = self.agent.messages.copy()

                return response

        except KeyboardInterrupt:
            with self.spinner_lock:
                self.spinner.stop()
            return "Operation was cancelled."

        finally:
            with self.spinner_lock:
                self.spinner.stop()

    def reflect_and_learn(self, task: str) -> None:
        """Run reflection on the last interaction and update playbook.

        Args:
            task: The original user task
        """
        if not self.reflector or not self.playbook:
            return

        if not self.agent or not self.agent.messages:
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

    def _inject_playbook_context(self, prompt: str) -> str:
        """Inject task-specific playbook strategies into the prompt.

        Args:
            prompt: Original prompt

        Returns:
            Prompt with playbook context prepended
        """
        relevant_entries = self.playbook.get_relevant_entries(
            task=prompt,
            top_k=config.get("PLAYBOOK", {}).get("MAX_INJECT", 10),
            include_failures=True,
        )

        if relevant_entries:
            playbook_text = self.playbook.format_for_prompt(relevant_entries)
            return f"{playbook_text}\n\n{prompt}"

        return prompt

    def _get_conversation_context(self) -> str:
        """Extract text context from current conversation.

        Returns:
            Concatenated text from recent messages
        """
        if not self.agent or not self.agent.messages:
            return ""

        context_parts = []
        for msg in self.agent.messages[-6:]:
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content:
                context_parts.append(content[:1000])
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        context_parts.append(item["text"][:1000])

        return " ".join(context_parts)

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts.

        Uses embeddings if available, falls back to lexical similarity.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0

        # Try semantic similarity with embeddings
        if config.get("EMBED_MODEL_ID"):
            try:
                from models.embeddings_controller import EmbeddingsController

                embeddings = EmbeddingsController()
                emb = embeddings.embed([text1, text2])
                emb1, emb2 = emb[0], emb[1]
                similarity = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )
                return float(similarity)
            except Exception:
                pass

        # Fallback: Jaccard similarity on word sets
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _inject_episodic_context(self, prompt: str) -> str:
        """Inject episodic memory context into the prompt.

        Uses similarity (semantic or lexical) to determine:
        1. If the query relates to the current conversation (skip injection)
        2. If retrieved episodes add new information vs. being redundant

        Args:
            prompt: Original prompt

        Returns:
            Prompt with episodic context prepended (if relevant and non-redundant)
        """
        conversation_context = self._get_conversation_context()

        # If there's existing conversation, check if query relates to it
        if conversation_context:
            query_to_conv_similarity = self._compute_similarity(
                prompt, conversation_context
            )
            follow_up_threshold = config.get("EPISODIC_MEMORY", {}).get(
                "FOLLOW_UP_THRESHOLD", 0.4  # Lower for Jaccard fallback
            )
            if query_to_conv_similarity > follow_up_threshold:
                return prompt

        # Retrieve similar episodes
        similar_episodes = self.episodic_memory.retrieve_similar_episodes(
            prompt, top_k=5
        )
        retrieval_threshold = config.get("EPISODIC_MEMORY", {}).get(
            "RETRIEVAL_THRESHOLD", 0.7
        )
        relevant_episodes = [
            ep
            for ep in similar_episodes
            if ep.get("similarity", 0) > retrieval_threshold
        ]

        if not relevant_episodes:
            return prompt

        # Filter episodes redundant with current conversation
        if conversation_context:
            redundancy_threshold = config.get("EPISODIC_MEMORY", {}).get(
                "REDUNDANCY_THRESHOLD", 0.5
            )
            filtered_episodes = []
            for ep in relevant_episodes:
                ep_task = ep.get("task", "")
                ep_to_conv_similarity = self._compute_similarity(
                    ep_task, conversation_context
                )
                if ep_to_conv_similarity < redundancy_threshold:
                    filtered_episodes.append(ep)
            relevant_episodes = filtered_episodes

        if not relevant_episodes:
            return prompt

        # Format and inject
        context = "[Episodic Memory - Similar Past Tasks]\n"
        for i, ep in enumerate(relevant_episodes, 1):
            task = ep.get("task", "Unknown task")[:70]
            tools = ep.get("tools", "")
            tool_names = []
            if isinstance(tools, str):
                try:
                    tools_list = ast.literal_eval(tools)
                    tool_names = [
                        t.get("name", "") for t in tools_list if isinstance(t, dict)
                    ]
                except:
                    pass
            tools_str = ", ".join(tool_names) if tool_names else "no tools"
            similarity = ep.get("similarity", 0)
            context += f'{i}. "{task}" → {tools_str} (similarity: {similarity:.2f})\n'

        return f"{context}\n\n{prompt}"

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
                [{"content": str(m.content)} for m in self.agent.messages], default=str
            )
            total_tokens += count_tokens(messages_str)
        return total_tokens

    def clear_context(self) -> None:
        """Clear conversation history but keep system prompt."""
        if self.agent:
            self.agent.clear_messages()

        profile_name = config.get("PROFILE", {}).get("NAME", "default")
        self.session_id = f"{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.system_prompt = self._build_system_prompt()
        if self.agent:
            self.agent.system_prompt = self.system_prompt

        # Flush RAG database when clearing context
        if config.get("ENABLE_RAG", False):
            self._flush_rag_store()

        self._flush_chunk_cache_store()

        # Flush RAG database if enabled
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

            session_file = os.path.join(rag_dir, "chunk_session_id.txt")
            with open(session_file, "w") as f:
                f.write(self.session_id)

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
        """Flush the chunk cache database."""
        try:
            from server.tools.readers.chunking_helper import reset_session_chunk_cache

            reset_session_chunk_cache()

            user_home = os.path.expanduser("~")
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            rag_dir = os.path.join(user_home, "agent-conversations", profile_name)

            if os.path.exists(rag_dir):
                for file in os.listdir(rag_dir):
                    if file.startswith("chunk_cache_"):
                        file_path = os.path.join(rag_dir, file)
                        try:
                            os.remove(file_path)
                            logger.debug(f"Deleted session file: {file}")
                        except Exception as e:
                            logger.debug(f"Failed to delete {file}: {e}")

            logger.debug("Chunk cache store cleared")
        except Exception as e:
            logger.warning(f"Failed to reset chunk cache: {e}")

    def _flush_rag_store(self) -> None:
        """Flush the RAG database."""
        try:
            from server.tools.rag import reset_session_rag

            reset_session_rag()

            user_home = os.path.expanduser("~")
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            rag_dir = os.path.join(user_home, "agent-conversations", profile_name)

            if os.path.exists(rag_dir):
                for file in os.listdir(rag_dir):
                    if file.startswith("rag_store_"):
                        file_path = os.path.join(rag_dir, file)
                        try:
                            if os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                            else:
                                os.remove(file_path)
                            logger.debug(f"Deleted session file/dir: {file}")
                        except Exception as e:
                            logger.debug(f"Failed to delete {file}: {e}")

            logger.debug("RAG store cleared")
        except Exception as e:
            logger.warning(f"Failed to reset RAG store: {e}")

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

            timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(conversations_dir, f"conversation_{timestamp}.json")

            strands_messages = convert_langchain_messages_to_strands(
                self.agent.messages
            )
            conversation_data = {
                "messages": [
                    {"role": "system", "content": [{"text": self.system_prompt}]}
                ]
                + strands_messages,
                "tools": (
                    [{"name": t.name, "description": t.description} for t in self.tools]
                    if self.tools
                    else []
                ),
            }

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
            timestamp: Optional timestamp for filename
            quality_markers: List of quality labels for each message
        """
        if not self.agent:
            return

        try:
            user_home = os.path.expanduser("~")
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            save_dir = os.path.join(
                user_home, "agent-conversations", profile_name, "conversations"
            )
            os.makedirs(save_dir, exist_ok=True)

            timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(save_dir, f"conversation_{timestamp}.json")

            strands_messages = convert_langchain_messages_to_strands(
                self.agent.messages
            )
            conversation_data = {
                "messages": [
                    {"role": "system", "content": [{"text": self.system_prompt}]}
                ]
                + strands_messages,
                "tools": (
                    [{"name": t.name, "description": t.description} for t in self.tools]
                    if self.tools
                    else []
                ),
                "quality_markers": quality_markers or [],
            }

            with open(filepath, "w") as f:
                json.dump(conversation_data, f, indent=2, default=str)

            print(f"Conversation saved to {filepath}")

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

            messages = (
                conversation_data
                if isinstance(conversation_data, list)
                else conversation_data.get("messages", [])
            )
            conversation_messages = [m for m in messages if m.get("role") != "system"]
            langchain_messages = convert_strands_messages_to_langchain(
                conversation_messages
            )

            if self.agent:
                self.agent.messages.clear()
                self.agent.messages.extend(langchain_messages)
                logger.info(
                    f"Loaded {len(langchain_messages)} messages from {normalized_path}"
                )
                print(
                    f"\n\033[90m[Context: {self._count_context_tokens()} tokens]\033[0m"
                )
                return True

            logger.error("Agent not initialized")
            return False

        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return False

    def __enter__(self):
        """Context manager entry."""
        self.start(verbose=self.verbose_mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
