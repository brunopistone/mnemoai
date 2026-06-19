"""Episodic memory manager for storing and retrieving task solutions."""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import tiktoken

from mnemoai.utils.config import config
from mnemoai.utils.logger import logger

from .chroma_store import ChromaEpisodicStore
from .faiss_store import FAISSEpisodicStore


class EpisodicMemoryManager:
    """Manages episodic memory - past task solutions with tool usage patterns."""

    def __init__(
        self,
        persist_path: str = None,
        store_type: str = "chroma",
        embeddings_controller=None,
    ):
        """Initialize episodic memory manager.

        Args:
            persist_path: Path to persist vector store data
            store_type: Type of vector store ("chroma" or "faiss")
            embeddings_controller: Required for both FAISS and ChromaDB stores
        """
        if not embeddings_controller:
            raise ValueError("embeddings_controller is required for episodic memory")

        self.encoder = tiktoken.encoding_for_model("gpt-4")  # For token counting

        # Load configuration
        self.config = config.get("EPISODIC_MEMORY", {})
        self.duplicate_threshold = self.config.get("DUPLICATE_THRESHOLD", 0.95)
        self.retrieval_threshold = self.config.get("RETRIEVAL_THRESHOLD", 0.7)
        self.max_tokens = self.config.get("MAX_TOKENS_PER_EPISODE", 400)
        self.semantic_weight = self.config.get("SEMANTIC_WEIGHT", 0.7)
        self.keyword_weight = self.config.get("KEYWORD_WEIGHT", 0.3)

        logger.debug(
            f"Episodic memory initialized with duplicate_threshold={self.duplicate_threshold}, "
            f"retrieval_threshold={self.retrieval_threshold}, max_tokens={self.max_tokens}"
        )

        if store_type == "faiss":
            self.store = FAISSEpisodicStore(persist_path, embeddings_controller)
        else:
            self.store = ChromaEpisodicStore(persist_path, embeddings_controller)

    def count_tokens(self, text: str) -> int:
        """Count tokens with model-specific approximation.

        For Ollama models, uses character-based approximation.
        For OpenAI/Bedrock models, uses tiktoken encoder.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        model_type = config.get("MODEL_ID", {}).get("TYPE", "ollama")

        if model_type == "ollama":
            # Ollama approximation: ~1.3 chars per token (configurable)
            multiplier = (
                config.get("LLM", {})
                .get("TOKEN_COUNTING", {})
                .get("OLLAMA_APPROXIMATION", 1.3)
            )
            return int(len(text) / multiplier)
        else:
            # Use tiktoken for OpenAI/Bedrock/SageMaker
            return len(self.encoder.encode(text))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens

        Returns:
            Truncated text
        """
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.encoder.decode(truncated_tokens)
        return truncated_text + "..."

    def store_episode(
        self,
        task: str,
        tools_used: List[Dict[str, Any]],
        outcome: str = "success",
    ) -> None:
        """Store a task completion pattern.

        Args:
            task: User's original query/task
            tools_used: List of tools invoked with args and results
            outcome: Success indicator
        """
        # Check for near-duplicate episodes (configurable threshold)
        similar = self.store.search(query=task, top_k=1)
        if similar and similar[0].get("similarity", 0) > self.duplicate_threshold:
            logger.debug(
                f"Skipping duplicate episode (similarity: {similar[0]['similarity']:.2f}, "
                f"threshold: {self.duplicate_threshold})"
            )
            return

        metadata = {
            "task": task,
            "tools": str(tools_used),
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
        }

        # Create compact searchable text - just task and tool names
        tool_names = [t.get("name", "") for t in tools_used]
        tool_summary = ", ".join(tool_names) if tool_names else "no tools"

        text = f"Task: {task}\nTools used: {tool_summary}\nOutcome: {outcome}"

        # Truncate to fit embedding model context (configurable)
        token_count = self.count_tokens(text)

        if token_count > self.max_tokens:
            logger.debug(
                f"Truncating episode from {token_count} to {self.max_tokens} tokens"
            )
            text = self._truncate_to_tokens(text, self.max_tokens)

        logger.debug(
            f"Storing episode: {task[:50]}... ({self.count_tokens(text)} tokens)"
        )
        self.store.add(text=text, metadata=metadata)

    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms for better retrieval.

        Args:
            query: Original query

        Returns:
            Expanded query with synonyms
        """
        if not self.config.get("ENABLE_QUERY_EXPANSION", True):
            return query

        # Simple synonym mapping for common actions
        synonyms = {
            "create": ["make", "generate"],
            "delete": ["remove", "clear"],
            "search": ["find", "locate"],
            "update": ["modify", "edit"],
            "fix": ["repair", "resolve"],
            "install": ["setup", "add"],
            "run": ["execute", "start"],
            "read": ["view", "show"],
            "write": ["save", "store"],
            "list": ["show", "display"],
        }

        query_lower = query.lower()
        expanded_terms = []
        max_terms = self.config.get("QUERY_EXPANSION_TERMS", 3)

        # Find matching words and add their synonyms
        for word, variants in synonyms.items():
            if word in query_lower and len(expanded_terms) < max_terms:
                # Add first variant only
                expanded_terms.append(variants[0])

        if expanded_terms:
            expanded_query = f"{query} {' '.join(expanded_terms)}"
            logger.debug(
                f"Query expansion: '{query[:50]}...' -> '{expanded_query[:80]}...'"
            )
            return expanded_query

        return query

    def retrieve_similar_episodes(
        self, task: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Find similar past task solutions with query expansion.

        Args:
            task: Current task to find similar episodes for
            top_k: Number of results to return

        Returns:
            List of similar episodes with metadata
        """
        logger.debug(f"Retrieving similar episodes for: {task[:50]}...")

        # Apply query expansion if enabled
        expanded_task = self._expand_query(task)

        return self.store.search(query=expanded_task, top_k=top_k)

    def cleanup(self, max_episodes: int = 1000, max_age_days: int = 90) -> None:
        """Remove old episodes and enforce size limit.

        Args:
            max_episodes: Maximum number of episodes to keep
            max_age_days: Maximum age in days
        """
        self.store.cleanup(max_episodes=max_episodes, max_age_days=max_age_days)


def is_task_successful(
    agent_response: str,
    agent_messages: List[Dict[str, Any]],
    next_user_message: Optional[str] = None,
) -> bool:
    """Determine if a task was successfully completed.

    Args:
        agent_response: Agent's final response
        agent_messages: Full conversation messages
        next_user_message: User's next message (if available)

    Returns:
        True if task appears successful
    """
    logger.debug("Evaluating task success...")

    # Load configurable markers
    episodic_config = config.get("EPISODIC_MEMORY", {})
    success_markers = episodic_config.get(
        "SUCCESS_MARKERS", ["thanks", "thank you", "perfect", "great", "worked", "good"]
    )
    correction_markers = episodic_config.get(
        "CORRECTION_MARKERS",
        ["wrong", "error", "fix", "actually", "instead", "incorrect"],
    )
    error_patterns = episodic_config.get(
        "ERROR_PATTERNS", ["error:", "failed:", "exception:", "could not", "unable to"]
    )

    # 1. Check for explicit success markers
    if next_user_message:
        next_lower = next_user_message.lower()

        # Check for success markers (whole words). Coerce to str: unquoted
        # YAML values like `no`/`yes`/`off` parse as bool/None, which would
        # otherwise crash on .lower().
        success_markers_lower = [str(m).lower() for m in success_markers]
        if any(
            f" {marker} " in f" {next_lower} "
            or next_lower.startswith(marker + " ")
            or next_lower.endswith(" " + marker)
            for marker in success_markers_lower
        ):
            logger.debug(
                f"✓ Success marker found in user message: {next_user_message[:50]}"
            )
            return True

        # Check for correction requests (whole words only)
        correction_markers_lower = [str(m).lower() for m in correction_markers]
        if any(
            f" {marker} " in f" {next_lower} "
            or next_lower.startswith(marker + " ")
            or next_lower.endswith(" " + marker)
            for marker in correction_markers_lower
        ):
            logger.debug(f"✗ Correction marker found: {next_user_message[:50]}")
            return False

        # Check for standalone "no" at start of message
        if next_lower.startswith("no ") or next_lower == "no":
            logger.debug(f"✗ Correction marker found: {next_user_message[:50]}")
            return False

    # 2. Check for error markers in response (configurable)
    response_lower = agent_response.lower()
    error_patterns_lower = [str(p).lower() for p in error_patterns]
    if any(pattern in response_lower for pattern in error_patterns_lower):
        logger.debug(f"✗ Error marker found in response")
        return False

    # 3. Check if tools succeeded
    for msg in agent_messages:
        # Handle both dict format and LangChain message objects
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", [])
        else:
            # LangChain message object
            role = getattr(msg, "type", None)
            if role == "human":
                role = "user"
            content = getattr(msg, "content", "")

        if role == "user":
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "toolResult" in item:
                        result = item["toolResult"]
                        if isinstance(result, dict) and result.get("error"):
                            logger.debug(f"✗ Tool execution failed")
                            return False

    logger.debug("✓ Task appears successful (no negative indicators)")
    return True


def extract_tools_from_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """Extract tool usage information from conversation messages.

    Supports both Strands format (dict) and LangChain format (BaseMessage).

    Args:
        messages: Conversation messages in either format

    Returns:
        List of tool usage records
    """
    # Try to import LangChain types
    try:
        from langchain_core.messages import AIMessage, ToolMessage

        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False

    tools_used = []

    for msg in messages:
        # Handle LangChain AIMessage with tool_calls
        if LANGCHAIN_AVAILABLE and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(
                    {
                        "name": tc.get("name"),
                        "args": tc.get("args", {}),
                        "id": tc.get("id"),
                    }
                )
            continue

        # Handle LangChain ToolMessage (contains result)
        if LANGCHAIN_AVAILABLE and hasattr(msg, "tool_call_id"):
            for tool in tools_used:
                if tool.get("id") == msg.tool_call_id:
                    tool["result"] = str(msg.content)
            continue

        # Handle Strands dict format
        if isinstance(msg, dict):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "toolUse" in item:
                            tool_use = item["toolUse"]
                            tools_used.append(
                                {
                                    "name": tool_use.get("name"),
                                    "args": tool_use.get("input", {}),
                                    "id": tool_use.get("toolUseId"),
                                }
                            )

            elif msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "toolResult" in item:
                            result = item["toolResult"]
                            # Find matching tool by ID
                            for tool in tools_used:
                                if tool.get("id") == result.get("toolUseId"):
                                    tool["result"] = result.get("content", [{}])[0].get(
                                        "text", ""
                                    )

    return tools_used
