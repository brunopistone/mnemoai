"""Episodic memory manager for storing and retrieving task solutions."""

from .chroma_store import ChromaEpisodicStore
from .faiss_store import FAISSEpisodicStore
from datetime import datetime
from typing import List, Dict, Any, Optional
from utils.logger import logger


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

        if store_type == "faiss":
            self.store = FAISSEpisodicStore(persist_path, embeddings_controller)
        else:
            self.store = ChromaEpisodicStore(persist_path, embeddings_controller)

    def store_episode(
        self,
        task: str,
        solution: str,
        tools_used: List[Dict[str, Any]],
        outcome: str = "success",
        full_conversation: List[Dict[str, Any]] = None,
    ) -> None:
        """Store a task completion pattern.

        Args:
            task: User's original query/task
            solution: Agent's final response/solution
            tools_used: List of tools invoked with args and results
            outcome: Success indicator
            full_conversation: Complete conversation messages leading to solution
        """
        # Check for near-duplicate episodes (similarity > 0.95)
        similar = self.store.search(query=task, top_k=1)
        if similar and similar[0].get('similarity', 0) > 0.95:
            logger.debug(f"Skipping duplicate episode (similarity: {similar[0]['similarity']:.2f})")
            return
        
        metadata = {
            "task": task,
            "solution": solution,
            "tools": str(tools_used),  # Convert to string for ChromaDB
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
            "conversation": str(full_conversation) if full_conversation else "",
        }

        # Create searchable text from full conversation
        tool_names = [t.get("name", "") for t in tools_used]

        # Build conversation text from agent.messages format
        conv_text = ""
        if full_conversation:
            for msg in full_conversation:
                role = msg.get("role", "")
                content = msg.get("content", [])

                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            # Extract text content
                            if "text" in item:
                                conv_text += f"{role}: {item['text']}\n"
                            # Note tool usage
                            elif "toolUse" in item:
                                tool_name = item["toolUse"].get("name", "unknown")
                                conv_text += f"{role}: [Used tool: {tool_name}]\n"
                            elif "toolResult" in item:
                                conv_text += f"{role}: [Tool result received]\n"

        text = f"Task: {task}\nConversation:\n{conv_text}\nSolution: {solution}\nTools: {', '.join(tool_names)}\nOutcome: {outcome}"

        logger.debug(f"Storing episode with full conversation: {task[:50]}...")
        self.store.add(text=text, metadata=metadata)

    def retrieve_similar_episodes(
        self, task: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Find similar past task solutions.

        Args:
            task: Current task to find similar episodes for
            top_k: Number of results to return

        Returns:
            List of similar episodes with metadata
        """
        logger.debug(f"Retrieving similar episodes for: {task[:50]}...")
        return self.store.search(query=task, top_k=top_k)

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

    # 1. Check for explicit success markers
    if next_user_message:
        next_lower = next_user_message.lower()

        # Check for success markers (whole words)
        success_markers = ["thanks", "thank you", "perfect", "great", "worked", "good"]
        if any(
            f" {marker} " in f" {next_lower} "
            or next_lower.startswith(marker + " ")
            or next_lower.endswith(" " + marker)
            for marker in success_markers
        ):
            logger.debug(
                f"✓ Success marker found in user message: {next_user_message[:50]}"
            )
            return True

        # Check for correction requests (whole words only)
        correction_markers = [
            "wrong",
            "error",
            "fix",
            "actually",
            "instead",
            "incorrect",
        ]
        if any(
            f" {marker} " in f" {next_lower} "
            or next_lower.startswith(marker + " ")
            or next_lower.endswith(" " + marker)
            for marker in correction_markers
        ):
            logger.debug(f"✗ Correction marker found: {next_user_message[:50]}")
            return False

        # Check for standalone "no" at start of message
        if next_lower.startswith("no ") or next_lower == "no":
            logger.debug(f"✗ Correction marker found: {next_user_message[:50]}")
            return False

    # 2. Check for error markers in response (only at sentence start or after punctuation)
    error_patterns = [
        "error:",
        "failed:",
        "exception:",
        "could not",
        "unable to",
        "error occurred",
        "failed to",
        "an error",
        "the error",
    ]
    response_lower = agent_response.lower()
    if any(pattern in response_lower for pattern in error_patterns):
        logger.debug(f"✗ Error marker found in response")
        return False

    # 3. Check if tools succeeded
    for msg in agent_messages:
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "toolResult" in item:
                        result = item["toolResult"]
                        if isinstance(result, dict) and result.get("error"):
                            logger.debug(f"✗ Tool execution failed")
                            return False

    logger.debug("✓ Task appears successful (no negative indicators)")
    return True


def extract_tools_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract tool usage information from conversation messages.

    Args:
        messages: Conversation messages

    Returns:
        List of tool usage records
    """
    tools_used = []

    for msg in messages:
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
