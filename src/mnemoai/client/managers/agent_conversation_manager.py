"""Conversation manager that uses simple text-based summaries."""

import json
import textwrap
from datetime import date
from typing import Any, Dict, List, Union

import tiktoken

from mnemoai.utils.config import config
from mnemoai.utils.logger import logger

# Try to import LangChain message types for compatibility
try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

MODEL_ID = "gpt-4"  # Default model for token counting

# ANSI color codes for green text
GREEN = "\033[92m"
RESET = "\033[0m"


def log_green(message: str, level: str = "info") -> None:
    """Print a user-facing status line in green (e.g. compaction progress).

    These are results/progress the user asked for, not diagnostics, so they go
    to stdout via ``print()`` — clean, no timestamp/level prefix, and always
    visible regardless of ``LOG_LEVEL``. ``level`` is accepted for backward
    compatibility but ignored. Operational diagnostics should use ``logger``.

    Args:
        message: Message to show the user
        level: Ignored (kept for backward compatibility)
    """
    print(f"{GREEN}{message}{RESET}")


def messages_to_dict_list(messages: List[Any]) -> List[Dict]:
    """Convert messages to list of dictionaries.

    Handles both Strands format (dict) and LangChain format (BaseMessage).

    Args:
        messages: List of messages in either format

    Returns:
        List of message dictionaries
    """
    result = []
    for msg in messages:
        if isinstance(msg, dict):
            result.append(msg)
        elif LANGCHAIN_AVAILABLE and isinstance(msg, BaseMessage):
            # Map LangChain message type -> role, preserving tool interactions.
            msg_type = getattr(msg, "type", None)
            entry: Dict[str, Any] = {"content": [{"text": str(msg.content)}]}

            if msg_type == "ai":
                entry["role"] = "assistant"
                # Preserve tool calls so the summary knows what was invoked.
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    entry["tool_calls"] = [
                        {"name": tc.get("name"), "args": tc.get("args", {})}
                        for tc in tool_calls
                    ]
            elif msg_type == "system":
                entry["role"] = "system"
            elif msg_type == "tool":
                # Tool result: keep its own role + the tool name for context.
                entry["role"] = "tool"
                entry["tool_name"] = getattr(msg, "name", None)
            else:
                entry["role"] = "user"

            result.append(entry)
        else:
            # Fallback: try to convert to string
            result.append({
                "role": "user",
                "content": [{"text": str(msg)}]
            })
    return result


class AgentConversationManager:
    def __init__(self, max_tokens: int = 5000) -> None:
        """Initialize conversation manager.

        Args:
            max_tokens: Maximum tokens before summarization
        """
        self.max_tokens = max_tokens
        self.encoder = tiktoken.encoding_for_model(MODEL_ID)
        self.previous_summary = None
        logger.info(f"Initialized conversation manager with max_tokens={max_tokens}")

    def count_tokens(self, messages: List[Dict]) -> int:
        """Count tokens with model-specific approximation.

        For Ollama models, uses character-based approximation.
        For OpenAI/Bedrock models, uses tiktoken encoder.

        Args:
            messages: List of message dictionaries

        Returns:
            Estimated token count
        """
        text = json.dumps(messages, default=str)
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

    @staticmethod
    def _message_text_for_summary(msg: Dict) -> str:
        """Render one message dict to text for the summary input.

        Includes tool calls (assistant) and tool results so tool interactions
        are captured, not just user/assistant prose.
        """
        parts = []
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
        elif isinstance(content, str):
            parts.append(content)

        text = "".join(parts).strip()
        role = msg.get("role")

        if role == "tool":
            name = msg.get("tool_name") or "tool"
            return f"[tool result from {name}]: {text}" if text else ""

        if role == "assistant" and msg.get("tool_calls"):
            calls = "; ".join(
                f"{tc.get('name')}({tc.get('args', {})})" for tc in msg["tool_calls"]
            )
            tool_line = f"[called tools: {calls}]"
            return f"{text}\n{tool_line}".strip() if text else tool_line

        return text

    async def generate_summary(
        self, messages: List[Dict], model: Any, focus_instructions: str = ""
    ) -> str:
        """Use the model to generate a natural language summary.

        Args:
            messages: List of conversation messages
            model: Model instance for generating summary (LangChain or Strands)
            focus_instructions: Optional user guidance on what to emphasize
                (used by the manual /compact command).

        Returns:
            Summary text
        """
        log_green(f"Generating summary ...")

        summary_prompt = f"""
        Create a detailed summary of this conversation that preserves important information for future reference. Include:

        1. Specific topics, requests, and questions discussed
        2. Key facts, data, or findings that were discovered
        3. Important decisions, solutions, or conclusions reached
        4. Any specific tools, commands, or technical details mentioned (including which tools were called, their inputs, and their results)
        5. Files read or modified, and any pending or in-progress work
        6. Context that would be valuable for continuing this conversation later

        Write in a structured format that maintains the essential details while being concise. Focus on preserving actionable information and specific context rather than just high-level themes.
        """

        summary_prompt = textwrap.dedent(summary_prompt).strip()
        if focus_instructions:
            summary_prompt += (
                "\n\nThe user asked you to focus the summary on the following — "
                f"prioritize this:\n{focus_instructions.strip()}"
            )

        try:
            summary_response = ""

            # Check if this is a LangChain model
            if LANGCHAIN_AVAILABLE and hasattr(model, 'ainvoke'):
                # LangChain model - use ainvoke
                from langchain_core.messages import HumanMessage, SystemMessage

                # Build LangChain messages
                lc_messages = []
                if self.previous_summary:
                    lc_messages.append(SystemMessage(content=self.previous_summary))

                # Add conversation context. Tool calls and tool results are
                # flattened into text so the summary captures what tools did,
                # not just the user/assistant prose.
                for msg in messages:
                    role = msg.get("role")
                    content = self._message_text_for_summary(msg)
                    if not content:
                        continue
                    if role == "assistant":
                        lc_messages.append(AIMessage(content=content))
                    else:
                        # user, tool results, and anything else become context
                        lc_messages.append(HumanMessage(content=content))

                # Add summary prompt
                lc_messages.append(HumanMessage(content=summary_prompt))

                # Generate summary
                response = await model.ainvoke(lc_messages)
                summary_response = str(response.content)

            else:
                # Strands model - use stream
                messages.append({"role": "user", "content": [{"text": summary_prompt}]})
                think_param = config.get("LLM", {}).get("SUMMARIZATION_THINK", False)

                async for event in model.stream(
                    messages, system_prompt=self.previous_summary, think=think_param
                ):
                    if (
                        "contentBlockDelta" in event
                        and "delta" in event["contentBlockDelta"]
                        and "text" in event["contentBlockDelta"]["delta"]
                    ):
                        summary_response += event["contentBlockDelta"]["delta"]["text"]

            # Clean the response
            clean_summary = summary_response.strip()
            return clean_summary

        except Exception as e:
            log_green(f"Failed to generate model summary: {e}", "error")
            return "Previous conversation covered multiple topics and requests."

    def _build_system_with_summary(self, clean_summary: str) -> str:
        """Embed a conversation summary into the configured system prompt."""
        summary_block = textwrap.dedent(
            f"""
            <conversation_summary>
            Previous conversation summary:
            {clean_summary}
            </conversation_summary>
            """
        ).strip()
        self.previous_summary = summary_block

        original_system_prompt = config.get("SYSTEM_PROMPT", None)
        if original_system_prompt:
            current_date = date.today().strftime("%Y-%m-%d")
            original_system_prompt = original_system_prompt.format(
                current_date=current_date
            )
            return f"{original_system_prompt}\n\n{summary_block}"
        return summary_block

    async def manage_messages(self, client: Any, model: Any, agent: Any) -> None:
        """Auto-compact: summarize if the conversation exceeds the token limit.

        Only triggers when over ``max_tokens``; otherwise no-op.
        """
        raw_messages = agent.messages.copy() if hasattr(agent, "messages") else []
        messages = messages_to_dict_list(raw_messages)

        if self.count_tokens(messages) <= self.max_tokens:
            return

        log_green(
            f"Token limit exceeded ({self.count_tokens(messages)} > "
            f"{self.max_tokens}); compacting conversation"
        )
        keep_recent = config.get("LLM", {}).get("KEEP_RECENT_MESSAGES", 6)
        await self._compact(client, model, agent, keep_recent=keep_recent)

    async def compact(
        self, client: Any, model: Any, agent: Any, focus_instructions: str = ""
    ) -> bool:
        """Manually compact the conversation now (the /compact command).

        A manual compact is an explicit request to shrink context, so it keeps
        a smaller recent window (``MANUAL_COMPACT_KEEP_RECENT``) than the
        automatic threshold path — otherwise a short conversation would have
        nothing older than the keep window and the command would no-op.

        Args:
            focus_instructions: Optional guidance on what to emphasize.

        Returns:
            True if older messages were actually summarized, else False.
        """
        keep_recent = config.get("LLM", {}).get("MANUAL_COMPACT_KEEP_RECENT", 2)
        return await self._compact(
            client,
            model,
            agent,
            keep_recent=keep_recent,
            focus_instructions=focus_instructions,
        )

    def _split_keep_recent(
        self, raw_messages: List[Any], keep_recent: int
    ) -> int:
        """Decide how many trailing messages to keep verbatim.

        Bounded by BOTH a message count (``keep_recent``) and a token budget
        (``KEEP_RECENT_TOKEN_BUDGET``), walking newest -> oldest and stopping as
        soon as either limit would be exceeded. This guarantees that a single
        oversized recent message (e.g. a pasted document that alone fills the
        context window) is NOT kept verbatim — it falls into the 'older' set and
        gets summarized instead.

        Returns:
            The split index: messages[:split] are summarized, messages[split:]
            are kept verbatim.
        """
        n = len(raw_messages)
        if keep_recent <= 0:
            return n  # keep nothing verbatim; summarize everything

        token_budget = config.get("LLM", {}).get(
            "KEEP_RECENT_TOKEN_BUDGET", max(1, int(self.max_tokens * 0.25))
        )

        kept = 0
        used = 0
        # Walk from the newest message backwards.
        for msg in reversed(raw_messages):
            if kept >= keep_recent:
                break
            msg_tokens = self.count_tokens(messages_to_dict_list([msg]))
            # Stop if adding this message would blow the token budget — but
            # always allow it to be summarized (it stays in 'older').
            if used + msg_tokens > token_budget:
                break
            used += msg_tokens
            kept += 1

        return n - kept

    async def _compact(
        self,
        client: Any,
        model: Any,
        agent: Any,
        keep_recent: int = 6,
        focus_instructions: str = "",
    ) -> bool:
        """Summarize older messages while keeping the most recent turns verbatim.

        The kept window is bounded by both a message count and a token budget
        (see ``_split_keep_recent``), so an oversized recent message is folded
        into the summary rather than kept verbatim. Everything not kept is
        summarized into the system prompt.

        Returns:
            True if there were older messages to summarize, else False.
        """
        raw_messages = agent.messages.copy() if hasattr(agent, "messages") else []
        if not raw_messages:
            return False

        split = self._split_keep_recent(raw_messages, keep_recent)
        older = raw_messages[:split]
        recent = raw_messages[split:]
        if not older:
            # Everything is within the keep window; nothing to summarize.
            return False

        client.spinner.start()
        try:
            summary = await self.generate_summary(
                messages_to_dict_list(older), model, focus_instructions
            )
            clean_summary = "".join(c for c in summary if c.isprintable())

            new_system_content = self._build_system_with_summary(clean_summary)

            # Keep recent turns verbatim; drop only the summarized older ones.
            agent.messages = list(recent)
            client.system_prompt = new_system_content
            agent.system_prompt = new_system_content
            log_green(
                f"Compacted: summarized {len(older)} older messages, "
                f"kept {len(recent)} recent."
            )
            return True
        finally:
            client.spinner.stop()
