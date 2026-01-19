"""Conversation manager that uses simple text-based summaries."""

from datetime import date
import json
import textwrap
import tiktoken
from typing import Any, Dict, List, Union
from utils.logger import logger
from utils.config import config

# Try to import LangChain message types for compatibility
try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

MODEL_ID = "gpt-4"  # Default model for token counting

# ANSI color codes for green text
GREEN = "\033[92m"
RESET = "\033[0m"


def log_green(message: str, level: str = "info") -> None:
    """Log message in green color.

    Args:
        message: Message to log
        level: Log level (info, error, etc.)
    """
    colored_message = f"{GREEN}{message}{RESET}"
    getattr(logger, level)(colored_message)


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
            # Convert LangChain message to dict
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            if hasattr(msg, '_message_type'):
                if msg._message_type == "human":
                    role = "user"
                elif msg._message_type == "ai":
                    role = "assistant"
                elif msg._message_type == "system":
                    role = "system"
            result.append({
                "role": role,
                "content": [{"text": str(msg.content)}]
            })
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

    async def generate_summary(self, messages: List[Dict], model: Any) -> str:
        """Use the model to generate a natural language summary.

        Args:
            messages: List of conversation messages
            model: Model instance for generating summary (LangChain or Strands)

        Returns:
            Summary text
        """
        log_green(f"Generating summary ...")

        summary_prompt = f"""
        Create a detailed summary of this conversation that preserves important information for future reference. Include:

        1. Specific topics, requests, and questions discussed
        2. Key facts, data, or findings that were discovered
        3. Important decisions, solutions, or conclusions reached
        4. Any specific tools, commands, or technical details mentioned
        5. Context that would be valuable for continuing this conversation later

        Write in a structured format that maintains the essential details while being concise. Focus on preserving actionable information and specific context rather than just high-level themes.
        """

        summary_prompt = textwrap.dedent(summary_prompt).strip()

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

                # Add conversation context
                for msg in messages:
                    content = ""
                    if isinstance(msg.get("content"), list):
                        for item in msg["content"]:
                            if isinstance(item, dict) and "text" in item:
                                content += item["text"]
                    elif isinstance(msg.get("content"), str):
                        content = msg["content"]

                    if msg.get("role") == "user":
                        lc_messages.append(HumanMessage(content=content))
                    elif msg.get("role") == "assistant":
                        lc_messages.append(AIMessage(content=content))

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

    async def manage_messages(self, client: Any, model: Any, agent: Any) -> None:
        """Manage messages, summarizing if needed.

        Args:
            client: Client instance
            model: Model instance for generating summary
            agent: Agent instance with messages
        """
        # Get messages and convert to dict format if needed
        raw_messages = agent.messages.copy() if hasattr(agent, 'messages') else []
        messages = messages_to_dict_list(raw_messages)

        current_tokens = self.count_tokens(messages)

        if current_tokens <= self.max_tokens:
            return

        print("\n")
        log_green(
            f"Token limit exceeded {current_tokens}, starting summarization process"
        )

        # Start spinner during summarization
        client.spinner.start()

        try:
            # Generate summary using the model
            summary = await self.generate_summary(messages, model)

            # Ensure summary is clean
            clean_summary = "".join(c for c in summary if c.isprintable())

            self.previous_summary = textwrap.dedent(
                f"""
                <conversation_summary>
                Previous conversation summary:
                {clean_summary}
                </conversation_summary>
                """
            ).strip()

            original_system_prompt = config.get("SYSTEM_PROMPT", None)

            if original_system_prompt:
                current_date = date.today().strftime("%Y-%m-%d")
                original_system_prompt = original_system_prompt.format(
                    current_date=current_date
                )

                new_system_content = f"""
                {original_system_prompt}

                <conversation_summary>
                Previous conversation summary:
                {clean_summary}
                </conversation_summary>
                """
            else:
                new_system_content = f"""
                <conversation_summary>
                Previous conversation summary:
                {clean_summary}
                </conversation_summary>
                """

            new_system_content = textwrap.dedent(new_system_content).strip()

            agent.messages.clear()
            client.system_prompt = new_system_content
            agent.system_prompt = new_system_content

        finally:
            # Always stop spinner
            client.spinner.stop()
