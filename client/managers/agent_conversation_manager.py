"""Conversation manager that uses simple text-based summaries."""

from datetime import date
import json
import textwrap
import tiktoken
from typing import Any, Dict, List
from utils.logger import logger
from utils.config import config

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
        """Count tokens in messages by converting to JSON string.

        Args:
            messages: List of message dictionaries

        Returns:
            Token count
        """
        return len(self.encoder.encode(json.dumps(messages, default=str)))

    async def generate_summary(self, messages: List[Dict], model: Any) -> str:
        """Use the model to generate a natural language summary.

        Args:
            messages: List of conversation messages
            model: Model instance for generating summary

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

        # Create clean messages for summarization
        messages.append({"role": "user", "content": [{"text": summary_prompt}]})

        try:
            summary_response = ""

            # Use the model to generate summary
            async for event in model.stream(
                messages, system_prompt=self.previous_summary, think=False
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
        messages = agent.messages.copy()

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
