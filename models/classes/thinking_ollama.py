"""Custom Ollama model that preserves thinking content."""

import asyncio
import ollama
import os
from strands.models.ollama import OllamaModel
from strands.types.streaming import StreamEvent
import sys
from typing import AsyncGenerator, Optional, Any

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from utils.config import config
from utils.logger import logger


class ThinkingOllamaModel(OllamaModel):
    """Extended Ollama model that preserves thinking content."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize thinking Ollama model.

        Args:
            *args: Positional arguments for OllamaModel
            **kwargs: Keyword arguments for OllamaModel
        """
        self._last_thinking = None
        self.return_thinking = kwargs.get("additional_args", {}).get("think", True)
        self.open_tag = "<thinking>"
        self.close_tag = "</thinking>"
        self.thinking_open_tags = ["<thinking>", "<think>"]
        self.thinking_close_tags = ["</thinking>", "</think>"]
        logger.debug(f"Think: {self.return_thinking}")
        super().__init__(*args, **kwargs)

    async def _chat_with_retry(self, client, request):
        """Execute chat request with retry logic for connection failures.

        Args:
            client: Ollama async client
            request: Chat request parameters

        Returns:
            Async generator from client.chat

        Raises:
            Exception: After all retries exhausted
        """
        llm_config = config.get("LLM", {})
        max_retries = llm_config.get("MAX_RETRIES", 3)
        retry_delay = llm_config.get("RETRY_DELAY", 1.0)
        backoff_multiplier = llm_config.get("RETRY_BACKOFF", 2.0)

        current_delay = retry_delay
        last_exception = None

        for attempt in range(max_retries):
            try:
                return await client.chat(**request)
            except (ConnectionError, TimeoutError, Exception) as e:
                last_exception = e
                error_type = type(e).__name__

                if attempt < max_retries - 1:
                    logger.warning(
                        f"Ollama connection failed (attempt {attempt + 1}/{max_retries}, "
                        f"error: {error_type}). Retrying in {current_delay:.1f}s..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_multiplier
                else:
                    logger.error(
                        f"Ollama connection failed after {max_retries} attempts. "
                        f"Last error: {error_type}: {str(e)}"
                    )
                    raise

        # Should never reach here, but just in case
        if last_exception:
            raise last_exception

    async def stream(
        self,
        messages: Any,
        tool_specs: Any = None,
        system_prompt: Optional[str] = None,
        *,
        think: Optional[bool] = None,
        tool_choice: Any = None,
        **kwargs,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with thinking content included.

        Args:
            messages: List of conversation messages
            tool_specs: Optional tool specifications
            system_prompt: Optional system prompt
            tool_choice: Optional tool choice configuration
            think: Optional override for return_thinking (if None, uses self.return_thinking)
            **kwargs: Additional keyword arguments

        Returns:
            Async generator of stream events
        """
        # Use think parameter if provided, otherwise use self.return_thinking
        if think is not None:
            logger.debug(f"Think override: {think}")

        return_thinking = think if think is not None else self.return_thinking

        request = self.format_request(messages, tool_specs, system_prompt)
        request["stream"] = True

        if hasattr(self, "additional_args") and self.additional_args:
            request.update(self.additional_args)

        client = ollama.AsyncClient(self.host, **self.client_args)

        yield self.format_chunk({"chunk_type": "message_start"})
        yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})

        thinking_opened = False
        thinking_closed = False
        in_thinking_tag = False
        tool_requested = False
        last_chunk = None

        # Use retry wrapper for connection reliability
        async for chunk in await self._chat_with_retry(client, request):
            last_chunk = chunk

            if hasattr(chunk, "message"):
                # Handle thinking field from Ollama
                if hasattr(chunk.message, "thinking") and chunk.message.thinking:
                    thinking_content = chunk.message.thinking
                    self._last_thinking = (self._last_thinking or "") + thinking_content

                    if return_thinking:
                        if not thinking_opened:
                            yield self.format_chunk(
                                {
                                    "chunk_type": "content_delta",
                                    "data_type": "text",
                                    "data": self.open_tag,
                                }
                            )
                            thinking_opened = True
                        yield self.format_chunk(
                            {
                                "chunk_type": "content_delta",
                                "data_type": "text",
                                "data": thinking_content,
                            }
                        )

                # Handle content field
                elif hasattr(chunk.message, "content") and chunk.message.content:
                    content = chunk.message.content

                    if thinking_opened and not thinking_closed:
                        yield self.format_chunk(
                            {
                                "chunk_type": "content_delta",
                                "data_type": "text",
                                "data": self.close_tag,
                            }
                        )
                        thinking_closed = True

                    # Filter thinking tags from content when return_thinking=False
                    # This handles cases where LLM outputs thinking tags in content field
                    if not return_thinking:
                        # Check if any closing tag (</think>, </thinking>) appears in this chunk
                        closing_tag = next(
                            (tag for tag in self.thinking_close_tags if tag in content),
                            None,
                        )
                        # Check if any opening tag (<think>, <thinking>) appears in this chunk
                        opening_tag = next(
                            (tag for tag in self.thinking_open_tags if tag in content),
                            None,
                        )

                        if closing_tag and in_thinking_tag:
                            # Found closing tag while inside thinking - keep only content AFTER it
                            # Example: "reasoning</think>answer" -> "answer"
                            content = content.split(closing_tag, 1)[1]
                            in_thinking_tag = False
                        elif opening_tag:
                            # Found opening tag - keep only content BEFORE it
                            # Example: "text<think>reasoning" -> "text"
                            before_tag = content.split(opening_tag)[0]
                            after_tag = content.split(opening_tag, 1)[1]

                            # Check if closing tag is in the same chunk
                            if closing_tag and closing_tag in after_tag:
                                # Both open and close in same chunk - strip thinking entirely
                                after_close = after_tag.split(closing_tag, 1)[1]
                                content = before_tag + after_close
                            else:
                                # Only opening tag - keep before, start skipping
                                content = before_tag
                                in_thinking_tag = True
                        elif in_thinking_tag:
                            # We're inside thinking tags - skip this chunk
                            content = ""

                    if content:
                        yield self.format_chunk(
                            {
                                "chunk_type": "content_delta",
                                "data_type": "text",
                                "data": content,
                            }
                        )

                # Handle tool calls
                if hasattr(chunk.message, "tool_calls") and chunk.message.tool_calls:
                    for tool_call in chunk.message.tool_calls:
                        yield self.format_chunk(
                            {
                                "chunk_type": "content_start",
                                "data_type": "tool",
                                "data": tool_call,
                            }
                        )
                        yield self.format_chunk(
                            {
                                "chunk_type": "content_delta",
                                "data_type": "tool",
                                "data": tool_call,
                            }
                        )
                        yield self.format_chunk(
                            {
                                "chunk_type": "content_stop",
                                "data_type": "tool",
                                "data": tool_call,
                            }
                        )
                        tool_requested = True

        yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})
        yield self.format_chunk(
            {
                "chunk_type": "message_stop",
                "data": "tool_use" if tool_requested else "end_turn",
            }
        )
        if last_chunk and hasattr(last_chunk, "done") and last_chunk.done:
            yield self.format_chunk({"chunk_type": "metadata", "data": last_chunk})

    def get_last_thinking(self) -> Optional[str]:
        """Get the thinking content from the last response.

        Returns:
            Last thinking content or None
        """
        return self._last_thinking
