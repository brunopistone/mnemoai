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

    def _process_buffered_content(self, buffer: str, in_tag: bool) -> tuple:
        """Process buffered content, handling thinking tags that may span chunks.

        Returns: (content_to_emit, remaining_buffer, new_in_tag_state)
        """
        import re

        # Pattern to match complete thinking tags (case-insensitive)
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
                    # Keep potential partial tag (anything starting with <)
                    last_lt = remaining.rfind("<")
                    if (
                        last_lt >= 0 and last_lt > len(remaining) - 12
                    ):  # Max tag length is </thinking> = 11
                        # Potential partial tag at end - keep in buffer
                        remaining = remaining[last_lt:]
                    else:
                        # No partial tag - discard all (it's thinking content)
                        remaining = ""
                    break
            else:
                # Looking for opening tag
                match = open_pattern.search(remaining)
                if match:
                    # Found opening tag - emit content before, then process rest
                    result += remaining[: match.start()]
                    remaining = remaining[match.end() :]
                    current_in_tag = True
                else:
                    # No complete opening tag - check for partial at end
                    last_lt = remaining.rfind("<")
                    if (
                        last_lt >= 0 and last_lt > len(remaining) - 11
                    ):  # Max tag length is <thinking> = 10
                        # Potential partial tag at end - emit before, keep partial
                        result += remaining[:last_lt]
                        remaining = remaining[last_lt:]
                    else:
                        # No partial tag - emit all
                        result += remaining
                        remaining = ""
                    break

        return result, remaining, current_in_tag

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
        # Buffer for handling tags split across chunks
        content_buffer = ""

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
                    if not return_thinking:
                        # Add to buffer and process
                        content_buffer += content
                        to_emit, content_buffer, in_thinking_tag = (
                            self._process_buffered_content(
                                content_buffer, in_thinking_tag
                            )
                        )
                        content = to_emit

                    if content:
                        yield self.format_chunk(
                            {
                                "chunk_type": "content_delta",
                                "data_type": "text",
                                "data": content,
                            }
                        )

                # Handle tool calls (inside the streaming loop)
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

        # Emit any remaining buffer content at the end
        if not return_thinking and content_buffer:
            # Final processing - emit remaining non-thinking content
            if not in_thinking_tag:
                yield self.format_chunk(
                    {
                        "chunk_type": "content_delta",
                        "data_type": "text",
                        "data": content_buffer,
                    }
                )

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
