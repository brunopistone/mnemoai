"""Streaming spinner callback handler."""

import threading
from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler

from mnemoai.client.ui.spinner import Spinner


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for spinner control during streaming."""

    def __init__(
        self,
        spinner: Optional[Spinner] = None,
        spinner_lock: Optional[threading.Lock] = None,
    ) -> None:
        self.spinner = spinner
        self.spinner_lock = spinner_lock or threading.Lock()
        self.first_token_received = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Stop the spinner on the first VISIBLE ANSWER token, keeping it up
        through reasoning and tool-call building (both silent stretches)."""
        chunk = kwargs.get("chunk")
        message = getattr(chunk, "message", None)
        # Tool-call argument fragments aren't answer text.
        if message is not None and getattr(message, "tool_call_chunks", None):
            return
        if not self._chunk_has_visible_text(message, token):
            return
        if not self.first_token_received and self.spinner:
            with self.spinner_lock:
                if not self.first_token_received:
                    self.spinner.stop()
                    self.first_token_received = True

    @staticmethod
    def _chunk_has_visible_text(message, token: str) -> bool:
        """True if this chunk carries visible answer text (not reasoning/empty).

        Responses/Bedrock stream content-block LISTS; a reasoning block or `[]`
        is a truthy `token` string but not answer text, so check the blocks for a
        non-empty `text`. Plain-string providers (Ollama) fall back to the token.
        """
        content = getattr(message, "content", None)
        if isinstance(content, list):
            return any(
                isinstance(b, dict)
                and (b.get("type") == "text" or "text" in b)
                and str(b.get("text", "")).strip()
                for b in content
            )
        if isinstance(content, str):
            return bool(content.strip())
        return bool(token and str(token).strip())

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        """Stop the spinner when a tool starts."""
        if self.spinner:
            with self.spinner_lock:
                self.spinner.stop()

    def on_tool_end(self, output, **kwargs) -> None:
        """Restart the spinner when a tool finishes."""
        if self.spinner:
            with self.spinner_lock:
                self.first_token_received = False
                self.spinner.start()

    def reset(self) -> None:
        """Reset the callback handler state."""
        self.first_token_received = False
