"""Nebius chat completions, including reasoning across tool-call turns."""

from typing import Any

from mnemoai.models.chat_models.chat_openai_reasoning import (
    ChatOpenAIReasoning,
    reasoning_text,
)


class ChatNebius(ChatOpenAIReasoning):
    """Keep the reasoning fields Nebius accepts in assistant history."""

    def _get_request_payload(self, input_: Any, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        for original, encoded in zip(messages, payload.get("messages", [])):
            if encoded.get("role") == "assistant":
                reasoning = reasoning_text(getattr(original, "additional_kwargs", {}))
                if reasoning:
                    encoded["reasoning_content"] = reasoning
        return payload
