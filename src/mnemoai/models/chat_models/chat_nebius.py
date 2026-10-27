"""Nebius chat completions, including reasoning across tool-call turns."""

from typing import Any

from mnemoai.models.chat_models.chat_openai_reasoning import (
    ChatOpenAIReasoning,
    reasoning_text,
)


def _usage_delta(current, previous):
    """Translate cumulative counters into increments for LangChain aggregation."""
    delta = {}
    for key, value in current.items():
        if isinstance(value, dict):
            delta[key] = _usage_delta(value, previous.setdefault(key, {}))
        else:
            delta[key] = value - previous.get(key, 0)
            previous[key] = value
    return delta


class ChatNebius(ChatOpenAIReasoning):
    """Keep the reasoning fields Nebius accepts in assistant history."""

    def _stream(self, *args, **kwargs):
        previous = {}
        for chunk in super()._stream(*args, **kwargs):
            if chunk.message.usage_metadata is not None:
                chunk.message.usage_metadata = _usage_delta(chunk.message.usage_metadata, previous)
            yield chunk

    async def _astream(self, *args, **kwargs):
        previous = {}
        async for chunk in super()._astream(*args, **kwargs):
            if chunk.message.usage_metadata is not None:
                chunk.message.usage_metadata = _usage_delta(chunk.message.usage_metadata, previous)
            yield chunk

    def _get_request_payload(self, input_: Any, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        for original, encoded in zip(messages, payload.get("messages", [])):
            if encoded.get("role") == "assistant":
                reasoning = reasoning_text(getattr(original, "additional_kwargs", {}))
                if reasoning:
                    encoded["reasoning_content"] = reasoning
        return payload
