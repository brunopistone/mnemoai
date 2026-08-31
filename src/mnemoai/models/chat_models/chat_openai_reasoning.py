"""ChatOpenAI that keeps the reasoning an OpenAI-compatible server sends.

`ChatOpenAI` targets the official OpenAI spec and, by its own documentation,
drops every non-standard response field: only `function_call`/`tool_calls` are
copied into `additional_kwargs`. But a local runner with a reasoning parser
(MLX, llama-server, vLLM, SGLang, LM Studio) splits a thinking model's
`<think>` block out of the answer and reports it in exactly such a field — so
the thinking arrives, is discarded in the adapter, and the turn renders as an
answer that appeared out of nowhere with no "Thought for Ns…" block.

The field NAME is not standardized and differs per server, and per reasoning
parser within one server: `reasoning_content` (mlx-openai-server, vLLM,
llama-server, DeepSeek), `reasoning` (mlx_lm's own server, OpenRouter),
`reasoning_details` (OpenRouter, structured), `thinking_blocks` (LiteLLM-shaped
proxies). `reasoning_text()` takes whichever is present, in whatever shape, and
the class normalizes it to ONE key — `additional_kwargs["reasoning_content"]`,
what `response_parsing.extract_content` already reads for Ollama and LiteLLM —
so nothing downstream needs to know which server produced the turn.

Safe in both directions: purely additive (a response with no reasoning field is
untouched, so real OpenAI behaves exactly as before), and never echoed back —
`_convert_message_to_dict` copies only `name`/`tool_calls`/`function_call` out
of `additional_kwargs`, so the recovered text is never re-sent to a server
whose chat template would reject it.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_openai import ChatOpenAI

# Checked in order; the FIRST non-empty one wins and they are never joined —
# a provider commonly sends the same thinking under two names (OpenRouter emits
# `reasoning` AND `reasoning_details`), so concatenating would duplicate it.
_REASONING_KEYS = (
    "reasoning_content",
    "reasoning",
    "reasoning_details",
    "thinking",
    "thinking_blocks",
    "reasoning_text",
)

# Where the text sits inside a structured reasoning block. `data` is absent on
# purpose: an encrypted/redacted block carries base64 ciphertext, not prose.
_TEXT_KEYS = (
    "text",
    "thinking",
    "reasoning_content",
    "reasoning",
    "summary_text",
    "summary",
    "content",
)

_MAX_DEPTH = 4


def _coerce(value: Any, depth: int = 0) -> str:
    """Pull display text out of a str / block list / block dict."""
    if depth > _MAX_DEPTH:
        return ""
    if isinstance(value, str):
        return value  # never stripped: a streamed fragment may be " the"
    if isinstance(value, Mapping):
        for key in _TEXT_KEYS:
            if key in value:
                text = _coerce(value[key], depth + 1)
                if text:
                    return text
        return ""
    # Blocks of ONE reasoning stream do join — unlike the top-level key choice.
    if isinstance(value, Sequence):
        return "".join(_coerce(item, depth + 1) for item in value)
    return ""


def reasoning_text(payload: Any) -> str:
    """Reasoning text carried by a delta/message/choice mapping, else ''."""
    if not isinstance(payload, Mapping):
        return ""
    for key in _REASONING_KEYS:
        text = _coerce(payload.get(key))
        if text:
            return text
    return ""


def _first_choice(chunk: Any) -> Mapping[str, Any]:
    """The first choice of a raw chunk dict, tolerating the beta stream shape."""
    if not isinstance(chunk, Mapping):
        return {}
    choices = chunk.get("choices")
    if not choices:
        nested = chunk.get("chunk")
        choices = nested.get("choices") if isinstance(nested, Mapping) else None
    if isinstance(choices, Sequence) and not isinstance(choices, str) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            return first
    return {}


def _raw_choices(response: Any) -> list[Any]:
    """The response's choices as plain dicts, whether it arrived as one or not."""
    if isinstance(response, Mapping):
        payload = response
    else:
        try:
            payload = response.model_dump()
        except Exception:
            return []
    choices = payload.get("choices") if isinstance(payload, Mapping) else None
    if isinstance(choices, Sequence) and not isinstance(choices, str):
        return list(choices)
    return []


def _attach_reasoning(message: Any, *payloads: Any) -> None:
    """Normalize any found reasoning onto `additional_kwargs`, first hit wins."""
    kwargs = getattr(message, "additional_kwargs", None)
    if kwargs is None or kwargs.get("reasoning_content"):
        return  # already set (a future adapter may extract it itself)
    for payload in payloads:
        text = reasoning_text(payload)
        if text:
            kwargs["reasoning_content"] = text
            return


class ChatOpenAIReasoning(ChatOpenAI):
    """ChatOpenAI that preserves a third-party server's reasoning field."""

    def _convert_chunk_to_generation_chunk(self, chunk: Any, *args: Any, **kwargs: Any):
        """Streamed path (sync + async share it); `chunk` is already a dict."""
        generation_chunk = super()._convert_chunk_to_generation_chunk(chunk, *args, **kwargs)
        if generation_chunk is None or generation_chunk.message is None:
            return generation_chunk
        choice = _first_choice(chunk)
        _attach_reasoning(generation_chunk.message, choice.get("delta"), choice)
        return generation_chunk

    def _create_chat_result(self, response: Any, generation_info: Any = None):
        """Non-streamed path: same recovery over the response's choices."""
        result = super()._create_chat_result(response, generation_info)
        choices = _raw_choices(response)
        if not choices:
            return result
        for generation, choice in zip(result.generations, choices):
            if not isinstance(choice, Mapping):
                continue
            _attach_reasoning(generation.message, choice.get("message"), choice)
        return result
