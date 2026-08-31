"""Reasoning recovery from OpenAI-compatible servers (`ChatOpenAIReasoning`).

The field name is not standardized — each server, and each reasoning parser
within a server, picks its own — so the tests are a shape matrix rather than one
happy path. Everything here is offline: a model is constructed with a dummy key
and base URL and its converters are called with hand-built wire payloads.
"""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from mnemoai.client.agent import stream_policy
from mnemoai.client.agent.response_parsing import extract_content
from mnemoai.models.chat_models.chat_openai_reasoning import (
    ChatOpenAIReasoning,
    _first_choice,
    reasoning_text,
)


@pytest.fixture
def model():
    """A model that can convert payloads without ever reaching a server."""
    return ChatOpenAIReasoning(
        model="test-model",
        api_key="sk-test",
        base_url="http://127.0.0.1:1/v1",
    )


def _delta_chunk(delta):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


# --- the pure extractor: one shape per server we know of ------------------


@pytest.mark.parametrize(
    "payload,expected",
    [
        # mlx-openai-server, vLLM, llama-server, SGLang, DeepSeek
        ({"reasoning_content": "weighing options"}, "weighing options"),
        # mlx_lm's own server, OpenRouter's flat field
        ({"reasoning": "weighing options"}, "weighing options"),
        # OpenRouter structured
        (
            {"reasoning_details": [{"type": "reasoning.text", "text": "step one"}]},
            "step one",
        ),
        # Anthropic-shaped proxies
        ({"thinking": "step one"}, "step one"),
        (
            {"thinking_blocks": [{"type": "thinking", "thinking": "step one"}]},
            "step one",
        ),
        ({"reasoning_text": "step one"}, "step one"),
        # A nested summary list (Responses-style blocks arriving over chat)
        (
            {"reasoning": {"summary": [{"type": "summary_text", "text": "brief"}]}},
            "brief",
        ),
        # Several blocks of ONE stream do join
        (
            {"reasoning_details": [{"text": "a"}, {"text": "b"}, {"text": "c"}]},
            "abc",
        ),
        # Nothing to recover
        ({}, ""),
        ({"content": "just an answer"}, ""),
        ({"reasoning_content": ""}, ""),
        ({"reasoning": None}, ""),
        ({"reasoning": 17}, ""),
        # An encrypted/redacted block carries ciphertext, not prose
        (
            {"reasoning_details": [{"type": "reasoning.encrypted", "data": "b64=="}]},
            "",
        ),
        ({"thinking_blocks": [{"type": "redacted_thinking", "data": "b64=="}]}, ""),
    ],
)
def test_reasoning_text_shapes(payload, expected):
    assert reasoning_text(payload) == expected


def test_duplicate_fields_are_not_concatenated():
    """OpenRouter sends the same thinking twice; joining would double it."""
    payload = {
        "reasoning": "the whole thought",
        "reasoning_details": [{"type": "reasoning.text", "text": "the whole thought"}],
    }
    assert reasoning_text(payload) == "the whole thought"


def test_reasoning_text_tolerates_non_mappings():
    for value in (None, "text", 3, [], object()):
        assert reasoning_text(value) == ""


def test_fragment_whitespace_is_preserved():
    """Streamed fragments are concatenated downstream; stripping would fuse words."""
    assert reasoning_text({"reasoning_content": " then "}) == " then "


def test_deeply_nested_payload_does_not_recurse_forever():
    nested = {"reasoning": None}
    cursor = nested
    for _ in range(50):
        cursor["reasoning"] = {"text": {"text": None}}
        cursor = cursor["reasoning"]["text"]
    assert reasoning_text(nested) == ""  # bounded, and never raises


# --- the streamed seam ----------------------------------------------------


@pytest.mark.parametrize(
    "delta",
    [
        {"role": "assistant", "content": "", "reasoning_content": "thinking…"},
        {"role": "assistant", "content": None, "reasoning": "thinking…"},
        {"content": "", "reasoning_details": [{"text": "thinking…"}]},
    ],
)
def test_streamed_reasoning_reaches_additional_kwargs(model, delta):
    chunk = model._convert_chunk_to_generation_chunk(
        _delta_chunk(delta), AIMessageChunk, None
    )
    assert chunk.message.additional_kwargs["reasoning_content"] == "thinking…"


def test_streamed_reasoning_at_choice_level(model):
    """Some servers attach it beside the delta rather than inside it."""
    raw = {"choices": [{"index": 0, "delta": {"content": ""}, "reasoning": "hmm"}]}
    chunk = model._convert_chunk_to_generation_chunk(raw, AIMessageChunk, None)
    assert chunk.message.additional_kwargs["reasoning_content"] == "hmm"


def test_content_chunk_is_untouched(model):
    """No reasoning field means the parent's result is passed through as-is."""
    chunk = model._convert_chunk_to_generation_chunk(
        _delta_chunk({"role": "assistant", "content": "Hello"}), AIMessageChunk, None
    )
    assert chunk.message.content == "Hello"
    assert "reasoning_content" not in chunk.message.additional_kwargs


def test_tool_call_chunk_still_converts(model):
    """The override must not disturb what the parent already extracts."""
    delta = {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "fs_read", "arguments": '{"path":'},
            }
        ],
        "reasoning_content": "picking a tool",
    }
    chunk = model._convert_chunk_to_generation_chunk(
        _delta_chunk(delta), AIMessageChunk, None
    )
    assert chunk.message.additional_kwargs["reasoning_content"] == "picking a tool"
    assert chunk.message.tool_call_chunks[0]["name"] == "fs_read"


def test_beta_stream_shape_is_handled(model):
    """The parent falls back to chunk["chunk"]["choices"]; so must this."""
    raw = {"chunk": {"choices": [{"index": 0, "delta": {"reasoning": "nested"}}]}}
    chunk = model._convert_chunk_to_generation_chunk(raw, AIMessageChunk, None)
    assert chunk.message.additional_kwargs["reasoning_content"] == "nested"


def test_chunk_without_choices_does_not_raise(model):
    for raw in ({}, {"choices": []}):
        model._convert_chunk_to_generation_chunk(raw, AIMessageChunk, None)


def test_first_choice_tolerates_junk():
    """Guards the lookup itself: the parent raises on these, so it must be the
    parent that decides, never an exception thrown while hunting for reasoning."""
    for raw in (None, "nope", {}, {"choices": None}, {"choices": "nope"},
                {"choices": [None]}, {"chunk": "nope"}, {"chunk": {}}):
        assert _first_choice(raw) == {}


# --- the non-streamed seam -----------------------------------------------


def _response(message):
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_non_streamed_reasoning_reaches_additional_kwargs(model):
    result = model._create_chat_result(
        _response(
            {
                "role": "assistant",
                "content": "42",
                "reasoning_content": "counted twice",
            }
        )
    )
    message = result.generations[0].message
    assert message.content == "42"
    assert message.additional_kwargs["reasoning_content"] == "counted twice"


def test_non_streamed_without_reasoning_is_untouched(model):
    result = model._create_chat_result(_response({"role": "assistant", "content": "42"}))
    assert "reasoning_content" not in result.generations[0].message.additional_kwargs


# --- the assumption everything above rests on ----------------------------


class TestTheFieldSurvivesTheSdk:
    """The tests above hand the converters a dict, which assumes the openai SDK
    keeps a field its own schema doesn't declare. If it ever stripped extras, or
    langchain stopped passing `model_dump()` in, every test above would still
    pass while the reasoning silently vanished again — so pin it against the real
    SDK models, on the same path `_stream`/`_generate` take.
    """

    RAW_CHUNK = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": None,
                "delta": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "thinking…",
                },
            }
        ],
    }

    RAW_RESPONSE = {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "42",
                    "reasoning_content": "counted twice",
                },
            }
        ],
    }

    def test_streamed_chunk_survives_validation_and_dump(self, model):
        from openai.types.chat import ChatCompletionChunk

        dumped = ChatCompletionChunk.model_validate(self.RAW_CHUNK).model_dump()
        assert dumped["choices"][0]["delta"]["reasoning_content"] == "thinking…"

        chunk = model._convert_chunk_to_generation_chunk(dumped, AIMessageChunk, None)
        assert chunk.message.additional_kwargs["reasoning_content"] == "thinking…"

    def test_non_streamed_response_survives_as_a_pydantic_object(self, model):
        from openai.types.chat import ChatCompletion

        response = ChatCompletion.model_validate(self.RAW_RESPONSE)
        result = model._create_chat_result(response)
        message = result.generations[0].message
        assert message.content == "42"
        assert message.additional_kwargs["reasoning_content"] == "counted twice"


# --- composition with the rest of the app --------------------------------


def test_extract_content_reads_the_normalized_key():
    """The whole point: `reasoning_content` is the key the turn view consumes."""
    chunk = AIMessageChunk(
        content="answer", additional_kwargs={"reasoning_content": "thinking…"}
    )
    content, reasoning = extract_content(chunk)
    assert content == "answer"
    assert reasoning == "thinking…"


def test_streamed_fragments_accumulate_when_chunks_merge(model):
    """Each chunk carries a fragment; adding chunks must join them, not clobber."""
    merged = None
    for fragment in ("think", "ing", "…"):
        chunk = model._convert_chunk_to_generation_chunk(
            _delta_chunk({"content": "", "reasoning_content": fragment}),
            AIMessageChunk,
            None,
        )
        merged = chunk.message if merged is None else merged + chunk.message
    assert merged.additional_kwargs["reasoning_content"] == "thinking…"


def test_reasoning_chunk_counts_as_stream_payload():
    """A thinking-only chunk proves the model is producing, so the first-token
    window may narrow to the per-chunk one (1.17.1's watchdog)."""
    chunk = AIMessageChunk(
        content="", additional_kwargs={"reasoning_content": "thinking…"}
    )
    assert stream_policy.chunk_has_payload(chunk) is True


def test_recovered_reasoning_is_never_sent_back():
    """Pins the safety property: a chat template that rejects `reasoning_content`
    on an input message must never see it (the adapter drops all but three keys)."""
    from langchain_openai.chat_models.base import _convert_message_to_dict

    payload = _convert_message_to_dict(
        AIMessage(content="hi", additional_kwargs={"reasoning_content": "thinking…"})
    )
    assert "reasoning_content" not in payload
    assert "reasoning" not in payload
