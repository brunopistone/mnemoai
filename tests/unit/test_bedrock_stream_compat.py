"""A Converse stream survives a block langchain-aws has no branch for.

The events below are the real wire shape captured from `converse_stream` for
`global.openai.gpt-5.6-luna` in us-east-1: those models put their reasoning in
`reasoningContent.redactedContent` (encrypted), and langchain-aws' parser indexes
its own converter's result unguarded — so the FIRST event of every answer raised
`IndexError`. Offline: the model's boto client is replaced with a stub.
"""

import pytest

from mnemoai.models.chat_models.bedrock_stream_compat import (
    _FilteredEventStream,
    _is_parseable,
    _StreamHardenedClient,
    harden_converse_stream,
)

# The encrypted reasoning block, as sent. Bytes, not str — botocore decodes the
# blob shape for us and the ciphertext is not text.
REDACTED = {
    "contentBlockDelta": {
        "contentBlockIndex": 0,
        "delta": {"reasoningContent": {"redactedContent": b"rsn_kLljGjpgy37aTjtAGI2"}},
    }
}

GPT_STREAM = [
    {"messageStart": {"role": "assistant"}},
    REDACTED,
    {"contentBlockStop": {"contentBlockIndex": 0}},
    {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "There are "}}},
    {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "**3**"}}},
    {"contentBlockStop": {"contentBlockIndex": 1}},
    {"messageStop": {"stopReason": "end_turn"}},
    {"metadata": {"usage": {"inputTokens": 24, "outputTokens": 57, "totalTokens": 81}}},
]


class _StubStream:
    """Stands in for botocore's EventStream, and records that it was closed."""

    def __init__(self, events):
        self._events = events
        self.closed = False

    def __iter__(self):
        return iter(self._events)

    def close(self):
        self.closed = True


class _StubClient:
    """The bedrock-runtime client, answering one canned stream."""

    def __init__(self, events):
        self.stream = _StubStream(events)
        self.calls = []
        self.meta = object()  # an unrelated attribute the proxy must pass through

    def converse_stream(self, **kwargs):
        self.calls.append(kwargs)
        return {"ResponseMetadata": {"HTTPStatusCode": 200}, "stream": self.stream}


# --- the predicate ---------------------------------------------------------


@pytest.mark.parametrize(
    "event,parseable",
    [
        ({"contentBlockDelta": {"delta": {"text": "hi"}}}, True),
        ({"contentBlockDelta": {"delta": {"reasoningContent": {"text": "why"}}}}, True),
        (REDACTED, False),
        # Not a single-block event, so not ours to judge.
        ({"messageStart": {"role": "assistant"}}, True),
        ({"metadata": {"usage": {}}}, True),
        # Malformed: langchain decides, exactly as it does today.
        ({}, True),
        ({"contentBlockDelta": {}}, True),
        ({"contentBlockDelta": {"delta": None}}, True),
        ("not an event", True),
    ],
)
def test_is_parseable(event, parseable):
    assert _is_parseable(event) is parseable


def test_content_block_start_is_judged_too():
    """The other event that gets indexed with [0], so it is guarded as well."""
    start = {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "1", "name": "x"}}}}
    assert _is_parseable(start) is True
    assert _is_parseable({"contentBlockStart": {"start": {"reasoningContent": {}}}}) is False


def test_an_unknown_block_type_is_left_to_langchain():
    """The guard's boundary. An EMPTY conversion means the library knowingly
    produced nothing (undisplayable ciphertext), which the non-streamed path
    drops too — so dropping it is consistent. A block it cannot classify at all
    RAISES, and that is a new capability to handle, not noise to swallow: it must
    surface exactly as it does on the non-streamed path."""
    assert _is_parseable({"contentBlockDelta": {"delta": {"somethingNew": {}}}}) is True


# --- the stream + client wrappers -----------------------------------------


def test_filtered_stream_drops_only_the_unparseable_event():
    stream = _StubStream(GPT_STREAM)
    kept = list(_FilteredEventStream(stream))
    assert REDACTED not in kept
    assert kept == [e for e in GPT_STREAM if e is not REDACTED]


def test_filtered_stream_closes_the_real_one():
    stream = _StubStream(GPT_STREAM)
    _FilteredEventStream(stream).close()
    assert stream.closed is True


def test_client_proxy_delegates_everything_else():
    client = _StubClient(GPT_STREAM)
    proxy = _StreamHardenedClient(client)
    assert proxy.meta is client.meta
    response = proxy.converse_stream(messages=[], system=[])
    assert client.calls == [{"messages": [], "system": []}]
    assert response["ResponseMetadata"] == {"HTTPStatusCode": 200}
    assert isinstance(response["stream"], _FilteredEventStream)


def test_hardening_is_idempotent():
    class _Model:
        client = _StubClient(GPT_STREAM)

    model = _Model()
    once = harden_converse_stream(model).client
    assert harden_converse_stream(model).client is once


def test_hardening_a_model_without_a_client_is_a_no_op():
    class _Model:
        client = None

    assert harden_converse_stream(_Model()).client is None


# --- the whole point: the same turn, through the real adapter --------------


class TestAgainstTheRealAdapter:
    """Pins the library hole AND the fix. Without both halves this file could
    pass while the turn still died: the first test proves the crash is real (so
    the guard is not decoration), the second that a real `ChatBedrockConverse`
    streams the answer once hardened.
    """

    def _model(self, events):
        from langchain_aws import ChatBedrockConverse

        model = ChatBedrockConverse(
            model="global.openai.gpt-5.6-luna",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            disable_streaming=False,
        )
        model.client = _StubClient(events)
        return model

    def test_unhardened_stream_still_raises(self):
        with pytest.raises(IndexError):
            list(self._model(GPT_STREAM).stream("hi"))

    def test_hardened_stream_yields_the_answer(self):
        model = harden_converse_stream(self._model(GPT_STREAM))
        chunks = list(model.stream("hi"))
        text = "".join(
            block.get("text", "")
            for chunk in chunks
            for block in (
                chunk.content if isinstance(chunk.content, list) else [chunk.content]
            )
            if isinstance(block, dict)
        )
        assert text == "There are **3**"

    def test_a_claude_style_stream_is_untouched(self):
        """Readable reasoning must still arrive: the guard drops nothing here."""
        events = [
            {"messageStart": {"role": "assistant"}},
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"text": "counting letters"}},
                }
            },
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "3"}}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
        plain = list(self._model(events).stream("hi"))
        hardened = list(harden_converse_stream(self._model(events)).stream("hi"))
        assert [c.content for c in hardened] == [c.content for c in plain]
        assert any(
            block.get("type") == "reasoning_content"
            for chunk in hardened
            for block in chunk.content
            if isinstance(block, dict)
        )
