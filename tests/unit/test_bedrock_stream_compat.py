"""A Converse stream survives a block langchain-aws has no branch for.

The events below are the real wire shape captured from `converse_stream` for
`global.openai.gpt-5.6-luna` in us-east-1: those models put their reasoning in
`reasoningContent.redactedContent` (encrypted), and langchain-aws' parser indexes
its own converter's result unguarded — so the FIRST event of every answer raised
`IndexError`. Offline: the model's boto client is replaced with a stub.

**The library's coverage is a moving target, so these tests must not assume one
version of it.** langchain-aws 1.7.4 added the missing `redactedContent` branch;
on it the guard correctly drops nothing, while on ≤1.7.3 the guard is the only
reason the turn completes. So the guard's own contract is pinned against a STUB
converter (empty → drop, block → keep, raises → keep) and the real-wire-shape
cases use either a payload every version converts to nothing (an empty
`reasoningContent`) or a skip gated on what the INSTALLED library actually does.
Events are built per test, never shared at module level: `_extract_usage_metadata`
pops `usage` out of the metadata dict in place, so one shared list would leak
between tests — the first consumer empties it and the next sees `{"metadata": {}}`.
"""

import copy

import pytest

from mnemoai.models.chat_models.bedrock_stream_compat import (
    _FilteredEventStream,
    _is_parseable,
    _StreamHardenedClient,
    harden_converse_stream,
)

# The encrypted reasoning block, as sent. Bytes, not str — botocore decodes the
# blob shape for us and the ciphertext is not text.
REDACTED_BLOCK = {"reasoningContent": {"redactedContent": b"rsn_kLljGjpgy37aTjtAGI2"}}

# A block EVERY version converts to nothing, so a test of "the guard drops what
# the library can't turn into anything" doesn't depend on the installed version.
EMPTY_REASONING = {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {}}}}


def redacted_delta():
    """The first event of every GPT-on-Bedrock answer."""
    return {"contentBlockDelta": {"contentBlockIndex": 0, "delta": copy.deepcopy(REDACTED_BLOCK)}}


def gpt_stream():
    """The captured sequence, fresh each call — langchain mutates the events."""
    return [
        {"messageStart": {"role": "assistant"}},
        redacted_delta(),
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "There are "}}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "**3**"}}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 24, "outputTokens": 57, "totalTokens": 81}}},
    ]


def _library_handles(block):
    """Does the INSTALLED langchain-aws turn this block into something?

    Mirrors the guard's own question, including its "an error is not our call"
    rule, so a skip reason can never disagree with the behavior under test.
    """
    from langchain_aws.chat_models.bedrock_converse import _bedrock_to_lc

    try:
        return bool(_bedrock_to_lc([copy.deepcopy(block)]))
    except Exception:
        return True


REDACTED_IS_HANDLED = _library_handles(REDACTED_BLOCK)

needs_the_hole = pytest.mark.skipif(
    REDACTED_IS_HANDLED,
    reason="this langchain-aws converts redactedContent (1.7.4+): no hole to guard here",
)


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
        (EMPTY_REASONING, False),
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
    assert _is_parseable(copy.deepcopy(event)) is parseable


@pytest.mark.parametrize("converted,parseable", [([], False), ([{"type": "text"}], True)])
def test_the_predicate_asks_the_library(monkeypatch, converted, parseable):
    """The guard in one line: ask the converter, drop only an EMPTY answer.

    Stubbed on purpose — the real block types a given langchain-aws covers change
    between releases (1.7.4 grew a `redactedContent` branch), and this rule must
    hold on all of them. `_is_parseable` imports the converter per call, so the
    patch belongs on its module.
    """
    from langchain_aws.chat_models import bedrock_converse

    monkeypatch.setattr(bedrock_converse, "_bedrock_to_lc", lambda blocks: converted)
    assert _is_parseable(redacted_delta()) is parseable


def test_a_converter_error_is_not_the_guards_call(monkeypatch):
    """A block the library cannot classify RAISES, and must keep raising: that is
    a new capability to handle, not noise to swallow."""
    from langchain_aws.chat_models import bedrock_converse

    def _boom(blocks):
        raise ValueError("Unexpected content block type in content")

    monkeypatch.setattr(bedrock_converse, "_bedrock_to_lc", _boom)
    assert _is_parseable(redacted_delta()) is True


def test_content_block_start_is_judged_too():
    """The other event that gets indexed with [0], so it is guarded as well."""
    start = {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "1", "name": "x"}}}}
    assert _is_parseable(start) is True
    assert _is_parseable({"contentBlockStart": {"start": {"reasoningContent": {}}}}) is False


def test_an_unknown_block_type_is_left_to_langchain():
    """The guard's boundary, through the real converter. An EMPTY conversion means
    the library knowingly produced nothing (undisplayable ciphertext), which the
    non-streamed path drops too — so dropping it is consistent. A block it cannot
    classify at all raises, and must surface exactly as it does when not streaming."""
    assert _is_parseable({"contentBlockDelta": {"delta": {"somethingNew": {}}}}) is True


# --- the stream + client wrappers -----------------------------------------


def test_filtered_stream_drops_only_the_unparseable_event():
    events = [
        {"messageStart": {"role": "assistant"}},
        copy.deepcopy(EMPTY_REASONING),
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "hi"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    unparseable = events[1]
    kept = list(_FilteredEventStream(_StubStream(events)))
    assert unparseable not in kept
    assert kept == [e for e in events if e is not unparseable]


def test_filtered_stream_closes_the_real_one():
    stream = _StubStream(gpt_stream())
    _FilteredEventStream(stream).close()
    assert stream.closed is True


def test_client_proxy_delegates_everything_else():
    client = _StubClient(gpt_stream())
    proxy = _StreamHardenedClient(client)
    assert proxy.meta is client.meta
    response = proxy.converse_stream(messages=[], system=[])
    assert client.calls == [{"messages": [], "system": []}]
    assert response["ResponseMetadata"] == {"HTTPStatusCode": 200}
    assert isinstance(response["stream"], _FilteredEventStream)


def test_hardening_is_idempotent():
    class _Model:
        client = _StubClient(gpt_stream())

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
    streams the answer whatever the installed version does with the block.
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

    @needs_the_hole
    def test_unhardened_stream_still_raises(self):
        with pytest.raises(IndexError):
            list(self._model(gpt_stream()).stream("hi"))

    def test_hardened_stream_yields_the_answer(self):
        """On a library without the branch the guard is what gets us here at all;
        on one with it, nothing is dropped and the answer is the same."""
        model = harden_converse_stream(self._model(gpt_stream()))
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

    @pytest.mark.skipif(
        not REDACTED_IS_HANDLED,
        reason="this langchain-aws has no redactedContent branch (≤1.7.3)",
    )
    def test_a_block_the_library_learned_to_convert_is_kept(self):
        """The forward half of asking rather than remembering: the moment the
        library grows a branch for a block, the guard stops dropping it — no
        allowlist here to update, and nothing silently lost once it has meaning."""
        events = gpt_stream()
        redacted = events[1]
        kept = list(_FilteredEventStream(_StubStream(events)))
        assert redacted in kept
        assert len(kept) == len(events)

    def test_a_claude_style_stream_is_untouched(self):
        """Readable reasoning must still arrive: the guard drops nothing here."""

        def events():
            return [
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

        plain = list(self._model(events()).stream("hi"))
        hardened = list(harden_converse_stream(self._model(events())).stream("hi"))
        assert [c.content for c in hardened] == [c.content for c in plain]
        assert any(
            block.get("type") == "reasoning_content"
            for chunk in hardened
            for block in chunk.content
            if isinstance(block, dict)
        )
