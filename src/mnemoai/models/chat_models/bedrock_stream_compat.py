"""Keep a Converse stream alive when Bedrock sends a block langchain-aws can't parse.

`_parse_stream_event` turns each `contentBlockDelta` / `contentBlockStart` into
one LangChain block with `_bedrock_to_lc([payload])[0]` — an **unguarded index**
into a list its own converter is free to return EMPTY. And it does: the
converter reads reasoning as `reasoningContent.reasoningText` (invoke shape) or
`reasoningContent.text` (stream shape), but has no branch for
`reasoningContent.redactedContent` — the ENCRYPTED reasoning that OpenAI's GPT
models on Bedrock emit as the FIRST content block of every answer. So the turn
dies on its first event with `IndexError: list index out of range`, which is a
deterministic failure (not transient, hence not retried): those models cannot
complete a single turn. It surfaces only when streaming, and the app forces
streaming on by design, so for `TYPE: bedrock` it is unconditional.

Dropping the block is what the library already does everywhere else — the
non-streamed path runs the same converter over the whole content list and simply
gets one block fewer — and there is nothing in it to display in any case: the
payload is a KMS-wrapped `rsn_…` blob, not prose.

The predicate ASKS the converter rather than keeping its own list of block types,
which is what lets this retire itself as upstream catches up: langchain-aws 1.7.4
added the missing branch, so from there on the block converts and nothing is
dropped, while an install on 1.7.3 or older (the floor is `>=1.4.1`) still needs
the guard to complete a turn. Hence also the version-gated tests — a test that
pins the crash must skip where the library no longer has the hole.

Applied to EVERY Converse model rather than to the families known to send it: it
is a pure "an unconvertible event must not kill the turn" guard (a no-op for a
model that never sends one), the same hole is one new block type away for any
other family, and being on the common path means it can't rot unnoticed.
"""

from typing import Any

from mnemoai.utils.logger import logger

# Both carry a single block payload that langchain-aws indexes with [0].
_BLOCK_PAYLOADS = (("contentBlockDelta", "delta"), ("contentBlockStart", "start"))


def _is_parseable(event: Any) -> bool:
    """False only when langchain-aws' own converter yields nothing for a block.

    The library is the authority on what it can convert, so ask it rather than
    re-deriving the list of block types here. Anything unexpected counts as
    parseable, leaving today's behavior (and any error) to langchain.
    """
    if not isinstance(event, dict):
        return True
    try:
        from langchain_aws.chat_models.bedrock_converse import _bedrock_to_lc

        for name, key in _BLOCK_PAYLOADS:
            body = event.get(name)
            if isinstance(body, dict) and key in body:
                return bool(_bedrock_to_lc([body[key]]))
    except Exception:
        return True
    return True


class _FilteredEventStream:
    """botocore `EventStream` stand-in that skips events langchain can't parse."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def __iter__(self):
        for event in self._stream:
            if _is_parseable(event):
                yield event
            else:
                logger.debug(
                    "Skipping a Converse stream event langchain-aws cannot convert "
                    "(keys=%s); most likely encrypted reasoning, which carries no "
                    "displayable text.",
                    sorted(event) if isinstance(event, dict) else type(event).__name__,
                )

    def close(self) -> None:
        """`_stream` closes the response stream in a finally; pass it through."""
        close = getattr(self._stream, "close", None)
        if close is not None:
            close()


class _StreamHardenedClient:
    """The bedrock-runtime client, with only `converse_stream` wrapped."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def converse_stream(self, *args: Any, **kwargs: Any) -> Any:
        response = self._client.converse_stream(*args, **kwargs)
        if isinstance(response, dict) and response.get("stream") is not None:
            return {**response, "stream": _FilteredEventStream(response["stream"])}
        return response


def harden_converse_stream(model: Any) -> Any:
    """Wrap a `ChatBedrockConverse`'s client so an unparseable event is skipped.

    Wrapping the transport rather than subclassing, because the offending index
    lives in a module-level function that `_stream` looks up directly (and
    `_astream` runs `_stream` in an executor, so one seam covers both). Returns
    the model so it can be used inline; idempotent, and never raises.
    """
    client = getattr(model, "client", None)
    if client is None or isinstance(client, _StreamHardenedClient):
        return model
    try:
        model.client = _StreamHardenedClient(client)
    except Exception as exc:  # a future pydantic config could refuse the assignment
        logger.debug("Could not harden the Converse stream client: %s", exc)
    return model
