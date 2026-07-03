"""StreamingCallbackHandler spinner control.

The spinner stops only on real ANSWER text — not on empty reasoning tokens or
tool-call argument tokens (a big fs_write streams its file_text as arg tokens,
which used to freeze the spinner for the whole write). On the Responses/Bedrock
block protocols the token is the repr of a content-block LIST, so a reasoning
block or `[]` is a truthy string but NOT answer text; the handler inspects the
chunk's blocks instead of trusting the token string.
"""

from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from mnemoai.client.client import StreamingCallbackHandler


class _FakeSpinner:
    def __init__(self):
        self.stops = 0

    def stop(self):
        self.stops += 1

    def start(self, label="Thinking"):
        pass


def _tool_arg_chunk(fragment):
    """A chunk streaming tool-call argument JSON (no visible content)."""
    msg = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": "fs_write", "args": fragment, "id": "1", "index": 0}
        ],
    )
    return ChatGenerationChunk(message=msg)


def _answer_chunk(text):
    """A chunk streaming plain answer text."""
    return ChatGenerationChunk(message=AIMessageChunk(content=text))


def test_tool_arg_tokens_do_not_stop_spinner():
    # fs_write arg tokens must not stop the spinner.
    sp = _FakeSpinner()
    h = StreamingCallbackHandler(spinner=sp)
    for frag in ('{"file_text": "<svg ', 'width=\\"1600\\" ', 'height=\\"1100\\">'):
        h.on_llm_new_token(frag, chunk=_tool_arg_chunk(frag))
    assert sp.stops == 0
    assert h.first_token_received is False


def test_answer_token_stops_spinner_once():
    sp = _FakeSpinner()
    h = StreamingCallbackHandler(spinner=sp)
    h.on_llm_new_token("Hello", chunk=_answer_chunk("Hello"))
    h.on_llm_new_token(" there", chunk=_answer_chunk(" there"))
    assert sp.stops == 1
    assert h.first_token_received is True


def test_answer_after_tool_args_still_stops_spinner():
    # Arg tokens (no stop), then answer text (stops once).
    sp = _FakeSpinner()
    h = StreamingCallbackHandler(spinner=sp)
    h.on_llm_new_token('{"file_text": "x"}', chunk=_tool_arg_chunk('{"file_text": "x"}'))
    assert sp.stops == 0
    h.on_llm_new_token("Done.", chunk=_answer_chunk("Done."))
    assert sp.stops == 1


def test_empty_token_never_stops_spinner():
    sp = _FakeSpinner()
    h = StreamingCallbackHandler(spinner=sp)
    h.on_llm_new_token("", chunk=_answer_chunk(""))
    h.on_llm_new_token("   ", chunk=_answer_chunk("   "))
    assert sp.stops == 0


def test_missing_chunk_kwarg_still_stops_on_text():
    # No chunk kwarg → fall back to stopping on any text.
    sp = _FakeSpinner()
    h = StreamingCallbackHandler(spinner=sp)
    h.on_llm_new_token("Hello")
    assert sp.stops == 1


def _blocks_chunk(blocks):
    """A chunk whose content is a block LIST (Responses/Bedrock protocols)."""
    return ChatGenerationChunk(message=AIMessageChunk(content=blocks))


def test_reasoning_block_does_not_stop_spinner():
    # The regression: a reasoning block's repr is a truthy token, but it isn't
    # answer text — the spinner must stay up (no dead pause before tool calls).
    sp = _FakeSpinner()
    h = StreamingCallbackHandler(spinner=sp)
    block = [{"id": "rs_1", "type": "reasoning"}]
    h.on_llm_new_token(str(block), chunk=_blocks_chunk(block))
    h.on_llm_new_token("[]", chunk=_blocks_chunk([]))
    assert sp.stops == 0
    assert h.first_token_received is False


def test_text_block_stops_spinner():
    sp = _FakeSpinner()
    h = StreamingCallbackHandler(spinner=sp)
    block = [{"type": "text", "text": "Hello"}]
    h.on_llm_new_token(str(block), chunk=_blocks_chunk(block))
    assert sp.stops == 1


def test_empty_text_block_does_not_stop_spinner():
    sp = _FakeSpinner()
    h = StreamingCallbackHandler(spinner=sp)
    block = [{"type": "text", "text": ""}]
    h.on_llm_new_token(str(block), chunk=_blocks_chunk(block))
    assert sp.stops == 0


def test_reasoning_blocks_then_text_block_stops_once():
    # Several reasoning blocks (no stop), then a text block (stops once).
    sp = _FakeSpinner()
    h = StreamingCallbackHandler(spinner=sp)
    h.on_llm_new_token("x", chunk=_blocks_chunk([{"id": "rs_1", "type": "reasoning"}]))
    h.on_llm_new_token("x", chunk=_blocks_chunk([]))
    assert sp.stops == 0
    h.on_llm_new_token("x", chunk=_blocks_chunk([{"type": "text", "text": "Answer"}]))
    assert sp.stops == 1
