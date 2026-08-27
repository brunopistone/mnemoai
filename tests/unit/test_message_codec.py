"""Unit tests for the Strands↔LangChain codec (client/agent/message_codec.py).

The round trip is not a convenience — every resume of a session DECODES the whole
transcript and re-ENCODES it into the new session file, so any asymmetry compounds
once per resume. A tool result used to come back as ``str()`` of its own block
list, which re-escaped every quote in it on the way back out: each cycle roughly
doubled the payload, and real ``fs_read`` results reached 1.07M chars in a
resumed conversation (reported as ``[Context: 12650351 tokens]``, ~90% of it
backslashes).
"""

import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from mnemoai.client.agent import message_codec
from mnemoai.client.agent.message_codec import (
    convert_langchain_messages_to_strands as enc,
)
from mnemoai.client.agent.message_codec import (
    convert_strands_messages_to_langchain as dec,
)


def _legacy_wrap(text: str, layers: int) -> str:
    """The pre-fix decode: ``str()`` of the block list, once per round trip."""
    for _ in range(layers):
        text = str([{"text": text}])
    return text


class TestToolResultRoundTripIsIdempotent:
    def test_repeated_round_trips_do_not_grow_the_payload(self):
        original = "def hello():\n    return 'world'  # \"quoted\" \\ backslash"
        msgs = [ToolMessage(content=original, tool_call_id="t1")]
        sizes = []
        for _ in range(50):
            msgs = dec(enc(msgs))
            sizes.append(len(msgs[0].content))
        assert len(set(sizes)) == 1, sizes[:5]  # stable from the first cycle
        assert msgs[0].content == original  # and byte-identical
        assert msgs[0].tool_call_id == "t1"

    def test_payload_that_looks_like_a_block_list_survives_one_trip(self):
        # A result whose text happens to start with "[{" must not be mangled by
        # the block-list check — it isn't a single-`text`-block literal.
        original = '[{"id": 1, "name": "x"}]'
        out = dec(enc([ToolMessage(content=original, tool_call_id="t1")]))
        assert out[0].content == original

    def test_multi_block_tool_result_is_read_not_reserialized(self):
        strands = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "t1",
                            "content": [{"text": "part one "}, {"text": "part two"}],
                        }
                    }
                ],
            }
        ]
        assert dec(strands)[0].content == "part one part two"

    def test_json_block_is_serialized_as_json_not_python_repr(self):
        strands = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "t1",
                            "content": [{"json": {"ok": True, "n": 2}}],
                        }
                    }
                ],
            }
        ]
        assert json.loads(dec(strands)[0].content) == {"ok": True, "n": 2}

    def test_encoding_list_content_emits_blocks_not_a_repr(self):
        out = enc([ToolMessage(content=[{"text": "a"}, {"text": "b"}], tool_call_id="t1")])
        blocks = out[0]["content"][0]["toolResult"]["content"]
        assert blocks == [{"text": "a"}, {"text": "b"}]
        assert dec(out)[0].content == "ab"


class TestLegacyInflatedTranscriptsAreRepaired:
    """Sessions recorded before the fix still hold the nested form on disk."""

    def test_nested_layers_are_collapsed_back_to_the_payload(self):
        original = "grep found 3 matches in 'file.py'"
        strands = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "t1",
                            "content": [{"text": _legacy_wrap(original, 8)}],
                        }
                    }
                ],
            }
        ]
        inflated = len(strands[0]["content"][0]["toolResult"]["content"][0]["text"])
        assert inflated > 10 * len(original)  # the bug, reproduced
        assert dec(strands)[0].content == original

    def test_collapse_is_bounded(self, monkeypatch):
        # The depth bound is a safety net, not a tuning knob: past it the result is
        # merely still-wrapped (and smaller), never a hang. Nesting grows
        # exponentially, so the bound is exercised by lowering it rather than by
        # building 64 layers — which is why the loop needs a bound in the first place.
        monkeypatch.setattr(message_codec, "_MAX_COLLAPSE_DEPTH", 2)
        deep = _legacy_wrap("payload", 6)
        out = message_codec._collapse_wrapped_tool_result(deep)
        assert len(out) < len(deep)  # progress made
        assert out == _legacy_wrap("payload", 4)  # exactly two layers removed

    def test_only_an_exact_single_text_block_literal_is_collapsed(self):
        # Two blocks, or a block with a second key, is real content — left alone.
        for literal in (
            str([{"text": "a"}, {"text": "b"}]),
            str([{"text": "a", "type": "text"}]),
            str([{"other": "a"}]),
        ):
            strands = [
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": "t1",
                                "content": [{"text": literal}],
                            }
                        }
                    ],
                }
            ]
            assert dec(strands)[0].content == literal


class TestOtherMessageKindsStillRoundTrip:
    def test_human_ai_system_and_tool_calls_are_preserved(self):
        msgs = [
            SystemMessage(content="SYS"),
            HumanMessage(content="hello"),
            AIMessage(
                content="calling",
                tool_calls=[{"id": "t1", "name": "fs_read", "args": {"path": "a.py"}}],
            ),
            ToolMessage(content="file body", tool_call_id="t1"),
        ]
        out = dec(enc(msgs))
        assert [type(m) for m in out] == [type(m) for m in msgs]
        assert [m.content for m in out] == ["SYS", "hello", "calling", "file body"]
        assert out[2].tool_calls[0]["name"] == "fs_read"
        assert out[2].tool_calls[0]["args"] == {"path": "a.py"}

    def test_reasoning_content_survives_the_trip(self):
        msgs = [
            AIMessage(content="answer", additional_kwargs={"reasoning_content": "why"})
        ]
        out = dec(enc(msgs))
        assert out[0].additional_kwargs.get("reasoning_content") == "why"
