"""Unit tests for provider-agnostic reasoning-block repair (message_sanitizer).

A reasoning block re-fed to the model on a later turn can make the provider
reject the whole request. The rules differ per provider shape, so the repair is
NORMALIZE (not blanket-drop):
  - Anthropic {type:thinking}: the API requires the inner `thinking` field; a
    signature-bearing block that lost its text is normalized (thinking:"") and
    KEPT (dropping would strip the signature and break thinking-first ordering
    on a tool-use turn). Only a block with NEITHER text NOR signature is dropped.
  - Bedrock {type:reasoning_content}: keep if it has text or a signature.
  - OpenAI {type:reasoning}: keep if it has summary text, an id, or
    encrypted_content (these carry the reasoning chain); drop only a bare stub.
  - Ollama / plain-string / additional_kwargs reasoning: untouched.
These are pure transforms — no live API.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mnemoai.client.agent.message_sanitizer import (
    sanitize_tool_pairs,
    strip_malformed_reasoning,
)

R = strip_malformed_reasoning


def _thinking_blocks(msg):
    return [b for b in msg.content if isinstance(b, dict) and b.get("type") == "thinking"]


class TestAnthropicThinking:
    def test_signature_only_block_normalized_and_stays_first(self):
        # RISK 2: a signature-only thinking block on a tool-use turn must be
        # NORMALIZED (not dropped) so it keeps leading the turn.
        ai = AIMessage(content=[
            {"type": "thinking", "signature": "s", "index": 0},
            {"type": "text", "text": "answer"},
        ])
        ai.tool_calls = [{"name": "x", "args": {}, "id": "c0", "type": "tool_call"}]
        out = R([ai])
        blocks = out[0].content
        assert blocks[0]["type"] == "thinking"  # still first
        assert blocks[0]["thinking"] == "" and blocks[0]["signature"] == "s"
        assert out[0].tool_calls[0]["id"] == "c0"  # tool_calls untouched

    def test_empty_thinking_with_signature_gets_key_kept(self):
        ai = AIMessage(content=[{"type": "thinking", "thinking": "", "signature": "s"}])
        out = R([ai])
        assert out[0].content[0] == {"type": "thinking", "thinking": "", "signature": "s"}

    def test_healthy_block_is_identity_no_copy(self):
        ai = AIMessage(content=[
            {"type": "thinking", "thinking": "real reasoning", "signature": "s"},
            {"type": "text", "text": "a"},
        ])
        out = R([ai])
        assert out[0] is ai  # no needless model_copy

    def test_no_text_no_signature_dropped(self):
        ai = AIMessage(content=[
            {"type": "thinking"},
            {"type": "text", "text": "a"},
        ])
        out = R([ai])
        assert _thinking_blocks(out[0]) == []
        assert any(b.get("type") == "text" for b in out[0].content)

    def test_idempotent(self):
        ai = AIMessage(content=[{"type": "thinking", "signature": "s"}])
        once = R([ai])
        twice = R(once)
        assert twice[0] is once[0]  # second pass is a no-op (same object)


class TestBedrockReasoningContent:
    def test_signed_empty_preserved(self):
        ai = AIMessage(content=[
            {"type": "reasoning_content",
             "reasoning_content": {"text": "", "signature": "s"}},
        ])
        out = R([ai])
        assert out[0] is ai  # kept (Bedrock needs the signature)

    def test_empty_no_signature_dropped(self):
        ai = AIMessage(content=[
            {"type": "reasoning_content", "reasoning_content": {"text": ""}},
            {"type": "text", "text": "a"},
        ])
        out = R([ai])
        assert not any(b.get("type") == "reasoning_content" for b in out[0].content)

    def test_with_text_preserved(self):
        ai = AIMessage(content=[
            {"type": "reasoning_content", "reasoning_content": {"text": "reasoned"}},
        ])
        out = R([ai])
        assert out[0] is ai


class TestOpenAIReasoning:
    def test_encrypted_content_preserved_even_without_summary(self):
        # RISK regression guard: an OpenAI reasoning item carrying id/
        # encrypted_content MUST survive even with no summary.
        ai = AIMessage(content=[
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc"},
        ])
        out = R([ai])
        assert out[0] is ai

    def test_summary_text_preserved(self):
        ai = AIMessage(content=[
            {"type": "reasoning",
             "summary": [{"type": "summary_text", "text": "did X"}]},
        ])
        out = R([ai])
        assert out[0] is ai

    def test_bare_stub_dropped(self):
        ai = AIMessage(content=[
            {"type": "reasoning"},
            {"type": "text", "text": "a"},
        ])
        out = R([ai])
        assert not any(b.get("type") == "reasoning" for b in out[0].content)


class TestUntouchedAndPurity:
    def test_plain_string_content_untouched(self):
        m = AIMessage(content="just text")
        out = R([m])
        assert out[0] is m

    def test_additional_kwargs_reasoning_untouched(self):
        # Ollama/LiteLLM put reasoning in additional_kwargs, not the content list.
        m = AIMessage(content="answer", additional_kwargs={"reasoning_content": "hmm"})
        out = R([m])
        assert out[0] is m and out[0].additional_kwargs["reasoning_content"] == "hmm"

    def test_non_ai_messages_untouched(self):
        msgs = [HumanMessage(content="hi"), ToolMessage(content="ok", tool_call_id="c0")]
        out = R(msgs)
        assert out[0] is msgs[0] and out[1] is msgs[1]

    def test_input_list_not_mutated_in_place(self):
        original_block = {"type": "thinking", "signature": "s"}
        ai = AIMessage(content=[original_block])
        msgs = [ai]
        out = R(msgs)
        # A new list is returned; the ORIGINAL block dict is not mutated.
        assert out is not msgs
        assert original_block == {"type": "thinking", "signature": "s"}  # unchanged
        assert out[0].content[0]["thinking"] == ""  # normalized on the COPY


class TestSanitizeAgreement:
    """sanitize_tool_pairs shares the same _strip_malformed_thinking helper, so
    it must normalize identically AND still repair orphan tool pairs (RISK 1)."""

    def test_sanitize_normalizes_thinking_like_strip(self):
        ai = AIMessage(content=[{"type": "thinking", "signature": "s"}])
        via_strip = R([ai])[0].content
        via_sanitize = sanitize_tool_pairs([ai])[0].content
        assert via_strip == via_sanitize
        assert via_sanitize[0] == {"type": "thinking", "signature": "s", "thinking": ""}

    def test_sanitize_still_repairs_orphan_tool_result(self):
        # A ToolMessage with no matching call is still dropped (orphan repair
        # unaffected by the thinking normalize).
        orphan = ToolMessage(content="x", tool_call_id="missing")
        out = sanitize_tool_pairs([HumanMessage(content="hi"), orphan])
        assert not any(isinstance(m, ToolMessage) for m in out)


class TestFlattenToolBlocks:
    """Auxiliary calls (router / decomposer / summarizer) bind NO tools, so a
    replayable tool_use/tool_result block in the history has no matching schema.
    Bedrock Converse then logs "Tool messages … detected without toolConfig" and
    raises a RuntimeWarning that leaks into the TUI (and a strict provider can
    400). flatten_tool_blocks renders them as text instead.
    """

    def _history(self):
        ai = AIMessage(content="")
        ai.tool_calls = [
            {"name": "fs_read", "args": {"path": "/tmp/x"}, "id": "t1",
             "type": "tool_call"}
        ]
        return [
            HumanMessage(content="read it"),
            ai,
            ToolMessage(content="file contents", tool_call_id="t1", name="fs_read"),
        ]

    def test_no_tool_blocks_survive(self):
        from mnemoai.client.agent.message_sanitizer import flatten_tool_blocks

        out = flatten_tool_blocks(self._history())
        assert not any(getattr(m, "tool_calls", None) for m in out)
        assert not any(isinstance(m, ToolMessage) for m in out)

    def test_content_is_preserved_as_text(self):
        from mnemoai.client.agent.message_sanitizer import flatten_tool_blocks

        out = flatten_tool_blocks(self._history())
        joined = "\n".join(str(m.content) for m in out)
        assert "fs_read" in joined            # the call is still described
        assert "file contents" in joined      # the result text survives
        assert "read it" in joined            # the user turn is untouched

    def test_tool_blocks_inside_block_list_content_flattened(self):
        # Bedrock/Anthropic can carry tool_use inside block-list content on a
        # message with no `tool_calls` attribute.
        from mnemoai.client.agent.message_sanitizer import flatten_tool_blocks

        msg = AIMessage(content=[
            {"type": "text", "text": "done"},
            {"type": "tool_use", "id": "t2", "name": "x", "input": {}},
        ])
        out = flatten_tool_blocks([msg])
        assert out[0].content == "done"
        assert not isinstance(out[0].content, list)

    def test_plain_messages_pass_through_unchanged(self):
        from mnemoai.client.agent.message_sanitizer import flatten_tool_blocks

        msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
        out = flatten_tool_blocks(msgs)
        assert out[0] is msgs[0] and out[1] is msgs[1]  # same objects, no copy

    def test_does_not_mutate_input(self):
        from mnemoai.client.agent.message_sanitizer import flatten_tool_blocks

        hist = self._history()
        flatten_tool_blocks(hist)
        assert hist[1].tool_calls[0]["id"] == "t1"  # original still has its call
        assert isinstance(hist[2], ToolMessage)

    def test_decomposer_sends_no_tool_blocks(self):
        # End-to-end guard on the actual regression path: _decompose_task binds no
        # tools, so whatever it hands the model must be tool-block free.
        from mnemoai.client.agent.agent import LangGraphAgent

        a = LangGraphAgent.__new__(LangGraphAgent)
        seen = {}

        class _M:
            callbacks = None

            def invoke(self, messages, config=None):
                seen["messages"] = messages
                return AIMessage(content="[]")

        a.model = _M()
        a._disable_reasoning = lambda: {}
        a._restore_reasoning = lambda saved: None
        a._decompose_task("now what?", "decompose", {"full"}, history=self._history())

        msgs = seen["messages"]
        assert not any(getattr(m, "tool_calls", None) for m in msgs)
        assert not any(isinstance(m, ToolMessage) for m in msgs)
        assert not any(
            isinstance(m.content, list)
            and any(
                isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result")
                for b in m.content
            )
            for m in msgs
        )
