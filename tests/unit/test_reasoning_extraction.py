"""Unit tests for reasoning extraction across provider shapes.

The agent surfaces model reasoning (gray text) by recognizing several content
shapes. This covers the OpenAI Responses API reasoning-SUMMARY block
(``{"type":"reasoning","summary":[{"type":"summary_text","text":…}]}``) used by
Mantle GPT-5/Grok, alongside the existing Bedrock ``thinking`` block and the
``additional_kwargs["reasoning_content"]`` path — and confirms reasoning never
leaks into the visible answer.
"""

from langchain_core.messages import AIMessage

from mnemoai.client.agent.agent import LangGraphAgent


def _agent():
    return LangGraphAgent.__new__(LangGraphAgent)


class _Chunk:
    def __init__(self, content, additional_kwargs=None):
        self.content = content
        self.additional_kwargs = additional_kwargs or {}


class TestResponsesReasoningSummary:
    def test_extract_content_splits_summary_from_text(self):
        a = _agent()
        chunk = _Chunk([
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking… "}]},
            {"type": "text", "text": "the answer"},
        ])
        content, reasoning = a._extract_content(chunk)
        assert content == "the answer"
        assert reasoning == "thinking… "

    def test_extract_content_multiple_summary_parts(self):
        a = _agent()
        chunk = _Chunk([
            {"type": "reasoning", "summary": [
                {"type": "summary_text", "text": "a"},
                {"type": "summary_text", "text": "b"},
            ]},
        ])
        _, reasoning = a._extract_content(chunk)
        assert reasoning == "ab"

    def test_reasoning_block_without_summary_is_safe(self):
        a = _agent()
        # An id-only reasoning block (no summary yet) must not crash or leak.
        chunk = _Chunk([{"type": "reasoning", "id": "rs_1"}, {"type": "text", "text": "x"}])
        content, reasoning = a._extract_content(chunk)
        assert content == "x"
        assert reasoning == ""

    def test_extract_thinking_from_final_message(self):
        a = _agent()
        msg = AIMessage(content=[
            {"type": "reasoning", "summary": [
                {"type": "summary_text", "text": "step1 "},
                {"type": "summary_text", "text": "step2"},
            ]},
            {"type": "text", "text": "answer"},
        ])
        assert a._extract_thinking(msg) == "step1 step2"

    def test_reasoning_never_leaks_into_visible(self):
        a = _agent()
        content = [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "secret reasoning"}]},
            {"type": "text", "text": "public answer"},
        ]
        assert a._extract_visible(content) == "public answer"


class TestBedrockReasoningContentBlock:
    """Bedrock Converse shape: {"type":"reasoning_content",
    "reasoning_content":{"text":…,"signature":…}} (Sonnet 5, Opus 4.6+)."""

    def test_extract_content_reads_nested_text(self):
        a = _agent()
        chunk = _Chunk([
            {"type": "reasoning_content",
             "reasoning_content": {"text": "step-by-step ", "signature": "sig"}},
            {"type": "text", "text": "answer"},
        ])
        content, reasoning = a._extract_content(chunk)
        assert content == "answer"
        assert reasoning == "step-by-step "

    def test_extract_thinking_from_final_message(self):
        a = _agent()
        msg = AIMessage(content=[
            {"type": "reasoning_content", "reasoning_content": {"text": "reasoned"}},
            {"type": "text", "text": "answer"},
        ])
        assert a._extract_thinking(msg) == "reasoned"

    def test_signature_only_block_is_safe(self):
        a = _agent()
        # A signature-only delta (empty text) must not crash or leak.
        chunk = _Chunk([
            {"type": "reasoning_content", "reasoning_content": {"signature": "sig"}},
            {"type": "text", "text": "x"},
        ])
        content, reasoning = a._extract_content(chunk)
        assert content == "x"
        assert reasoning == ""


class TestOtherReasoningShapesStillWork:
    def test_bedrock_thinking_block(self):
        a = _agent()
        chunk = _Chunk([
            {"type": "thinking", "thinking": "deliberating "},
            {"type": "text", "text": "done"},
        ])
        content, reasoning = a._extract_content(chunk)
        assert content == "done"
        assert reasoning == "deliberating "

    def test_additional_kwargs_reasoning_content(self):
        a = _agent()
        chunk = _Chunk("visible", additional_kwargs={"reasoning_content": "hidden"})
        content, reasoning = a._extract_content(chunk)
        assert content == "visible"
        assert reasoning == "hidden"
