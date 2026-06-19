"""Unit tests for vision content normalization.

Regression for the 'list' object has no attribute 'strip' crash: the OpenAI
Responses API (and Anthropic) return content as a list of blocks rather than a
plain string. VisionModelController._content_to_text must flatten any shape to
text and skip non-text blocks.
"""

from mnemoai.models.controllers.vision_model_controller import (
    VisionModelController as V,
)


class TestContentToText:
    def test_plain_string_passthrough(self):
        assert V._content_to_text("a red square") == "a red square"

    def test_responses_api_text_block_list(self):
        content = [{"type": "text", "text": "a red square", "annotations": []}]
        assert V._content_to_text(content) == "a red square"

    def test_skips_non_text_blocks(self):
        content = [
            {"type": "reasoning", "text": "let me think"},
            {"type": "text", "text": "final answer"},
        ]
        assert V._content_to_text(content) == "final answer"

    def test_bedrock_block_without_type_defaults_text(self):
        # Bedrock-style blocks may omit "type"; treat as text.
        assert V._content_to_text([{"text": "hello"}]) == "hello"

    def test_list_of_plain_strings(self):
        assert V._content_to_text(["foo", "bar"]) == "foobar"

    def test_empty_list(self):
        assert V._content_to_text([]) == ""

    def test_result_is_always_strippable_string(self):
        # The original crash was calling .strip() on a list; the result must
        # always be a str so callers can safely .strip() it.
        out = V._content_to_text([{"type": "text", "text": "  spaced  "}])
        assert isinstance(out, str)
        assert out.strip() == "spaced"
