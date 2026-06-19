"""Unit tests for AI response parsing (utils/formatting/response_parser.py)."""

from personal_ai_assistant.utils.formatting.response_parser import (
    extract_answer,
    extract_thinking,
    format_response,
)


class TestExtractAnswer:
    def test_extracts_from_answer_tags(self):
        assert extract_answer("<answer>42</answer>") == "42"

    def test_strips_whitespace_inside_answer_tags(self):
        assert extract_answer("<answer>\n  hi  \n</answer>") == "hi"

    def test_returns_text_after_closing_think_tag(self):
        assert extract_answer("<think>reasoning</think>final") == "final"

    def test_returns_text_after_closing_thinking_tag(self):
        assert extract_answer("<thinking>reasoning</thinking>  result") == "result"

    def test_no_tags_returns_original(self):
        assert extract_answer("plain text") == "plain text"

    def test_only_thinking_no_answer_returns_original(self):
        # When think tag closes but nothing follows, original is returned.
        text = "<think>reasoning</think>"
        assert extract_answer(text) == text


class TestExtractThinking:
    def test_extracts_thinking_tags(self):
        assert extract_thinking("<thinking>my reasoning</thinking>answer") == "my reasoning"

    def test_extracts_think_tags(self):
        assert extract_thinking("<think>brief</think>answer") == "brief"

    def test_no_thinking_returns_none(self):
        assert extract_thinking("just an answer") is None


class TestFormatResponse:
    def test_returns_thinking_and_answer_tuple(self):
        thinking, answer = format_response("<think>reasoning</think>the answer")
        assert thinking == "reasoning"
        assert answer == "the answer"

    def test_no_tags(self):
        thinking, answer = format_response("plain")
        assert thinking is None
        assert answer == "plain"
