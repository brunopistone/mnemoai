"""Unit tests for subtask parsing (client/orchestrator.py)."""

from mnemoai.client.agent.orchestrator import parse_subtasks

VALID = {"simple_qa", "code", "research", "knowledge", "full"}
FALLBACK = "original query"


class TestParseSubtasks:
    def test_parses_clean_json_array(self):
        content = (
            '[{"description": "read the file", "category": "code"}, '
            '{"description": "summarize it", "category": "simple_qa"}]'
        )
        result = parse_subtasks(content, FALLBACK, VALID)
        assert len(result) == 2
        assert result[0] == {"description": "read the file", "category": "code"}
        assert result[1]["category"] == "simple_qa"

    def test_strips_markdown_code_fences(self):
        content = '```json\n[{"description": "do x", "category": "code"}]\n```'
        result = parse_subtasks(content, FALLBACK, VALID)
        assert result == [{"description": "do x", "category": "code"}]

    def test_strips_thinking_tags_before_json(self):
        content = (
            "<think>let me decompose this</think>"
            '[{"description": "task a", "category": "research"}]'
        )
        result = parse_subtasks(content, FALLBACK, VALID)
        assert result == [{"description": "task a", "category": "research"}]

    def test_invalid_category_normalized_to_full(self):
        content = '[{"description": "do y", "category": "nonsense"}]'
        result = parse_subtasks(content, FALLBACK, VALID)
        assert result[0]["category"] == "full"

    def test_missing_category_defaults_to_full(self):
        content = '[{"description": "no category here"}]'
        result = parse_subtasks(content, FALLBACK, VALID)
        assert result[0]["category"] == "full"

    def test_malformed_json_falls_back_to_single_subtask(self):
        result = parse_subtasks("this is not json at all", FALLBACK, VALID)
        assert result == [{"description": FALLBACK, "category": "full"}]

    def test_empty_string_falls_back(self):
        result = parse_subtasks("", FALLBACK, VALID)
        assert result == [{"description": FALLBACK, "category": "full"}]

    def test_non_list_json_falls_back(self):
        result = parse_subtasks('{"description": "x"}', FALLBACK, VALID)
        assert result == [{"description": FALLBACK, "category": "full"}]

    def test_entries_without_description_are_skipped(self):
        content = '[{"category": "code"}, {"description": "keep me", "category": "code"}]'
        result = parse_subtasks(content, FALLBACK, VALID)
        assert result == [{"description": "keep me", "category": "code"}]

    def test_bedrock_list_content_blocks(self):
        content = [
            {"type": "thinking", "thinking": "decomposing"},
            {"type": "text", "text": '[{"description": "t", "category": "full"}]'},
        ]
        result = parse_subtasks(content, FALLBACK, VALID)
        assert result == [{"description": "t", "category": "full"}]
