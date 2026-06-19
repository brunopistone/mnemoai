"""Unit tests for episodic memory heuristics (client/memory/episodic_memory.py).

These cover the pure success-detection and tool-extraction logic. They rely on
the built-in default markers (no config.yaml required).
"""

from personal_ai_assistant.client.memory.episodic_memory import (
    is_task_successful,
    extract_tools_from_messages,
)


class TestIsTaskSuccessful:
    def test_success_marker_in_next_message(self):
        assert is_task_successful("here you go", [], "thanks that worked") is True

    def test_correction_marker_in_next_message(self):
        assert is_task_successful("here you go", [], "that's wrong, fix it") is False

    def test_standalone_no_is_correction(self):
        assert is_task_successful("done", [], "no") is False

    def test_no_prefix_is_correction(self):
        assert is_task_successful("done", [], "no that's not right") is False

    def test_error_pattern_in_response_is_failure(self):
        assert is_task_successful("Error: file not found", []) is False

    def test_unable_to_in_response_is_failure(self):
        assert is_task_successful("I was unable to complete this", []) is False

    def test_clean_response_no_next_message_is_success(self):
        assert is_task_successful("Here is the answer you requested.", []) is True

    def test_failed_tool_result_is_failure(self):
        messages = [
            {
                "role": "user",
                "content": [{"toolResult": {"error": True, "msg": "boom"}}],
            }
        ]
        assert is_task_successful("ok", messages) is False

    def test_successful_tool_result_is_success(self):
        messages = [
            {
                "role": "user",
                "content": [{"toolResult": {"error": False, "data": "x"}}],
            }
        ]
        assert is_task_successful("ok", messages) is True

    def test_success_marker_matches_as_whole_word(self):
        # "good" as a standalone word triggers success.
        assert is_task_successful("done", [], "good") is True

    def test_robust_to_non_string_markers_from_yaml(self, monkeypatch):
        # Regression: an unquoted `no` in config.yaml parses as bool False,
        # which previously crashed is_task_successful on .lower(). Markers
        # must be coerced to str so this stays robust.
        import personal_ai_assistant.client.memory.episodic_memory as em

        bad_config = {
            "SUCCESS_MARKERS": ["thanks", True],
            "CORRECTION_MARKERS": ["wrong", False],
            "ERROR_PATTERNS": ["error:", None],
        }
        monkeypatch.setattr(
            em.config, "get", lambda key, default=None: bad_config
            if key == "EPISODIC_MEMORY"
            else default,
        )
        # Should not raise, and should still detect the real markers.
        assert is_task_successful("done", [], "thanks!") is True
        assert is_task_successful("done", [], "that's wrong") is False


class TestExtractToolsFromMessages:
    def test_strands_format_tool_use(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "name": "execute_bash",
                            "input": {"command": "ls"},
                            "toolUseId": "abc",
                        }
                    }
                ],
            }
        ]
        tools = extract_tools_from_messages(messages)
        assert len(tools) == 1
        assert tools[0]["name"] == "execute_bash"
        assert tools[0]["args"] == {"command": "ls"}
        assert tools[0]["id"] == "abc"

    def test_langchain_format_tool_calls(self):
        from langchain_core.messages import AIMessage, ToolMessage

        ai = AIMessage(
            content="",
            tool_calls=[
                {"name": "glob_search", "args": {"pattern": "*.py"}, "id": "t1"}
            ],
        )
        tool_msg = ToolMessage(content="found 3 files", tool_call_id="t1", name="glob_search")
        tools = extract_tools_from_messages([ai, tool_msg])
        assert len(tools) == 1
        assert tools[0]["name"] == "glob_search"
        assert tools[0]["result"] == "found 3 files"

    def test_no_tools_returns_empty(self):
        from langchain_core.messages import AIMessage

        assert extract_tools_from_messages([AIMessage(content="just text")]) == []
