"""Unit tests for tool-error translation (LangGraphAgent._tool_error_message).

When the model calls a tool with a required argument missing (e.g. file_edit
without new_string — a real GPT-5.5 failure when trying to DELETE text), pydantic
raises an opaque "Field required" error the model tends to retry verbatim. The
agent translates that into a plain instruction naming the missing field, so the
model fixes the call instead of looping.
"""

from mnemoai.client.agent.agent import LangGraphAgent

# The exact shape pydantic raised in the reported incident.
MISSING_NEW_STRING = Exception(
    "1 validation error for File_EditArgs\n"
    "new_string\n"
    "  Field required [type=missing, input_value={'file_path': 'docs/index.md', "
    "'old_string': '...'}, input_type=dict]\n"
    "    For further information visit https://errors.pydantic.dev/2.12/v/missing"
)


class TestToolErrorMessage:
    def test_missing_field_named_in_guidance(self):
        msg = LangGraphAgent._tool_error_message("file_edit", MISSING_NEW_STRING)
        assert "missing required argument" in msg
        assert "new_string" in msg
        # Mentions the delete-with-"" hint so the model learns the right shape.
        assert 'pass ""' in msg or 'pass ""' in msg.replace("“", '"').replace("”", '"')

    def test_multiple_missing_fields_listed(self):
        exc = Exception(
            "2 validation errors for File_EditArgs\n"
            "old_string\n  Field required [type=missing, ...]\n"
            "new_string\n  Field required [type=missing, ...]\n"
        )
        msg = LangGraphAgent._tool_error_message("file_edit", exc)
        assert "new_string" in msg and "old_string" in msg

    def test_non_validation_error_passes_through(self):
        msg = LangGraphAgent._tool_error_message("execute_bash", Exception("boom"))
        assert msg == "Error: boom"

    def test_includes_tool_name(self):
        msg = LangGraphAgent._tool_error_message("file_edit", MISSING_NEW_STRING)
        assert "file_edit" in msg
