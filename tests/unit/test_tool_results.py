"""Tool outcomes distinguish failures from document contents."""

import pytest
from langchain_core.messages import ToolMessage

from mnemoai.utils.tool_results import is_error_result


@pytest.mark.parametrize("result", [
    '{"error":true}', '{"success":false}', '{"exit_status":2}',
    '{"blocked":true}', "User declined to run this command.",
    ToolMessage(content="rejected", tool_call_id="t", status="error"),
])
def test_explicit_failure_is_recognized(result):
    assert is_error_result(result)


@pytest.mark.parametrize("result", [
    '{"content":"Error: example from a manual"}', '{"exit_status":0}',
    '{"success":true}', "The documentation explains errors.",
])
def test_document_contents_are_not_failures(result):
    assert not is_error_result(result)
