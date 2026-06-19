"""Unit tests for the tool error handler decorator (server/tools/error_handler.py)."""

import asyncio
import json

from mnemoai.server.error_handler import tool_error_handler, create_error_response


def run(coro):
    return asyncio.run(coro)


class TestToolErrorHandler:
    def test_passes_through_successful_result(self):
        @tool_error_handler
        async def ok_tool(x):
            return f"result: {x}"

        assert run(ok_tool("hi")) == "result: hi"

    def test_file_not_found_structured_error(self):
        @tool_error_handler
        async def bad_tool():
            raise FileNotFoundError("[Errno 2] No such file: '/nope.txt'")

        result = json.loads(run(bad_tool()))
        assert result["error"] is True
        assert result["error_type"] == "FileNotFoundError"
        assert "next_steps" in result and isinstance(result["next_steps"], list)

    def test_permission_error(self):
        @tool_error_handler
        async def bad_tool():
            raise PermissionError("denied: '/etc/shadow'")

        result = json.loads(run(bad_tool()))
        assert result["error_type"] == "PermissionError"

    def test_value_error(self):
        @tool_error_handler
        async def bad_tool():
            raise ValueError("bad value")

        result = json.loads(run(bad_tool()))
        assert result["error_type"] == "ValueError"
        assert "bad value" in result["original_error"]

    def test_unexpected_error_caught_by_catchall(self):
        @tool_error_handler
        async def bad_tool():
            raise RuntimeError("something weird")

        result = json.loads(run(bad_tool()))
        assert result["error"] is True
        assert result["error_type"] == "RuntimeError"
        assert result["function"] == "bad_tool"

    def test_preserves_function_name_via_wraps(self):
        @tool_error_handler
        async def my_named_tool():
            return "ok"

        assert my_named_tool.__name__ == "my_named_tool"


class TestCreateErrorResponse:
    def test_builds_standard_shape(self):
        result = json.loads(
            create_error_response("ConfigError", "missing key", ["check config"])
        )
        assert result["error"] is True
        assert result["error_type"] == "ConfigError"
        assert result["message"] == "missing key"
        assert result["next_steps"] == ["check config"]

    def test_extra_fields_merged(self):
        result = json.loads(
            create_error_response("X", "msg", [], detail="extra", code=7)
        )
        assert result["detail"] == "extra"
        assert result["code"] == 7
