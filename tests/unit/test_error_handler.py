"""Unit tests for the tool error handler decorator (server/error_handler.py)."""

import ast
import asyncio
import builtins
import inspect
import json

from mnemoai.server import error_handler
from mnemoai.server.error_handler import create_error_response, tool_error_handler


def run(result):
    """Resolve a tool result: await a coroutine, else pass the value through.

    A tool with a blocking body is a plain ``def`` (server/tools/thread_offload.py
    offloads it to a thread at registration), so calling it directly here returns
    the string rather than a coroutine.
    """
    return asyncio.run(result) if asyncio.iscoroutine(result) else result


class TestToolErrorHandler:
    def test_exception_handlers_are_not_shadowed_by_an_earlier_parent(self):
        tree = ast.parse(inspect.getsource(error_handler._error_response))
        chain = next(node for node in ast.walk(tree) if isinstance(node, ast.Try))
        earlier = []
        for handler in chain.handlers:
            names = ast.unparse(handler.type).split(".")
            exception = vars(error_handler).get(
                names[0], getattr(builtins, names[0], None)
            )
            for name in names[1:]:
                exception = getattr(exception, name)
            assert isinstance(exception, type)
            assert not any(issubclass(exception, parent) for parent in earlier), (
                f"{exception.__name__} is shadowed by an earlier exception handler"
            )
            earlier.append(exception)

    def test_specific_errors_are_not_hidden_by_parent_exception_classes(self):
        for error, expected in (
            (json.JSONDecodeError("bad", "x", 0), "JSONDecodeError"),
            (TimeoutError("deadline"), "TimeoutError"),
        ):
            @tool_error_handler
            def fail():
                raise error

            assert json.loads(fail())["error_type"] == expected

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
