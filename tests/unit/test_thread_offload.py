"""Unit tests for keeping blocking tool bodies off the MCP server's event loop.

The server is one stdio subprocess with one event loop, and the SDK dispatches
tool calls concurrently — so a tool body that blocks freezes every OTHER agent's
in-flight call until the client's MCP_CALL_TIMEOUT kills it. These tests pin both
halves of the contract: the offload actually moves work to a thread without
changing the tool's MCP surface, and no tool is declared ``async def`` while
having a blocking (await-free) body.
"""

import ast
import asyncio
import inspect
import pathlib
import threading
import time

import pytest

from mnemoai.server.tools.thread_offload import ThreadedToolServer, offload_blocking


class _CapturingMCP:
    """Minimal stand-in for FastMCP that captures registered tool functions."""

    def __init__(self):
        self.registered = {}
        self.tool_args = []

    def tool(self, *args, **kwargs):
        self.tool_args.append((args, kwargs))

        def decorator(func):
            self.registered[func.__name__] = func
            return func

        return decorator


class TestOffloadBlocking:
    def test_a_coroutine_function_is_passed_through_unchanged(self):
        async def already_async(x: int) -> str:
            return str(x)

        # Identity, not a wrapper: a genuinely-async tool must keep running on
        # the loop (that's where its awaits belong).
        assert offload_blocking(already_async) is already_async

    def test_a_sync_function_becomes_a_coroutine_function(self):
        def blocking(x: int) -> str:
            return str(x)

        wrapped = offload_blocking(blocking)
        assert wrapped is not blocking
        # The SDK decides whether to await via inspect.iscoroutinefunction, which
        # does NOT follow __wrapped__ — so the wrapper must report as async.
        assert inspect.iscoroutinefunction(wrapped)

    def test_the_body_runs_off_the_calling_thread(self):
        def blocking() -> str:
            return threading.current_thread().name

        caller = threading.current_thread().name
        ran_on = asyncio.run(offload_blocking(blocking)())
        assert ran_on != caller

    def test_arguments_and_return_value_pass_through(self):
        def blocking(a, b=2, *, c=3) -> str:
            return f"{a}-{b}-{c}"

        wrapped = offload_blocking(blocking)
        assert asyncio.run(wrapped(1, c=9)) == "1-2-9"

    def test_an_exception_propagates_out_of_the_thread(self):
        def blocking() -> str:
            raise ValueError("kaboom")

        with pytest.raises(ValueError, match="kaboom"):
            asyncio.run(offload_blocking(blocking)())

    def test_metadata_is_preserved_for_schema_generation(self):
        def blocking(pattern: str, path: str = None) -> str:
            """Docstring the model reads."""
            return ""

        wrapped = offload_blocking(blocking)
        assert wrapped.__name__ == "blocking"
        assert wrapped.__doc__ == "Docstring the model reads."
        # inspect.signature must unwrap to the ORIGINAL, so FastMCP builds the
        # input schema from the real parameters. Setting __signature__ on the
        # wrapper would break that (annotations would resolve in the wrong module).
        assert not hasattr(wrapped, "__signature__")
        assert str(inspect.signature(wrapped)) == str(inspect.signature(blocking))


class TestTheMCPContractIsUnchanged:
    """The offload must be invisible to the model: same schema, same description."""

    def test_fastmcp_builds_an_identical_tool_from_the_wrapper(self):
        pytest.importorskip("mcp.server.fastmcp")
        from mcp.server.fastmcp.tools.base import Tool

        def sample(pattern: str, path: str = None, max_results: int = 1000) -> str:
            """Fast file pattern matching.

            Args:
                pattern: Glob pattern
            """
            return ""

        plain = Tool.from_function(sample)
        wrapped = Tool.from_function(offload_blocking(sample))
        assert wrapped.name == plain.name
        assert wrapped.description == plain.description
        assert wrapped.parameters == plain.parameters
        assert wrapped.is_async and not plain.is_async

    def test_a_sync_tool_is_callable_through_a_real_fastmcp_server(self):
        pytest.importorskip("mcp.server.fastmcp")
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("test")
        server = ThreadedToolServer(mcp)

        @server.tool()
        def echo_thread(value: str) -> str:
            """Echo the value plus where it ran."""
            return f"{value}:{threading.current_thread() is threading.main_thread()}"

        tools = asyncio.run(mcp.list_tools())
        tool = next(t for t in tools if t.name == "echo_thread")
        assert set(tool.inputSchema["properties"]) == {"value"}
        assert "Echo the value" in tool.description

        out = str(asyncio.run(mcp.call_tool("echo_thread", {"value": "hi"})))
        # Ran, returned its value, and did NOT run on the loop's thread.
        assert "hi:False" in out


class TestThreadedToolServer:
    def test_it_registers_the_offloaded_wrapper(self):
        mcp = _CapturingMCP()

        @ThreadedToolServer(mcp).tool()
        def blocking() -> str:
            return threading.current_thread().name

        registered = mcp.registered["blocking"]
        assert inspect.iscoroutinefunction(registered)

    def test_it_returns_the_original_function_to_the_caller(self):
        mcp = _CapturingMCP()

        def blocking() -> str:
            return "sync result"

        returned = ThreadedToolServer(mcp).tool()(blocking)
        # The name inside each register_* function stays bound to the real
        # implementation (what the unit tests capture and call directly).
        assert returned is blocking
        assert returned() == "sync result"

    def test_an_async_tool_is_registered_as_is(self):
        mcp = _CapturingMCP()

        async def genuinely_async() -> str:
            await asyncio.sleep(0)
            return "ok"

        ThreadedToolServer(mcp).tool()(genuinely_async)
        assert mcp.registered["genuinely_async"] is genuinely_async

    def test_tool_decorator_arguments_are_forwarded(self):
        mcp = _CapturingMCP()
        ThreadedToolServer(mcp).tool(name="renamed")(lambda: "x")
        assert mcp.tool_args == [((), {"name": "renamed"})]

    def test_other_attributes_are_forwarded(self):
        mcp = _CapturingMCP()
        mcp.some_server_attr = "value"
        assert ThreadedToolServer(mcp).some_server_attr == "value"


class TestConcurrencyIsRestored:
    """The actual bug: one blocking tool must not stall every other agent's call."""

    def test_a_blocking_body_on_the_loop_starves_a_concurrent_call(self):
        # The failure mode being fixed, pinned so the reason stays legible: an
        # `async def` that never awaits runs start-to-finish on the loop, so the
        # concurrent call cannot even begin until it returns.
        order = []

        async def slow_on_the_loop():
            order.append("slow-start")
            time.sleep(0.05)
            order.append("slow-end")

        async def fast():
            order.append("fast")

        async def drive():
            await asyncio.gather(slow_on_the_loop(), fast())

        asyncio.run(drive())
        assert order == ["slow-start", "slow-end", "fast"]

    def test_an_offloaded_body_lets_a_concurrent_call_through(self):
        # Same shape, offloaded: "fast" now lands BETWEEN the slow tool's start
        # and end — the loop stayed free while the blocking body ran.
        order = []
        started = threading.Event()
        release = threading.Event()

        def slow_in_a_thread():
            order.append("slow-start")
            started.set()
            assert release.wait(10), "release was never signalled"
            order.append("slow-end")

        async def fast():
            order.append("fast")

        async def drive():
            task = asyncio.create_task(offload_blocking(slow_in_a_thread)())
            deadline = time.monotonic() + 10
            while not started.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.01)  # the loop is alive: this progresses
            assert started.is_set(), "the offloaded body never started"
            await asyncio.wait_for(fast(), timeout=5)
            release.set()
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(drive())
        assert order == ["slow-start", "fast", "slow-end"]


class TestNoToolReintroducesTheBug:
    """A guard, not a behavior test: it fails on the NEXT blocking `async def`."""

    @staticmethod
    def _tool_functions():
        """Every ``@mcp.tool()``-decorated function in server/tools, by AST.

        Parsed rather than imported so the check also covers the config-gated
        groups (web_search, web_crawler, rag) that a unit run never registers,
        and needs no config.yaml or heavy optional dependency.
        """
        import mnemoai.server.tools as tools_pkg

        root = pathlib.Path(tools_pkg.__file__).parent
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                decorated = any(
                    (isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "tool")
                    or getattr(d, "attr", "") == "tool"
                    for d in node.decorator_list
                )
                if decorated:
                    yield path.name, node

    def test_the_scan_finds_the_tools(self):
        found = {name for _, node in self._tool_functions() for name in [node.name]}
        # Sanity-check the AST scan itself, so a silent zero-match can't make the
        # guard below pass vacuously.
        assert {"execute_bash", "glob_search", "web_search", "fs_read"} <= found
        assert len(found) > 25

    def test_no_tool_is_async_with_an_await_free_body(self):
        offenders = [
            f"{filename}::{node.name}"
            for filename, node in self._tool_functions()
            if isinstance(node, ast.AsyncFunctionDef)
            and "Await(" not in ast.dump(node)
        ]
        assert not offenders, (
            "These tools are `async def` but never await, so their whole body "
            "runs inline on the MCP server's event loop and blocks every other "
            "agent's tool call. Make them plain `def` — thread_offload runs a "
            f"sync tool on a worker thread: {offenders}"
        )
