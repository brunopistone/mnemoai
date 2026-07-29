"""MCP Tool wrapper for LangChain/LangGraph integration."""

import asyncio
import atexit
import contextlib
import json
import threading
import time
from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, Field, create_model

from mnemoai.utils.config import config
from mnemoai.utils.console import print_error
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import open_mcp_log

# Default upper bound for a single MCP tool call, in seconds. This is a FLOOR,
# not a ceiling: a tool that takes its own timeout argument raises the transport
# deadline above it (see _call_deadline), because a transport that gives up while
# the tool is still legitimately working can never let that call succeed.
MCP_CALL_TIMEOUT = config.get("LLM", {}).get("MCP_CALL_TIMEOUT", 300)

# Tool arguments that state how long the TOOL intends to run. The transport
# deadline is derived from these plus headroom, so `wait_for_task(1500)` isn't
# killed at the 300s default while the server is still dutifully waiting.
_TIMEOUT_ARGS = ("timeout_seconds", "timeout")

# Added to a tool's own timeout to get the transport deadline: the tool needs
# time to notice its deadline, build the timeout payload, and ship it back. Too
# small and we abort the very response we asked it to produce.
_TIMEOUT_HEADROOM = 30

# How long each slice of the blocking wait lasts. Short enough that Esc feels
# immediate, long enough that a multi-minute call isn't thousands of wakeups.
_CANCEL_POLL_INTERVAL = 0.2

# Hard ceiling on a DERIVED deadline. Nothing validates a tool's timeout argument
# on either side, so a hallucinated `timeout_seconds=999999999` would otherwise
# make the transport wait ~32 years. The old single 300s cap contained this by
# accident; deriving the deadline from the argument removed that safety net, so
# state the bound explicitly. One hour is far above any real tool wait and still
# guarantees the call eventually ends.
_MAX_DERIVED_TIMEOUT = 3600


def _call_deadline(arguments: Dict[str, Any], default: float = MCP_CALL_TIMEOUT) -> float:
    """Transport deadline for one call: the tool's own timeout + headroom, or the default.

    A single global cap is wrong for tools that are *asked* to wait: the client
    aborted `wait_for_task(timeout_seconds=1500)` after 300s with an empty
    `concurrent.futures.TimeoutError`, so a legitimate long wait could never
    complete and the failure carried no message explaining why.

    Never returns LESS than ``default`` — a tool declaring a short timeout still
    gets the normal transport budget, since the argument describes the tool's own
    internal deadline, not how long the round trip may take. And never more than
    ``_MAX_DERIVED_TIMEOUT``, because the argument is model-supplied and unvalidated
    at both ends.
    """
    for key in _TIMEOUT_ARGS:
        raw = arguments.get(key) if isinstance(arguments, dict) else None
        if raw is None:
            continue
        try:
            wanted = float(raw)
        except (TypeError, ValueError):
            continue  # a non-numeric timeout is the tool's problem, not ours
        if wanted > 0:
            # Clamp the DERIVED part only, then floor at `default`: the ceiling
            # guards against an absurd model-supplied argument, and must never
            # drag the budget below what the user explicitly configured.
            derived = min(wanted + _TIMEOUT_HEADROOM, _MAX_DERIVED_TIMEOUT)
            return max(default, derived)
    return default


class MCPCallTimeout(Exception):
    """A tool call exceeded the transport deadline.

    Exists because ``concurrent.futures.TimeoutError`` stringifies to the empty
    string: every log/raise site interpolates ``{e}``, so a timeout surfaced as
    a bare ``Tool execution error:`` with nothing after the colon — impossible to
    tell from a crash. This one always carries what timed out and what to change.
    """


class MCPToolWrapper(BaseTool):
    """Wrapper that converts an MCP tool to a LangChain tool."""

    name: str = ""
    description: str = ""
    mcp_tool: Any = None
    mcp_client: Any = None
    args_schema: Optional[Type[BaseModel]] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, mcp_tool: MCPTool, mcp_client: Any, **kwargs) -> None:
        """Initialize MCP tool wrapper.

        Args:
            mcp_tool: The MCP tool definition
            mcp_client: The MCP client for executing tools
            **kwargs: Additional arguments
        """
        args_schema = self._build_args_schema(mcp_tool)
        super().__init__(
            name=mcp_tool.name,
            description=mcp_tool.description or f"Tool: {mcp_tool.name}",
            mcp_tool=mcp_tool,
            mcp_client=mcp_client,
            args_schema=args_schema,
            **kwargs,
        )

    def _build_args_schema(self, mcp_tool: MCPTool) -> Type[BaseModel]:
        """Build a Pydantic model from MCP tool input schema.

        Args:
            mcp_tool: The MCP tool definition

        Returns:
            Pydantic model class for the tool arguments
        """
        input_schema = mcp_tool.inputSchema or {}
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        fields = {}
        for prop_name, prop_def in properties.items():
            python_type = type_mapping.get(prop_def.get("type", "string"), str)
            prop_desc = prop_def.get("description", "")

            if prop_name in required:
                fields[prop_name] = (python_type, Field(description=prop_desc))
            else:
                default_value = prop_def.get("default")
                fields[prop_name] = (
                    Optional[python_type],
                    Field(default=default_value, description=prop_desc),
                )

        model_name = f"{mcp_tool.name.replace('-', '_').replace(' ', '_').title()}Args"
        return create_model(model_name, **fields)

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs,
    ) -> str:
        """Execute the MCP tool synchronously.

        Args:
            run_manager: Callback manager for the tool run
            **kwargs: Tool arguments

        Returns:
            Tool execution result as string
        """
        try:
            # Call by the server-side tool name (mcp_tool.name), which is stable
            # even when `.name` is namespaced for collisions (e.g. server__tool).
            return self.mcp_client.call_tool_sync(self.mcp_tool.name, kwargs)
        except Exception as e:
            # `or repr(e)` because some exceptions (notably the stdlib
            # TimeoutError) have an EMPTY str(), which rendered this as a bare
            # "Tool execution error:" with nothing after the colon.
            detail = str(e) or repr(e)
            logger.error(f"Tool execution error: {detail}")
            raise ToolException(f"Error executing tool {self.name}: {detail}")

    async def _arun(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs,
    ) -> str:
        """Execute the MCP tool asynchronously.

        Args:
            run_manager: Callback manager for the tool run
            **kwargs: Tool arguments

        Returns:
            Tool execution result as string
        """
        try:
            return await self.mcp_client.call_tool(self.mcp_tool.name, kwargs)
        except Exception as e:
            detail = str(e) or repr(e)  # see _run: some exceptions stringify to ""
            logger.error(f"Async tool execution error: {detail}")
            raise ToolException(f"Error executing tool {self.name}: {detail}")


class MCPClientWrapper:
    """Wrapper for MCP client with background event loop."""

    def __init__(self, server_params: StdioServerParameters) -> None:
        """Initialize MCP client wrapper.

        Args:
            server_params: Parameters for the MCP server subprocess
        """
        self.server_params = server_params
        self._tools: List[MCPToolWrapper] = []
        self._connected = False

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session = None
        self._errlog = None  # file handle for the subprocess's stderr
        # The single task that owns the stdio + session contexts, and the event
        # used to ask it to exit them. Both contexts must be entered and exited
        # on the SAME task (anyio cancel-scope affinity), so no other code may
        # touch them — see _serve_session.
        self._session_task: Optional[asyncio.Task] = None
        self._close: Optional[asyncio.Event] = None
        # Callable returning True when the UI has asked to cancel the turn. Set by
        # the client once the agent exists; without it a blocking tool call can
        # only end at its deadline (see _run_coroutine).
        self._cancel_probe = None

        atexit.register(self.shutdown)

    def __enter__(self):
        """Sync context manager entry."""
        if not self._connected:
            self._start_background_loop()
            self._run_coroutine(self._connect())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit (teardown deferred to atexit shutdown)."""
        return False

    def _start_background_loop(self) -> None:
        """Start a background thread with its own event loop."""
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        """Run the event loop in the background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coroutine(self, coro, timeout: float = MCP_CALL_TIMEOUT, cancellable: bool = True):
        """Run a coroutine on the background loop and wait, cancellably.

        Waits in short slices rather than one ``future.result(timeout)`` call,
        because that single wait is **uninterruptible**: it parks in
        ``threading.Condition.wait``, a C-level acquire that only notices the
        ``KeyboardInterrupt`` the UI injects when it RETURNS. So pressing Esc
        during a tool call did nothing until the deadline expired — measured, an
        interrupt injected 0.5s into an 8s wait landed at 8.0s. Since the
        deadline is now derived from the tool's own timeout, that window could be
        ten minutes (``execute_bash(timeout=600)`` → 630s) with the UI showing
        "(cancelling…)" the whole time.

        Slicing gives the interpreter a chance to deliver that async exception
        between slices, and lets us notice the cooperative cancel event even when
        no exception is delivered at all.

        Args:
            coro: Coroutine to run
            timeout: Max seconds to wait before cancelling the coroutine
            cancellable: Whether a pending UI cancel should abort this wait. False
                for teardown (see ``shutdown``), which must still run *after* the
                user cancelled a turn — otherwise the cancel flag that ended the
                turn also aborts the disconnect and orphans the subprocess.

        Returns:
            Result of the coroutine
        """
        if self._loop is None:
            raise RuntimeError("Background loop not started")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        deadline = time.monotonic() + timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                try:
                    return future.result(timeout=min(_CANCEL_POLL_INTERVAL, remaining))
                except TimeoutError:
                    # This slice expired, not the call. Check for a cancel, then
                    # keep waiting — the loop's own deadline check ends it.
                    if cancellable and self._cancel_requested():
                        future.cancel()
                        raise KeyboardInterrupt from None
        except TimeoutError:
            # Cancel the orphaned coroutine on the background loop so it can't
            # keep mutating session state after we've given up on it.
            future.cancel()
            # Re-raise as something that has a message: the stdlib TimeoutError
            # renders as "" and made this failure indistinguishable from a crash.
            raise MCPCallTimeout(
                f"no response from the MCP server after {timeout:.0f}s"
            ) from None
        except BaseException:
            # Cancelled (injected KeyboardInterrupt landed between slices) or any
            # other abort: don't leave the coroutine running against a session
            # nobody is waiting on any more.
            future.cancel()
            raise

    def _cancel_requested(self) -> bool:
        """True if the UI asked to abort the running turn.

        Reads the agent's cooperative cancel event, which the Esc/Ctrl+C handler
        sets before injecting the KeyboardInterrupt. Consulting it here is what
        makes a cancel land *during* a blocking tool call rather than after it.
        Best-effort: no provider (embedded use, tests) simply means no cancel.
        """
        probe = self._cancel_probe
        if probe is None:
            return False
        try:
            return bool(probe())
        except Exception:  # noqa: BLE001 — a broken probe must not break the call
            return False

    async def _serve_session(self, ready: "asyncio.Future") -> None:
        """Own the stdio + session contexts for the WHOLE life of the connection.

        This is the one place they may be entered and exited, and it is why the
        method exists: ``stdio_client`` and ``ClientSession`` are built on
        ``anyio`` task groups, whose cancel scopes are **task-affine** — exiting
        one from a different task than entered it raises ``RuntimeError:
        Attempted to exit cancel scope in a different task than it was entered
        in``. The previous design entered them in ``_connect()`` and exited them
        in ``_disconnect()``, each submitted through its own
        ``run_coroutine_threadsafe`` call and therefore running as a DIFFERENT
        task, so any reconnect after a server crash dumped that traceback into
        the user's terminal and left the dead subprocess's pipes unreaped.

        Keeping both `async with` blocks inside a single coroutine makes entry
        and exit the same task by construction. Callers no longer touch the
        contexts at all: they resolve ``ready`` to learn the session is usable,
        and set ``_close`` to ask for teardown.
        """
        errlog = None
        try:
            # Route the subprocess's stderr to a log file instead of the terminal,
            # so its noise (e.g. `npm notice` from an npx server) can't corrupt the
            # pinned UI, while staying available for debugging a startup failure.
            try:
                errlog = open_mcp_log()  # size-rotated, line-buffered
            except OSError as e:
                logger.debug(f"Could not open MCP stderr log ({e}); using default")

            client_cm = (
                stdio_client(self.server_params, errlog=errlog)
                if errlog is not None
                else stdio_client(self.server_params)
            )
            async with client_cm as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._connected = True
                    if not ready.done():
                        ready.set_result(True)
                    # Park here for the connection's lifetime. Everything the
                    # caller does happens on OTHER tasks against `self._session`;
                    # this task exists purely to hold the contexts open so that
                    # the eventual `__aexit__` runs where `__aenter__` did.
                    await self._close.wait()
        except BaseException as e:  # noqa: BLE001 — must reach the waiter
            # A startup failure (missing binary, server that exits immediately)
            # has to surface on the caller's thread, not vanish into a task
            # nobody awaits — MultiMCPClient relies on it to skip a bad server.
            if not ready.done():
                ready.set_exception(e)
            elif not isinstance(e, asyncio.CancelledError):
                logger.debug(f"MCP session task ended: {type(e).__name__}: {e}")
        finally:
            self._connected = False
            self._session = None
            if errlog is not None:
                try:
                    errlog.close()
                except OSError:
                    pass
            self._errlog = None
            # Unblock anyone waiting on teardown of a task that died on its own.
            if not ready.done():
                ready.set_result(False)

    async def _connect(self) -> None:
        """Start the session task and wait until the session is usable."""
        if self._connected:
            return
        loop = asyncio.get_running_loop()
        self._close = asyncio.Event()
        ready: asyncio.Future = loop.create_future()
        # Strong reference: a bare create_task() can be garbage-collected
        # mid-flight, which would tear down the connection at random.
        self._session_task = loop.create_task(self._serve_session(ready))
        await ready  # raises whatever the task failed with

    async def _disconnect(self) -> None:
        """Ask the session task to exit its contexts, and wait for it to finish.

        Only ever SIGNALS: the `__aexit__` itself happens inside
        ``_serve_session`` (see that method for why that distinction matters).
        """
        task = self._session_task
        self._session_task = None
        if task is None:
            self._connected = False
            self._session = None
            return
        if self._close is not None:
            self._close.set()
        try:
            # Bounded: a wedged server must not hang shutdown forever. Cancelling
            # is safe here — the cancellation is delivered to the task that owns
            # the scopes, which is exactly what anyio requires.
            await asyncio.wait_for(asyncio.shield(task), timeout=10)
        except (asyncio.TimeoutError, TimeoutError):
            logger.debug("MCP session task did not exit in time; cancelling")
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        except BaseException as e:  # noqa: BLE001 — teardown must not propagate
            logger.debug(f"MCP session task teardown error (ignored): {e}")
        finally:
            self._connected = False
            self._session = None

    def list_tools_sync(self) -> List[MCPToolWrapper]:
        """Synchronously list available tools from the MCP server.

        Returns:
            List of LangChain-compatible tool wrappers
        """
        return self._run_coroutine(self._list_tools())

    async def _list_tools(self) -> List[MCPToolWrapper]:
        """List available tools from the MCP server.

        Returns:
            List of LangChain-compatible tool wrappers
        """
        if not self._session:
            raise RuntimeError("Not connected to MCP server")

        result = await self._session.list_tools()
        self._tools = [
            MCPToolWrapper(mcp_tool=tool, mcp_client=self) for tool in result.tools
        ]
        return self._tools

    def call_tool_sync(self, name: str, arguments: Dict[str, Any]) -> str:
        """Synchronously call an MCP tool.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result as string
        """
        deadline = _call_deadline(arguments)
        try:
            return self._run_coroutine(self.call_tool(name, arguments), timeout=deadline)
        except MCPCallTimeout as e:
            # Name the tool and the knob. Deliberately NOT retried: the request
            # was already delivered, so the tool may well have run — replaying it
            # would duplicate side effects (a second commit, a re-applied edit, a
            # second background build), and we publish no idempotency hints that
            # would let us tell a safe retry from a destructive one.
            raise MCPCallTimeout(
                f"'{name}' did not respond within {deadline:.0f}s. The call was "
                "not retried — it may already have run, so repeating it could "
                "duplicate its effects. Raise LLM.MCP_CALL_TIMEOUT if this tool "
                "legitimately needs longer."
            ) from e

    async def _reconnect(self) -> None:
        """Tear down a dead session and establish a fresh one.

        Called when a tool invocation fails with a transport-level error
        (e.g. the server subprocess crashed). Without this, a single crash
        would make every subsequent tool call fail for the rest of the session.
        """
        logger.warning("MCP session appears dead; attempting reconnect")
        # _disconnect only SIGNALS the session task, which exits the contexts on
        # the task that entered them — so this no longer raises anyio's
        # "exit cancel scope in a different task" and the dead subprocess's
        # pipes are actually closed instead of leaked.
        try:
            await self._disconnect()
        except Exception as e:
            logger.debug(f"Error during reconnect teardown (ignored): {e}")
        await self._connect()
        # Refresh tool handles bound to the new session.
        await self._list_tools()
        logger.info("MCP session reconnected")

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool, reconnecting once if the session has died.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result as string
        """
        if not self._session:
            raise RuntimeError("Not connected to MCP server")

        logger.debug(f"Executing MCP tool: {name} with args: {arguments}")
        try:
            result = await self._session.call_tool(name, arguments)
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as e:
            # Transport/connection failure: try one reconnect, then retry once.
            logger.warning(f"MCP tool call failed ({type(e).__name__}: {e}); retrying")
            await self._reconnect()
            result = await self._session.call_tool(name, arguments)

        return self._parse_tool_result(result)

    @staticmethod
    def _parse_tool_result(result: Any) -> str:
        """Convert an MCP tool result into a plain string.

        Args:
            result: Raw MCP call_tool result

        Returns:
            Result content as string
        """
        if hasattr(result, "content"):
            content = result.content
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                    else:
                        text_parts.append(str(block))
                return "\n".join(text_parts)
            return str(content)
        return (
            json.dumps(result, default=str) if not isinstance(result, str) else result
        )

    def shutdown(self) -> None:
        """Shutdown the background loop and disconnect.

        Disconnects first so the MCP server subprocess is terminated (its
        stdio context is exited) before the loop stops. This matters for an
        in-process restart (os.execv), which does NOT reap child processes:
        without the explicit disconnect the server subprocess would be
        orphaned.
        """
        if self._loop:
            if self._connected:
                try:
                    # 15s: _disconnect itself waits up to 10s for the session
                    # task to unwind before cancelling it, so a shorter budget
                    # here would abandon that teardown and orphan the subprocess.
                    # cancellable=False: shutdown often runs right after the user
                    # cancelled a turn, and that still-set cancel flag would
                    # otherwise abort the disconnect — leaking the subprocess.
                    self._run_coroutine(
                        self._disconnect(), timeout=15, cancellable=False
                    )
                except Exception as e:
                    logger.debug(f"MCP disconnect during shutdown failed: {e}")
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5)
            self._loop = None
            self._thread = None
            self._connected = False
            self._session = None
            self._session_task = None


class MultiMCPClient:
    """Aggregates the built-in MCP server with optional external ones.

    Presents the SAME interface the client already uses for a single server —
    the context-manager protocol (``with mcp_client:``) and
    ``list_tools_sync()`` — so callers don't change. Internally it owns one
    :class:`MCPClientWrapper` per server (built-in first, then each external
    server from ``mcp.json``), connects/disconnects them together, and merges
    their tools into one list.

    Resilience: a server that fails to connect (bad command, missing binary,
    crash on startup) is logged and skipped; the app still runs with the
    servers that did connect. Tool-name collisions are resolved by namespacing
    the EXTERNAL tool as ``servername__tool`` — the built-in tools keep their
    names so the app's core tools are never shadowed.
    """

    def __init__(self, builtin_params, external_servers=None) -> None:
        """Initialize with the built-in server params and optional externals.

        Args:
            builtin_params: StdioServerParameters for mnemoai's own server.
            external_servers: list of ExternalServer (name, params) from mcp.json.
        """
        # (display_name, wrapper). The built-in server has no namespace prefix.
        self._members = [("builtin", MCPClientWrapper(builtin_params))]
        for server in external_servers or []:
            self._members.append((server.name, MCPClientWrapper(server.params)))
        self._tools: List[MCPToolWrapper] = []
        self._cancel_probe = None

    def set_cancel_probe(self, probe) -> None:
        """Route the UI's cancel signal to every member wrapper.

        Without this an Esc during a tool call can't land until the call's
        deadline — a member wrapper has no other way to learn about it.
        """
        self._cancel_probe = probe
        for _, wrapper in self._members:
            wrapper._cancel_probe = probe

    def __enter__(self):
        """Connect every server; skip (with a warning) any that fail."""
        live = []
        for name, wrapper in self._members:
            try:
                wrapper.__enter__()
                live.append((name, wrapper))
            except Exception as e:
                if name == "builtin":
                    # The built-in server is essential — re-raise.
                    raise
                print_error(f"MCP server '{name}' failed to start; skipping. ({e})")
        self._members = live
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit every connected server's context."""
        for _, wrapper in self._members:
            try:
                wrapper.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.debug(f"MCP server exit error (ignored): {e}")

    def list_tools_sync(self) -> List[MCPToolWrapper]:
        """List tools across all connected servers, namespacing collisions.

        Built-in tool names win; an external tool whose name already exists is
        renamed ``servername__tool`` so it stays callable without shadowing a
        core tool. The server-side call still uses the original name (see
        ``MCPToolWrapper._run``).
        """
        merged: List[MCPToolWrapper] = []
        seen = set()
        for name, wrapper in self._members:
            try:
                tools = wrapper.list_tools_sync()
            except Exception as e:
                print_error(f"MCP server '{name}': could not list tools; skipping. ({e})")
                continue
            for tool in tools:
                display = tool.name
                if display in seen:
                    display = f"{name}__{tool.name}"
                    logger.info(
                        "Tool name collision: '%s' from '%s' exposed as '%s'.",
                        tool.name, name, display,
                    )
                tool.name = display
                seen.add(display)
                merged.append(tool)
        self._tools = merged
        return self._tools

    def shutdown(self) -> None:
        """Shut down every server's background loop."""
        for _, wrapper in self._members:
            try:
                wrapper.shutdown()
            except Exception as e:
                logger.debug(f"MCP server shutdown error (ignored): {e}")
