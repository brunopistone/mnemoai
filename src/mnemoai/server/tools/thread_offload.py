"""Keep blocking tool bodies OFF the MCP server's event loop.

The server is ONE stdio subprocess with ONE event loop, and the SDK dispatches
every incoming request concurrently (``tg.start_soon(self._handle_message, …)``
in ``mcp.server.lowlevel.server``). So parallel agents — orchestrator waves,
``spawn_agent`` sub-agents — genuinely have several tool calls in flight at
once. But a tool function that does synchronous work never yields: while it
runs, that single loop cannot dispatch another call, read a new request, or even
write back a response that is already finished. One blocking tool freezes the
whole tool layer for every agent.

Declaring the tool ``async def`` does NOT help — it's the body that matters. 24
of the 31 tools were ``async def`` with no ``await`` anywhere, so each one ran
start-to-finish inline on the loop, and FastMCP runs a plain ``def`` inline too
(``func_metadata.call_fn_with_arg_validation``: ``return fn(**arguments)`` with
no thread offload). The result was a hang the user had to break by hand: with
three orchestrator workers and two ``explore`` sub-agents all calling tools, a
single ``execute_bash`` freezes the server, and every other in-flight call dies
at the client's ``LLM.MCP_CALL_TIMEOUT`` (300s) — including a ``glob_search``
whose real work is milliseconds. Worst case is a tool the client explicitly
supports waiting on: ``wait_for_task(timeout_seconds=1500)`` polls with
``time.sleep(1)``, so it used to block the entire server for 25 minutes while
``mcp_tool_wrapper._call_deadline`` raised only the CALLING agent's deadline —
every parallel agent kept the 300s default and was guaranteed to time out.

**The convention this module enforces: a tool is ``async def`` only if it really
awaits.** Anything else is a plain ``def`` and is run on a worker thread. Both
halves are load-bearing, so keep them together:

- ``def`` (blocking) -> wrapped by :func:`offload_blocking`, runs in a thread.
- ``async def`` (truly non-blocking) -> passed through, runs on the loop.

:class:`ThreadedToolServer` applies that at the single registration chokepoint
(``ToolManager.register_tools``) instead of per tool, so a tool added later is
covered by construction and the two halves can't drift.
"""

import functools
import inspect
from typing import Any, Callable

import anyio.to_thread


def offload_blocking(fn: Callable) -> Callable:
    """Wrap a SYNC ``fn`` so it runs on a worker thread; pass a coroutine fn through.

    ``functools.wraps`` is what keeps the tool's MCP contract identical, and it
    must NOT be paired with an explicit ``__signature__``: FastMCP builds the
    input schema with ``inspect.signature(fn, eval_str=True)``, which follows
    ``__wrapped__`` back to the original function and therefore evaluates its
    annotations in the ORIGINAL module's globals. Setting ``__signature__`` stops
    that unwrapping and would resolve annotations against this module instead.
    ``iscoroutinefunction`` does not follow ``__wrapped__``, so the SDK still
    sees an async callable and awaits it.
    """
    if inspect.iscoroutinefunction(fn):
        return fn

    @functools.wraps(fn)
    async def _threaded(*args: Any, **kwargs: Any) -> Any:
        # anyio (not asyncio.to_thread): the SDK's server runs on anyio, so this
        # uses its capacity limiter and works under either backend.
        return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))

    return _threaded


class ThreadedToolServer:
    """FastMCP proxy whose ``tool()`` offloads sync tool bodies to a thread.

    A proxy rather than a subclass or a monkeypatch: it needs no knowledge of
    FastMCP internals (only that ``tool()`` returns a decorator), so an SDK
    upgrade can't silently disarm it. Every other attribute is forwarded, so the
    ``register_*`` functions treat it exactly like the real server.
    """

    def __init__(self, mcp: Any) -> None:
        self._mcp = mcp

    def tool(self, *args: Any, **kwargs: Any) -> Callable:
        """Register the offloaded wrapper, but hand the ORIGINAL function back.

        Returning the original keeps the name inside each ``register_*`` function
        bound to the real implementation (what tests capture and call directly),
        while the server only ever invokes the thread-offloaded wrapper.
        """
        decorate = self._mcp.tool(*args, **kwargs)

        def _register(fn: Callable) -> Callable:
            decorate(offload_blocking(fn))
            return fn

        return _register

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mcp, name)
