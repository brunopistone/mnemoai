"""Built-in MCP tools over memory streams on the client's background loop."""

import asyncio
import threading
import uuid
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar

from mnemoai.client.mcp_tool_wrapper import MCPClientWrapper
from mnemoai.server import tool_state

_caller_cancel = ContextVar("mnemoai_local_call_cancel", default=None)


@contextmanager
def _cancellable_request():
    event = _caller_cancel.get()
    if event is None:
        event = threading.Event()
    token = _caller_cancel.set(event)
    try:
        yield event
    except BaseException:
        # Signal from the caller's thread before cancellation is reported,
        # without waiting for the server loop to process Future.cancel().
        event.set()
        raise
    finally:
        _caller_cancel.reset(token)


class LocalMCPClient(MCPClientWrapper):
    def __init__(self):
        super().__init__(None)
        self._builtin = True

    def call_tool_sync(self, name, arguments):
        with _cancellable_request():
            return super().call_tool_sync(name, arguments)

    async def call_tool(self, name, arguments):
        with _cancellable_request():
            return await super().call_tool(name, arguments)

    @asynccontextmanager
    async def _open_session(self):
        from mcp.shared.memory import create_connected_server_and_client_session

        from mnemoai.server.factory import create_server

        # A connection outlives the call that may have triggered a reconnect.
        _caller_cancel.set(None)
        # A reconnect gets fresh server state, as a replacement subprocess did.
        # MCP request tasks and AnyIO worker threads inherit this context; the
        # foreground agent's own file reads cannot grant server write permission.
        state = tool_state.ToolServerState()
        self._local_state = state
        with tool_state.use(state):
            server = create_server()
            async with create_connected_server_and_client_session(server) as session:
                try:
                    yield session
                finally:
                    for event in state.calls.values():
                        event.set()
                    pending = tuple(state.pending)
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

    async def _request_tool(self, session, name, arguments, metadata):
        state = self._local_state
        token = uuid.uuid4().hex
        event = _caller_cancel.get() or threading.Event()
        state.calls[token] = event
        metadata = {
            **metadata,
            "meta": {**metadata.get("meta", {}), tool_state.CANCEL_META_KEY: token},
        }
        task = asyncio.create_task(session.call_tool(name, arguments, **metadata))
        state.pending.add(task)

        def finished(task):
            state.pending.discard(task)
            state.calls.pop(token, None)
            if not task.cancelled():
                task.exception()  # consume errors after a caller abandoned its wait

        task.add_done_callback(finished)
        try:
            # Keep the protocol response receiver alive after cancellation. A
            # queued request then still sees its cancellation token before any
            # action, and a running request's late result is safely consumed.
            return await asyncio.shield(task)
        except BaseException:
            event.set()
            raise
