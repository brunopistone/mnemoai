"""Background schema discovery; publication and failure reporting stay foreground."""

import threading
from concurrent.futures import Future
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerStatus:
    name: str
    state: str
    tool_count: int = 0


class MCPStartup:
    """Discover external servers without holding the first interactive prompt."""

    def __init__(self, members):
        self.members = list(members)
        # Display names are not identities: an external entry may be "builtin".
        self.results = [Future() for _ in self.members]
        self.closed = threading.Event()
        self.threads = []

    def _discover(self, index, wrapper):
        future = self.results[index]
        try:
            if self.closed.is_set():
                raise RuntimeError("MCP startup was closed")
            wrapper.__enter__()
            tools = wrapper.list_tools_sync()
            if self.closed.is_set():
                raise RuntimeError("MCP startup was closed")
            future.set_result(tools)
        except BaseException as exc:
            future.set_exception(exc)

    def start(self):
        for index, (name, wrapper) in enumerate(self.members[1:], start=1):
            thread = threading.Thread(
                target=self._discover, args=(index, wrapper),
                name=f"mcp-start-{name}", daemon=True,
            )
            self.threads.append(thread)
            thread.start()
        _, wrapper = self.members[0]
        self._discover(0, wrapper)
        # The built-in schema is essential, not an optional failed connection.
        return self.results[0].result()

    def snapshot(self):
        statuses = []
        for (name, wrapper), future in zip(self.members, self.results):
            if self.closed.is_set():
                state, count = "closed", 0
            elif not future.done():
                state, count = "connecting", 0
            elif future.exception() is not None:
                state, count = "failed", 0
            else:
                state = "ready" if getattr(wrapper, "_connected", True) else "disconnected"
                count = len(future.result())
            statuses.append(ServerStatus(name, state, count))
        return tuple(statuses)

    def wait(self, cancel_probe=None):
        """Cancelling a turn stops its wait, not tool discovery for later turns."""
        for future in self.results:
            while not future.done():
                if self.closed.wait(0.1):
                    raise RuntimeError("MCP startup was closed")
                if cancel_probe is not None and cancel_probe():
                    raise KeyboardInterrupt
        if self.closed.is_set():
            raise RuntimeError("MCP startup was closed")

    def close(self):
        self.closed.set()
