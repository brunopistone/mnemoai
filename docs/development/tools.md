# Add tools and file readers

## Adding New Tools { #adding-new-tools }

1. Create tool file in `server/tools/`:

```python
from mcp.server.fastmcp import FastMCP

def register_your_tool(mcp: FastMCP):
    @mcp.tool()
    def your_tool(param: str) -> str:
        """Tool description for the LLM."""
        # Implementation
        return result
```

Write it as a plain `def` unless the body genuinely `await`s something. The MCP
server connection has one event loop, so a blocking body declared
`async def` runs start-to-finish on that loop and freezes every other agent's
in-flight tool call; a sync tool is moved to a worker thread automatically where
the tools are registered. The same rule applies to any **helper** the tool calls —
awaiting a coroutine that never suspends is still blocking the loop. Two AST
guards in `tests/unit/test_thread_offload.py` fail on an await-free `async def`
anywhere under `server/tools/`.

Built-in tools use MCP memory streams in the application process. External
servers keep their stdio transport, and `python -m mnemoai.server.server`
remains available for standalone use. Both transports use the same server
factory, schemas, validation, safety checks, and worker-thread dispatch.
`server/tool_state.py` scopes mutable tool state to a connection; use that
context for new in-memory state rather than adding process-global registries.
Local workers check request cancellation before starting. Long-running
synchronous tools should also observe `tool_state.cancel_event()` at safe
boundaries; shell execution and document summarization do this. Cancellation
does not undo a completed write, and tool actions must never be replayed.

Keep optional provider and reader imports inside the path that uses them.
Registering a tool must not start its provider, load a browser SDK, or create
an unused tokenizer. Startup measurements must include full readiness and the
first real operation so deferred imports cannot conceal a slower first request.

2. Register in `tools_manager.py`:

```python
from .your_tool import register_your_tool
register_your_tool(mcp)
```

## Adding New File Readers { #adding-new-file-readers }

1. Create reader in `server/tools/readers/`:

```python
def read_your_format(path: str) -> str:
    """Read your custom format."""
    # Implementation
    return content
```

2. Register in `fs_read.py`:

```python
from .readers.your_reader import read_your_format
# Add to file type detection logic
```
