"""MCP Tool wrapper for LangChain/LangGraph integration."""

import asyncio
import atexit
import json
import threading
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field, create_model
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPTool

from utils.config import config
from utils.logger import logger

# Upper bound for any single MCP tool call, in seconds. Must comfortably exceed
# the longest tool-level timeout (e.g. execute_bash allows up to 120s) so the
# transport doesn't abort a call the tool itself considers valid.
MCP_CALL_TIMEOUT = config.get("LLM", {}).get("MCP_CALL_TIMEOUT", 300)


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
            return self.mcp_client.call_tool_sync(self.name, kwargs)
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            raise ToolException(f"Error executing tool {self.name}: {e}")

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
            return await self.mcp_client.call_tool(self.name, kwargs)
        except Exception as e:
            logger.error(f"Async tool execution error: {e}")
            raise ToolException(f"Error executing tool {self.name}: {e}")


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
        self._context_depth = 0

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session = None
        self._client_cm = None

        atexit.register(self.shutdown)

    def __enter__(self):
        """Sync context manager entry."""
        if not self._connected:
            self._start_background_loop()
            self._run_coroutine(self._connect())
        self._context_depth += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        self._context_depth -= 1

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

    def _run_coroutine(self, coro, timeout: float = MCP_CALL_TIMEOUT):
        """Run a coroutine in the background event loop and wait for result.

        Args:
            coro: Coroutine to run
            timeout: Max seconds to wait before cancelling the coroutine

        Returns:
            Result of the coroutine
        """
        if self._loop is None:
            raise RuntimeError("Background loop not started")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            # Cancel the orphaned coroutine on the background loop so it can't
            # keep mutating session state after we've given up on it.
            future.cancel()
            raise

    async def _connect(self) -> None:
        """Connect to the MCP server."""
        if self._connected:
            return

        self._client_cm = stdio_client(self.server_params)
        read, write = await self._client_cm.__aenter__()

        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        self._connected = True

    async def _disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if not self._connected:
            return
        try:
            if self._session:
                await self._session.__aexit__(None, None, None)
            if self._client_cm:
                await self._client_cm.__aexit__(None, None, None)
        finally:
            self._connected = False
            self._session = None
            self._client_cm = None

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
        return self._run_coroutine(self.call_tool(name, arguments))

    async def _reconnect(self) -> None:
        """Tear down a dead session and establish a fresh one.

        Called when a tool invocation fails with a transport-level error
        (e.g. the server subprocess crashed). Without this, a single crash
        would make every subsequent tool call fail for the rest of the session.
        """
        logger.warning("MCP session appears dead; attempting reconnect")
        self._connected = False
        try:
            await self._disconnect()
        except Exception as e:
            logger.debug(f"Error during reconnect teardown (ignored): {e}")
        self._session = None
        self._client_cm = None
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

    def get_tools(self) -> List[MCPToolWrapper]:
        """Get the cached list of tools.

        Returns:
            List of LangChain-compatible tool wrappers
        """
        return self._tools

    def shutdown(self) -> None:
        """Shutdown the background loop and disconnect."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5)
            self._loop = None
            self._thread = None
            self._connected = False
            self._session = None
            self._client_cm = None
