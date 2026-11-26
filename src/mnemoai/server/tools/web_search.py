"""Register the configured web-search provider under the stable MCP tool name."""

from mcp.server.fastmcp import FastMCP

from .web_search_controller import WebSearchController


def register_web_search_tools(
    mcp: FastMCP,
    provider: str = "brave",
    api_key: str | None = None,
    include_web_tools: bool = False,
) -> None:
    """Expose provider-specific parameters without changing existing Brave configs."""
    WebSearchController(provider, api_key).register(mcp, include_web_tools)
