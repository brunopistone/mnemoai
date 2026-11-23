"""Construct the same tool server for local and stdio transports."""

import os


def create_server():
    from mcp.server.fastmcp import FastMCP

    from mnemoai.server.tools import ToolManager

    server = FastMCP("MCP Server", log_level=os.getenv("LOG_LEVEL", "WARNING").upper())
    ToolManager().register_tools(server)
    return server
