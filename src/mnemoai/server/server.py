import os

from mcp.server.fastmcp import FastMCP

from mnemoai.server.tools import register_tools

# FastMCP configures its own logger (separate from utils.logger). Default it to
# WARNING so its per-request lines ("Processing request of type ...") don't leak
# into the chat UI; honor LOG_LEVEL so diagnostics can be turned back on.
mcp = FastMCP("MCP Server", log_level=os.getenv("LOG_LEVEL", "WARNING").upper())

# Register all tools
register_tools(mcp)

if __name__ == "__main__":
    mcp.run(transport="stdio")
