from mcp.server.fastmcp import FastMCP
from mnemoai.server.tools import register_tools

mcp = FastMCP("MCP Server")

# Register all tools
register_tools(mcp)

if __name__ == "__main__":
    mcp.run(transport="stdio")
