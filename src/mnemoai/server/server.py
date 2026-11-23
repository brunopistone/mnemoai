from mnemoai.server.factory import create_server

mcp = create_server()

if __name__ == "__main__":
    mcp.run(transport="stdio")
