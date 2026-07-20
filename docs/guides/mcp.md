# External MCP Servers

mnemoai always runs its own built-in MCP server (file ops, bash, git, web, RAG,
vision, planning). You can add **more** MCP servers by creating
`~/.mnemoai/mcp/mcp.json` with the standard `mcpServers` schema (an
`mcp.json.example` is seeded there on first run). Their tools are merged with the
built-in ones and made available to the agent.

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": { "BRAVE_API_KEY": "your_brave_api_key" }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
      "disabled": true
    }
  }
}
```

Per-server fields: `command` (required), `args` (optional list), `env`
(optional; merged over the process environment), and `disabled` (optional;
`true` skips the server). A template ships at
`~/.mnemoai/mcp/mcp.json.example` (seeded on first run from the bundled
`src/mnemoai/utils/mcp.json.example`).

Behavior:

- **Additive** — the built-in server is always on; external servers run
  alongside it. Tools from all servers are merged into one list.
- **Resilient** — if an external server fails to start (bad command, missing
  binary, crash), it's logged in red and skipped; the app still runs with the
  built-in server and any others that connected.
- **No shadowing** — if an external tool's name collides with a built-in one,
  the external tool is exposed as `servername__tool` so core tools are never
  overridden (the server is still called with the original tool name).
- **Works with routing & orchestration** — external tools are appended to every
  non-empty query route, and when orchestration is enabled the task decomposer
  is told which external tools exist and steers subtasks that need them to the
  `full` category (which binds every tool). So external tools stay reachable
  whether routing/orchestration is on or off.
- Run **`/mcp`** in the chat to see configured servers, status, and tool counts.
