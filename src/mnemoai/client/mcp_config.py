"""Loader for external MCP servers declared in ``~/.mnemoai/mcp.json``.

mnemoai always runs its own built-in MCP server (the one under
``server/server.py``). On top of that, users can declare *additional* stdio MCP
servers in an ``mcp.json`` file using the same ``mcpServers`` schema as Claude
Code / Claude Desktop / kiro:

    {
      "mcpServers": {
        "brave-search": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-brave-search"],
          "env": {"BRAVE_API_KEY": "..."},
          "disabled": false
        }
      }
    }

This module reads that file and turns each enabled entry into a named
``StdioServerParameters``. It is intentionally tolerant: a missing file yields
no servers, and a malformed file or a bad individual entry is reported (red)
and skipped rather than crashing startup — one broken server must not block the
app.
"""

import json
import os
from pathlib import Path
from typing import List, NamedTuple, Optional

from mcp import StdioServerParameters

from mnemoai.utils.console import print_error
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import legacy_mcp_config_path, mcp_config_path


class ExternalServer(NamedTuple):
    """A named external MCP server to launch as a stdio subprocess."""

    name: str
    params: StdioServerParameters


def _parse_entry(name: str, entry: dict) -> Optional[ExternalServer]:
    """Turn one ``mcpServers`` entry into an ExternalServer, or None if invalid.

    ``command`` is required; ``args`` (list) and ``env`` (dict) are optional.
    The entry's ``env`` is merged over the current process environment so the
    child inherits PATH etc. and overrides only the keys it specifies.
    """
    if not isinstance(entry, dict):
        print_error(f"MCP server '{name}': entry must be an object; skipping.")
        return None

    command = entry.get("command")
    if not command or not isinstance(command, str):
        print_error(f"MCP server '{name}': missing/invalid 'command'; skipping.")
        return None

    args = entry.get("args", [])
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        print_error(f"MCP server '{name}': 'args' must be a list of strings; skipping.")
        return None

    env_overrides = entry.get("env", {})
    if not isinstance(env_overrides, dict):
        print_error(f"MCP server '{name}': 'env' must be an object; skipping.")
        return None

    env = os.environ.copy()
    # JSON values may be non-strings (e.g. numbers); StdioServerParameters env
    # must be str->str, so coerce.
    env.update({str(k): str(v) for k, v in env_overrides.items()})

    return ExternalServer(
        name=name,
        params=StdioServerParameters(command=command, args=args, env=env),
    )


def load_external_servers(path: Optional[Path] = None) -> List[ExternalServer]:
    """Load enabled external MCP servers from ``mcp.json``.

    Args:
        path: Override the config path (defaults to ``mcp/mcp.json`` in the app
            home, falling back to the legacy ``mcp.json`` at the home root).

    Returns:
        One ExternalServer per enabled, valid entry. Empty when the file is
        absent, empty, unreadable, or has no valid entries. Entries with
        ``"disabled": true`` are skipped silently.
    """
    if path is not None:
        cfg_path = path
    else:
        cfg_path = mcp_config_path()
        if not cfg_path.is_file():
            # Fall back to the pre-subfolder location for older installs.
            cfg_path = legacy_mcp_config_path()
    if not cfg_path.is_file():
        return []

    try:
        data = json.loads(cfg_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print_error(f"Could not read MCP config {cfg_path}: {e}")
        return []

    servers_obj = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers_obj, dict):
        if servers_obj is not None:
            print_error(f"MCP config {cfg_path}: 'mcpServers' must be an object.")
        return []

    servers: List[ExternalServer] = []
    for name, entry in servers_obj.items():
        if isinstance(entry, dict) and entry.get("disabled"):
            logger.debug(f"MCP server '{name}' is disabled; skipping.")
            continue
        server = _parse_entry(name, entry)
        if server:
            servers.append(server)

    if servers:
        logger.info(
            "Loaded %d external MCP server(s): %s",
            len(servers),
            ", ".join(s.name for s in servers),
        )
    return servers
