"""Compact command results; details stay available without changing behavior."""

import textwrap

_BOLD = "\033[1m"
_GREEN = "\033[92m"
_DIM = "\033[90m"
_RESET = "\033[0m"


def _inline(value) -> str:
    printable = "".join(
        char for char in str(value or "") if char.isprintable() or char.isspace()
    )
    return " ".join(printable.split())


def render_mcp_status(
    members, tools, *, verbose=False, config_path=None, width=88
) -> str:
    """Render already-loaded connection/tool data, without making MCP requests.

    Ownership comes from the wrapper reference, not a name prefix: most tools
    need no namespace, and a server can itself return a prefixed tool name.
    """
    lines = [f"{_BOLD}MCP Tools{_RESET}", ""]
    for name, wrapper in members:
        owned = [tool for tool in tools if getattr(tool, "mcp_client", None) is wrapper]
        connected = getattr(wrapper, "_connected", True)
        status = "connected" if connected else "disconnected"
        color = _GREEN if connected else _DIM
        noun = "tool" if len(owned) == 1 else "tools"
        lines.append(
            f"  • {_inline(name)}: {color}{status}{_RESET} "
            f"({_DIM}{len(owned)} {noun}{_RESET})"
        )
        if verbose:
            labels = []
            for tool in owned:
                exposed = _inline(tool.name)
                original = _inline(getattr(getattr(tool, "mcp_tool", None), "name", ""))
                labels.append(
                    f"{original} → {exposed}"
                    if original and original != exposed
                    else exposed
                )
            lines.extend(
                textwrap.wrap(
                    "Tools: " + (", ".join(labels) or "(none)"),
                    width=max(32, width),
                    initial_indent="    ",
                    subsequent_indent="           ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
    if not members:
        lines.append("  No MCP servers connected.")
    lines.extend(["", f"  {_DIM}{len(tools)} tools available.{_RESET}"])
    if verbose:
        lines.extend(
            [
                "",
                f"  Config: {config_path}",
                '  Format: {"mcpServers": {"name": {"command": ..., "args": [...], "env": {...}}}}',
            ]
        )
    else:
        lines.append(f"  {_DIM}Use /mcp verbose for tools and setup details.{_RESET}")
    return "\n".join(lines)


def render_model_update(change, *, applied: bool) -> str:
    """Describe a saved model selection without claiming a pending restart is done."""
    label = _inline(change.label)
    target = _inline(change.model_name)
    if target and change.model_type:
        target += f" ({_inline(change.model_type)})"
    if change.follows_chat:
        action = "now follows Chat" if applied else "saved to follow Chat"
        message = f"{label} {action}"
        if target:
            message += f": {target}"
    else:
        action = "changed to" if applied else "saved:"
        message = (
            f"{label} {action} {target}"
            if target
            else f"{label} {'updated' if applied else 'saved'}"
        )
    lines = [f"{_GREEN}• {message}{_RESET}"]
    if change.parameters_reset:
        lines.append(
            f"  {_DIM}Parameters reset to defaults · /params to adjust.{_RESET}"
        )
    if not applied:
        lines.append(f"  {_DIM}Restarting to apply…{_RESET}")
    return "\n".join(lines)
