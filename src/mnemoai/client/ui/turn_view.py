"""Styled rendering of a turn's reasoning and tool calls (pinned-input UI).

Pure string builders — no I/O, no prompt_toolkit — so they're unit-testable and
usable from either the agent (which prints above the pinned region) or tests.
The visual language mirrors modern agentic CLIs:

- a reasoning block with a green left border, a ``Thought for Ns…`` header, and
  the reasoning text indented under a ``↳`` connector;
- tool invocations as ``ToolName`` on its own line with each argument indented
  under a ``↳ key=value`` connector.

ANSI is used directly (not prompt_toolkit widgets) because these lines are
written into native scrollback ABOVE the pinned input, not drawn in the live
region. Colors degrade harmlessly on terminals that ignore them.
"""

# ANSI helpers. Kept local so this module has no dependencies.
_GREEN = "\033[32m"
_GRAY = "\033[90m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

# The green vertical bar drawn at the left of a reasoning block, and the
# connector that introduces an indented detail line.
_BAR = f"{_GREEN}▌{_RESET}"
_CONNECTOR = "↳"


def format_duration(seconds: float) -> str:
    """Human, compact duration for the ``Thought for …`` header.

    ``0.4`` → ``"0s"``, ``1`` → ``"1s"``, ``12.7`` → ``"12s"``, ``90`` → ``"1m30s"``.
    """
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    return f"{minutes}m{secs}s"


def render_reasoning_block(reasoning: str, seconds: float) -> str:
    """Build the green-border reasoning block.

    Args:
        reasoning: The model's reasoning text (may be multi-line / empty).
        seconds: How long the reasoning took, for the header.

    Returns:
        A multi-line string ready to print, or ``""`` when there's no reasoning
        (a non-reasoning model / empty reasoning collapses to nothing — no empty
        block, per the design).
    """
    text = (reasoning or "").strip()
    if not text:
        return ""

    header = f"{_BAR} {_BOLD}Thought for {format_duration(seconds)}…{_RESET}"
    lines = [header]
    # Each reasoning line is dimmed and indented under a connector. Blank lines
    # inside the reasoning are preserved (as a bare bar) so paragraphs read.
    for i, raw in enumerate(text.split("\n")):
        connector = _CONNECTOR if i == 0 else " "
        if raw.strip():
            lines.append(f"{_BAR}   {_GRAY}{connector} {raw}{_RESET}")
        else:
            lines.append(_BAR)
    return "\n".join(lines)


def render_tool_call(name: str, args: dict) -> str:
    """Build the styled block for one tool invocation.

    ``ToolName`` in bold on its own line, then each argument on an indented
    ``↳ key=value`` line (dimmed). Long values are shown in full here (the
    compact ``[⚙ …]`` marker elsewhere is the elided one). Newlines in a value
    are flattened to keep one arg per line.
    """
    lines = [f"{_BOLD}{name or 'tool'}{_RESET}"]
    for key, value in (args or {}).items():
        flat = str(value).replace("\n", " ")
        lines.append(f"  {_GRAY}{_CONNECTOR} {key}={flat}{_RESET}")
    return "\n".join(lines)
