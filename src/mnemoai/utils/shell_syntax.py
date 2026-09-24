"""Conservative parsing for shell approval and read-only command policies."""

import shlex

_SHELL_OPERATORS = ("\n", "\r", ">", "<", "|", ";", "&", "`", "$", "(", ")")


def simple_command_tokens(command: str) -> list[str]:
    """Return one literal command, or no tokens if shell evaluation is required."""
    if not isinstance(command, str) or any(op in command for op in _SHELL_OPERATORS):
        return []
    try:
        return shlex.split(command)
    except ValueError:
        return []
