"""Server-side safety policies for tools that run shell commands or write files.

These policies enforce, *inside the MCP server*, the hard limits that the tool
docstrings advertise. They are deliberately independent of the client's
confirmation/plan-mode gating: the MCP server is a subprocess that another client
could drive, so the catastrophic-action blocks must live below the agent layer.

Scope (intentional): these block only *catastrophic, almost-never-intended*
actions — wiping the disk, formatting a device, writing over system directories,
reading credential files. Ordinary mutations (``rm file.txt``, editing a project
file) are NOT blocked here; those remain gated by the client's ``_confirm_tool``
prompt and plan mode.
"""

from .bash_policy import (
    BashPolicyResult,
    classify_shell_command,
    classify_shell_write_targets,
)
from .path_policy import PathPolicyResult, classify_write_path
from .url_policy import UrlPolicyResult, classify_url

__all__ = [
    "BashPolicyResult",
    "classify_shell_command",
    "classify_shell_write_targets",
    "PathPolicyResult",
    "classify_write_path",
    "UrlPolicyResult",
    "classify_url",
]
