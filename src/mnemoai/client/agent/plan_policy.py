"""Plan-mode enforcement policy (pure logic).

Plan mode is a user-toggled client-side hard gate: while active, mutating/
shell-executing tools are blocked at the tool chokepoint. Functions take an
explicit ``plan_active`` flag (not agent state) to stay pure and testable; the
agent keeps thin delegating methods over them.
"""

import os
from pathlib import Path

from mnemoai.utils.paths import plans_dir

# Tools hard-blocked in plan mode. execute_bash/fs_write/file_edit are blocked
# CONDITIONALLY (see is_blocked_by_plan_mode); the rest unconditionally.
PLAN_BLOCKED_TOOLS = {
    "execute_bash",
    "fs_write",
    "file_edit",
    "git_safe",
    "git_commit_safe",
    "start_background_task",
}

# fs_write/file_edit are allowed only for the plan file (a .md under plans_dir).
PLAN_FILE_SUFFIX = ".md"

# Leading programs allowed in plan mode (subject to is_readonly_bash checks).
READONLY_BASH_CMDS = {
    "ls", "cat", "head", "tail", "less", "more", "pwd", "echo", "find",
    "grep", "rg", "egrep", "fgrep", "wc", "stat", "file", "tree", "du",
    "df", "which", "type", "whoami", "hostname", "date", "env", "printenv",
    "ps", "uname", "id", "diff", "sort", "uniq", "cut", "awk", "sed",
    "realpath", "readlink", "basename", "dirname", "git",
}
# Shell operators that could chain/redirect a mutation → treat as non-read-only.
BASH_MUTATION_OPS = (">", ">>", "|", ";", "&&", "||", "`", "$(", "&")
# git subcommands that are unambiguously read-only (others can mutate → blocked).
READONLY_GIT_SUBCMDS = {
    "status", "log", "diff", "show", "rev-parse", "describe", "blame",
    "ls-files", "ls-tree", "cat-file", "shortlog",
}
# Per-program flags that make an allowlisted command mutating (keyed by program
# since `-i` means in-place for sed but is read-only for grep/ls).
BASH_MUTATING_FLAGS = {
    "sed": ("-i", "--in-place"),  # also matches `-i.bak` (prefix check)
    "find": ("-delete", "-exec", "-execdir", "-fprint", "-fprintf", "-fls"),
    "awk": ("-i",),  # gawk -i inplace
}


def is_readonly_bash(command: str) -> bool:
    """Heuristically decide if a shell command is read-only (plan-mode safe).

    Conservative — leading program allowlisted, no mutation operator, read-only
    git subcommand, no in-place flags; anything uncertain returns False.
    """
    cmd = (command or "").strip()
    if not cmd:
        return False
    if any(op in cmd for op in BASH_MUTATION_OPS):
        return False
    tokens = cmd.split()
    prog = tokens[0]
    if prog not in READONLY_BASH_CMDS:
        return False
    # Reject program-specific mutating flags (e.g. `sed -i`, `sed -i.bak`,
    # `find … -delete`/`-exec`) even though the program is allowlisted.
    bad_flags = BASH_MUTATING_FLAGS.get(prog, ())
    for tok in tokens[1:]:
        if any(tok == f or tok.startswith(f) for f in bad_flags):
            return False
    if prog == "git":
        sub = tokens[1] if len(tokens) > 1 else ""
        return sub in READONLY_GIT_SUBCMDS
    return True


def is_plan_file(path: str) -> bool:
    """True if ``path`` is the writable plan file (a .md under the plans dir)."""
    if not path:
        return False
    try:
        target = Path(os.path.expanduser(path)).resolve()
        base = plans_dir().resolve()
        return target.suffix == PLAN_FILE_SUFFIX and base in target.parents
    except Exception:
        return False


def is_blocked_by_plan_mode(
    tool_name: str, tool_args: dict = None, *, plan_active: bool
) -> bool:
    """True when plan mode is active and this tool/call would mutate.

    Read-only tools and the memory notebook always pass. Three tools are
    CONDITIONAL: ``execute_bash`` is allowed when the command is read-only, and
    ``fs_write``/``file_edit`` are allowed only when writing the plan file.
    Everything else in ``PLAN_BLOCKED_TOOLS`` is unconditionally blocked.
    """
    if not plan_active:
        return False
    if tool_name not in PLAN_BLOCKED_TOOLS:
        return False

    args = tool_args or {}
    if tool_name == "execute_bash":
        return not is_readonly_bash(str(args.get("command", "")))
    if tool_name in ("fs_write", "file_edit"):
        return not is_plan_file(str(args.get("path", "")))
    return True


def plan_mode_block_message(tool_name: str) -> str:
    """ToolMessage for a plan-mode block, tailored per tool to point at the
    read-only escape hatch (read-only shell, or writing the plan file)."""
    if tool_name == "execute_bash":
        return (
            "Blocked: plan mode is active (read-only). Only read-only shell "
            "commands (e.g. ls, cat, grep, git status/log/diff) are allowed "
            "while planning. Investigate with read-only tools and present a "
            "plan; the user must exit plan mode (/plan) before mutating "
            "commands can run."
        )
    if tool_name in ("fs_write", "file_edit"):
        try:
            plan_hint = str(plans_dir())
        except Exception:
            plan_hint = "the plans directory"
        return (
            "Blocked: plan mode is active (read-only). You may only write your "
            f"plan as a Markdown file under {plan_hint}. Editing other files is "
            "blocked — present a plan for the user to review; the user must "
            "exit plan mode (/plan) before other changes can be made."
        )
    return (
        "Blocked: plan mode is active (read-only). Present a plan for the user "
        "to review; the user must exit plan mode (/plan) before this tool can "
        "run."
    )
