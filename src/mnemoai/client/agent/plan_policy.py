"""Plan-mode enforcement policy (pure logic).

Plan mode is a user-toggled, client-side hard gate: while active, mutating /
shell-executing tools are blocked at the tool chokepoint, regardless of what the
model does. This module holds the *decision* logic and its data tables; the agent
keeps thin delegating methods (`_is_blocked_by_plan_mode`, `_is_readonly_bash`,
`_is_plan_file`) that call in here, so its public surface — and the unit tests
that build a bare agent — are unchanged.

The functions here take an explicit ``plan_active`` flag rather than reading agent
state, which keeps them pure and directly testable.
"""

import os
from pathlib import Path

from mnemoai.utils.paths import plans_dir

# Mutating / shell-executing tools hard-blocked while plan mode is active.
# Read-only tools (fs_read, glob/grep search, web, document readers) and the
# `memory` notebook are allowed.
# NOTE: execute_bash, fs_write, and file_edit are blocked CONDITIONALLY (see
# is_blocked_by_plan_mode): read-only bash and writes to the plan file are
# permitted; everything else here is an unconditional block.
PLAN_BLOCKED_TOOLS = {
    "execute_bash",
    "fs_write",
    "file_edit",
    "git_safe",
    "git_commit_safe",
    "start_background_task",
}

# In plan mode, fs_write/file_edit are allowed ONLY for the plan file (a single
# writable path under the plans dir). Everything else writing is blocked.
PLAN_FILE_SUFFIX = ".md"

# Read-only shell commands permitted in plan mode (the leading program must be
# one of these AND the command must contain no shell operator that could chain a
# mutation — see is_readonly_bash). Conservative on purpose: when in doubt, the
# command is treated as NOT read-only and blocked.
READONLY_BASH_CMDS = {
    "ls", "cat", "head", "tail", "less", "more", "pwd", "echo", "find",
    "grep", "rg", "egrep", "fgrep", "wc", "stat", "file", "tree", "du",
    "df", "which", "type", "whoami", "hostname", "date", "env", "printenv",
    "ps", "uname", "id", "diff", "sort", "uniq", "cut", "awk", "sed",
    "realpath", "readlink", "basename", "dirname", "git",
}
# Shell operators that could append/redirect a mutation onto a read-only
# command. Their presence forces the command to be treated as non-read-only.
BASH_MUTATION_OPS = (">", ">>", "|", ";", "&&", "||", "`", "$(", "&")
# git subcommands that are unambiguously read-only (others — commit/push/
# checkout/tag/stash/config-set/branch -d… — can mutate, so are blocked).
READONLY_GIT_SUBCMDS = {
    "status", "log", "diff", "show", "rev-parse", "describe", "blame",
    "ls-files", "ls-tree", "cat-file", "shortlog",
}
# Per-program flags that turn an otherwise-read-only allowlisted command into a
# mutating one. `-i` is program-specific (sed: in-place edit; but grep/ls `-i`
# are read-only), so it's keyed by program rather than global.
BASH_MUTATING_FLAGS = {
    "sed": ("-i", "--in-place"),  # also matches `-i.bak` (prefix check)
    "find": ("-delete", "-exec", "-execdir", "-fprint", "-fprintf", "-fls"),
    "awk": ("-i",),  # gawk -i inplace
}


def is_readonly_bash(command: str) -> bool:
    """Heuristically decide if a shell command is read-only (plan-mode safe).

    Conservative: the leading program must be in the read-only allowlist, the
    command must contain no operator that could chain/redirect a mutation, and a
    ``git`` command must use a read-only subcommand. Anything uncertain returns
    False (so it stays blocked).
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
    """The ToolMessage returned when a tool is blocked by plan mode.

    Tailored per tool so the model knows the read-only escape hatch (run a
    read-only shell command, or write the plan file) rather than just erroring
    out.
    """
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
