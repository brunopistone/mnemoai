"""Catastrophic-command classifier for shell tools.

Shared by ``execute_bash`` and ``start_background_task`` so shell safety is
defined in exactly one place. The classifier blocks a *small, curated* set of
commands that destroy the system or are essentially never what a user wants an
assistant to run unattended:

- recursive force-deletes of a root-ish path (``rm -rf /``, ``rm -rf ~``, ``/*``)
- filesystem creation / raw-device overwrite (``mkfs*``, ``dd of=/dev/...``)
- whole-disk wipes (``shred``/``wipefs`` on a device)
- power-state changes (``shutdown``, ``reboot``, ``halt``, ``poweroff``, ``init 0/6``)
- classic fork bomb ``:(){ :|:& };:``
- writes into a protected system directory (``echo x > /etc/hosts``), which
  reuse ``path_policy.classify_write_path`` so the shell and the file tools
  cannot disagree about what "protected" means

Everything else is allowed at this layer. Ordinary destructive-but-scoped
commands (``rm build/``, ``git reset --hard``) are intentionally NOT blocked
here — they stay gated by the client's confirmation prompt. The goal is a
hard floor against irreversible system damage, not a second confirmation gate.

The matching is regex/word based and errs toward *allowing* when unsure (a
false block is more annoying than a rare false allow that the client prompt
still catches), EXCEPT for the handful of patterns above where the downside is
catastrophic and irreversible.
"""

import re
import shlex
from dataclasses import dataclass

from .path_policy import classify_write_path


@dataclass(frozen=True)
class BashPolicyResult:
    """Outcome of classifying a shell command.

    Attributes:
        blocked: True if the command must be refused outright.
        reason: Human-readable explanation (empty when not blocked).
        rule: Short machine id of the matched rule (empty when not blocked).
    """

    blocked: bool
    reason: str = ""
    rule: str = ""


# Each entry: (compiled pattern, rule id, human reason). Patterns run against the
# raw command string with IGNORECASE. Order does not matter — first match wins.
_BLOCK_RULES: list[tuple[re.Pattern, str, str]] = [
    # rm -rf / , rm -rf /* , rm -rf ~ , rm -rf $HOME , rm -fr .  (root-ish wipes)
    # Require the recursive+force flags (in either order, combined or separate)
    # AND a root-ish target, so `rm -rf build/` is NOT caught.
    (
        re.compile(
            r"\brm\b[^\n|;&]*?"
            r"(?:-[a-eg-qs-z]*[rR][a-eg-qs-z]*f|-[a-eg-qs-z]*f[a-eg-qs-z]*[rR]"
            r"|--recursive[^\n]*--force|--force[^\n]*--recursive"
            r"|-[rR]\s+-f|-f\s+-[rR])"
            r"\s+"
            r"(?:-[-\w]+\s+)*"  # skip any further flags before the target
            r"['\"]?"  # a quoted target is the same target
            r"(?:/|/\*|~/?|~/\*|\$HOME/?|\$\{HOME\}/?|\.|\./\*|\*)"
            r"['\"]?"
            # The target does NOT have to end the command: `rm -rf /` alone is
            # refused by GNU rm, so the form that actually wipes the filesystem
            # is `rm -rf / --no-preserve-root`. Anchoring straight after the
            # target therefore missed the only variant that works. Look past
            # trailing flags and output redirections (`2>/dev/null`) before
            # requiring end-of-command, so neither can defeat the rule.
            r"(?:\s+-[-\w]+|\s*\d*>{1,2}\s*[^\s;|&]+)*\s*"
            r"(?:$|[;|&])",
            re.IGNORECASE,
        ),
        "rm_rf_root",
        "Refusing recursive force-delete of a root/home/current-dir target — "
        "this can wipe the entire filesystem or home directory. Delete a "
        "specific subdirectory by name instead.",
    ),
    # mkfs, mkfs.ext4, mke2fs — creating a filesystem destroys the target device.
    (
        re.compile(r"\b(?:mkfs(?:\.\w+)?|mke2fs)\b", re.IGNORECASE),
        "mkfs",
        "Refusing to create a filesystem (mkfs) — this irreversibly erases the "
        "target device.",
    ),
    # dd writing to a raw device (of=/dev/sd*, /dev/disk*, /dev/nvme*, /dev/rdisk*).
    (
        re.compile(
            r"\bdd\b[^\n]*\bof=\s*/dev/(?:sd|hd|disk|rdisk|nvme|mmcblk|vd)\w*",
            re.IGNORECASE,
        ),
        "dd_device",
        "Refusing dd write to a raw disk device — this overwrites the disk and "
        "is unrecoverable.",
    ),
    # shred / wipefs targeting a device node.
    (
        re.compile(
            r"\b(?:shred|wipefs)\b[^\n]*\s/dev/(?:sd|hd|disk|rdisk|nvme|mmcblk|vd)\w*",
            re.IGNORECASE,
        ),
        "wipe_device",
        "Refusing to wipe a raw disk device.",
    ),
    # Power-state changes: shutdown/reboot/halt/poweroff, and `init 0` / `init 6`.
    (
        re.compile(r"\b(?:shutdown|reboot|halt|poweroff)\b", re.IGNORECASE),
        "power_state",
        "Refusing a power-state command (shutdown/reboot/halt/poweroff).",
    ),
    (
        re.compile(r"\binit\b\s+[06]\b", re.IGNORECASE),
        "power_state",
        "Refusing a runlevel change (init 0/6) that halts or reboots the machine.",
    ),
    # Classic fork bomb, tolerant of internal whitespace: :(){ :|:& };:
    (
        re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),
        "fork_bomb",
        "Refusing a fork bomb — it would exhaust system process resources.",
    ),
]

# --- System-write floor -------------------------------------------------------
# `fs_write`/`file_edit` refuse to write under /etc, /usr, /System, … but a shell
# command reaching the same path was never checked, so `echo x > /etc/hosts` was
# simply not a "write tool" call. Shell redirections and the handful of commands
# whose arguments ARE their write targets are extracted here and classified with
# the same `classify_write_path`, so the two tool families cannot disagree.
#
# This is deliberately not a shell parser. It covers the unambiguous forms an
# assistant actually produces; anything cleverer (eval, printf into a variable,
# a script written elsewhere and then run) is out of reach at this layer and
# stays the client confirmation prompt's problem.

# Character devices that are always a safe write target. `2>/dev/null` appears in
# a large share of real commands and /dev is otherwise a protected prefix, so
# these MUST be exempt or the policy would block ordinary work.
_SAFE_DEVICE_TARGETS: frozenset[str] = frozenset(
    {
        "/dev/null",
        "/dev/zero",
        "/dev/stdout",
        "/dev/stderr",
        "/dev/tty",
        "/dev/fd",
    }
)

# Output redirections, used ONLY when the command can't be tokenized (unbalanced
# quoting). The token scan below is preferred because it respects quoting: this
# regex would read `echo 'x > /etc/hosts'` as a redirection. The target class
# excludes `&`, so an fd duplication (`1>&2`) yields no path at all.
_REDIRECT_RE = re.compile(r"(?:\d+|&)?>{1,2}\|?\s*(?P<target>[^\s;|&<>()]+)")

# Shell metacharacters that shlex(punctuation_chars=True) groups into their own
# tokens, so a token made only of these is an operator, never a path.
_OPERATOR_CHARS = "<>|&();\n"

# Commands where EVERY non-flag operand is written to.
_WRITE_ALL_OPERANDS: frozenset[str] = frozenset(
    {"tee", "truncate", "touch", "mkdir", "mkfifo", "chmod", "chown", "chgrp"}
)
# Commands where the LAST non-flag operand is the destination.
_WRITE_LAST_OPERAND: frozenset[str] = frozenset(
    {"cp", "mv", "install", "ln", "rsync"}
)
# Wrappers to look through when finding the command being run.
_WRAPPERS: frozenset[str] = frozenset(
    {"sudo", "doas", "env", "command", "nohup", "time", "xargs", "nice"}
)
# Shells whose `-c` argument is itself a command to scan (the `sh -c '… > /etc/x'`
# form that made this whole check necessary).
_SHELLS: frozenset[str] = frozenset({"sh", "bash", "zsh", "dash", "ksh"})


def _unquote(token: str) -> str:
    """Strip one layer of surrounding quotes from a shell token."""
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        return token[1:-1]
    return token


def _is_exempt_device(path: str) -> bool:
    """True for the standard sinks and terminal devices under ``/dev``."""
    return path in _SAFE_DEVICE_TARGETS or path.startswith("/dev/fd/")


def _is_operator(token: str) -> bool:
    """True when ``token`` is a run of shell metacharacters, not a word."""
    return bool(token) and all(ch in _OPERATOR_CHARS for ch in token)


def _split_segments(tokens: list[str]) -> list[list[str]]:
    """Split a token list on shell separators into per-command segments."""
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token and all(ch in ";|&\n" for ch in token):
            segments.append([])
        else:
            segments[-1].append(token)
    return [seg for seg in segments if seg]


def _partition_redirections(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Split one segment into ``(operands, redirection_targets)``.

    Redirection targets are consumed here so they are neither classified twice
    nor mistaken for a command operand (``tee log > /dev/null``).
    """
    operands: list[str] = []
    targets: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if _is_operator(token):
            if ">" in token:
                # A trailing `&` means fd duplication (`2>&1`): no path follows.
                if not token.endswith("&") and i + 1 < len(tokens):
                    target = tokens[i + 1]
                    if not target.startswith("&") and not target.isdigit():
                        targets.append(target)
                i += 2  # skip the operator and whatever it redirects to
                continue
            i += 1
            continue
        operands.append(token)
        i += 1
    return operands, targets


def _operand_targets(operands: list[str], depth: int) -> list[str]:
    """Write targets implied by one command's operands (recursing into ``sh -c``)."""
    tokens = list(operands)
    while tokens and tokens[0] in _WRAPPERS:
        tokens.pop(0)
    if not tokens:
        return []

    verb = tokens[0].rsplit("/", 1)[-1]  # /bin/cp -> cp
    args = tokens[1:]

    if verb in _SHELLS and depth < 2:
        # Scan the -c payload as a command in its own right.
        nested: list[str] = []
        for i, token in enumerate(args):
            if token == "-c" and i + 1 < len(args):
                nested.extend(_iter_write_targets(args[i + 1], depth + 1))
        return nested

    operands = [a for a in args if not a.startswith("-")]
    if verb in _WRITE_ALL_OPERANDS:
        return operands
    if verb == "sed" and any(
        a == "-i" or a.startswith("-i") or a == "--in-place" for a in args
    ):
        return operands
    if verb in _WRITE_LAST_OPERAND and operands:
        return operands[-1:]
    if verb == "dd":
        return [a.split("=", 1)[1] for a in args if a.startswith("of=")]
    return []


def _iter_write_targets(command: str, depth: int = 0) -> list[str]:
    """Collect every path ``command`` plausibly writes to.

    Tokenizes so quoting is respected: a redirection inside a quoted string is
    just text (``echo 'x > /etc/hosts'``), while a quoted ``sh -c`` payload stays
    one token and is scanned as a command in its own right.
    """
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        # Unbalanced quoting: fall back to the raw-string redirection scan rather
        # than giving up on the check entirely.
        return [_unquote(m.group("target")) for m in _REDIRECT_RE.finditer(command)]

    targets: list[str] = []
    for segment in _split_segments(tokens):
        operands, redirections = _partition_redirections(segment)
        targets.extend(redirections)
        targets.extend(_operand_targets(operands, depth))
    return [t for t in targets if t]


def classify_shell_write_targets(command: str) -> BashPolicyResult:
    """Classify the write targets of a shell command against the path policy.

    Args:
        command: The full shell command string as it would be passed to the shell.

    Returns:
        A :class:`BashPolicyResult` blocked when any extracted write target
        lands in a protected system directory (``/dev/null`` and friends are
        exempt). Commands whose targets can't be determined are allowed.
    """
    for target in _iter_write_targets(command):
        if _is_exempt_device(target):
            continue
        verdict = classify_write_path(target)
        if verdict.blocked:
            return BashPolicyResult(
                blocked=True,
                reason=(
                    f"{verdict.reason} This command writes to '{target}'. Use a "
                    f"path in your home, project, or temp directory instead."
                ),
                rule="system_write",
            )
    return BashPolicyResult(blocked=False)


def classify_shell_command(command: str) -> BashPolicyResult:
    """Classify a shell command for catastrophic, irreversible actions.

    Args:
        command: The full shell command string as it would be passed to the shell.

    Returns:
        A :class:`BashPolicyResult`. ``blocked`` is True only for the curated set
        of system-destroying patterns and for writes into a protected system
        directory; all other commands are allowed at this layer (and remain
        subject to the client's confirmation gate).
    """
    if not command or not command.strip():
        return BashPolicyResult(blocked=False)

    for pattern, rule, reason in _BLOCK_RULES:
        if pattern.search(command):
            return BashPolicyResult(blocked=True, reason=reason, rule=rule)

    return classify_shell_write_targets(command)
