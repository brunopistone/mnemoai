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
from dataclasses import dataclass


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
            r"(?:/|/\*|~/?|~/\*|\$HOME/?|\$\{HOME\}/?|\.|\./\*|\*)\s*"
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


def classify_shell_command(command: str) -> BashPolicyResult:
    """Classify a shell command for catastrophic, irreversible actions.

    Args:
        command: The full shell command string as it would be passed to the shell.

    Returns:
        A :class:`BashPolicyResult`. ``blocked`` is True only for the curated set
        of system-destroying patterns; all other commands are allowed at this
        layer (and remain subject to the client's confirmation gate).
    """
    if not command or not command.strip():
        return BashPolicyResult(blocked=False)

    for pattern, rule, reason in _BLOCK_RULES:
        if pattern.search(command):
            return BashPolicyResult(blocked=True, reason=reason, rule=rule)

    return BashPolicyResult(blocked=False)
