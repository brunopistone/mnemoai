"""Auto-approve mode: which confirmation categories run without asking.

The user-facing counterpart to plan mode. Plan mode BLOCKS the mutating tools;
this one lets them through without a prompt — the same session-scoped flag
pattern (``client.auto_approve_mode`` → a provider lambda → the agent), read at
one place in the gate so nothing has to re-derive it.

Pure: a mode string plus the tool's own args in, a bool out. No agent, no
config, no I/O — so the ladder is unit-testable on its own and the gate keeps
its single decision point.

**It is a session TRUST, not a new permission.** It sits at the same rung as the
"a" answer (``_trusted_confirm_categories``): below the server-side safety
floors, below the plan-mode hard block, and below a hook's ``deny`` — every one
of those is decided in ``tool_loop.run_tool_calls`` before the gate is ever
reached, so no mode here can widen them. What it replaces is only the keypress.

**The ``git`` category is never auto-approved, at any tier** (``NEVER_AUTO``).
That category gates ``allow_dangerous=True`` and a tool's own
``requires_confirmation`` refusal — i.e. a server-side safety check the model is
asking to override. ``confirmation_gate`` moved exactly that decision out of the
model's hands on purpose (a tool refused, and the model's documented next step
was to re-call with the override set, approving its own dangerous call); a mode
that handed it back would undo that, so the tiers stop short of it. A flagged
operation still asks, in every mode.

Tiers are ordered and cycle, so one command can step through them:

``off``     nothing is auto-approved — the shipped behavior.
``edits``   file writes whose target is inside the working directory. The safe
            default: the thing that interrupts most, scoped to the tree the user
            is working in, so an edit to ``~/.ssh`` or ``/etc`` still asks.
``writes``  every file write (any path) plus curated-memory writes.
``all``     the above plus shell commands.
"""

import os
from typing import Optional, Tuple

# Ordered ladder — `cycle` walks this, so the order is the user-visible one.
MODES: Tuple[str, ...] = ("off", "edits", "writes", "all")

DEFAULT_MODE = "off"

# Categories a tier may cover, keyed by mode. `edits` is additionally scoped to
# the working directory (see `covers`); the others are path-agnostic.
_TIER_CATEGORIES = {
    "off": frozenset(),
    "edits": frozenset({"write"}),
    "writes": frozenset({"write", "memory"}),
    "all": frozenset({"write", "memory", "bash"}),
}

# Never auto-approved, whatever the mode: this is the override of a server-side
# safety refusal, which must stay a human decision. See the module docstring.
NEVER_AUTO = frozenset({"git"})

# Only `edits` scopes writes to the working directory; the wider tiers are the
# user explicitly asking for any path.
_WORKSPACE_SCOPED = frozenset({"edits"})

# (label, prompt-toolkit color) per mode, for the input badge. `off` has none —
# an indicator for the default state is noise.
_BADGES = {
    "edits": ("⏵ auto-edits", "ansigreen"),
    "writes": ("⏵⏵ auto-writes", "ansiyellow"),
    "all": ("⏵⏵ auto-all", "ansired"),
}

# One line each, shown when the mode is switched on.
_NOTICES = {
    "off": "every destructive tool asks again",
    "edits": "file edits inside this directory run without asking",
    "writes": "all file writes and memory updates run without asking",
    "all": "file writes, memory updates and shell commands run without asking",
}


def normalize(mode: Optional[str]) -> str:
    """Coerce anything to a known mode, falling back to ``off``.

    Tolerant on purpose: the mode can arrive from a config file or a provider
    lambda on a bare test object, and an unrecognized value must degrade to the
    shipped behavior rather than to a wider one.
    """
    if not isinstance(mode, str):
        return DEFAULT_MODE
    candidate = mode.strip().lower()
    return candidate if candidate in _TIER_CATEGORIES else DEFAULT_MODE


def cycle(mode: Optional[str]) -> str:
    """The next mode in the ladder, wrapping ``all`` back to ``off``."""
    current = normalize(mode)
    return MODES[(MODES.index(current) + 1) % len(MODES)]


def covers(
    mode: Optional[str],
    category: str,
    target: Optional[str] = None,
    cwd: Optional[str] = None,
) -> bool:
    """True when ``mode`` auto-approves ``category`` for this call.

    ``target`` is the path a write would touch (``None`` for the categories that
    have none); ``cwd`` defaults to the process working directory. A
    workspace-scoped tier with an unknown or outside target returns False — the
    prompt is the safe answer whenever the scope can't be established.
    """
    if category in NEVER_AUTO:
        return False
    current = normalize(mode)
    if category not in _TIER_CATEGORIES[current]:
        return False
    if category == "write" and current in _WORKSPACE_SCOPED:
        return _within_workspace(target, cwd)
    return True


def _within_workspace(target: Optional[str], cwd: Optional[str]) -> bool:
    """True when ``target`` resolves inside ``cwd``. Unknown paths are False.

    Resolves symlinks on both sides so a link out of the tree can't smuggle a
    write past the scope, and compares on a separator boundary so a sibling
    directory sharing a name prefix (``/repo-old`` vs ``/repo``) doesn't match.
    """
    if not target or not str(target).strip():
        return False
    try:
        root = os.path.realpath(cwd or os.getcwd())
        path = os.path.realpath(os.path.expanduser(str(target).strip()))
    except (OSError, ValueError):
        return False
    return path == root or path.startswith(root + os.sep)


def badge(mode: Optional[str]) -> Optional[Tuple[str, str]]:
    """``(label, color)`` for the input prompt, or None when nothing to show."""
    return _BADGES.get(normalize(mode))


def notice(mode: Optional[str]) -> str:
    """One line describing what the mode does, for the switch confirmation."""
    return _NOTICES[normalize(mode)]
