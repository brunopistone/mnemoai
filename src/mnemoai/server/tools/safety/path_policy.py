"""Write-path classifier for file-mutating tools.

Shared by ``fs_write`` and ``file_edit`` so the "don't write over the operating
system" rule the docstrings advertise is actually enforced server-side, in one
place.

Scope (intentional): this blocks writes *into* critical system directories
(``/etc``, ``/bin``, ``/usr``, ``/boot``, ``/System``, ``/dev``, the root of the
filesystem itself, …) where clobbering a file can break the machine. It does NOT
restrict writes to the user's home, project trees, temp dirs, or the app home —
those remain allowed and gated by the client's write-confirmation prompt. The
goal is a hard floor against system corruption, not a workspace jail.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PathPolicyResult:
    """Outcome of classifying a write target.

    Attributes:
        blocked: True if writing to the path must be refused.
        reason: Human-readable explanation (empty when not blocked).
        resolved: The absolute, normalized path that was classified.
    """

    blocked: bool
    reason: str = ""
    resolved: str = ""


# Absolute directory prefixes that are off-limits for writes. A target is blocked
# if it IS one of these or lives underneath one. Kept conservative: these are
# OS/system locations, not anything under the user's home. Both Linux and macOS
# system roots are listed so the policy is correct on either platform.
_BLOCKED_PREFIXES: tuple[str, ...] = (
    "/etc",
    "/bin",
    "/sbin",
    "/usr",
    "/lib",
    "/lib64",
    "/boot",
    "/dev",
    "/proc",
    "/sys",
    "/var/log",
    "/var/run",
    "/var/db",
    "/run",
    "/System",  # macOS
    "/Library",  # macOS system-wide (user Library at ~/Library is fine)
    "/private/etc",  # macOS: /etc is a symlink here
    # NOTE: NOT "/private/var" — the macOS per-user temp dir lives at
    # /private/var/folders/..., a legitimate and common write target. Only the
    # genuinely-system subpaths under it are blocked.
    "/private/var/db",
    "/private/var/log",
)


def _normalize(path: str) -> str:
    """Expand ``~`` and make the path absolute + normalized (no filesystem probe)."""
    expanded = os.path.expanduser(path.strip())
    # abspath resolves against CWD for relative paths and collapses .. / . segments
    # so a target like "/usr/../etc/passwd" normalizes to "/etc/passwd".
    return os.path.abspath(expanded)


def classify_write_path(path: str) -> PathPolicyResult:
    """Classify a file path as a safe or forbidden write target.

    Args:
        path: The destination path (may be relative, may contain ``~`` or ``..``).

    Returns:
        A :class:`PathPolicyResult`. ``blocked`` is True for the filesystem root
        itself and for any path inside a critical system directory; everything
        else (home, projects, temp, app home) is allowed at this layer.
    """
    if not path or not path.strip():
        # An empty path isn't a system-write; let the tool's own validation handle it.
        return PathPolicyResult(blocked=False, resolved="")

    resolved = _normalize(path)

    # Writing directly AT the filesystem root.
    if resolved == "/":
        return PathPolicyResult(
            blocked=True,
            reason="Refusing to write at the filesystem root '/'.",
            resolved=resolved,
        )

    for prefix in _BLOCKED_PREFIXES:
        if resolved == prefix or resolved.startswith(prefix + os.sep):
            return PathPolicyResult(
                blocked=True,
                reason=(
                    f"Refusing to write under the protected system directory "
                    f"'{prefix}'. Writes are allowed in your home, project, temp, "
                    f"and app directories."
                ),
                resolved=resolved,
            )

    return PathPolicyResult(blocked=False, resolved=resolved)
