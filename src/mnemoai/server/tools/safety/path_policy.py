"""Path classifiers for file tools.

Shared by ``fs_write``/``file_edit`` (write side) and ``fs_read`` (read side) so
the "don't write over the operating system" and "don't read the user's secrets"
rules are enforced server-side, in one place.

Write scope (intentional): this blocks writes *into* critical system directories
(``/etc``, ``/bin``, ``/usr``, ``/boot``, ``/System``, ``/dev``, the root of the
filesystem itself, …) where clobbering a file can break the machine. It does NOT
restrict writes to the user's home, project trees, temp dirs, or the app home —
those remain allowed and gated by the client's write-confirmation prompt. The
goal is a hard floor against system corruption, not a workspace jail.

Read scope (intentional): this blocks the handful of files that are *only* ever
secrets — cloud credentials, private keys, ``.env`` files. The motivation is
specific to an LLM tool: anything read here is pasted into a prompt and shipped
to a model provider, so a single incurious ``fs_read`` exfiltrates a long-lived
credential. Everything else is readable.

Both classifiers resolve symlinks (``realpath``), because a symlink is the
cheapest way around a prefix check: ``ln -s /etc ~/tmp/etc`` would otherwise
turn a "home" write into an ``/etc`` write.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PathPolicyResult:
    """Outcome of classifying a path.

    Attributes:
        blocked: True if the operation on the path must be refused.
        reason: Human-readable explanation (empty when not blocked).
        resolved: The absolute, normalized, symlink-resolved path classified.
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
    """Expand ``~``, make absolute, and resolve symlinks.

    ``realpath`` (not ``abspath``) is the security-relevant part: it collapses
    ``..``/``.`` AND follows symlinks, so ``~/link-to-etc/passwd`` classifies as
    ``/etc/passwd`` instead of sailing past the prefix check. Non-existent
    trailing components are preserved, so a not-yet-created file still resolves
    through its real parent directory.
    """
    expanded = os.path.expanduser(path.strip())
    return os.path.realpath(expanded)


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
