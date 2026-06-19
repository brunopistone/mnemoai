"""Small helpers for user-facing terminal output (stdout, colored).

These are for messages the *user* should see as part of using the app —
distinct from operational diagnostics, which go through ``utils.logger``
(stderr, off by default). Errors are printed in red so failures stand out.
Everything here goes to stdout to stay in order with the rest of the chat UI
(the welcome box, status lines, model responses).
"""

_RED = "\033[91m"
_GREEN = "\033[92m"
_RESET = "\033[0m"


def print_error(message: str) -> None:
    """Print a user-facing error in red (prefixed with ✗)."""
    print(f"{_RED}✗ {message}{_RESET}")


def print_success(message: str) -> None:
    """Print a user-facing success/status line in green."""
    print(f"{_GREEN}{message}{_RESET}")
