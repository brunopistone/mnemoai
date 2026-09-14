"""Small helpers for user-facing terminal output (stdout, colored).

These are for messages the *user* should see as part of using the app —
distinct from operational diagnostics, which go through ``utils.logger``
(stderr, off by default). Errors are printed in red so failures stand out.
Everything here goes to stdout to stay in order with the rest of the chat UI
(the welcome box, status lines, model responses).
"""

_RED = "\033[91m"
_GREEN = "\033[92m"
_DIM = "\033[90m"
_RESET = "\033[0m"

# The boot spinner, while it's running. A message printed from under it would
# otherwise be chased by the next animation frame, stranding a stale
# "⠿ Connecting model…" in the scrollback — see StartupLoader.write_above.
_active_loader = None


def set_active_loader(loader) -> None:
    """Route console output through ``loader`` (None to unset) while it spins."""
    global _active_loader
    _active_loader = loader


def _emit(text: str) -> None:
    """Print a line, pausing the boot spinner around it if one is animating."""
    loader = _active_loader
    if loader is not None:
        loader.write_above(text)
    else:
        print(text)


def print_error(message: str) -> None:
    """Print a user-facing error in red (prefixed with ✗)."""
    _emit(f"{_RED}✗ {message}{_RESET}")


def print_success(message: str) -> None:
    """Print a user-facing success/status line in green."""
    _emit(f"{_GREEN}{message}{_RESET}")


def print_notice(message: str) -> None:
    """Print a dim user-facing status line (prefixed with ·, as /doctor's rows).

    For work the user did not ask for but is waiting on — a one-off startup
    repair, say. ``logger.info`` cannot do this job: the console handler sits at
    ``LOG_LEVEL`` (WARNING by default), so an INFO record reaches the log file
    and nothing else, and no filter can lift it — the level is checked before
    filters run, and the logger's own level discards the record first. A notice
    that explains a pause therefore has to be printed. Pair it with the
    ``logger.info`` rather than replacing it: the screen line scrolls away, the
    file record is what a later question about that startup is answered from.
    """
    _emit(f"{_DIM}· {message}{_RESET}")
