"""Logging utilities for the AI application."""

import logging
import os
import sys

# ANSI colors per level. ERROR/CRITICAL red, WARNING yellow; DEBUG/INFO plain.
_LEVEL_COLORS = {
    logging.WARNING: "\033[93m",   # yellow
    logging.ERROR: "\033[91m",     # red
    logging.CRITICAL: "\033[91m",  # red
}
_RESET = "\033[0m"


class _CursorTracker:
    """Wraps a text stream to remember whether the cursor is mid-line.

    The chat UI streams answer chunks to stdout without a trailing newline, so a
    log record written to stderr afterwards lands ON THE SAME visual line. By
    tracking whether the last character written to stdout was a newline, the log
    handler can prepend one when needed — keeping logs on their own lines.
    """

    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped
        # Start "at line start" so a log before any output doesn't get a blank
        # line prepended.
        self.at_line_start = True

    def write(self, s):
        n = self._wrapped.write(s)
        if s:
            self.at_line_start = s.endswith("\n")
        return n

    def __getattr__(self, name):
        # Delegate everything else (flush, isatty, fileno, encoding, …).
        return getattr(self._wrapped, name)


# Install the tracker on stdout once, so log handlers can consult it. Only wrap
# a real stream (skip when stdout is already wrapped or missing).
if not isinstance(sys.stdout, _CursorTracker) and sys.stdout is not None:
    sys.stdout = _CursorTracker(sys.stdout)


class _NewlineGuardHandler(logging.StreamHandler):
    """StreamHandler that ensures a log record starts on a fresh line.

    If stdout is mid-line (the chat UI streamed text without a trailing
    newline), emit a leading newline to stderr first so the log message isn't
    appended to the user-facing output. No-op when stdout isn't a TTY (piped
    output stays clean) or when already at line start.
    """

    def emit(self, record: logging.LogRecord) -> None:
        out = sys.stdout
        try:
            mid_line = (
                isinstance(out, _CursorTracker)
                and not out.at_line_start
                and hasattr(self.stream, "isatty")
                and self.stream.isatty()
            )
            if mid_line:
                self.stream.write("\n")
                self.stream.flush()
                out.at_line_start = True
        except Exception:
            pass
        super().emit(record)


class _ColorFormatter(logging.Formatter):
    """Formatter that colors the whole record by level when writing to a TTY.

    Colors are applied only when the stream is a terminal (``use_color``), so
    redirected/piped logs stay free of ANSI escape codes.
    """

    def __init__(self, fmt: str, use_color: bool) -> None:
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        color = _LEVEL_COLORS.get(record.levelno)
        if self.use_color and color:
            return f"{color}{text}{_RESET}"
        return text


def setup_logger(name: str = "ai_app", level: int = None) -> logging.Logger:
    """Set up and configure a logger.

    Operational diagnostics (model init, tool loading, summary generation,
    etc.) go through this logger to stderr and are **off by default** (level
    WARNING) so the chat UI stays clean; set ``LOG_LEVEL=INFO`` or
    ``LOG_LEVEL=DEBUG`` to surface them for troubleshooting. User-facing output
    (results, prompts, status the user asked for) should use ``print()``
    instead of this logger.

    Args:
        name: The name of the logger
        level: The logging level (defaults to WARNING, or the LOG_LEVEL env var)

    Returns:
        The configured logger
    """
    # Get log level from environment variable if not specified
    if level is None:
        log_level_str = os.getenv("LOG_LEVEL", "WARNING").upper()
        level = getattr(logging, log_level_str, logging.WARNING)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Create console handler and set level. The newline-guard handler ensures a
    # log line never lands inline with streamed chat output (which has no
    # trailing newline) — it prepends a newline when stdout is mid-line.
    if not logger.handlers:
        console_handler = _NewlineGuardHandler(sys.stderr)
        console_handler.setLevel(level)

        # Color the record by level on a TTY (red for errors, yellow for
        # warnings); stay plain when stderr is redirected to a file/pipe.
        use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        formatter = _ColorFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            use_color=use_color,
        )

        # Add formatter to handler
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

    # Suppress Brave Search client logs
    logging.getLogger("brave_search_python_client").setLevel(logging.WARNING)
    logging.getLogger("brave_search_python_client.boot").setLevel(logging.WARNING)

    return logger


# Create a default logger instance
logger = setup_logger()
