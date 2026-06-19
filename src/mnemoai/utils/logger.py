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

    # Create console handler and set level
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stderr)
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
