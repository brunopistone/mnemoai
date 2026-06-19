"""Logging utilities for the AI application."""

import logging
import sys
import os


def setup_logger(name: str = "ai_app", level: int = None) -> logging.Logger:
    """Set up and configure a logger.

    Args:
        name: The name of the logger
        level: The logging level (defaults to INFO, or LOG_LEVEL env var)

    Returns:
        The configured logger
    """
    # Get log level from environment variable if not specified
    if level is None:
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, log_level_str, logging.INFO)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Create console handler and set level
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
