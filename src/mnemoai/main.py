"""Main entry point for the LangGraph chat application."""

import argparse
import sys
from typing import Optional

from mnemoai.client.client import LangGraphClient
from mnemoai.client.ui.chat_interface import ChatInterface
from mnemoai.utils.configurator import config_exists, run_first_run_setup
from mnemoai.utils.console import print_error
from mnemoai.utils.paths import seed_example_files

# Global client reference for cleanup
_client: Optional[LangGraphClient] = None


def main(verbose: bool = False) -> None:
    """Initialize the application and start the chat loop.

    Args:
        verbose: Enable verbose mode to show thinking process

    Returns:
        None
    """
    global _client

    # Initialize LangGraph client (it launches the MCP server subprocess via
    # `python -m mnemoai.server.server`).
    _client = LangGraphClient(verbose=verbose)

    # Start client
    _client.start(verbose)

    # Create and run chat interface
    chat_interface = ChatInterface(_client)

    # Register cleanup function using chat interface method. Enable if you need to save conversation automatically on closure
    # atexit.register(lambda: chat_interface.client.save_conversation(chat_interface.chat_timestamp))

    chat_interface.run_chat_loop()


def cli() -> None:
    """Console-script entry point (used by the ``mnemoai`` command).

    Parses CLI args and starts the app. Kept zero-arg so it can be referenced
    as ``main:cli`` in pyproject's [project.scripts].
    """
    parser = argparse.ArgumentParser(
        prog="mnemoai", description="Mnemo AI — local agentic AI assistant"
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable verbose mode (hide thinking process)",
    )
    args = parser.parse_args()

    seed_example_files()

    if not config_exists() and sys.stdin.isatty():
        if run_first_run_setup() is not None:
            from mnemoai.utils.config import config

            config.reload()
        elif not config_exists():
            # Setup was declined/cancelled and there's still no config to run
            # with — exit cleanly rather than crashing deep in client init.
            print_error("No config available. Exiting.")
            return

    # Default is verbose=True, unless --no-verbose is specified
    verbose = not args.no_verbose
    main(verbose=verbose)


if __name__ == "__main__":
    cli()
