"""Main entry point for the LangGraph chat application."""

import argparse
import os
from typing import Optional
from client.client import LangGraphClient
from client.ui.chat_interface import ChatInterface

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

    # Get the absolute path to the server.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(current_dir, "server", "server.py")

    # Initialize LangGraph client
    _client = LangGraphClient(server_path=server_path, verbose=verbose)

    # Start client
    _client.start(verbose)

    # Create and run chat interface
    chat_interface = ChatInterface(_client)

    # Register cleanup function using chat interface method. Enable if you need to save conversation automatically on closure
    # atexit.register(lambda: chat_interface.client.save_conversation(chat_interface.chat_timestamp))

    chat_interface.run_chat_loop()


def cli() -> None:
    """Console-script entry point (used by the ``personal-ai-assistant`` command).

    Parses CLI args and starts the app. Kept zero-arg so it can be referenced
    as ``main:cli`` in pyproject's [project.scripts].
    """
    parser = argparse.ArgumentParser(description="AI Chat Application")
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable verbose mode (hide thinking process)",
    )
    args = parser.parse_args()

    # Default is verbose=True, unless --no-verbose is specified
    verbose = not args.no_verbose
    main(verbose=verbose)


if __name__ == "__main__":
    cli()
