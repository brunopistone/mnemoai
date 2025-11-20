"""Main entry point for the Strands chat application."""

import argparse
import os
from typing import Optional
from client.client import StrandsClient
from client.ui.chat_interface import ChatInterface

# Global client reference for cleanup
_client: Optional[StrandsClient] = None


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

    # Initialize Strands client
    _client = StrandsClient(server_path=server_path, verbose=verbose)

    # Start client
    _client.start(verbose)

    # Create and run chat interface
    chat_interface = ChatInterface(_client)

    # Register cleanup function using chat interface method. Enable if you need to save conversation automatically on closure
    # atexit.register(lambda: chat_interface.client.save_conversation(chat_interface.chat_timestamp))

    chat_interface.run_chat_loop()


if __name__ == "__main__":
    # Parse command line arguments
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
