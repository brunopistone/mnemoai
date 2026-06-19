"""Utility for formatting clickable URLs in terminal output."""

import os
import re
from typing import Any


def make_urls_clickable(text: str) -> str:
    """Convert URLs in text to clickable terminal hyperlinks.

    Args:
        text: Text containing URLs

    Returns:
        Text with clickable URLs
    """

    # Check if terminal supports hyperlinks
    term_program = os.environ.get("TERM_PROGRAM", "")
    supports_hyperlinks = (
        term_program in ["iTerm.app", "vscode"] or "ITERM" in os.environ
    )

    if not supports_hyperlinks:
        # For terminals that don't support hyperlinks, just highlight URLs
        return highlight_urls(text)

    # First handle markdown links [text](url)
    markdown_pattern = r"\[([^\]]+)\]\((https?://[^\s)]+)\)"

    def format_markdown_url(match: Any) -> str:
        """Format markdown link as clickable hyperlink.

        Args:
            match: Regex match object

        Returns:
            Formatted hyperlink string
        """
        display_text = match.group(1)
        url = match.group(2)
        # ANSI escape sequence for hyperlinks
        return f"\033]8;;{url}\033\\{display_text}\033]8;;\033\\"

    # Replace markdown links first
    text = re.sub(markdown_pattern, format_markdown_url, text)

    # Then handle plain URLs that aren't already formatted
    url_pattern = (
        r'(?<!\033]8;;)https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;:!?](?!\033\\)'
    )

    def format_plain_url(match: Any) -> str:
        """Format plain URL as clickable hyperlink.

        Args:
            match: Regex match object

        Returns:
            Formatted hyperlink string
        """
        url = match.group(0)
        return f"\033]8;;{url}\033\\{url}\033]8;;\033\\"

    text = re.sub(url_pattern, format_plain_url, text)

    return text


def highlight_urls(text: str) -> str:
    """Highlight URLs with color for terminals that don't support hyperlinks.

    Args:
        text: Text containing URLs

    Returns:
        Text with highlighted URLs
    """

    # First handle markdown links [text](url)
    markdown_pattern = r"\[([^\]]+)\]\((https?://[^\s)]+)\)"

    def format_markdown_url(match: Any) -> str:
        """Format markdown link with color highlighting.

        Args:
            match: Regex match object

        Returns:
            Formatted colored string
        """
        display_text = match.group(1)
        url = match.group(2)
        # Light blue/cyan underlined for the display text and URL
        return f"\033[36;4m{display_text}\033[0m (\033[36;4m{url}\033[0m)"

    # Replace markdown links first
    text = re.sub(markdown_pattern, format_markdown_url, text)

    # Then handle plain URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;:!?]'

    def format_plain_url(match: Any) -> str:
        """Format plain URL with color highlighting.

        Args:
            match: Regex match object

        Returns:
            Formatted colored string
        """
        url = match.group(0)
        # Light blue/cyan underlined for URLs (36 = cyan, 4 = underline)
        return f"\033[36;4m{url}\033[0m"

    text = re.sub(url_pattern, format_plain_url, text)

    return text


def format_url(url: str, display_text: str = None) -> str:
    """Format a single URL as clickable terminal hyperlink.

    Args:
        url: URL to format
        display_text: Optional display text (defaults to URL)

    Returns:
        Formatted clickable URL
    """
    if display_text is None:
        display_text = url
    return f"\033]8;;{url}\033\\{display_text}\033]8;;\033\\"
