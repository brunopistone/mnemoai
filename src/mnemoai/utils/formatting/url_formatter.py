"""Utility for formatting clickable URLs in terminal output."""

import os
import re
from typing import Any, Callable

# An SGR (color/style) escape. Link formatting runs LAST in a span — emphasis and
# inline code are already ANSI by then — so these patterns must treat escapes as
# structure, not as text.
_SGR = r"\x1b\[[0-9;]*m"

# Markdown link `[text](url)`. Two guards, each a bug this pattern used to have:
# the lookbehind stops the opening bracket from being the `[` of an escape
# sequence (`**bold**` renders to `\x1b[1m…`, whose `[` was matched as a link
# start — the ESC was left stranded outside the replacement and the terminal
# printed a literal `1m`); and the display-text class admits whole SGR runs but
# no bare ESC, so emphasis *inside* link text still survives while a stray
# escape can no longer be absorbed as display text.
_MD_LINK = (
    r"(?<!\x1b)\[(?P<text>(?:" + _SGR + r"|[^\]\x1b])+)\]"
    r"\((?P<href>https?://[^\s)\x1b]+)\)"
)

# Plain URL. `\x1b` is excluded from BOTH classes: without it the trailing class
# swallowed the ESC of a following reset, so the match ran one byte past the URL
# and the terminal printed a literal `[0m`.
_PLAIN_URL = r'https?://[^\s<>"{}|\\^`\[\]\x1b]+[^\s<>"{}|\\^`\[\].,;:!?\x1b]'


def _format_links(
    text: str,
    on_markdown: Callable[[str, str], str],
    on_plain: Callable[[str], str],
    plain_guard: str = "",
) -> str:
    """Rewrite markdown links and plain URLs in ONE pass.

    A single alternation is what keeps a markdown link's URL from being rewritten
    twice: the link alternative starts at the `[`, so it consumes the whole
    `[text](url)` before the plain-URL alternative can see the URL inside it.
    Two sequential passes re-wrapped every markdown URL. ``plain_guard`` is a
    fixed-width lookbehind making a second call on already-formatted text a
    no-op.

    Args:
        text: Text containing URLs
        on_markdown: Called with (display text, url) for a markdown link
        on_plain: Called with the url for a bare URL
        plain_guard: Optional lookbehind prefixed to the plain-URL alternative

    Returns:
        Text with links rewritten
    """
    pattern = re.compile(
        r"(?:" + _MD_LINK + r")|(?P<url>" + plain_guard + _PLAIN_URL + r")"
    )

    def replace(match: Any) -> str:
        """Dispatch on which alternative matched."""
        href = match.group("href")
        if href is not None:
            return on_markdown(match.group("text"), href)
        return on_plain(match.group("url"))

    return pattern.sub(replace, text)


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

    return _format_links(
        text,
        lambda display, url: f"\033]8;;{url}\033\\{display}\033]8;;\033\\",
        lambda url: f"\033]8;;{url}\033\\{url}\033]8;;\033\\",
        # A URL already inside an OSC 8 hyperlink is not a link to re-wrap —
        # neither the target (after `]8;;`) nor the visible copy (after the ST
        # that closes the target). Two chained lookbehinds, since a single
        # alternation would need one fixed width.
        plain_guard=r"(?<!\033]8;;)(?<!\033\\)",
    )


def highlight_urls(text: str) -> str:
    """Highlight URLs with color for terminals that don't support hyperlinks.

    Args:
        text: Text containing URLs

    Returns:
        Text with highlighted URLs
    """

    return _format_links(
        text,
        # Light blue/cyan underlined for the display text and URL.
        lambda display, url: f"\033[36;4m{display}\033[0m (\033[36;4m{url}\033[0m)",
        lambda url: f"\033[36;4m{url}\033[0m",
        # A URL this function already highlighted is not re-highlighted.
        plain_guard=r"(?<!\033\[36;4m)",
    )


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
