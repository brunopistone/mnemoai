"""Formatting utilities for console output and text processing."""

from .response_parser import *

# Deliberate public re-export (documented in ARCHITECTURE.md) — kept so
# `from mnemoai.utils.formatting import make_urls_clickable` keeps working. No
# `__all__` here: it would also narrow the `response_parser` star re-export above.
from .url_formatter import make_urls_clickable  # noqa: F401
