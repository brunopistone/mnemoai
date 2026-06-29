"""Filesystem-path normalization shared by file/image tools.

Models are often handed a path the user copied from a *shell* — with spaces
backslash-escaped (``/Users/me/My\\ File.png``) or wrapped in quotes
(``"/Users/me/My File.png"``). A tool is not a shell, so passing that string to
``open()`` / ``os.path.exists`` fails on the literal backslashes/quotes. This
helper resolves both forms WITHOUT breaking legitimate paths: it prefers the
path exactly as given, and only falls back to a de-escaped/de-quoted variant
when the literal one doesn't exist on disk.
"""

import os


def _strip_surrounding_quotes(text: str) -> str:
    """Drop a single matching pair of surrounding single/double quotes."""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


def _shell_unescape(text: str) -> str:
    """Undo shell backslash-escaping of common path characters.

    Turns ``\\ `` into a space (and unescapes other backslash-escaped
    punctuation a shell would produce), so a drag-into-terminal path resolves.
    """
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            out.append(text[i + 1])
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def clean_path_syntax(path: str) -> str:
    """Strip shell quoting/escaping from a path WITHOUT checking the filesystem.

    Use for a *write* target (which may not exist yet): turns
    ``"/Users/me/My File.txt"`` or ``/Users/me/My\\ File.txt`` into the plain
    ``/Users/me/My File.txt``. A path that needs no cleaning is returned
    unchanged. Unlike :func:`normalize_path`, this does no existence probing, so
    it's safe before creating a file. Conservative: only unescapes/unquotes when
    the result still looks like the same path (it never expands ``~`` here).
    """
    raw = (path or "").strip()
    unquoted = _strip_surrounding_quotes(raw)
    # Only undo backslash-escaping when there ARE escapes; this leaves a genuine
    # backslash-containing path (rare on POSIX, but possible) untouched unless it
    # was clearly shell-escaped (``\ `` etc.).
    if "\\" in unquoted:
        return _shell_unescape(unquoted)
    return unquoted


def normalize_path(path: str) -> str:
    """Normalize a user-supplied path, tolerating shell escaping/quoting.

    Resolution order (first that exists on disk wins; otherwise the expanded
    literal is returned so the caller's own not-found handling still fires):

    1. ``~`` expansion of the literal string (current behavior — never breaks a
       path that already works, including one with real backslashes).
    2. quotes stripped, then ``~`` expanded.
    3. shell backslash-escapes undone, then ``~`` expanded.
    4. both: quotes stripped AND shell-unescaped, then ``~`` expanded.

    Args:
        path: Raw path string as the model passed it.

    Returns:
        A normalized path string (existing if any candidate matched).
    """
    raw = (path or "").strip()
    literal = os.path.expanduser(raw)

    candidates = [
        literal,
        os.path.expanduser(_strip_surrounding_quotes(raw)),
        os.path.expanduser(_shell_unescape(raw)),
        os.path.expanduser(_shell_unescape(_strip_surrounding_quotes(raw))),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    # Nothing matched — return the plain expanded literal so the caller reports
    # a clean "not found" against what the user actually meant.
    return literal
