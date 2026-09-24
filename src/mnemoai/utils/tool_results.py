"""Recognize explicit tool failures without classifying document text as errors."""

import json
from typing import Any

_EXIT_CODES = ("exit_status", "return_code", "returncode", "exit_code")
_FAILURE_PREFIXES = (
    "error:",
    "error executing ",
    "error reading ",
    "error writing ",
    "error listing ",
    "error searching ",
    "error parsing ",
    "blocked:",
    "user declined",
    "tool not found:",
    "memory not updated:",
)


def is_error_result(result: Any) -> bool:
    """Read native message status, structured payloads, and explicit error notices."""
    if getattr(result, "status", None) == "error":
        return True
    if hasattr(result, "content"):
        result = result.content
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (ValueError, TypeError):
            return result.lstrip().lower().startswith(_FAILURE_PREFIXES)
    if not isinstance(result, dict):
        return False
    if result.get("error") or result.get("isError") or result.get("blocked"):
        return True
    if result.get("success") is False or result.get("status") in ("error", "failed"):
        return True
    for key in _EXIT_CODES:
        if result.get(key) is not None:
            try:
                return int(result[key]) != 0
            except (TypeError, ValueError):
                continue
    return False
