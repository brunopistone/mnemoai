"""Identify automated reviewer messages without treating them as user requests."""

import json

FEEDBACK_PREFIX = "[MnemoAI automated peer-review feedback; not a user request]"
DATA_MARKER = "\n\nReview data:\n"


class ReviewStopped(Exception):
    """A supervisor budget ended; stop new work without undoing completed work."""


def is_feedback(content):
    return isinstance(content, str) and content.startswith(FEEDBACK_PREFIX)


def feedback_summary(content):
    """Human-readable historical feedback, without replaying the control framing."""
    try:
        data = json.loads(content.split(DATA_MARKER, 1)[1])
        summary = str(data["summary"])
        return "".join(c for c in summary if c.isprintable() or c.isspace())[:1000]
    except (IndexError, KeyError, TypeError, ValueError, RecursionError):
        return "Automated peer-review feedback"
