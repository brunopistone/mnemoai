"""What a failed turn LEAVES BEHIND in history (pure logic).

The marker that closes a dead turn out lives here, away from the several paths
that can report one, so they can't drift.

It exists because history is otherwise left ending on a dangling user message.
Nothing recorded that the turn produced no answer — so the transcript read as if
the question had been answered, and, worse, the provider adapters MERGE
consecutive user messages (``langchain-aws`` joins two ``HumanMessage``s into a
single Bedrock ``user`` block, texts separated by a newline), which makes the
failed prompt ride along silently as a prefix of the NEXT question. An explicit
assistant marker separates the two, exactly as ``INTERRUPTED_MARKER`` does for a
cancel.

It is deliberately SHORT and carries only the exception's CLASS name, because it
is sent to the model like any other assistant turn. The provider's own error
prose is a different thing entirely — it can be a whole JSON body — and belongs
on screen and in the log file, never in the context of every later turn.
"""

from typing import Union

# One sentence, one class name — see the module docstring on why it stays small.
_MARKER_PREFIX = "[Turn failed before it produced an answer"


def failure_marker(exc: Union[BaseException, str]) -> str:
    """The assistant message that closes out a turn which produced nothing.

    ``exc`` may be the exception itself or an already-resolved class name, for a
    caller that can name the failure better than the exception which escaped.
    """
    name = exc if isinstance(exc, str) else type(exc).__name__
    name = (name or "").strip()
    return f"{_MARKER_PREFIX}: {name}.]" if name else f"{_MARKER_PREFIX}.]"


def is_failure_marker(text: object) -> bool:
    """True if ``text`` is one of our failure markers (bookkeeping, not an answer)."""
    return isinstance(text, str) and text.strip().startswith(_MARKER_PREFIX)
