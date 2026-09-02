"""What a failed turn LEAVES BEHIND and what the user is told to do (pure logic).

Two facts about a turn that died live here so the several paths reporting one
can't drift: the **marker** that closes the turn out in history, and the
**recovery** the user is pointed at.

The marker exists because history is otherwise left ending on a dangling user
message. Nothing recorded that the turn produced no answer — so the transcript
read as if the question had been answered, and, worse, the provider adapters
MERGE consecutive user messages (``langchain-aws`` joins two ``HumanMessage``s
into a single Bedrock ``user`` block, texts separated by a newline), which makes
the failed prompt ride along silently as a prefix of the NEXT question. An
explicit assistant marker separates the two, exactly as ``INTERRUPTED_MARKER``
does for a cancel.

It is deliberately SHORT and carries only the exception's CLASS name, because it
is sent to the model like any other assistant turn. The provider's own error
prose is a different thing entirely — it can be a whole JSON body — and belongs
on screen and in the log file, never in the context of every later turn.

The recovery is **named, not implied**. "An error I can't recover from
automatically" is true and useless: the commands that actually resolve each class
already exist — ``/compact`` when the prompt outgrew the model, ``/rewind`` to
take back the exchange the provider rejected, ``/model`` when this model is the
one refusing — and nothing pointed at any of them.
"""

from typing import Union

from mnemoai.client.agent import stream_policy

# One sentence, one class name — see the module docstring on why it stays small.
_MARKER_PREFIX = "[Turn failed before it produced an answer"

# Provider phrasings for "I understood the request and will not serve it": the
# content itself was rejected, so repeating it verbatim fails identically. Kept
# separate from stream_policy's tuples, which answer a different question (is
# another attempt worth making) — here nothing is retried and the point is which
# command gets the user unstuck.
_REJECTED_MARKERS = (
    "validation",  # ValidationException (Bedrock) — a malformed/unacceptable body
    "invalid_request",
    "invalid request",
    "refusal",
    "usage policy",
    "content_filter",
    "content filter",
    "unsupported",
    "malformed",
)

# What each class is called; the wording for each lives in `recovery_advice`.
OVERSIZED = "oversized"
REJECTED = "rejected"
CONNECTION = "connection"
UNKNOWN = "unknown"


def failure_marker(exc: Union[BaseException, str]) -> str:
    """The assistant message that closes out a turn which produced nothing.

    ``exc`` may be the exception or an already-resolved class name (the
    diagnostic probe can name the failure better than the exception that escaped).
    """
    name = exc if isinstance(exc, str) else type(exc).__name__
    name = (name or "").strip()
    return f"{_MARKER_PREFIX}: {name}.]" if name else f"{_MARKER_PREFIX}.]"


def is_failure_marker(text: object) -> bool:
    """True if ``text`` is one of our failure markers (bookkeeping, not an answer)."""
    return isinstance(text, str) and text.strip().startswith(_MARKER_PREFIX)


def classify(exc: Exception) -> str:
    """Which kind of dead end this is — the input to :func:`recovery_advice`.

    Order matters: an overflow and a rejection are both deterministic, and an
    oversized prompt is the more specific (and more actionable) reading of the
    two, so it is tested first.
    """
    if stream_policy.is_context_overflow_error(exc):
        return OVERSIZED
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(m in text for m in _REJECTED_MARKERS):
        return REJECTED
    if stream_policy.is_transient_network_error(exc):
        return CONNECTION
    return UNKNOWN


def recovery_advice(exc: Exception) -> str:
    """One sentence naming the command(s) that resolve this failure, or ``""``.

    Empty for a connection failure: the caller's own wording ("just send your
    message again") already IS the recovery, and a list of commands beside it
    would suggest the conversation needs repairing when it doesn't.

    Command names are written bare rather than in backticks — this text is shown
    both through the markdown renderer and as a plain line, and only one of the
    two would render the ticks away.
    """
    kind = classify(exc)
    if kind == OVERSIZED:
        return (
            "The conversation no longer fits this model's context window: "
            "/compact summarizes the older turns, or /rewind takes back the last "
            "exchange."
        )
    if kind == REJECTED:
        return (
            "The model rejected the conversation's contents rather than failing "
            "to reach it, so sending the same thing again will fail the same way: "
            "/rewind takes back the last exchange, /compact drops older messages "
            "from the prompt if one of those is the cause, and /model switches to "
            "a different model."
        )
    if kind == CONNECTION:
        return ""
    return (
        "Your conversation is intact — try again, and if it keeps happening "
        "/rewind takes back the last exchange."
    )
