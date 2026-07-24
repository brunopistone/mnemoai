"""Streaming retry/error classification policy (pure logic).

Decides — from an exception's text alone — whether a streaming model call failed
because the prompt overflowed the context window (the caller must compact, never
retry) or because of a transient connection/5xx failure (retry on a fresh
connection with backoff), and computes the backoff delay. Provider-agnostic:
matches phrasings rather than exception types, so it works across every
LangChain provider stack (httpx/requests/boto3). Pure functions with no agent
state; the agent keeps thin delegating methods over them and re-exports the
marker tuples as class attributes (the ``plan_policy`` pattern).

``network_retry_delay`` and ``sleep_or_cancel`` take their config-derived inputs
(base/factor) and the cancel event as explicit arguments, so the agent-side
config read stays at its module (where tests patch it) and this module needs no
config import.
"""

import time

# Provider phrasings for "the prompt exceeded the model's context window".
CONTEXT_OVERFLOW_MARKERS = (
    "prompt is too long",              # Anthropic / Bedrock Mantle
    "context length",                  # OpenAI-compatible
    "maximum context",
    "context window",
    "too many tokens",
    "model_context_window_exceeded",   # Bedrock/Converse stop reason
    "input is too long",
    "exceeds the maximum",
)
# Substrings marking a transient connection failure — a dropped/dead socket
# (laptop sleep), a reset, or a server-side 5xx/overload — that a fresh retry
# can recover, as opposed to a deterministic 4xx the same request would repeat.
TRANSIENT_NETWORK_MARKERS = (
    "connection reset",
    "connection aborted",
    "connection error",
    "econnreset",
    "epipe",
    "broken pipe",
    "etimedout",
    "timed out",
    "timeout",
    "read timed out",
    "server disconnected",
    "connection closed",
    "peer closed connection",
    "remotedisconnected",
    "incomplete read",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "overloaded",
    "overloaded_error",
    "internal server error",
    "api_error",
    "502",
    "503",
    "504",
    "529",
)


def is_context_overflow_error(exc: Exception) -> bool:
    """True if ``exc`` is a context-window-exceeded error (not a generic 400).

    Matches the provider phrasings so the backstop can compact + terminate
    instead of retrying the same oversized prompt in a loop.
    """
    text = str(exc).lower()
    return any(m in text for m in CONTEXT_OVERFLOW_MARKERS)


def is_transient_network_error(exc: Exception) -> bool:
    """True if ``exc`` looks like a transient connection/network failure worth
    retrying on a fresh connection (dead socket, reset, timeout, 5xx/overload).

    Kept provider-agnostic (matches the exception text) so it works for every
    LangChain provider — a dead socket surfaces differently per httpx/requests/
    boto3 stack but the phrasings above cover them."""
    text = str(exc).lower()
    return any(m in text for m in TRANSIENT_NETWORK_MARKERS)


def network_retry_delay(attempt: int, base: float, factor: float) -> float:
    """Exponential backoff (seconds) for a network-error stream retry, capped so
    a sleep-recovery retry never waits absurdly long. ``base``/``factor`` are the
    caller's LLM.RETRY_DELAY / RETRY_BACKOFF knobs (read at the call site)."""
    return min(base * (factor ** attempt), 30.0)


def sleep_or_cancel(cancel_event, delay: float) -> bool:
    """Sleep up to ``delay`` seconds, waking early if ``cancel_event`` is set.

    Returns True if cancelled during the wait, False if the full delay elapsed.
    Uses the event's ``wait`` (interruptible) instead of ``time.sleep`` (a C-level
    block the async KeyboardInterrupt can't preempt). Falls back to ``time.sleep``
    when there's no event (a bare object)."""
    if cancel_event is None:
        time.sleep(delay)
        return False
    return cancel_event.wait(delay)
