"""Provider retry/error classification policy (pure logic).

Decides — from an exception's text alone — whether a model call failed because
the prompt overflowed the context window (the caller must compact, never retry)
or because of a transient connection/5xx failure (retry on a fresh connection
with backoff), and computes the backoff delay. Provider-agnostic: matches
phrasings rather than exception types, so it works across every LangChain
provider stack (httpx/requests/boto3). Pure functions with no agent state; the
agent keeps thin delegating methods over them and re-exports the marker tuples
as class attributes (the ``plan_policy`` pattern).

Named for the streamed turn it was written for, it now covers every provider
call: ``call_with_transient_retry`` / ``acall_with_transient_retry`` drive the
NON-streamed auxiliary calls too — route classification, task decomposition, the
compaction summary. Each of those has a graceful fallback, which is exactly why
they had no retry at all and quietly took the fallback on the first 529 while the
streamed turn beside them recovered on its second attempt.

``network_retry_delay`` and ``sleep_or_cancel`` take their config-derived inputs
(base/factor) and the cancel event as explicit arguments, so the agent-side
config read stays at its module (where tests patch it) and this module needs no
config import.
"""

import asyncio
import random
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
    # botocore's own wording for the same failures, which its exceptions spell out
    # in prose rather than reusing the phrasings above. Only ReadTimeoutError and
    # ConnectTimeoutError contain a marker already ("timeout"), so without these
    # FIVE of its seven transport errors read as deterministic and got NO retry —
    # including ConnectionClosedError ("Connection WAS closed …", which the
    # "connection closed" marker above does not substring-match), the way a
    # dropped Bedrock converse-stream socket actually surfaces.
    "connection was closed",              # ConnectionClosedError
    "could not connect",                  # EndpointConnectionError
    "failed to establish a connection",   # ConnectionError
    "reading from response stream",       # ResponseStreamingError
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

# Fraction of the computed delay added at random. An overloaded provider rejects
# every concurrent caller within milliseconds of the others (orchestrator waves,
# spawned sub-agents, the compaction map all fan out), so a purely deterministic
# backoff makes them retry in lockstep and re-collide.
RETRY_JITTER = 0.25
# Attempt cap for an auxiliary call. Each has a working fallback, so a quick
# second chance beats a long stall — unlike the streamed turn, where a give-up
# costs the user the whole answer and the full LLM.MAX_RETRIES budget is right.
AUX_RETRY_ATTEMPTS = 3
# A provider's own retry-after is authoritative but must not park a turn forever.
_RETRY_AFTER_CAP = 60.0
# Grace added on top of the request timeout when deriving the first-token window,
# so a request that really did exceed its budget fails with the provider's own
# specific error rather than our generic "no data" one.
_FIRST_TOKEN_GRACE = 30.0


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


def network_retry_delay(
    attempt: int,
    base: float,
    factor: float,
    jitter: float = 0.0,
    rand=random.random,
) -> float:
    """Exponential backoff (seconds) for a network-error retry, capped so a
    sleep-recovery retry never waits absurdly long. ``base``/``factor`` are the
    caller's LLM.RETRY_DELAY / RETRY_BACKOFF knobs (read at the call site).

    ``jitter`` adds up to that fraction of the delay at random, spreading callers
    an overloaded provider rejected simultaneously. Applied AFTER the cap (a
    capped delay is still a collision point) and defaulting to 0 so the function
    stays pure/deterministic unless a caller asks for spread.
    """
    delay = min(base * (factor ** attempt), 30.0)
    if jitter:
        delay += delay * jitter * rand()
    return delay


def first_token_timeout(
    idle: float, request_timeout: float, grace: float = _FIRST_TOKEN_GRACE
) -> float:
    """Seconds to allow before a stream's FIRST chunk, from the per-chunk ``idle``
    watchdog and the configured ``LLM.REQUEST_TIMEOUT``.

    The idle watchdog is for a stream that goes silent once RUNNING; the wait for
    the first token is a different thing entirely — the provider spends prefill +
    reasoning on the whole prompt before emitting a byte, which on a large context
    is minutes, not seconds (a ~440k-token turn at max effort measured ~123s).
    Policing that window with the per-chunk timeout aborts a perfectly healthy
    turn, and each retry then re-sends the same oversized prompt and re-pays the
    same doomed wait, so the turn can never complete — the same trap as botocore's
    60s default ``read_timeout``, one layer up.

    Derived from ``REQUEST_TIMEOUT`` (already documented as the knob that must
    exceed the largest expected time-to-first-byte) rather than a new config key,
    so the fix reaches installs whose ``config.yaml`` will never mention it. Never
    shorter than ``idle``, so raising the watchdog raises this too.
    """
    if request_timeout and request_timeout > 0:
        return max(idle, request_timeout + grace)
    return idle


def aux_attempts(max_retries, cap: int = AUX_RETRY_ATTEMPTS) -> int:
    """Attempt count for an auxiliary call: the configured LLM.MAX_RETRIES,
    clamped to ``cap`` (and to at least one attempt — the call itself)."""
    try:
        configured = int(max_retries)
    except (TypeError, ValueError):
        configured = cap
    return max(1, min(cap, configured))


def retry_after_seconds(exc: Exception, cap: float = _RETRY_AFTER_CAP):
    """The provider's own ``retry-after`` hint for ``exc`` in seconds, or None.

    Reads the two shapes the supported stacks produce — an httpx/requests
    response object with a header mapping (the Anthropic/OpenAI SDKs) and
    botocore's response dict — and ignores anything that isn't a positive number:
    the HTTP-date form is legal but rare here, and guessing across a clock skew is
    worse than falling back to the backoff.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None and isinstance(response, dict):
        headers = (response.get("ResponseMetadata") or {}).get("HTTPHeaders")
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    raw = getter("retry-after")
    if raw is None:
        raw = getter("Retry-After")  # plain dicts aren't case-insensitive
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    return min(seconds, cap) if seconds > 0 else None


def transient_retry_delay(
    exc: Exception,
    attempt: int,
    base: float,
    factor: float,
    jitter: float = RETRY_JITTER,
) -> float:
    """Seconds to wait before retrying ``exc``: the provider's own retry-after
    when it sent one (it knows its own load), else jittered backoff."""
    hinted = retry_after_seconds(exc)
    if hinted is not None:
        return hinted
    return network_retry_delay(attempt, base, factor, jitter=jitter)


def retry_notice(
    label: str, exc: Exception, delay: float, attempt: int, attempts: int
) -> str:
    """One wording for every retry log line, so the call sites read alike without
    this module having to import the logger."""
    return (
        f"{label} hit a transient provider error ({exc}); retrying in "
        f"{delay:.1f}s (attempt {attempt}/{attempts})"
    )


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


def call_with_transient_retry(
    call,
    attempts: int,
    base: float,
    factor: float,
    cancel_event=None,
    on_retry=None,
):
    """Run ``call()``, retrying only a transient provider failure, and re-raise
    anything else immediately.

    For the non-streamed auxiliary calls (route classification, task
    decomposition). The last transient failure is re-raised too, so the caller's
    existing fallback still runs — a 529 storm degrades gracefully instead of
    aborting the turn. ``on_retry(exc, delay, attempt, attempts)`` reports each
    wait (this module logs nothing itself); the backoff waits on
    ``cancel_event`` so Esc/Ctrl+C wakes it, and a cancel aborts.
    """
    attempts = max(1, attempts)
    for attempt in range(attempts):
        try:
            return call()
        except Exception as e:
            if attempt >= attempts - 1 or not is_transient_network_error(e):
                raise
            delay = transient_retry_delay(e, attempt, base, factor)
            if on_retry is not None:
                on_retry(e, delay, attempt + 1, attempts)
            if sleep_or_cancel(cancel_event, delay):
                raise KeyboardInterrupt("cancelled during retry backoff") from e


async def acall_with_transient_retry(
    call,
    attempts: int,
    base: float,
    factor: float,
    on_retry=None,
):
    """Async twin of :func:`call_with_transient_retry` for ``ainvoke`` callers
    (the compaction summary map/reduce).

    No cancel event: an async caller is cancelled through
    ``asyncio.CancelledError``, a BaseException the ``except Exception`` here
    deliberately lets through.
    """
    attempts = max(1, attempts)
    for attempt in range(attempts):
        try:
            return await call()
        except Exception as e:
            if attempt >= attempts - 1 or not is_transient_network_error(e):
                raise
            delay = transient_retry_delay(e, attempt, base, factor)
            if on_retry is not None:
                on_retry(e, delay, attempt + 1, attempts)
            await asyncio.sleep(delay)
