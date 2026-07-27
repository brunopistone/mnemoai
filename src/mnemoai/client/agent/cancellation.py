"""Cooperative cancel for a running turn (thread-safe helpers).

``agent._cancel_event`` (a ``threading.Event``) is set from the UI thread on
Esc/Ctrl+C; the blocking waits ``.wait()`` on it and poll it so a turn tears down
immediately instead of after a C-level blocking wait returns.

Functions take the agent as the first arg and read those fields ON the agent (so
``__init__``/``clear_messages`` are unchanged and bare ``__new__`` test stubs
still work via the getattr fallbacks). The agent keeps thin delegating methods —
the ``plan_policy``/``tool_formatting`` collaborator pattern.

This module was named ``steering.py`` until 1.8.0, when it also held the
**mid-turn steering queue** (``enqueue``/``drain``/``has_pending``/``clear``).
That queue was removed because the UI never fed it: a message submitted mid-turn
is queued FIFO and run as its own turn instead. Draining only at tool-round
boundaries could never be correct — a message typed during the final,
tool-call-free model call was never drained and leaked into the following turn —
so re-enabling it requires a different drain point, not a re-wire. See
``_execute_tools`` in agent.py for the full rationale.
"""


def request_cancel(agent) -> None:
    """Signal a cooperative cancel of the running turn (set the cancel event).

    Idempotent; no-op on a bare object with no event. The blocking waits
    ``.wait()`` on this event (waking instantly) and check it at each retry, so
    the turn tears down immediately instead of after a C-level wait returns."""
    ev = getattr(agent, "_cancel_event", None)
    if ev is not None:
        ev.set()


def is_cancelled(agent) -> bool:
    """True if a cooperative cancel was requested for this turn."""
    ev = getattr(agent, "_cancel_event", None)
    return ev is not None and ev.is_set()
