"""Cooperative cancel for a running turn (thread-safe helpers).

``agent._cancel_event`` (a ``threading.Event``) is set from the UI thread on
Esc/Ctrl+C; the blocking waits ``.wait()`` on it and poll it so a turn tears down
immediately instead of after a C-level blocking wait returns.

Functions take the agent as the first arg and read those fields ON the agent (so
``__init__``/``clear_messages`` are unchanged and bare ``__new__`` test stubs
still work via the getattr fallbacks). The agent keeps thin delegating methods —
the ``plan_policy``/``tool_formatting`` collaborator pattern.

This module was named ``steering.py`` until 1.8.0, when it also held the queue for
messages the user sends mid-turn. Cancelling and redirecting are two different
things, so they are two modules now: that queue lives in ``mid_turn.py``, which
drains at tool-round boundaries **and** at turn end — the missing second drain
point being why the first version was removed rather than re-wired.

The two still meet at one point: a cancel must not leave a mid-turn message
half-owned. ``is_cancelled`` ends the graph immediately, so the turn stops before
its next drain point and the UI reclaims the undelivered text
(``agent.reclaim_mid_turn``) as its own turn — a message the user steered into a
cancelled turn must neither vanish nor surface inside an unrelated later one.
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
