"""Mid-turn steering queue + cooperative cancel (thread-safe helpers).

Two async-abort primitives the streaming waits and graph loop honor:

- **Steering** — messages the user types WHILE a turn runs are enqueued on
  ``agent._steer_queue`` (guarded by ``agent._steer_lock``) and drained at
  tool-round boundaries as wrapped ``HumanMessage``s, so the model addresses them
  without ending the turn.
- **Cancel** — ``agent._cancel_event`` (a ``threading.Event``) is set from the UI
  thread on Esc/Ctrl+C; the blocking waits ``.wait()`` on it and poll it so a
  turn tears down immediately instead of after a C-level blocking wait returns.

Functions take the agent as the first arg and read/mutate those fields ON the
agent (so ``__init__``/``clear_messages`` are unchanged and bare ``__new__`` test
stubs still work via the getattr fallbacks). The agent keeps thin delegating
methods — the ``plan_policy``/``tool_formatting`` collaborator pattern.
"""

from typing import List

from langchain_core.messages import BaseMessage, HumanMessage


def enqueue(agent, text: str) -> None:
    """Queue a mid-turn user message on the agent's steer queue. Thread-safe;
    ignores blank text. No-op-safe on a bare object without a lock."""
    text = (text or "").strip()
    if not text:
        return
    lock = getattr(agent, "_steer_lock", None)
    if lock is None:
        agent._steer_queue.append(text)
        return
    with lock:
        agent._steer_queue.append(text)


def drain(agent) -> List[BaseMessage]:
    """Pop all pending steering messages as wrapped ``HumanMessage``s (or []).

    The model treats each as a new user request to address after the current
    work rather than as narration. Consumed atomically so a concurrent enqueue
    isn't lost.
    """
    lock = getattr(agent, "_steer_lock", None)
    pending = getattr(agent, "_steer_queue", None)
    if not pending:
        return []
    if lock is not None:
        with lock:
            if not agent._steer_queue:
                return []
            texts = agent._steer_queue[:]
            agent._steer_queue = []
    else:
        texts = pending[:]
        agent._steer_queue = []
    return [
        HumanMessage(
            content=(
                "The user sent a new message while you were working:\n"
                f"{t}\n\n"
                "IMPORTANT: After finishing your current step, address the "
                "user's message above. Do not ignore it."
            )
        )
        for t in texts
    ]


def has_pending(agent) -> bool:
    """True if any mid-turn steering message is pending."""
    return bool(getattr(agent, "_steer_queue", None))


def clear(agent) -> None:
    """Discard all pending steering messages.

    Called when a turn is cancelled: a message steered into the cancelled turn
    must NOT leak into the next one (else the model answers a question the user
    meant for an aborted turn). Thread-safe.
    """
    lock = getattr(agent, "_steer_lock", None)
    if lock is not None:
        with lock:
            agent._steer_queue = []
    else:
        agent._steer_queue = []


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
