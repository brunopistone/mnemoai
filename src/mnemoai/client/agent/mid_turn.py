"""A message the user sends while a turn is already running.

The default is FIFO: a submission waits for the running turn to finish and then
runs as its own turn. That is correct and it is also the wrong shape for the
common case — "also check the other file", "no, in Italian", "stop looking at
tests" — where the message is a CORRECTION of the work in flight. Waiting means
the turn spends minutes finishing the thing the user just redirected, and the
correction then arrives with the wrong work already done.

So the running turn gets first refusal: the message is folded into it at the next
**drain point** and the model addresses it without the turn ending.

Two drain points, and the second one is why the first attempt at this was retired
(1.8.0): draining only between tool rounds leaves the turn's FINAL, tool-call-free
model call uncovered, so a message typed during it was never drained and surfaced
inside an unrelated later turn. Hence

* after a tool round (``agent._execute_tools``), and
* at turn end (the ``deliver`` graph node, routed from ``_should_continue``),

plus an acceptance **window** that closes atomically at the last drain point: once
no drain point is left, ``accept`` refuses and the caller queues the message
normally. Every path therefore ends in delivery or in the caller taking the text
back (``reclaim``) — never in loss, never in a leak into a turn it wasn't meant
for.

Shape of what is delivered (all pending texts in ONE message, since the provider
adapters merge consecutive user messages and two would be indistinguishable
anyway): a ``<mid-turn-message>`` framing block, stripped again before storage, so
history keeps what the user actually typed. Functions take the agent as their
first argument like the other agent collaborators (``cancellation``, ``tool_loop``)
and tolerate a bare ``__new__`` stub that never ran ``__init__``.
"""

import re
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage

# The framing block. Public because the agent's ephemeral-block regex and
# _commit_turn both key on it, and a rename in one place would silently stop the
# stripping (the model-facing framing would then be stored as the user's words).
BLOCK_TAG = "mid-turn-message"

_FRAMING = (
    f"<{BLOCK_TAG}>\n"
    "The user sent this while you were working. Finish the step you are on, "
    "then address it — do not ignore it.\n"
    f"</{BLOCK_TAG}>\n"
)

_BLOCK_RE = re.compile(rf"\A\s*<{BLOCK_TAG}>.*?</{BLOCK_TAG}>\s*", re.DOTALL)


def open_window(agent) -> None:
    """Start accepting mid-turn messages for the turn that is starting.

    Also discards anything a previous turn left undelivered: the caller reclaims
    those (see :func:`reclaim`), so whatever is still here belongs to a turn that
    is over and must not be answered inside this one.
    """
    lock = getattr(agent, "_mid_turn_lock", None)
    if lock is None:
        agent._mid_turn_queue = []
        agent._mid_turn_open = True
        return
    with lock:
        agent._mid_turn_queue = []
        agent._mid_turn_open = True


def close(agent) -> None:
    """Stop accepting; keep whatever is pending for the caller to reclaim."""
    lock = getattr(agent, "_mid_turn_lock", None)
    if lock is None:
        agent._mid_turn_open = False
        return
    with lock:
        agent._mid_turn_open = False


def close_if_empty(agent) -> bool:
    """Close the window iff nothing is pending; True when it closed.

    Called at the LAST drain point, where "is anything pending" and "can anything
    still be accepted" must be answered in ONE critical section: a message
    accepted after this point would have no drain point left to ride on, which is
    exactly how the pre-1.8.0 version leaked one into the next turn.
    """
    lock = getattr(agent, "_mid_turn_lock", None)
    if lock is None:
        if getattr(agent, "_mid_turn_queue", None):
            return False
        agent._mid_turn_open = False
        return True
    with lock:
        if agent._mid_turn_queue:
            return False
        agent._mid_turn_open = False
        return True


def accept(agent, text: str) -> bool:
    """Offer `text` to the running turn; True once the turn owns it.

    False means the caller keeps it (no turn running, the window has closed, or
    the text is empty) and must queue it as a turn of its own.
    """
    text = (text or "").strip()
    if not text:
        return False
    lock = getattr(agent, "_mid_turn_lock", None)
    if lock is None:
        if not getattr(agent, "_mid_turn_open", False):
            return False
        agent._mid_turn_queue = list(getattr(agent, "_mid_turn_queue", [])) + [text]
        return True
    with lock:
        if not agent._mid_turn_open:
            return False
        agent._mid_turn_queue.append(text)
        return True


def has_pending(agent) -> bool:
    """True when a mid-turn message is waiting for a drain point."""
    return bool(getattr(agent, "_mid_turn_queue", None))


def drain(agent) -> List[BaseMessage]:
    """Take everything pending as ONE model-facing message (empty list if none).

    The caller places it AFTER every ``ToolMessage`` of the round it drains into:
    an unanswered ``tool_call_id`` makes the provider reject the next call.
    """
    texts = _take(agent)
    if not texts:
        return []
    notify = getattr(agent, "_on_mid_turn_delivered", None)
    if notify is not None:
        try:
            notify(list(texts))
        except Exception:
            pass  # a delivery notice is cosmetic; the message still goes through
    return [HumanMessage(content=_FRAMING + "\n\n".join(texts))]


def reclaim(agent) -> List[str]:
    """Take back what this turn never delivered, closing the window.

    The counterpart to ``accept``'s promise: a turn cancelled with Esc, or one
    that died, hands its pending text back so the caller can run it as its own
    turn instead of the user's words vanishing with the turn.
    """
    close(agent)
    return _take(agent)


def stored_text(message: BaseMessage) -> str:
    """The user's own text out of a delivered mid-turn message ("" if not one).

    How ``_commit_turn`` tells this ``HumanMessage`` — real conversation that
    exists nowhere else, produced INSIDE the turn — from the reminder-bearing
    prompt the turn was seeded with, which is already stored.
    """
    content = getattr(message, "content", "")
    if not isinstance(content, str) or f"<{BLOCK_TAG}>" not in content:
        return ""
    return _BLOCK_RE.sub("", content).strip()


def _take(agent) -> List[str]:
    """Pop the pending texts (lock-guarded when the agent has a lock)."""
    pending = getattr(agent, "_mid_turn_queue", None)
    if not pending:
        return []
    lock = getattr(agent, "_mid_turn_lock", None)
    if lock is None:
        agent._mid_turn_queue = []
        return list(pending)
    with lock:
        if not agent._mid_turn_queue:
            return []
        texts = agent._mid_turn_queue[:]
        agent._mid_turn_queue = []
    return texts
