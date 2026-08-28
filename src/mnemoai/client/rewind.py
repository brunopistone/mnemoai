"""Take back the last exchange (`/rewind`).

Sometimes the prompt was the mistake: a wrong file, a wrong framing, a question
that sent the whole turn down a path you don't want in the context. `/clear`
throws the session away and `/compact` summarizes it; neither undoes one thing.
So this drops the last prompt **and everything the turn produced** from the live
conversation and from the transcript, leaving the conversation as it was the
moment before you pressed Enter.

**It moves the conversation only. Files on disk are never touched** — the same
boundary `/branch` draws, and the reason this is deliberately not a checkpoint /
restore feature: undoing an edit is `git` (or `/diff`) territory, and a command
that half-undid a turn — context rolled back, files not — would be worse than one
that says which half it does.

Two things it deliberately leaves alone, because both stay TRUE after a rewind:

* **The read-before-write gate** (`server/tools/read_state.py`) records that a
  file was read at mtime *T*. That is a staleness guard, not a record of what is
  in the context: the file is still at *T*, so the next write is still safe to
  allow. Un-recording the read would only make the model re-read a file it has
  correctly seen.
* **The file ledger** (`/files`) is a ledger, not a context inventory — "we
  looked at this" happened, and it stays happened. Its own report already says
  a listed file may have left the window.

What it cannot take back is what the session LEARNED from the turn: an episodic
memory entry, a playbook strategy, an auto-extracted `MEMORY.md` fact. Those
writes are lossy by construction — `store_episode` de-duplicates against what is
already there and `PlaybookStore.append` folds a repeat strategy into an existing
entry by raising its confidence, so there is no entry to delete, only a learned
signal to do arithmetic on. So the notice NAMES them instead (only the ones
actually enabled), and points at the one that is directly editable.
"""

from typing import Any, List, Optional

from langchain_core.messages import HumanMessage

from mnemoai.client.session_log import last_live_turn
from mnemoai.client.ui import turn_view
from mnemoai.utils.logger import logger

_GRAY = "\033[90m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

# Chars of the withdrawn prompt echoed back, so the user can see WHICH turn went.
_PREVIEW_CHARS = 72


def _text_of(content: Any) -> str:
    """Visible text from a message's content (string or provider block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and (b.get("type") == "text" or "text" in b)
        )
    return ""


def boundary(messages: List[Any]) -> int:
    """Index of the last message that is a prompt the USER actually typed (-1 if none).

    Pure. The same rule the reflector uses to find a turn boundary and the
    ``--resume`` picker uses to label a session: a ``role: user`` message can also
    be a tool result or an auto-delivered background sub-agent report, and the
    per-turn injections (steering, the plan reminder, the episodic block) are
    prefixes ON a real prompt rather than messages of their own — so what makes a
    message the start of a turn is that stripping every injection leaves
    something behind.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, HumanMessage):
            continue
        if turn_view.user_prompt_text(_text_of(getattr(msg, "content", ""))):
            return i
    return -1


def preview(message: Any, max_len: int = _PREVIEW_CHARS) -> str:
    """The withdrawn prompt, injection stripped, flattened and clipped."""
    text = turn_view.user_prompt_text(_text_of(getattr(message, "content", "")))
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def learning_sinks(client: Any) -> List[str]:
    """The learning stores a withdrawn turn may already have written to.

    Only what is actually running: naming a store the user disabled would invent
    a consequence, and the point of the line is that these are the parts a
    rewind can't reach.
    """
    out = []
    if getattr(client, "episodic_memory", None):
        out.append("episodic memory")
    if getattr(client, "playbook", None) and getattr(client, "reflector", None):
        out.append("the playbook")
    return out


def render(prompt: str, dropped: int, recorded: bool, sinks: List[str]) -> str:
    """The ``/rewind`` notice: what went, and what a rewind can't reach.

    Pure over its arguments (no client, no terminal), so it is unit-testable.
    """
    where = "conversation and transcript" if recorded else "conversation"
    out = [f"{_BOLD}⟲ withdrew your last prompt{_RESET}"]
    if prompt:
        out.append(f'  {_GRAY}"{prompt}"{_RESET}')
    out.append(
        f"  {_GRAY}{dropped} message{'s' if dropped != 1 else ''} dropped "
        f"from the {where}.{_RESET}"
    )
    out.append("")
    out.append(
        f"  {_GRAY}Files on disk are untouched — a rewind moves the "
        f"conversation only.{_RESET}"
    )
    if sinks:
        out.append(
            f"  {_GRAY}What the session learned from that turn stays "
            f"({', '.join(sinks)}).{_RESET}"
        )
    return "\n".join(out)


def withdraw(client: Any) -> str:
    """Drop the last exchange from live history + the transcript; returns the notice.

    Refuses rather than half-succeeds. The interesting refusal is a compaction:
    the graph state and the transcript are both append-only, so a summary that
    already stands for the withdrawn turn cannot be un-summarized — the turn
    would go from the message list while the context kept describing it.

    The transcript is kept in step two ways, because a withdrawn exchange is not
    always a turn of THIS session: a turn taken here is withdrawn by number,
    while one that arrived with a restored conversation (one ``restore`` blob) is
    recorded as the surviving history — see :meth:`SessionLog.log_rewind`.
    """
    agent = getattr(client, "agent", None)
    messages = list(getattr(agent, "messages", None) or []) if agent else []
    if not messages:
        return "Nothing to rewind — this conversation has no turns yet."

    index = boundary(messages)
    if index < 0:
        return (
            "Nothing to rewind — no prompt of yours is left in the live "
            "conversation (compaction summarized them all away)."
        )

    # The transcript knows whether a compaction landed on this turn; without one
    # (SESSION_MAX_AGE_DAYS: 0) there is no file to keep in step either, so the
    # live truncation is all there is to do.
    log = getattr(agent, "session_log", None)
    target: Optional[dict] = None
    if log is not None and getattr(log, "path", None) is not None:
        try:
            target = last_live_turn(log.path)
        except Exception as e:  # noqa: BLE001 — an unreadable log is not fatal
            logger.debug(f"Could not read the last logged turn: {e}")
    if target and target.get("compacted"):
        return (
            "Can't rewind that turn — the conversation compacted during or right "
            "after it, so the summary standing in for it can't be undone.\n"
            f"  {_GRAY}/compact and /rewind are the two ways to shrink the "
            f"context; only one of them is reversible.{_RESET}"
        )

    prompt = preview(messages[index])
    dropped = len(messages) - index
    del agent.messages[index:]

    recorded = False
    if log is not None and getattr(log, "path", None) is not None:
        try:
            # A turn of THIS session is withdrawn by number; an exchange that came
            # in with a restored conversation has no record of its own, so the
            # surviving history is pinned instead — else the withdrawal would hold
            # for this run and come back at the next --resume.
            if target:
                recorded = bool(log.log_rewind(target["n"]))
            else:
                recorded = bool(log.log_rewind(kept=agent.messages))
        except Exception as e:  # noqa: BLE001 — a log must never break a command
            logger.debug(f"Session log rewind write failed: {e}")

    # The cached provider token count measured the history that just shrank, and
    # the legacy episodic path would otherwise store the withdrawn exchange as an
    # episode on the NEXT prompt (it evaluates the previous one, delayed).
    client._forget_context_size()
    client.previous_query = None
    client.previous_response = None
    client.previous_messages = None

    return render(prompt, dropped, recorded, learning_sinks(client))
