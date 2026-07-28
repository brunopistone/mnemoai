"""Append-only per-directory session transcripts, for ``--resume``.

Sessions are scoped to the directory the app was launched from, so ``--resume``
in a project offers that project's sessions and nothing else
(:func:`mnemoai.utils.paths.sessions_dir`). One file per session, one JSON object
per line:

    {"t": "meta",  "session_id": …, "cwd": …, "started": …, "model": …}
    {"t": "turn",  "n": 1, "ts": …, "messages": [ …strands dicts… ]}
    {"t": "compact", "n": 3, "ts": …}

**Why append-only instead of mirroring the live message list:** compaction
*replaces* ``agent.messages`` wholesale (it summarizes older turns away), so a
log that mirrored the list would lose that history the moment it compacted — and
resuming would restore a summarized stub rather than the conversation. The log is
therefore the record of what was *said*, written once per turn and never
rewritten; a ``compact`` marker records that the live context was shrunk, without
discarding the transcript.

Sub-agent runs are deliberately NOT logged: they execute on their own isolated
message list and never enter ``agent.messages``, so writing them here would
interleave a second conversation into the chain.

Distinct from ``/save``: that writes a single user-curated snapshot to
``conversations/`` which is never swept. These logs are automatic and expire
(:data:`mnemoai.utils.paths.SESSION_MAX_AGE_DAYS`).
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.agent.message_codec import (
    convert_langchain_messages_to_strands,
)
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import sessions_dir

# Chars of the first user message kept as a session's preview label.
_PREVIEW_CHARS = 100

# The episodic-memory block the client prepends to a prompt before the agent
# stores it. It is injected context, not something the user typed, so it must
# never become a session's label — an unfiltered preview made every row read
# `[Episodic Memory - Similar Past Tasks] 1. "hello" → …`, so several unrelated
# sessions looked identical and none was identifiable.
_EPISODIC_PREFIX = "[Episodic Memory"


def first_user_prompt(messages: List[Dict[str, Any]], max_len: int = _PREVIEW_CHARS) -> str:
    """The first thing the USER actually typed, flattened and clipped.

    Skips every kind of injected/synthetic content, because none of it
    identifies the session to the person picking from a list: the ephemeral
    reminder blocks (steering, plan mode), the prepended episodic-memory block,
    tool results, and auto-delivered background sub-agent reports. Falls through
    to the next user message when one yields nothing, so a session whose first
    turn was pure injection is still labelled by its first real prompt.
    """
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # A tool-result-only message has no text block: skip it, don't stop.
            for block in content:
                if isinstance(block, str):
                    text = block
                    break
                if isinstance(block, dict) and block.get("text"):
                    text = str(block["text"])
                    break

        text = LangGraphAgent._strip_ephemeral(text)
        # The episodic block is prepended as "[Episodic Memory …]\n…\n\n<prompt>";
        # keep only the real prompt that follows it.
        if text.lstrip().startswith(_EPISODIC_PREFIX):
            text = text.split("\n\n", 1)[1] if "\n\n" in text else ""
        # A background sub-agent's report is auto-delivered as a user message;
        # it's the agent talking to itself, not a prompt.
        if text.lstrip().startswith("Your background sub-agent"):
            continue

        text = " ".join(text.split())
        if text:
            # max_len is the TOTAL width, ellipsis included — callers size it to
            # a column budget, so the marker must fit inside it, not overflow it.
            if len(text) <= max_len:
                return text
            return text[: max_len - 1] + "…"
    return ""


class SessionLog:
    """Writes one append-only transcript for the running session.

    Every write is best-effort: a session log is a convenience, so a failure to
    write it must never break the turn the user is waiting on.
    """

    def __init__(self, cwd: str = None, profile: str = None, model: str = None):
        self._turn = 0
        self.path: Optional[Path] = None
        try:
            # timestamp + pid + random suffix: the first two alone collide when
            # two sessions start in the same second (rapid relaunch, or two logs
            # in one process), which would append TWO conversations into one file.
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_id = f"{stamp}_{os.getpid()}_{uuid.uuid4().hex[:6]}"
            self.path = sessions_dir(cwd, profile) / f"session_{self.session_id}.jsonl"
            self._append(
                {
                    "t": "meta",
                    "session_id": self.session_id,
                    "cwd": str(cwd if cwd is not None else os.getcwd()),
                    "started": time.time(),
                    "model": model or "",
                }
            )
        except Exception as e:  # noqa: BLE001 — never break startup over a log
            logger.debug(f"Session log disabled (could not create): {e}")
            self.path = None

    def _append(self, record: Dict[str, Any]) -> None:
        if self.path is None:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Session log append failed: {e}")

    def seed_history(self, messages: List[Any], source: str = "") -> None:
        """Copy an ALREADY-RESTORED conversation into this session's transcript.

        Called after ``--resume`` or ``/load`` rehydrates history into the agent.
        A session file only ever held the turns that happened after it was
        created, so continuing a restored conversation produced a transcript
        starting mid-thread: resuming *that* file restored a stump, and each
        resume-of-a-resume truncated the chain further. Copying the restored
        messages in makes every session file self-contained — the same reason a
        resumed session is re-stamped into the fresh log rather than appended to
        the old one, which keeps the source file immutable so resuming it again
        still yields the original conversation.

        Recorded as one ``restore`` record (not a ``turn``) so the turn counter
        keeps meaning "turns the user took in THIS session", while
        :func:`read_session` still replays the messages in order.
        """
        if self.path is None or not messages:
            return
        try:
            payload = convert_langchain_messages_to_strands(list(messages))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Session log seed encode failed: {e}")
            return
        if not payload:
            return
        self._append(
            {"t": "restore", "ts": time.time(), "source": source, "messages": payload}
        )

    def log_turn(self, messages: List[Any]) -> None:
        """Record the messages this turn added (already-final LangChain messages)."""
        if self.path is None or not messages:
            return
        try:
            payload = convert_langchain_messages_to_strands(list(messages))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Session log encode failed: {e}")
            return
        if not payload:
            return
        self._turn += 1
        self._append(
            {"t": "turn", "n": self._turn, "ts": time.time(), "messages": payload}
        )

    def log_compaction(self) -> None:
        """Note that the live context was compacted (the transcript is unaffected)."""
        self._append({"t": "compact", "n": self._turn, "ts": time.time()})

    def discard_if_empty(self) -> bool:
        """Delete this session's file if no turn was ever recorded; True if removed.

        The `meta` record is written at startup, before we know whether the user
        will actually say anything — so a launch they immediately quit (or a
        cancelled ``--resume``) leaves a turn-less file. Those are already hidden
        from the picker, but without this they'd pile up on disk until they age
        out. Safe by construction: it only unlinks a file with zero `turn`
        records — a seeded ``restore`` record doesn't count, and dropping such a
        file loses nothing, because the session it copied from is never mutated.
        """
        if self.path is None or self._turn > 0:
            return False
        try:
            if read_session(self.path)["turns"] > 0:
                return False  # written by someone else — leave it alone
            self.path.unlink()
            self.path = None
            return True
        except OSError:
            return False


def _iter_records(path: Path):
    """Yield parsed records from a session file, skipping unparsable lines.

    A log can be truncated mid-write if the process was killed, so a bad final
    line must not make the whole session unreadable.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except (ValueError, TypeError):
                    continue
    except OSError:
        return


def read_session(path) -> Dict[str, Any]:
    """Read one session file into ``{meta, messages, turns}`` (strands dicts).

    Returns ``messages: []`` for a session that never completed a turn.
    """
    p = Path(path)
    meta: Dict[str, Any] = {}
    messages: List[Dict[str, Any]] = []
    turns = 0
    for rec in _iter_records(p):
        kind = rec.get("t")
        if kind == "meta":
            meta = rec
        elif kind in ("turn", "restore"):
            msgs = rec.get("messages")
            if isinstance(msgs, list):
                messages.extend(msgs)
                # A `restore` record is inherited history, not a turn the user
                # took here — it must not inflate the turn count the picker
                # shows, but it IS part of the conversation to replay.
                if kind == "turn":
                    turns += 1
    return {"meta": meta, "messages": messages, "turns": turns, "path": str(p)}


def _preview(messages: List[Dict[str, Any]]) -> str:
    """The picker's label: the user's first real prompt, injection stripped."""
    return first_user_prompt(messages) or "(no prompt)"


def list_sessions(cwd=None, profile: str = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Resumable sessions for this directory, newest first, capped at ``limit``.

    Sessions with no completed turn are skipped — resuming one would restore an
    empty conversation, and a resume the user never typed into is a pure
    DUPLICATE of the session it restored (same messages, one row each), so
    offering it would make the list grow by one every time they resume without
    asking anything. ``limit`` bounds only what's OFFERED; nothing is deleted
    here (age-based expiry owns deletion, see ``sweep_old_sessions``).
    """
    try:
        d = sessions_dir(cwd, profile)
        files = [f for f in d.glob("session_*.jsonl") if f.is_file()]
    except OSError:
        return []
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    out: List[Dict[str, Any]] = []
    for f in files:
        if len(out) >= max(1, limit):
            break
        data = read_session(f)
        if not data["messages"] or data["turns"] == 0:
            continue
        out.append(
            {
                "path": str(f),
                "session_id": data["meta"].get("session_id", f.stem),
                "modified": f.stat().st_mtime,
                "turns": data["turns"],
                "messages": data["messages"],
                "preview": _preview(data["messages"]),
            }
        )
    return out


def latest_session(cwd=None, profile: str = None) -> Optional[Dict[str, Any]]:
    """Most recent resumable session for this directory, or None."""
    found = list_sessions(cwd, profile, limit=1)
    return found[0] if found else None
