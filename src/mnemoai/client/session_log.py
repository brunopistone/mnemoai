"""Append-only per-directory session transcripts, for ``--resume``.

Sessions are scoped to the directory the app was launched from, so ``--resume``
in a project offers that project's sessions and nothing else
(:func:`mnemoai.utils.paths.sessions_dir`). One file per session, one JSON object
per line:

    {"t": "meta",  "session_id": …, "cwd": …, "started": …, "model": …}
    {"t": "turn",  "n": 1, "ts": …, "messages": [ …strands dicts… ]}
    {"t": "compact", "n": 3, "ts": …, "summary": "…", "messages": [ …kept… ]}
    {"t": "compact", "n": 4, "ts": …, "messages": [ …evicted history… ]}
    {"t": "label", "ts": …, "title": "…"}          # /rename, last one wins
    {"t": "rewind", "n": 4, "ts": …}               # /rewind, turn 4 withdrawn

**Why append-only instead of mirroring the live message list:** compaction
*replaces* ``agent.messages`` wholesale (it summarizes older turns away), so a
log that mirrored the list would lose that history the moment it compacted — and
the full text of those turns would be gone from disk for good. The log is
therefore the record of what was *said*, written once per turn and never
rewritten.

**But a restore must reproduce the COMPACTED state, not the raw history.** A
``compact`` record is therefore a checkpoint, carrying the summary that replaced
the older turns plus the window that stayed live: :func:`read_session` returns
that state as ``messages`` while still exposing everything ever logged as
``all_messages``. Without it, resuming a compacted conversation silently undid
the compaction — the live context jumped back to the full pre-compaction history
(measured: a chat the provider reported at 235,793 tokens came back as ~1.05M,
past the model's window), so the first turn after every resume had to summarize
the whole thing again, and the summary already paid for was thrown away.

**Two ways the context shrinks, so two checkpoint shapes** — what marks a record
as restorable is the ``messages`` key, not ``summary``. A summary checkpoint
replaces older turns; an **eviction** checkpoint (tool-result eviction, the cheap
layer that runs with no model call) drops no message at all, it rewrites old tool
results smaller — and since the transcript holds each result at its original
size, a restore replayed those and handed back every token eviction had just
reclaimed. Same defect, a path where no summary exists to record. Records written
before 1.12.6 carry neither key and stay purely informational (they restore
everything, which is what they always did).

**A withdrawn turn is a record too, not a deletion.** ``/rewind`` takes back the
last exchange, and append-only leaves exactly one way to say so: a ``rewind``
record naming the turn, which every reader then skips (:func:`read_session`,
:func:`turn_summaries`, :func:`branch_session`) — the same shape as a ``compact``
checkpoint, which also changes what comes back without rewriting a byte. The
withdrawn text stays on disk, which is the point: a rewind is an undo of the
conversation, not a promise that what was said was erased. And when the withdrawn
exchange is not a turn of this session — a restored conversation arrives as ONE
``restore`` blob, so its last exchange has no record to skip — the rewind pins the
surviving history exactly as a checkpoint does, or the withdrawal would hold for
the run and then come back at the next resume.

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

from mnemoai.client.agent.message_codec import (
    convert_langchain_messages_to_strands,
)
from mnemoai.client.ui import turn_view
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import sessions_dir

# Chars of the first user message kept as a session's preview label.
_PREVIEW_CHARS = 100
# Chars kept of a user-set session name (/rename). Long enough to be descriptive,
# short enough that a picker row still fits one line next to its metadata.
_LABEL_CHARS = 80

def _message_text(message: Dict[str, Any]) -> str:
    """First text block of a strands message ("" for a tool-result-only message)."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                return block
            if isinstance(block, dict) and block.get("text"):
                return str(block["text"])
    return ""


def _is_user_prompt(message: Dict[str, Any]) -> bool:
    """True for a message that is something the USER actually typed.

    Used to size a conversation for the picker. A tool-result-only message and an
    auto-delivered background sub-agent report both carry ``role: user`` but
    neither is a prompt, so counting them would overstate the length.
    """
    if message.get("role") != "user":
        return False
    return bool(turn_view.user_prompt_text(_message_text(message)))


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
        # Injected context (steering/plan reminders, the episodic block) and an
        # auto-delivered sub-agent report all resolve to "" — skip to the next
        # message rather than stopping, so a session whose first turn was pure
        # injection is still labelled by its first real prompt.
        text = " ".join(turn_view.user_prompt_text(_message_text(m)).split())
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
        # Set by any record that PINS a state narrower than the raw turns (a
        # compaction checkpoint, a `/rewind` rebase). Such a file is worth keeping
        # and offering even with no turn of its own: see `discard_if_empty`.
        self._pinned_state = False
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

    @classmethod
    def reopen(cls, path) -> "SessionLog":
        """Attach to an EXISTING session file and keep appending to it.

        Used by ``/branch``: the fork's file already holds the copied history, and
        this run must continue writing into it rather than create a third file.
        Skips ``__init__`` (which would write a second ``meta`` record) and
        restores the turn counter from what's on disk, so the next turn is
        numbered correctly instead of restarting at 1.
        """
        log = cls.__new__(cls)
        p = Path(path)
        log.path = p
        log._turn = 0
        log._pinned_state = False
        try:
            data = read_session(p)
            log._turn = data["turns"]
            log.session_id = data["meta"].get("session_id", p.stem)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Session log reopen fell back to a fresh counter: {e}")
            log.session_id = p.stem
        return log

    def _append(self, record: Dict[str, Any]) -> None:
        if self.path is None:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Session log append failed: {e}")

    def seed_history(
        self,
        messages: List[Any],
        source: str = "",
        summary: str = "",
        kept: Optional[List[Any]] = None,
    ) -> None:
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

        ``messages`` is the FULL history (so the new file keeps the whole text on
        disk); ``summary``/``kept`` re-state the compaction checkpoint that was
        active, so the next restore rebuilds the compacted context rather than the
        raw history it stands for. Pass ``kept`` only when there IS a checkpoint to
        carry — an eviction one has no summary, so a caller that always passed the
        live history would duplicate the whole conversation in every restore.
        """
        if self.path is None or not messages:
            return
        try:
            payload = convert_langchain_messages_to_strands(list(messages))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Session log seed encode failed: {e}")
            return
        if not payload:
            # The encoder silently yields nothing for input that isn't LangChain
            # messages (e.g. already-encoded strands dicts), which would drop the
            # restored history and resume a stump. Loud enough to diagnose.
            logger.warning(
                f"Session log seed produced no records from {len(messages)} "
                "message(s) — expected LangChain messages; history not recorded."
            )
            return
        self._append(
            {"t": "restore", "ts": time.time(), "source": source, "messages": payload}
        )
        if summary or kept is not None:
            self.log_compaction(summary=summary, kept=kept or [], seeded=True)

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

    def log_compaction(
        self, summary: str = "", kept: Optional[List[Any]] = None, seeded: bool = False
    ) -> None:
        """Record that the live context shrank — as a restorable checkpoint.

        One record, two shapes. **With a summary:** the turns it replaced stay on
        disk in their own ``turn`` records and ``kept`` is the window left
        verbatim, which may legitimately be empty. **Without one:** no message was
        dropped, they got SMALLER — tool-result eviction rewrote old results in
        place — and ``kept`` is that whole rewritten history.

        So the ``messages`` key, not ``summary``, is what marks a record as a
        checkpoint (:func:`read_session`). Eviction has no summary to record, yet
        its state is exactly as unreachable from the raw turns as a summary's is:
        the transcript holds each result at its ORIGINAL size, so a restore that
        replayed the turns re-inflated everything eviction had just reclaimed.

        Falls back to a bare marker if the window can't be encoded: an unusable
        checkpoint would drop real history, a marker only loses the optimization.

        ``seeded`` marks a checkpoint COPIED from the session this one resumed
        (:meth:`seed_history` re-states it so the chain doesn't re-inflate). That
        one is not a shrink this session performed — the parent holds the same
        state — so it must not make a turn-less file worth keeping.
        """
        record: Dict[str, Any] = {"t": "compact", "n": self._turn, "ts": time.time()}
        if summary or kept is not None:
            try:
                record["messages"] = convert_langchain_messages_to_strands(
                    list(kept or [])
                )
                if summary:
                    record["summary"] = str(summary)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Compaction checkpoint encode failed: {e}")
                record.pop("messages", None)
                record.pop("summary", None)
        self._append(record)
        if "messages" in record and not seeded:
            self._pinned_state = True

    def set_label(self, title: str) -> bool:
        """Name this session for the ``--resume`` picker; True if recorded.

        Appended as a record like everything else — the file is never rewritten,
        so renaming twice leaves two records and the LAST one wins
        (:func:`read_session`). An empty title is a valid record: it clears the
        name back to the first-prompt preview.
        """
        if self.path is None:
            return False
        self._append({"t": "label", "ts": time.time(), "title": str(title)[:_LABEL_CHARS]})
        return True

    def log_rewind(
        self, turn_n: Optional[int] = None, kept: Optional[List[Any]] = None
    ) -> bool:
        """Record that an exchange was WITHDRAWN (``/rewind``); True if recorded.

        Appended, never cut out: the file is the record of what was *said*, and a
        reader honors the withdrawal instead (:func:`read_session`). Two shapes,
        because a withdrawn turn is not always a ``turn`` record of this session:

        * **``turn_n``** — the ordinary case: name the turn, and every reader skips
          that record. The turn counter goes back down with it, so the next turn
          takes the number again and a session whose only turn was withdrawn is
          discardable at exit — which is why a stored number can appear twice in
          one file and :func:`_withdrawn_turns` matches the most recent one.
        * **``kept``** — the withdrawn exchange came in with a RESTORED
          conversation (``--resume`` / ``/load`` seed it as one ``restore`` blob),
          so there is no record to skip: the rewind pins the state that survived
          instead, exactly as a compaction checkpoint does. Without it the
          withdrawal would hold for this run and then quietly come back the next
          time the session was resumed.

        Never both. An unencodable window records NOTHING rather than a bare
        marker: that marker means "withdraw the last turn", which is a different
        (and wrong) instruction here. And since the encoder yields nothing at all
        for input that isn't LangChain messages, a non-empty window that encodes to
        nothing is a FAILURE, not a rewind back to zero — recorded as the latter it
        would drop the whole conversation at the next restore.
        """
        if self.path is None:
            return False
        record: Dict[str, Any] = {"t": "rewind", "ts": time.time()}
        if turn_n is not None:
            record["n"] = int(turn_n)
        elif kept is not None:
            try:
                payload = convert_langchain_messages_to_strands(list(kept))
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Session log rewind encode failed: {e}")
                return False
            if kept and not payload:
                logger.warning(
                    f"Session log rewind produced no records from {len(kept)} "
                    "message(s) — expected LangChain messages; not recorded."
                )
                return False
            record["messages"] = payload
        self._append(record)
        if turn_n is not None:
            self._turn = max(0, self._turn - 1)
        else:
            # Inherited history was rebased, so this file is the only place the
            # withdrawal exists — it must survive `discard_if_empty` and stay in
            # the picker even with no turn of its own.
            self._pinned_state = True
        return True

    def discard_if_empty(self) -> bool:
        """Delete this session's file if no turn was ever recorded; True if removed.

        The `meta` record is written at startup, before we know whether the user
        will actually say anything — so a launch they immediately quit (or a
        cancelled ``--resume``) leaves a turn-less file. Those are already hidden
        from the picker, but without this they'd pile up on disk until they age
        out. Safe by construction: it only unlinks a file with zero `turn`
        records — a seeded ``restore`` record doesn't count, and dropping such a
        file loses nothing, because the session it copied from is never mutated.

        The exception is a file that PINS a narrower state — a compaction
        checkpoint or a ``/rewind`` rebase over inherited history. It has no turn
        of its own, but it is the only place that shrink exists: dropping it would
        hand the next ``--resume`` the un-compacted / un-withdrawn conversation
        back.
        """
        if self.path is None or self._turn > 0 or self._pinned_state:
            return False
        try:
            if read_session(self.path)["turns"] > 0:
                return False  # written by someone else — leave it alone
            self.path.unlink()
            self.path = None
            return True
        except OSError:
            return False


def _withdrawn_turns(records: List[Dict[str, Any]]) -> set:
    """Record indices of ``turn`` records a later ``rewind`` record withdrew.

    Matched on the stored turn number, searched NEWEST first: the counter is
    restored from the surviving turns when a session is reopened and steps back
    on every rewind, so one number can legitimately appear on two records in one
    file, and a ``rewind`` always means the most recent turn still carrying it. A
    record with no usable number falls back to the last live turn, which is the
    only turn ``/rewind`` ever withdraws — EXCEPT one carrying ``messages``, which
    withdrew INHERITED history and pins what survived instead of naming a turn (so
    the fallback would take back a turn nobody asked about).
    """
    live: List[tuple] = []
    withdrawn = set()
    for i, rec in enumerate(records):
        kind = rec.get("t")
        if kind == "turn":
            live.append((rec.get("n"), i))
        elif kind == "rewind" and "messages" not in rec and live:
            n = rec.get("n")
            pos = next(
                (p for p in range(len(live) - 1, -1, -1) if live[p][0] == n),
                len(live) - 1,
            )
            withdrawn.add(live[pos][1])
            del live[pos]
    return withdrawn


def _checkpoint_indices(records: List[Dict[str, Any]]) -> set:
    """Indices of compaction CHECKPOINT records, excluding a seeded one.

    ``seed_history`` and ``branch_session`` write a checkpoint immediately after
    their ``restore`` record to re-state the state that was carried in — that one
    describes history this session INHERITED, not a compaction that ran here, so
    it must not make the first turn look uncompactable.
    """
    return {
        i
        for i, rec in enumerate(records)
        if rec.get("t") == "compact"
        and "messages" in rec
        and not (i and records[i - 1].get("t") == "restore")
    }


def last_live_turn(path) -> Optional[Dict[str, Any]]:
    """The most recent turn a ``rewind`` record could still withdraw.

    ``{"n": stored turn number, "preview": …, "compacted": bool}``, or None when
    no turn is left. ``compacted`` is the one thing that makes a turn
    unwithdrawable: a checkpoint written **after** it stands for a window that
    already contains it, and one written just **before** it may be a mid-turn
    compaction, whose window contains the turn's own prompt. The two are
    indistinguishable from the file (same position, and ``log_compaction`` stamps
    both with the previous turn's number), so both count — a rewind that left the
    restorable state describing a conversation that never happened is worse than
    one the user is told to skip.
    """
    records = list(_iter_records(Path(path)))
    withdrawn = _withdrawn_turns(records)
    turns = [
        i for i, r in enumerate(records) if r.get("t") == "turn" and i not in withdrawn
    ]
    if not turns:
        return None
    last = turns[-1]
    checkpoints = _checkpoint_indices(records)
    compacted = any(i > last for i in checkpoints)
    # …and walk back from the turn record until the previous turn: a checkpoint
    # reached first was written with this turn already (partly) in history.
    for i in range(last - 1, -1, -1):
        if records[i].get("t") in ("turn", "restore"):
            break
        if i in checkpoints:
            compacted = True
            break
    rec = records[last]
    msgs = rec.get("messages") if isinstance(rec.get("messages"), list) else []
    return {"n": rec.get("n"), "preview": first_user_prompt(msgs), "compacted": compacted}


def turn_summaries(path) -> List[Dict[str, Any]]:
    """Per-turn labels for a session, for the ``/branch`` turn picker.

    Returns ``[{"n": 1, "preview": …, "ts": …}, …]`` — one entry per ``turn``
    record, labelled by what the USER typed that turn (same stripping as the
    ``--resume`` picker). A ``restore`` record contributes no entry: its messages
    are inherited history, so they are not a turn you can branch *at* — branching
    before turn 1 would just duplicate the parent. A withdrawn turn contributes
    none either: ``/branch`` must not offer a point ``/rewind`` took back, and the
    numbering here counts SURVIVORS, so it stays the numbering every other reader
    uses.
    """
    records = list(_iter_records(Path(path)))
    withdrawn = _withdrawn_turns(records)
    out: List[Dict[str, Any]] = []
    n = 0
    for i, rec in enumerate(records):
        if rec.get("t") != "turn" or i in withdrawn:
            continue
        n += 1
        msgs = rec.get("messages") if isinstance(rec.get("messages"), list) else []
        out.append(
            {
                "n": n,
                "ts": rec.get("ts"),
                "preview": first_user_prompt(msgs) or "(no prompt)",
            }
        )
    return out


def branch_session(path, through_turn: int = None, cwd=None, profile: str = None) -> Optional[Path]:
    """Copy a session's transcript up to ``through_turn`` into a NEW session file.

    Returns the new path, or None if there was nothing to copy. ``through_turn``
    is inclusive and 1-based (None/0 or beyond the end = the whole session), so
    branching at turn 3 yields a session ending after turn 3 — the point to
    continue from in a different direction.

    **The source file is never touched.** That is the whole safety property: a
    branch is a copy, so the original conversation stays resumable exactly as it
    was, and a branch that goes nowhere costs nothing. The copy is written as a
    single ``restore`` record rather than replayed turn-by-turn, because it is
    inherited history for the new session — the same shape ``seed_history`` writes
    after ``--resume``, so the new file is self-contained and its own turn counter
    starts at the turns the user takes *in the branch*.

    A compaction checkpoint inside the copied range is copied too, so a fork of a
    compacted conversation starts from the context the source had rather than
    re-inflating the history the summary already stands for. A branch point BEFORE
    the compaction never reaches that record, and so forks from the raw history —
    which is the point of branching there.
    """
    src = Path(path)
    meta: Dict[str, Any] = {}
    messages: List[Dict[str, Any]] = []
    live: List[Dict[str, Any]] = []
    summary = ""
    checkpoint = False
    kept = 0
    limit = through_turn if (through_turn and through_turn > 0) else None
    records = list(_iter_records(src))
    withdrawn = _withdrawn_turns(records)
    for i, rec in enumerate(records):
        kind = rec.get("t")
        if kind == "meta":
            meta = rec
            continue
        if kind == "turn" and i in withdrawn:
            continue  # /rewind took it back — a fork must not resurrect it
        if kind == "rewind" and "messages" in rec:
            # A rewind of inherited history: same as a checkpoint, the restorable
            # window restarts from what it pinned.
            window = rec.get("messages")
            live = list(window) if isinstance(window, list) else []
            checkpoint = True
            continue
        if kind == "compact" and "messages" in rec:
            window = rec.get("messages")
            live = list(window) if isinstance(window, list) else []
            checkpoint = True
            if "summary" in rec:
                summary = str(rec.get("summary") or "")
            continue
        if kind not in ("turn", "restore"):
            continue
        msgs = rec.get("messages")
        if not isinstance(msgs, list):
            continue
        if kind == "turn":
            if limit is not None and kept >= limit:
                break  # records are in order, so the rest is past the branch point
            kept += 1
        messages.extend(msgs)
        live.extend(msgs)
    if not messages:
        return None

    # A branch must land in the SAME per-directory session dir as its source, or
    # it disappears from that project's picker. The source's own `meta.cwd` is
    # authoritative: `branch_session` can be called from anywhere (and a test's
    # process cwd is never the session's project), so falling back to the process
    # cwd silently files the fork under the wrong project.
    if cwd is None:
        cwd = meta.get("cwd") or None
    log = SessionLog(cwd=cwd, profile=profile, model=meta.get("model", ""))
    if log.path is None:
        return None
    if log.path.parent != src.parent:
        logger.debug(
            f"Branch filed under {log.path.parent} (source: {src.parent})"
        )
    log._append(
        {
            "t": "restore",
            "ts": time.time(),
            "source": str(src),
            "branched_from": {
                "session_id": meta.get("session_id", src.stem),
                "through_turn": kept,
            },
            "messages": messages,
        }
    )
    if checkpoint:
        # Written raw (not via log_compaction) because these are already-encoded
        # strands dicts, not LangChain messages. No `summary` key when the source
        # checkpoint had none (eviction) — the shape has to survive the copy, or
        # the fork restores a summary the conversation never had.
        record: Dict[str, Any] = {
            "t": "compact",
            "n": 0,
            "ts": time.time(),
            "messages": live,
        }
        if summary:
            record["summary"] = summary
        log._append(record)
    return log.path


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
    """Read one session file into ``{meta, messages, all_messages, summary, …}``.

    Returns ``messages: []`` for a session that never completed a turn.

    **``messages`` is the state to RESTORE, ``all_messages`` everything logged.**
    They differ once the session compacted: a ``compact`` checkpoint replaces the
    history before it with ``summary`` plus the window that stayed live, which is
    the context the user actually had. ``all_messages`` still holds every turn's
    full text — it is what the replay, the picker preview and ``exchanges`` read,
    so a compaction can't shorten how the conversation is displayed or sized.
    ``compacted_away`` is how many messages the checkpoint stands in for, and
    ``checkpoint`` whether one was found at all — an eviction checkpoint shrinks
    the state without a summary or a dropped message, so neither of the other two
    can tell a caller that ``messages`` is narrower than ``all_messages``.

    ``turns`` counts only turns taken in THIS file (inherited history doesn't
    inflate it). ``exchanges`` is what a reader actually cares about: how long the
    whole restorable conversation is, inherited history included — the picker shows
    that one, because you're choosing what to RESTORE, and a 243-message resumed
    conversation reporting "1 turn" reads as the shortest entry in the list.

    ``resumed_from`` is the session file a ``--resume``/``/load`` copied its history
    from, and ``branched_from`` the fork point for a ``/branch``. They differ in
    kind, so the picker treats them oppositely: a resume SUPERSEDES its source (the
    child contains all of it), while a branch DIVERGES from it (both are real).

    A ``rewind`` record narrows BOTH lists — the whole exchange is gone from the
    conversation, so it must not be restored, replayed or counted, even though its
    text is still on disk. That holds for either shape: a withdrawn ``turn`` record
    is skipped, and a rebase (the exchange arrived inside a ``restore`` blob) pins
    what survived. A rebase therefore also sets ``checkpoint`` — with no turn of its
    own, that flag is the only thing telling a caller this file holds a state its
    parent doesn't.
    """
    p = Path(path)
    meta: Dict[str, Any] = {}
    messages: List[Dict[str, Any]] = []
    all_messages: List[Dict[str, Any]] = []
    branched_from: Dict[str, Any] = {}
    resumed_from = ""
    label = ""
    summary = ""
    checkpoint = False
    compacted_away = 0
    turns = 0
    records = list(_iter_records(p))
    withdrawn = _withdrawn_turns(records)
    for i, rec in enumerate(records):
        kind = rec.get("t")
        if kind == "turn" and i in withdrawn:
            continue
        if kind == "meta":
            meta = rec
        elif kind == "label":
            # Last one wins: /rename appends, it never rewrites the file.
            label = str(rec.get("title") or "").strip()
        elif kind == "compact" and "messages" in rec:
            # A checkpoint: the restorable state restarts from the window this
            # record kept. Only `messages` is rewound — `all_messages` keeps the
            # full text. An EVICTION checkpoint carries no summary (it dropped no
            # message, it rewrote old tool results smaller), so it must not clear
            # the summary an earlier compaction left standing.
            kept = rec.get("messages")
            kept = kept if isinstance(kept, list) else []
            compacted_away += max(0, len(messages) - len(kept))
            messages = list(kept)
            checkpoint = True
            if "summary" in rec:
                summary = str(rec.get("summary") or "")
        elif kind == "rewind" and "messages" in rec:
            # A rewind that reached into INHERITED history (a `restore` blob holds
            # it, so there is no turn record to skip): the record pins the state
            # that survived. BOTH lists narrow here, unlike a compaction — a
            # withdrawal means the exchange did not happen, so it must not be
            # replayed, previewed or counted either, which is what the by-number
            # shape gets for free by skipping its turn record. Sound because a
            # rebase is only written when no live turn record is left, so
            # `all_messages` is the seeded blob and `kept` is what survived of it.
            kept = rec.get("messages")
            kept = list(kept) if isinstance(kept, list) else []
            messages = kept
            all_messages = list(kept)
            checkpoint = True
        elif kind in ("turn", "restore"):
            if kind == "restore":
                if isinstance(rec.get("branched_from"), dict):
                    branched_from = rec["branched_from"]
                else:
                    # A plain restore (resume / load), not a fork.
                    resumed_from = str(rec.get("source") or "")
            msgs = rec.get("messages")
            if isinstance(msgs, list):
                messages.extend(msgs)
                all_messages.extend(msgs)
                # A `restore` record is inherited history, not a turn the user
                # took here — it must not inflate the turn count the picker
                # shows, but it IS part of the conversation to replay.
                if kind == "turn":
                    turns += 1
    return {
        "meta": meta,
        "messages": messages,
        "all_messages": all_messages,
        "summary": summary,
        "checkpoint": checkpoint,
        "compacted_away": compacted_away,
        "turns": turns,
        "exchanges": sum(1 for m in all_messages if _is_user_prompt(m)),
        "path": str(p),
        "branched_from": branched_from,
        "resumed_from": resumed_from,
        "label": label,
    }


def _preview(messages: List[Dict[str, Any]]) -> str:
    """The picker's label: the user's first real prompt, injection stripped."""
    return first_user_prompt(messages) or "(no prompt)"


def list_sessions(
    cwd=None, profile: str = None, limit: int = 20, collapse_chains: bool = True
) -> List[Dict[str, Any]]:
    """Resumable conversations for this directory, newest first, capped at ``limit``.

    **One row per conversation, not per file.** Resuming writes a NEW file seeded
    with the whole prior conversation (so each file is self-contained), which means
    a chat resumed three times exists as four files — each a strict SUPERSET of the
    one before, all sharing the same opening prompt. Listing them per-file produced
    four identical-looking rows for one conversation, and the newest (longest) one
    reported the FEWEST turns because ``turns`` excludes inherited history. So a
    session that another session resumed is dropped: its content is entirely inside
    its successor, and the successor is what you want to continue.

    A ``/branch`` fork is deliberately NOT collapsed — it diverges from its parent
    rather than superseding it, so both remain real conversations and both are
    offered (the fork is tagged in the label).

    Sessions with no completed turn are skipped — resuming one would restore the
    same conversation its parent already offers — unless the file pins a narrower
    state (a compaction checkpoint or a ``/rewind`` rebase), which exists nowhere
    else. ``limit`` bounds only what's OFFERED; nothing is deleted here (age-based
    expiry owns deletion, see ``sweep_old_sessions``).

    ``collapse_chains=False`` returns every session including superseded links.
    That's for resolving an EXPLICIT ``--resume <id>``: hiding a row from the menu
    must not make its id unresolvable, since naming an id is a deliberate request
    for that exact point in the chain.
    """
    try:
        d = sessions_dir(cwd, profile)
        files = [f for f in d.glob("session_*.jsonl") if f.is_file()]
    except OSError:
        return []
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    # Read everything first: which files are superseded can only be known from the
    # whole set, so this can't be decided inside the (limited) emit loop below.
    scanned: List[Dict[str, Any]] = []
    superseded: set = set()
    for f in files:
        data = read_session(f)
        # Judge emptiness on everything logged: a compaction checkpoint can leave
        # `messages` empty (summary only) in a conversation that is very much real.
        # A file with no turn of its OWN is normally noise — the session it resumed
        # holds the same conversation — unless it pins a narrower state (a
        # checkpoint: compaction, or a `/rewind` that rebased inherited history).
        # Skipping that row offers the parent instead, which hands back the very
        # history the user just compacted or withdrew — including when a rewind
        # emptied the conversation outright, which is why a checkpoint survives the
        # emptiness test too (it restores nothing, and that is the honest answer).
        if not data["checkpoint"] and (not data["all_messages"] or data["turns"] == 0):
            continue
        scanned.append((f, data))
        parent = data.get("resumed_from") or ""
        if parent and collapse_chains:
            # Match on the resolved path: `source` is absolute, but normalize so a
            # symlinked/differently-spelled app home still matches.
            try:
                superseded.add(str(Path(parent).resolve()))
            except OSError:
                superseded.add(parent)

    out: List[Dict[str, Any]] = []
    for f, data in scanned:
        if len(out) >= max(1, limit):
            break
        try:
            if str(f.resolve()) in superseded:
                continue  # a later session resumed this one and contains all of it
        except OSError:
            pass
        out.append(
            {
                "path": str(f),
                "session_id": data["meta"].get("session_id", f.stem),
                "modified": f.stat().st_mtime,
                "turns": data["turns"],
                # What the row should SIZE itself by: the whole restorable
                # conversation, inherited history included.
                "exchanges": data.get("exchanges", data["turns"]),
                "messages": data["messages"],
                # Sized and labelled from the full text: a compaction shrinks the
                # live context, not the conversation — and the preview is the
                # OPENING prompt, which a checkpoint's kept window no longer holds.
                "preview": _preview(data["all_messages"]),
                # A name the user gave this session (/rename). Shown instead of the
                # first-prompt preview, which is why it survives a resume.
                "label": data.get("label", ""),
                # A fork shares its parent's opening prompt, so the preview alone
                # can't tell them apart — the picker appends a "(branch)" marker.
                "branched_from": data.get("branched_from") or {},
                "resumed_from": data.get("resumed_from", ""),
            }
        )
    return out


def latest_session(cwd=None, profile: str = None) -> Optional[Dict[str, Any]]:
    """Most recent resumable session for this directory, or None."""
    found = list_sessions(cwd, profile, limit=1)
    return found[0] if found else None
