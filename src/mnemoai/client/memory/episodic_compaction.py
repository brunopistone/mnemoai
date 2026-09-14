"""One-shot on-disk compaction of an episodic ChromaDB store — bytes only.

Rewrites the legacy ``tools`` payload (see :mod:`episode_tools`) to the name list
its readers actually use, in BOTH places Chroma keeps a copy — the metadata table
and the write-ahead log — then VACUUMs. Nothing is deleted: every row, id,
seq_id, vector and embedding survives, so the store stays exactly as searchable
and as replayable as it was; only the unread bytes inside each row go. Measured
on a real store: 272.0 MB → 4.5 MB with all 618 episodes intact.

**Raw SQL on a store nobody has opened yet, deliberately.** Going through the
Chroma API would append a fresh write-ahead record per update — the repair would
grow the file it is shrinking — and there is no API for this: ``SqliteDB.vacuum``
only runs ``VACUUM``, which reclaims nothing while the bytes are still live rows.

**Why the write-ahead log is half the problem.** Chroma's ``purge_log`` deletes
queue rows below the MIN of its segments' recorded ``max_seq_id``, counting a
segment that recorded none as ``-1``. The local persisted vector segment records
nothing in SQLite, so that minimum is always ``-1`` and the queue is never
purged, however often ``automatically_purge`` runs — every episode's metadata is
kept twice, forever. Hence rewriting the queue rows in place rather than waiting
for a purge that cannot happen.

Best-effort throughout: a locked, missing or unreadable store is skipped, never
raised — this runs at startup, and housekeeping must not be the reason the app
won't start.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from mnemoai.client.memory.episode_tools import compact_tools
from mnemoai.utils.logger import logger

_DB_NAME = "chroma.sqlite3"

# Below this a store holds nothing worth reclaiming, and the check itself must
# stay free: deciding by CONTENT means reading every stored value (85 MB of
# strings on the store above), which is the cost we are trying to avoid paying
# per startup. A compacted store with the full 1000-episode allowance is a few
# MB, so the gate never hides a bloated one. No sentinel file: the gate is the
# file size itself, so a store that regrows is repaired again.
_MIN_SIZE_BYTES = 16 * 1024 * 1024

# A `tools` value longer than this cannot be a name list (they are capped at
# MAX_TOOLS_CHARS), so it is a legacy payload.
_OVERSIZED_CHARS = 2000

_BUSY_TIMEOUT_MS = 5000


@dataclass
class Compaction:
    """What one store's compaction did (bytes are the file's actual sizes)."""

    path: str
    episodes: int
    queue_rows: int
    before: int
    after: int

    @property
    def reclaimed(self) -> int:
        return max(0, self.before - self.after)


def compact_store(store_path) -> Compaction | None:
    """Compact one episodic store dir. None when there was nothing to do."""
    db_path = Path(store_path) / _DB_NAME
    try:
        before = db_path.stat().st_size
    except OSError:
        return None
    if before < _MIN_SIZE_BYTES:
        return None

    try:
        # isolation_level=None: VACUUM cannot run inside a transaction, so the
        # rewrites take an explicit one and the vacuum runs outside it.
        con = sqlite3.connect(str(db_path), isolation_level=None, timeout=10)
    except sqlite3.Error as e:
        logger.debug(f"Episodic compaction skipped ({db_path}): {e}")
        return None

    try:
        con.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        con.execute("BEGIN IMMEDIATE")
        try:
            episodes = _compact_metadata(con)
            queue_rows = _compact_queue(con)
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
        if not (episodes or queue_rows):
            return None
        _vacuum(con, db_path)
    except (sqlite3.Error, OSError, ValueError) as e:
        logger.debug(f"Episodic compaction failed ({db_path}): {e}")
        return None
    finally:
        con.close()

    try:
        after = db_path.stat().st_size
    except OSError:
        after = before
    result = Compaction(str(store_path), episodes, queue_rows, before, after)
    logger.debug(
        f"Compacted {db_path}: {_mb(before)} → {_mb(after)} "
        f"({episodes} episodes, {queue_rows} log rows)"
    )
    return result


def compact_stores(models_root) -> list[Compaction]:
    """Compact every ``*/episodic_memory`` store under a profile's models dir.

    Covers the store this session is about to open AND the ones left behind by
    other models — the bytes are the same bytes, and a store nobody opens again
    would otherwise keep them forever.
    """
    try:
        stores = sorted(Path(models_root).glob(f"*/episodic_memory/{_DB_NAME}"))
    except OSError as e:
        logger.debug(f"Episodic compaction scan skipped: {e}")
        return []

    pending = [p.parent for p in stores if _is_oversized(p)]
    if not pending:
        return []

    total = sum(_size(p / _DB_NAME) for p in pending)
    # Announced BEFORE the work: it is seconds of startup the user did not ask
    # for, once, and a silent pause reads as a hang.
    logger.info(
        f"Compacting episodic memory storage ({_mb(total)} across "
        f"{len(pending)} store{'s' if len(pending) != 1 else ''}, one-off)…"
    )

    results = [r for r in (compact_store(p) for p in pending) if r]
    if results:
        reclaimed = sum(r.reclaimed for r in results)
        logger.info(f"Episodic memory storage compacted: {_mb(reclaimed)} reclaimed")
    return results


def _compact_metadata(con: sqlite3.Connection) -> int:
    """Rewrite oversized ``tools`` values in the metadata table; rows changed."""
    rows = con.execute(
        "SELECT id, string_value FROM embedding_metadata "
        "WHERE key = 'tools' AND string_value IS NOT NULL "
        "AND length(string_value) > ?",
        (_OVERSIZED_CHARS,),
    ).fetchall()
    changed = 0
    for row_id, value in rows:
        compacted = compact_tools(value)
        if compacted == value:
            continue
        con.execute(
            "UPDATE embedding_metadata SET string_value = ? "
            "WHERE id = ? AND key = 'tools'",
            (compacted, row_id),
        )
        changed += 1
    return changed


def _compact_queue(con: sqlite3.Connection) -> int:
    """Rewrite the same value inside the write-ahead log's JSON; rows changed.

    The row itself stays — its seq_id is what the vector segment replays from,
    and Chroma's own purge is unreachable here (see the module docstring).
    """
    rows = con.execute(
        "SELECT seq_id, metadata FROM embeddings_queue "
        "WHERE metadata IS NOT NULL AND length(metadata) > ?",
        (_OVERSIZED_CHARS,),
    ).fetchall()
    changed = 0
    for seq_id, blob in rows:
        try:
            payload = json.loads(blob)
        except (ValueError, TypeError):
            continue  # not ours to rewrite; leave it exactly as it is
        if not isinstance(payload, dict):
            continue
        tools = payload.get("tools")
        if not isinstance(tools, str) or len(tools) <= _OVERSIZED_CHARS:
            continue
        payload["tools"] = compact_tools(tools)
        con.execute(
            "UPDATE embeddings_queue SET metadata = ? WHERE seq_id = ?",
            (json.dumps(payload, separators=(",", ":")), seq_id),
        )
        changed += 1
    return changed


def _vacuum(con: sqlite3.Connection, db_path: Path) -> None:
    """Reclaim the freed pages. Tolerated failure: the rows are already small,
    so the next startup vacuums instead (the size gate still sees the file)."""
    try:
        con.execute("VACUUM")
    except sqlite3.Error as e:
        logger.debug(f"VACUUM skipped ({db_path}): {e}")


def _is_oversized(db_path: Path) -> bool:
    return _size(db_path) >= _MIN_SIZE_BYTES


def _size(db_path: Path) -> int:
    try:
        return db_path.stat().st_size
    except OSError:
        return 0


def _mb(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.1f} GB"
    return f"{size / 1024**2:.1f} MB"
