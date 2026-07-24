"""Per-instance session id + RAG/chunk-cache artifact lifecycle (client helpers).

Mints this instance's session id, sweeps its own prior-run artifacts on startup
(a ``/model`` restart re-execs and mints a new id, orphaning the old store/cache),
and flushes them on ``/clear`` — all scoped to a single ``session_id`` so a
concurrent instance's files are never touched.

Functions take the ``LangGraphClient`` as the first arg and read/write its
``session_id`` field, so the client stays the owner (also read by save/load/clear
+ _approve_plan). The client keeps thin delegating methods — the
``context_injection``/``plan_policy`` collaborator pattern. No import of the
client class (functions receive the instance), so there is no import cycle.
"""

import os
import shutil
import sqlite3
from datetime import datetime
from typing import Optional

from mnemoai.utils.config import config
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import (
    chunk_session_pointer_path,
    instance_id,
    profile_dir,
    rag_session_pointer_path,
)


def new_session_id(client) -> str:
    """A session id unique to THIS instance: ``{profile}_{ts}_{instance_id}``.

    The timestamp alone is second-granular, so two instances (terminal tabs)
    on the same profile started in the SAME second would otherwise mint an
    IDENTICAL session id — and since the per-session artifact filenames
    (``chunk_cache_{id}.db``, ``rag_store_{id}``) key off it with no other
    namespacing, they'd share the SAME files on disk and clobber/delete each
    other's data. Appending the instance id (unique per live process; see
    ``paths.instance_id``) makes every instance's artifacts physically
    distinct, which also makes this instance's own restart-orphan cleanup safe
    (a session id belongs to exactly one instance)."""
    profile_name = config.get("PROFILE", {}).get("NAME", "default")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{profile_name}_{ts}_{instance_id()}"


def prev_session_from_pointer(client, pointer_path) -> Optional[str]:
    """This instance's PREVIOUS session_id, read from its own per-instance
    pointer file, or None. A ``/model``/``/params`` restart re-execs in place
    (``os.execv`` preserves ``MNEMOAI_INSTANCE_ID``), so the pointer still
    names THIS instance's id — but the fresh startup mints a NEW ``session_id``.
    The old one identifies this instance's now-orphaned store/cache.

    Safe to delete ONLY because ``session_id`` embeds the instance id (see its
    generation), so it is unique per instance: a concurrent tab can never share
    this session_id, hence never share the artifact we remove. Returns None if
    the pointer is absent, empty/whitespace, or already equals the current id.
    """
    try:
        if pointer_path.is_file():
            prev = pointer_path.read_text().strip()
            if prev and prev != client.session_id:
                return prev
    except OSError:
        pass
    return None


def repoint_session(client, pointer_path, flush_fn) -> None:
    """Sweep this instance's own prior-session artifact, then claim the pointer.

    Sweeps BEFORE repointing (safe: ``session_id`` is instance-unique, so the
    swept artifact is never a concurrent sibling's), then writes the current
    ``session_id`` so the next run can find and sweep this one.
    """
    prev = prev_session_from_pointer(client, pointer_path)
    if prev is not None:
        flush_fn(prev)
    pointer_path.write_text(client.session_id)


def initialize_rag_session(client) -> None:
    """Initialize RAG session at application startup.

    Also cleans up THIS instance's own store left by a prior run (e.g. after a
    ``/model`` restart), so stale ``rag_store_*`` don't accumulate.
    """
    try:
        profile_dir()  # ensure the dir exists

        # Per-instance pointer so concurrent tabs don't overwrite each other.
        repoint_session(client, rag_session_pointer_path(), client._flush_rag_store)

        logger.debug(f"RAG session initialized: {client.session_id}")
    except Exception as e:
        logger.warning(f"Failed to initialize RAG session: {e}")


def initialize_chunk_cache(client) -> None:
    """Initialize chunk cache DB at application startup.

    Also deletes THIS instance's own chunk cache left by a prior run (e.g.
    after a ``/model`` restart re-execs and mints a new session_id), so stale
    ``chunk_cache_*.db`` don't accumulate.
    """
    try:
        rag_dir = str(profile_dir())

        # Per-instance pointer so concurrent tabs don't overwrite each other.
        repoint_session(
            client, chunk_session_pointer_path(), client._flush_chunk_cache_store
        )

        db_path = os.path.join(rag_dir, f"chunk_cache_{client.session_id}.db")
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_cache (
                    key TEXT PRIMARY KEY,
                    summary TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.commit()
            logger.debug(f"Chunk cache initialized: {os.path.basename(db_path)}")
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"Failed to initialize chunk cache: {e}")


def flush_chunk_cache_store(client, session_id: str = None) -> None:
    """Flush THIS instance's chunk cache — its own DB + pointer only.

    Scoped to ``session_id`` (defaults to the current one) so a concurrent
    instance's ``chunk_cache_*.db`` is never touched.
    """
    session_id = session_id or client.session_id
    try:
        from mnemoai.server.tools.readers.chunking_helper import (
            reset_session_chunk_cache,
        )

        reset_session_chunk_cache()  # removes this instance's pointer file

        db_path = os.path.join(
            str(profile_dir()), f"chunk_cache_{session_id}.db"
        )
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                logger.debug(f"Deleted chunk cache: {os.path.basename(db_path)}")
            except OSError as e:
                logger.debug(f"Failed to delete {db_path}: {e}")

        logger.debug("Chunk cache store cleared")
    except Exception as e:
        logger.warning(f"Failed to reset chunk cache: {e}")


def flush_rag_store(client, session_id: str = None) -> None:
    """Flush THIS instance's RAG store — its own store dir/file + pointer only.

    Scoped to ``session_id`` (defaults to the current one) so a concurrent
    instance's ``rag_store_*`` is never touched.
    """
    session_id = session_id or client.session_id
    try:
        from mnemoai.server.tools.rag import reset_session_rag

        reset_session_rag()  # removes this instance's pointer file

        rag_dir = str(profile_dir())
        # Both backends key the store by session_id: FAISS → a
        # ``rag_store_<id>.faiss`` file, ChromaDB → a ``rag_store_<id>`` dir.
        for name in (f"rag_store_{session_id}.faiss", f"rag_store_{session_id}"):
            path = os.path.join(rag_dir, name)
            if not os.path.exists(path):
                continue
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                logger.debug(f"Deleted RAG store: {name}")
            except OSError as e:
                logger.debug(f"Failed to delete {name}: {e}")

        logger.debug("RAG store cleared")
    except Exception as e:
        logger.warning(f"Failed to reset RAG store: {e}")
