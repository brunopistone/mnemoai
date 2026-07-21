"""Unit tests for FAISS episodic-store persistence resilience.

FAISS keeps its index in memory and persists via write_index + a JSON dump — so,
unlike ChromaDB's SQLite "database moved" (code 1032), its analogous failure is
the persist DIR being moved/removed under it (a backup/sync/restore, or the app
home relocating). ``_persist`` must self-heal by recreating the dir and retrying
once so a store never crashes the turn. No embedding model / network needed.
"""

import numpy as np

from mnemoai.client.memory.faiss_store import FAISSEpisodicStore


class _FakeEmb:
    """Deterministic embeddings — fixed dim, no model/network."""

    def embed(self, texts):
        return np.ones((len(texts), 8), dtype=np.float32)


def test_add_persists_normally(tmp_path):
    s = FAISSEpisodicStore(str(tmp_path / "ep"), _FakeEmb())
    s.add("first episode", {"task": "a"})
    assert (tmp_path / "ep" / "episodic.index").exists()
    assert (tmp_path / "ep" / "episodic_metadata.json").exists()
    assert len(s.metadata) == 1


def test_add_self_heals_when_dir_removed(tmp_path):
    import shutil

    d = tmp_path / "ep"
    s = FAISSEpisodicStore(str(d), _FakeEmb())
    s.add("first", {"task": "a"})
    # The persist dir is moved/removed under us (backup/sync/restore).
    shutil.rmtree(d)
    # add() must recreate the dir and persist, not raise (faiss wraps the file
    # error in RuntimeError, not OSError — both are handled).
    s.add("second", {"task": "b"})
    assert (d / "episodic.index").exists()
    assert len(s.metadata) == 2


def test_cleanup_self_heals_when_dir_removed(tmp_path):
    import shutil

    d = tmp_path / "ep"
    s = FAISSEpisodicStore(str(d), _FakeEmb())
    for i in range(3):
        s.add(f"episode {i}", {"task": str(i), "timestamp": ""})
    shutil.rmtree(d)
    # cleanup rebuilds the index + metadata and persists — must self-heal too.
    s.cleanup(max_episodes=1, max_age_days=99999)
    assert (d / "episodic.index").exists()
    assert len(s.metadata) == 1
