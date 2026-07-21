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


class _FakeEmbDim:
    """Fake embeddings with a model fingerprint (name + dim), like the real
    controller — so a same-DIM model swap is still detected."""

    def __init__(self, dim, name="modelA"):
        self.dim = dim
        self._name = name

    def embed(self, texts):
        return np.ones((len(texts), self.dim), dtype=np.float32)

    def runtime_dimension(self):
        return self.dim

    def fingerprint(self):
        return f"{self._name}|{self.dim}"


def test_reset_when_embedding_dimension_changes(tmp_path):
    # Build the index at dim 8, then reopen with a dim-16 model: the store must
    # RESET (drop old index + metadata) rather than let a later add/search crash
    # on the dimension mismatch. Episodic memory is re-learnable model-scoped
    # scratch, so a reset is the safe migration.
    d = str(tmp_path / "ep")
    s = FAISSEpisodicStore(d, _FakeEmbDim(8))
    s.add("old episode", {"task": "t"})
    assert s.index.d == 8 and len(s.metadata) == 1

    s2 = FAISSEpisodicStore(d, _FakeEmbDim(16))  # new model, different dim
    assert s2.index is None          # reset — index dropped
    assert s2.metadata == []         # metadata dropped
    s2.add("new episode", {"task": "t2"})  # works at the new dim
    assert s2.index.d == 16


def test_reset_on_same_dim_different_model(tmp_path):
    # Same dimension, different model → still incompatible vectors → must reset
    # (the case a dimension-only check misses).
    d = str(tmp_path / "ep")
    s = FAISSEpisodicStore(d, _FakeEmbDim(8, name="qwen"))
    s.add("old episode", {"task": "t"})
    assert len(s.metadata) == 1
    s2 = FAISSEpisodicStore(d, _FakeEmbDim(8, name="cohere"))  # same dim, new model
    assert s2.index is None and s2.metadata == []


def test_no_reset_when_same_model(tmp_path):
    d = str(tmp_path / "ep")
    s = FAISSEpisodicStore(d, _FakeEmbDim(8, name="modelA"))
    s.add("episode", {"task": "t"})
    s2 = FAISSEpisodicStore(d, _FakeEmbDim(8, name="modelA"))  # same model → keep
    assert s2.index is not None and len(s2.metadata) == 1
