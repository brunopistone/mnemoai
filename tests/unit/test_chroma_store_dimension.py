"""Unit tests for the ChromaDB episodic store's embedding-dimension migration.

An existing Chroma collection is locked to the embedding dimension it was built
with; switching to an embedding model that emits a DIFFERENT dimension (e.g. a
Cohere/Titan model, or resizing via DIMENSION) otherwise makes every query/add
raise "Collection expecting embedding with dimension of X, got Y" on every turn.
The store detects the mismatch and RESETS the collection (episodic memory is
re-learnable, model-scoped scratch). No network / real embedding model needed.
"""

import numpy as np

from mnemoai.client.memory.chroma_store import ChromaEpisodicStore


class _FakeEmb:
    """Fake embeddings with a model fingerprint (name + dim), like the real
    controller. ``name`` lets two same-DIM models be distinguished."""

    def __init__(self, dim, name="modelA"):
        self.dim = dim
        self._name = name

    def embed(self, texts):
        return np.ones((len(texts), self.dim), dtype=np.float32)

    def runtime_dimension(self):
        return self.dim

    def fingerprint(self):
        return f"{self._name}|{self.dim}"


def test_reset_when_dimension_changes(tmp_path):
    d = str(tmp_path / "ep")
    s = ChromaEpisodicStore(d, _FakeEmb(1536))
    s.add("old episode", {"task": "t"})
    assert s.collection.count() == 1

    # Reopen with a 1024-dim model → the collection must be reset, not crash.
    s2 = ChromaEpisodicStore(d, _FakeEmb(1024))
    assert s2.collection.count() == 0        # reset dropped old episodes
    s2.add("new episode", {"task": "t2"})    # works at the new dimension
    assert len(s2.search("new", top_k=3)) == 1


def test_no_reset_when_same_model(tmp_path):
    d = str(tmp_path / "ep")
    s = ChromaEpisodicStore(d, _FakeEmb(1024, name="modelA"))
    s.add("episode", {"task": "t"})
    s2 = ChromaEpisodicStore(d, _FakeEmb(1024, name="modelA"))  # same model → keep
    assert s2.collection.count() == 1


def test_reset_on_same_dim_different_model(tmp_path):
    # The key guarantee a dimension-only check would MISS: two different models
    # at the SAME dimension (e.g. Ollama qwen@1024 vs Cohere v4@1024) produce
    # incompatible vectors, so the store must still reset.
    d = str(tmp_path / "ep")
    s = ChromaEpisodicStore(d, _FakeEmb(1024, name="qwen"))
    s.add("episode", {"task": "t"})
    assert s.collection.count() == 1
    s2 = ChromaEpisodicStore(d, _FakeEmb(1024, name="cohere"))  # same dim, new model
    assert s2.collection.count() == 0  # reset despite identical dimension


def test_empty_collection_no_spurious_reset(tmp_path):
    # A brand-new (empty) collection has no locked-in dimension yet, so opening
    # it with any model must NOT trigger a reset path error.
    d = str(tmp_path / "ep")
    s = ChromaEpisodicStore(d, _FakeEmb(768))
    assert s.collection.count() == 0
    s.add("first", {"task": "t"})
    assert s.collection.count() == 1
