"""Existing episodic metadata stays searchable during an embedding outage."""

import numpy as np
import pytest

from mnemoai.client.memory.chroma_store import ChromaEpisodicStore
from mnemoai.client.memory.faiss_store import FAISSEpisodicStore
from tests.unit.test_embedding_integrity import (
    forget_integrity_stamp,
    legacy_vector,
    real_vector,
)


class Embedder:
    dim = 128
    online = True

    def fingerprint(self):
        if not self.online:
            raise RuntimeError("provider unavailable")
        return "real-embedding-model-128"

    def embed(self, texts):
        if not self.online:
            raise RuntimeError("provider unavailable")
        return np.vstack(
            [legacy_vector() if text == "legacy" else real_vector() for text in texts]
        )


@pytest.mark.parametrize("backend", [FAISSEpisodicStore, ChromaEpisodicStore])
def test_outage_uses_bm25_and_cannot_write_new_episodes(tmp_path, backend):
    embedder = Embedder()
    store = backend(str(tmp_path), embedder)
    store.add(
        "database", {"task": "configure postgres connection", "tools": "execute_bash"}
    )
    store.add("bread", {"task": "bake sourdough bread", "tools": ""})
    embedder.online = False
    hits = store.search("postgres")
    assert [hit["task"] for hit in hits] == ["configure postgres connection"]
    assert hits[0]["retrieval_method"] == "bm25"
    assert hits[0]["similarity"] == 1.0
    with pytest.raises(RuntimeError, match="unavailable"):
        store.add("new", {"task": "new episode"})
    assert store.search("absenttoken") == []


@pytest.mark.parametrize("backend", [FAISSEpisodicStore, ChromaEpisodicStore])
def test_offline_reopen_repairs_legacy_vectors_before_recall(tmp_path, backend):
    embedder = Embedder()
    store = backend(str(tmp_path), embedder)
    store.add("database", {"task": "postgres connection", "tools": ""})
    store.add("legacy", {"task": "postgres fabricated entry", "tools": ""})
    forget_integrity_stamp(store)
    embedder.online = False
    reopened = backend(str(tmp_path), embedder)
    hits = reopened.search("postgres")
    assert [hit["task"] for hit in hits] == ["postgres connection"]
    assert hits[0]["retrieval_method"] == "bm25"
    embedder.online = True
    assert all("retrieval_method" not in hit for hit in reopened.search("postgres"))
