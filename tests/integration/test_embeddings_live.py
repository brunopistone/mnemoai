"""Real-provider vectors and real stores, followed by a controlled local outage.

The successful embedding calls use the configured provider. Only the outage is
injected; these tests never interrupt a real service or open personal stores.
"""

import hashlib

import numpy as np
import pytest

from mnemoai.client.memory.chroma_store import ChromaEpisodicStore
from mnemoai.client.memory.faiss_store import FAISSEpisodicStore
from mnemoai.models.controllers.embeddings_controller import EmbeddingsController
from mnemoai.server.tools.rag.session import SessionRAG
from mnemoai.utils.config import config
from mnemoai.utils.embedding_integrity import is_legacy_synthetic
from tests.unit.test_embedding_integrity import forget_integrity_stamp

pytestmark = pytest.mark.integration


def _controller():
    model = (config.get("RAG", {}) or {}).get("EMBED_MODEL_ID")
    if not model:
        pytest.skip("no embedding model configured")
    controller = EmbeddingsController(model)
    # Exercise the provider, not a previously cached query, including recovery.
    controller.cache_enabled = False
    return controller


def _outage(_texts):
    raise ConnectionError("injected embedding service outage")


def _legacy_vector(dim):
    digest = hashlib.sha256(b"historical fabricated embedding").digest()
    vector = np.resize(np.frombuffer(digest, dtype=np.uint8).astype(np.float32), dim)
    return vector / np.linalg.norm(vector)


@pytest.mark.parametrize("backend", ["faiss", "chromadb"])
def test_live_document_embeddings_repair_outage_and_recovery(
    tmp_path, monkeypatch, backend, record_testsuite_property
):
    controller = _controller()
    record_testsuite_property("embedding_model", controller.embed_model_name)
    monkeypatch.setitem(
        config._config_data,
        "RAG",
        {**config.get("RAG", {}), "VECTOR_STORE": {"TYPE": backend}},
    )
    options = {
        "embed_model_config": controller.embed_model_config,
        "session_id": "live-documents",
        "rag_dir": str(tmp_path),
    }
    rag = SessionRAG(**options)
    rag.embeddings_controller = controller
    assert (
        rag.ingest("database.md", "Postgres connection pooling and SQL transactions.")
        > 0
    )
    assert rag.ingest("bread.md", "Sourdough bread baking with flour and starter.") > 0
    scores, hits = rag.query("postgres connection", top_k=2)
    assert hits[0]["doc_id"] == "database.md"
    assert all(np.isfinite(score) for score in scores)
    assert all("retrieval_method" not in hit for hit in hits)

    # Simulate a store written by the old release beside the live real vectors.
    rag.store.add(
        _legacy_vector(rag.store.dim)[None, :],
        [
            {
                "doc_id": "legacy.md",
                "chunk_idx": 0,
                "text": "postgres fabricated row",
                "session_id": rag.session_id,
            }
        ],
    )
    forget_integrity_stamp(rag.store.store)
    reopened = SessionRAG(**options)
    reopened.embeddings_controller = controller
    assert {m["doc_id"] for m in reopened.store.metadatas} == {
        "database.md",
        "bread.md",
    }
    assert list(tmp_path.rglob("*legacy-synthetic-metadata.json"))
    before = [dict(meta) for meta in reopened.store.metadatas]

    with monkeypatch.context() as outage:
        outage.setattr(controller, "_embed_raw", _outage)
        outage.setattr(controller, "_embed_retries", 1)
        scores, hits = reopened.query("postgres")
        assert scores == [1.0]
        assert [hit["doc_id"] for hit in hits] == ["database.md"]
        assert hits[0]["retrieval_method"] == "bm25"
        assert reopened.query("nomatchingwordxyz") == ([], [])
        with pytest.raises(RuntimeError, match="no vectors were stored"):
            reopened.ingest("database.md", "Replacement during the outage.")
        assert reopened.store.metadatas == before
        assert controller._embedding_cache == {}

    scores, hits = reopened.query("postgres connection")
    assert hits[0]["doc_id"] == "database.md"
    assert "retrieval_method" not in hits[0]
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("backend", [FAISSEpisodicStore, ChromaEpisodicStore])
def test_live_episode_embeddings_outage_and_recovery(
    tmp_path, monkeypatch, backend, record_testsuite_property
):
    controller = _controller()
    record_testsuite_property("embedding_model", controller.embed_model_name)
    vectors = controller.embed(
        ["Postgres connection pooling", "Sourdough bread baking"]
    )
    assert vectors.shape[0] == 2 and vectors.shape[1] > 0
    assert np.isfinite(vectors).all()
    assert not any(is_legacy_synthetic(vector) for vector in vectors)

    store = backend(str(tmp_path), controller)
    store.add(
        "Postgres connection pooling",
        {"task": "configure postgres connection pooling", "tools": "execute_bash"},
        episode_id="database",
    )
    store.add(
        "Sourdough bread baking",
        {"task": "bake sourdough bread", "tools": ""},
        episode_id="bread",
    )
    assert store.search("postgres connection")[0]["task"].startswith(
        "configure postgres"
    )

    with monkeypatch.context() as outage:
        outage.setattr(controller, "_embed_raw", _outage)
        outage.setattr(controller, "_embed_retries", 1)
        reopened = backend(str(tmp_path), controller)
        hits = reopened.search("postgres")
        assert len(hits) == 1 and hits[0]["retrieval_method"] == "bm25"
        assert hits[0]["task"] == "configure postgres connection pooling"
        assert reopened.search("nomatchingwordxyz") == []
        with pytest.raises(RuntimeError, match="no vectors were stored"):
            reopened.add("new episode", {"task": "new episode"}, episode_id="new")
        assert controller._embedding_cache == {}

    hits = reopened.search("postgres connection")
    assert hits[0]["task"] == "configure postgres connection pooling"
    assert "retrieval_method" not in hits[0]
    assert len(reopened.search("bread postgres", top_k=10)) == 2
