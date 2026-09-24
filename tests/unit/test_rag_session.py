"""Document replacement and session changes in the long-lived MCP process."""

from types import SimpleNamespace

import numpy as np
import pytest

from mnemoai.server.tools.rag import session


def test_new_session_pointer_invalidates_the_server_cache(tmp_path, monkeypatch):
    pointer = tmp_path / "pointer"
    pointer.write_text("new-session")
    monkeypatch.setattr(session, "rag_session_pointer_path", lambda: pointer)
    monkeypatch.setattr(session, "profile_dir", lambda: tmp_path)
    monkeypatch.setattr(
        session, "_rag_session", SimpleNamespace(session_id="old-session")
    )
    monkeypatch.setattr(
        session, "SessionRAG", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    assert session.get_rag_session().session_id == "new-session"


@pytest.mark.parametrize("backend", ["faiss", "chromadb"])
def test_reindexing_replaces_only_that_documents_old_chunks(
    tmp_path, monkeypatch, backend
):
    monkeypatch.setattr(
        session.config,
        "get",
        lambda key, default=None: (
            {
                "VECTOR_STORE": {"TYPE": backend},
            }
            if key == "RAG"
            else default
        ),
    )
    rag = session.SessionRAG(
        embed_model_config={"NAME": "fake", "TYPE": "openai", "DIMENSION": 3},
        session_id="replace-test",
        rag_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        rag, "_embed_batch", lambda texts: np.ones((len(texts), 3), dtype=np.float32)
    )
    rag.ingest("/a/notes.md", "old version")
    rag.ingest("/b/notes.md", "other document")
    rag.ingest("/a/notes.md", "new version")
    by_doc = {meta["doc_id"]: meta["text"] for meta in rag.store.metadatas}
    assert by_doc == {"/a/notes.md": "new version", "/b/notes.md": "other document"}
    assert len(rag.store.metadatas) == 2


@pytest.mark.parametrize("backend", ["faiss", "chromadb"])
def test_offline_search_uses_only_existing_text_and_repairs_old_vectors(
    tmp_path, monkeypatch, backend
):
    from tests.unit.test_embedding_integrity import (
        forget_integrity_stamp,
        legacy_vector,
        real_vector,
    )

    monkeypatch.setattr(
        session.config,
        "get",
        lambda key, default=None: (
            {
                "VECTOR_STORE": {"TYPE": backend},
            }
            if key == "RAG"
            else default
        ),
    )
    options = {
        "embed_model_config": {"NAME": "fake", "TYPE": "openai", "DIMENSION": 128},
        "session_id": "outage-test",
        "rag_dir": str(tmp_path),
    }
    rag = session.SessionRAG(**options)
    monkeypatch.setattr(
        rag,
        "_embed_batch",
        lambda texts: np.vstack(
            [legacy_vector() if text == "legacy" else real_vector() for text in texts]
        ),
    )
    rag.ingest("database.md", "postgres connection")
    rag.ingest("food.md", "sourdough bread")
    rag.ingest("poison.md", "legacy")
    forget_integrity_stamp(rag.store.store)
    reopened = session.SessionRAG(**options)
    monkeypatch.setattr(
        reopened,
        "_embed_batch",
        lambda texts: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
    )
    scores, hits = reopened.query("postgres")
    assert scores == [1.0]
    assert [hit["doc_id"] for hit in hits] == ["database.md"]
    assert hits[0]["retrieval_method"] == "bm25"
    assert len(reopened.store.metadatas) == 2
    with pytest.raises(RuntimeError, match="unavailable"):
        reopened.ingest("new.md", "new document")


@pytest.mark.parametrize("backend", ["faiss", "chromadb"])
def test_fully_poisoned_store_can_be_rebuilt_at_a_new_real_dimension(
    tmp_path, monkeypatch, backend
):
    from tests.unit.test_embedding_integrity import (
        forget_integrity_stamp,
        legacy_vector,
    )

    monkeypatch.setattr(
        session.config,
        "get",
        lambda key, default=None: (
            {
                "VECTOR_STORE": {"TYPE": backend},
            }
            if key == "RAG"
            else default
        ),
    )
    options = {
        "embed_model_config": {"NAME": "fake", "TYPE": "openai", "DIMENSION": 128},
        "session_id": "recovery-test",
        "rag_dir": str(tmp_path),
    }
    rag = session.SessionRAG(**options)
    monkeypatch.setattr(rag, "_embed_batch", lambda texts: legacy_vector()[None, :])
    rag.ingest("poison.md", "legacy")
    forget_integrity_stamp(rag.store.store)
    reopened = session.SessionRAG(**options)
    assert reopened.store.metadatas == []
    monkeypatch.setattr(
        reopened,
        "_embed_batch",
        lambda texts: np.ones((len(texts), 1024), dtype=np.float32),
    )
    reopened.ingest("real.md", "real document")
    assert [meta["doc_id"] for meta in reopened.store.metadatas] == ["real.md"]
    assert reopened.store.dim == 1024
