"""Document replacement and session changes in the long-lived MCP process."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from mnemoai.server.tools.rag import session
from mnemoai.server.tools.rag_tool import register_rag_tools


@pytest.mark.parametrize("backend", ["faiss", "chromadb"])
@pytest.mark.parametrize("already_initialized", [False, True])
def test_clear_waits_for_inflight_ingestion_and_leaves_no_late_rows(
    tmp_path, monkeypatch, backend, already_initialized,
):
    monkeypatch.setattr(session.config, "get", lambda key, default=None: (
        {"VECTOR_STORE": {"TYPE": backend}} if key == "RAG" else default
    ))
    rag = session.SessionRAG(
        embed_model_config={"NAME": "fixture", "TYPE": "openai", "DIMENSION": 3},
        session_id="clear-inflight", rag_dir=str(tmp_path),
    )
    monkeypatch.setattr(rag, "_embed_batch", lambda texts: np.ones((len(texts), 3), dtype=np.float32))
    if already_initialized:
        rag.ingest("existing", "old text")
    tools = {}
    class Registry:
        def tool(self):
            def register(fn):
                tools[fn.__name__] = fn
                return fn
            return register
    register_rag_tools(Registry())
    monkeypatch.setattr("mnemoai.server.tools.rag_tool.get_rag_session", lambda: rag)
    entered, finish_embedding, cleared = threading.Event(), threading.Event(), threading.Event()
    errors = []
    def embed(texts):
        entered.set()
        assert finish_embedding.wait(5)
        return np.ones((len(texts), 3), dtype=np.float32)
    monkeypatch.setattr(rag, "_embed_batch", embed)
    def ingest():
        try:
            rag.ingest("later", "new text")
        except BaseException as exc:
            errors.append(exc)
    def clear():
        try:
            assert "cleared" in tools["clear_documents"]()
            cleared.set()
        except BaseException as exc:
            errors.append(exc)
    writer = threading.Thread(target=ingest)
    cleaner = threading.Thread(target=clear)
    writer.start()
    try:
        assert entered.wait(5)
        cleaner.start()
        assert not cleared.wait(0.05)
    finally:
        finish_embedding.set()
        writer.join(timeout=5)
        if cleaner.ident is not None:
            cleaner.join(timeout=5)
    assert not errors
    assert cleared.is_set() and not rag.store.metadatas and rag.bm25 is None
    assert rag.query("text") == ([], [])


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
@pytest.mark.parametrize("backend", ["faiss", "chromadb"])
@pytest.mark.parametrize("changed", ["NAME", "API_BASE", "missing_stamp"])
def test_persisted_rag_rejects_mixed_embedding_identity_but_keeps_keyword_reads(
    tmp_path, monkeypatch, backend, changed,
):
    monkeypatch.setattr(session.config, "get", lambda key, default=None: (
        {"VECTOR_STORE": {"TYPE": backend}} if key == "RAG" else default
    ))
    model = {"TYPE": "openai", "NAME": "first", "DIMENSION": 3, "API_BASE": "https://first.example"}
    options = {"session_id": "identity", "rag_dir": str(tmp_path)}
    rag = session.SessionRAG(embed_model_config=model, **options)
    monkeypatch.setattr(rag, "_embed_batch", lambda texts: np.ones((len(texts), 3), dtype=np.float32))
    rag.ingest("database.md", "postgres connection")
    if changed == "missing_stamp":
        rag.store._identity_path.unlink()
    else:
        model = {**model, changed: "second" if changed == "NAME" else "https://second.example"}
    reopened = session.SessionRAG(embed_model_config=model, **options)
    monkeypatch.setattr(reopened, "_embed_batch", lambda texts: np.ones((len(texts), 3), dtype=np.float32))
    _, hits = reopened.query("postgres")
    assert hits[0]["retrieval_method"] == "bm25"
    assert hits[0]["doc_id"] == "database.md"
    with pytest.raises((ValueError, RuntimeError), match="identity"):
        reopened.ingest("new.md", "new content")
    assert len(reopened.store.metadatas) == 1
    reopened.clear()
    reopened.ingest("replacement.md", "postgres replacement")
    assert not reopened.store.integrity_error
