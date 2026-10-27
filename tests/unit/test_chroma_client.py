"""Local storage must never execute an embedding function from saved metadata."""

import chromadb
import pytest

from mnemoai.client.memory.chroma_store import ChromaEpisodicStore
from mnemoai.client.memory.model_dir_merge import _collection
from mnemoai.server.tools.rag.chroma_store import ChromaStore
from mnemoai.utils.chroma_client import PRECOMPUTED_EMBEDDINGS, persistent_client


class _Embeddings:
    def fingerprint(self):
        return "security-test|3"


def test_environment_cannot_redirect_embedded_storage_to_http(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_API_IMPL", "chromadb.api.fastapi.FastAPI")
    monkeypatch.setenv("CHROMA_SERVER_HOST", "untrusted.invalid")
    client = persistent_client(tmp_path)
    try:
        collection = client.create_collection(
            "local_only", embedding_function=PRECOMPUTED_EMBEDDINGS
        )
        collection.add(ids=["kept"], embeddings=[[1.0, -1.0, 0.0]])
        assert collection.get()["ids"] == ["kept"]
        assert (tmp_path / "chroma.sqlite3").exists()
        assert client.get_settings().chroma_api_impl == "chromadb.api.rust.RustBindingsAPI"
    finally:
        client.close()


def test_existing_default_embedding_collection_keeps_its_data(tmp_path):
    old_client = chromadb.PersistentClient(path=str(tmp_path))
    old = old_client.create_collection("existing")
    old.add(ids=["kept"], embeddings=[[1.0, -1.0, 0.0]], documents=["original"])
    old_client.close()
    client = persistent_client(tmp_path)
    try:
        collection = client.get_collection(
            "existing", embedding_function=PRECOMPUTED_EMBEDDINGS
        )
        assert collection.get()["documents"] == ["original"]
        assert collection.query(
            query_embeddings=[[1.0, -1.0, 0.0]], n_results=1
        )["ids"] == [["kept"]]
    finally:
        client.close()


@pytest.mark.parametrize("backend", ["rag", "episodic", "merge"])
@pytest.mark.parametrize("reset", [False, True])
def test_saved_embedding_configuration_cannot_execute(tmp_path, monkeypatch, backend, reset):
    from chromadb.utils.embedding_functions import known_embedding_functions

    class UntrustedEmbedding:
        @staticmethod
        def build_from_config(config):
            pytest.fail("saved embedding configuration was executed")

    monkeypatch.setitem(known_embedding_functions, "untrusted-test", UntrustedEmbedding)
    if backend == "rag":
        store = ChromaStore(dim=3, rag_dir=str(tmp_path))
    else:
        store = ChromaEpisodicStore(str(tmp_path), _Embeddings())
    if reset:
        store.clear()
    collection = _collection(tmp_path) if backend == "merge" else store.collection
    assert collection is not None
    # Raw persisted configuration consumed by Chroma's _embed. The sentinel
    # fails if a missing vector triggers executable-model deserialization.
    collection._model.configuration_json["embedding_function"] = {
        "type": "known",
        "name": "untrusted-test",
        "config": {"trust_remote_code": True},
    }
    collection.add(ids=["kept"], embeddings=[[1.0, -1.0, 0.0]], documents=["text"])
    assert collection.query(
        query_embeddings=[[1.0, -1.0, 0.0]], n_results=1
    )["ids"] == [["kept"]]
    with pytest.raises(ValueError, match="precomputed embeddings"):
        collection.add(ids=["unsafe"], documents=["missing vector"])
    with pytest.raises(ValueError, match="precomputed embeddings"):
        collection.query(query_texts=["missing vector"], n_results=1)
    assert collection.get()["ids"] == ["kept"]
    store.client.close()
