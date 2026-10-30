"""Local storage must never execute an embedding function from saved metadata."""

import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from pathlib import Path

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
    old = old_client.create_collection(
        "existing", configuration={"hnsw": {
            "space": "cosine", "ef_construction": 64, "max_neighbors": 8,
        }},
    )
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
        assert collection.configuration["hnsw"]["space"] == "cosine"
        assert collection.configuration["hnsw"]["ef_construction"] == 64
        assert collection.configuration["hnsw"]["max_neighbors"] == 8
    finally:
        client.close()


@pytest.mark.parametrize("acquisition", ["create", "get", "get_or_create", "by_id", "list"])
def test_all_collection_acquisitions_require_precomputed_vectors(tmp_path, acquisition):
    client = persistent_client(tmp_path)
    try:
        created = client.create_collection("precomputed")
        if acquisition == "create":
            collection = created
        elif acquisition == "get":
            collection = client.get_collection("precomputed")
        elif acquisition == "get_or_create":
            collection = client.get_or_create_collection("precomputed")
        elif acquisition == "by_id":
            if not hasattr(client, "get_collection_by_id"):
                pytest.skip("this Chroma version has no collection-by-ID API")
            collection = client.get_collection_by_id(created.id)
        else:
            collection, = client.list_collections()
        collection.add(ids=["kept"], embeddings=[[1.0, -1.0, 0.0]])
        assert collection.get()["ids"] == ["kept"]
        with pytest.raises(ValueError, match="precomputed embeddings"):
            collection.add(ids=["unsafe"], documents=["no implicit model"])
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


@pytest.mark.parametrize("backend", ["rag", "episodic", "merge"])
@pytest.mark.parametrize("location", [
    "configuration", "default_dense", "key_dense", "default_sparse", "key_sparse",
])
def test_reopened_store_never_deserializes_saved_embedding_functions(
    tmp_path, monkeypatch, backend, location,
):
    from chromadb.utils.embedding_functions import (
        known_embedding_functions,
        sparse_known_embedding_functions,
    )

    class UntrustedEmbedding:
        @staticmethod
        def build_from_config(config):
            pytest.fail("saved embedding factory executed")

    for registry in (known_embedding_functions, sparse_known_embedding_functions):
        monkeypatch.setitem(registry, "untrusted-test", UntrustedEmbedding)

    def open_store():
        if backend == "rag":
            return ChromaStore(dim=3, rag_dir=str(tmp_path))
        return ChromaEpisodicStore(str(tmp_path), _Embeddings())

    store = open_store()
    store.collection.add(
        ids=["kept"], embeddings=[[1.0, -1.0, 0.0]], documents=["original"],
        metadatas=[{"task": "original", "solution": "retained"}],
    )
    name = store.collection.name
    database = Path(store.persist_path) / "chroma.sqlite3"
    store.client.close()
    # Modify actual persisted JSON, then reopen through the application paths.
    # A metadata key named "embedding_function" must retain its schema too.
    with closing(sqlite3.connect(database)) as db, db:
        raw_config, raw_schema = db.execute(
            "SELECT config_json_str, schema_str FROM collections WHERE name=?", (name,),
        ).fetchone()
        configuration, schema = json.loads(raw_config), json.loads(raw_schema)
        payload = {"type": "known", "name": "untrusted-test", "config": {"trust_remote_code": True}}
        if location == "configuration":
            configuration["embedding_function"] = payload
        else:
            if location == "key_sparse":
                schema["keys"]["embedding_function"] = {
                    "sparse_vector": deepcopy(schema["defaults"]["sparse_vector"]),
                }
            types = schema["defaults"] if location.startswith("default") else schema["keys"][
                "#embedding" if location == "key_dense" else "embedding_function"
            ]
            value_type, index_type = (
                ("float_list", "vector_index") if location.endswith("dense")
                else ("sparse_vector", "sparse_vector_index")
            )
            types[value_type][index_type]["config"]["embedding_function"] = payload
        raw_config, raw_schema = json.dumps(configuration), json.dumps(schema)
        db.execute(
            "UPDATE collections SET config_json_str=?, schema_str=? WHERE name=?",
            (raw_config, raw_schema, name),
        )

    store = open_store()
    try:
        collection = _collection(tmp_path) if backend == "merge" else store.collection
        assert collection is not None
        # Inspecting configuration/schema also must not construct stored models.
        assert collection.configuration["embedding_function"] is None
        assert collection.schema is not None
        if location == "key_sparse":
            assert "embedding_function" in collection.schema.keys
        assert collection.get()["documents"] == ["original"]
        collection.add(ids=["added"], embeddings=[[0.0, 1.0, -1.0]], documents=["added"])
        collection.update(ids=["added"], embeddings=[[0.0, 1.0, -1.0]], documents=["updated"])
        collection.upsert(ids=["upserted"], embeddings=[[-1.0, 0.0, 1.0]], documents=["upserted"])
        assert collection.query(
            query_embeddings=[[1.0, -1.0, 0.0]], n_results=1,
        )["ids"] == [["kept"]]
        assert collection.count() == 3
        with pytest.raises(ValueError, match="precomputed embeddings"):
            collection.add(ids=["unsafe"], documents=["missing vector"])
        with pytest.raises(ValueError, match="precomputed embeddings"):
            collection.query(query_texts=["missing vector"])
        # Protection is an in-memory view, not a migration of users' databases.
        with closing(sqlite3.connect(database)) as db:
            assert db.execute(
                "SELECT config_json_str, schema_str FROM collections WHERE name=?", (name,),
            ).fetchone() == (raw_config, raw_schema)
    finally:
        store.client.close()
