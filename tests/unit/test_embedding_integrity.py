"""Repair only the legacy fabricated format, preserving real vectors and backups."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mnemoai.utils.embedding_integrity import (
    CHROMA_REPAIR_STAMP,
    is_legacy_synthetic,
    recover_faiss_repair,
    repair_chroma,
    repair_faiss,
)


def forget_integrity_stamp(store):
    """Model a pre-repair release when seeding synthetic vectors through a fixture."""
    if hasattr(store, "collection"):
        metadata = dict(store.collection.metadata or {})
        metadata.pop(CHROMA_REPAIR_STAMP, None)
        store.collection.modify(metadata=metadata or {"legacy_fixture": True})
    else:
        path = getattr(store, "index_path", store.persist_path)
        Path(str(path) + ".synthetic-checked.json").unlink(missing_ok=True)


def legacy_vector(dim=128):
    raw = np.frombuffer(
        hashlib.sha256(b"legacy fabricated vector").digest(), dtype=np.uint8
    )
    vector = np.resize(raw.astype(np.float32), dim)
    return vector / np.linalg.norm(vector)


def real_vector():
    vector = np.linspace(-1, 1, 128, dtype=np.float32)
    return vector / np.linalg.norm(vector)


@pytest.mark.parametrize("dim", [64, 128, 768, 1024, 1536])
def test_legacy_sha256_shape_is_identified(dim):
    assert is_legacy_synthetic(legacy_vector(dim))


def test_nonnegative_or_periodic_alone_is_not_enough():
    rng = np.random.default_rng(12)
    positive = rng.random(128)
    periodic = np.tile(rng.random(32), 4)
    for vector in (positive, periodic, np.ones(128), real_vector(), np.zeros(128)):
        norm = np.linalg.norm(vector)
        assert not is_legacy_synthetic(vector / norm if norm else vector)
    assert not is_legacy_synthetic(legacy_vector(32))


def test_faiss_repair_preserves_real_rows_and_is_idempotent(tmp_path, monkeypatch):
    import faiss

    path, metadata_path = tmp_path / "index.faiss", tmp_path / "metadata.json"
    index = faiss.IndexFlatIP(128)
    index.add(np.vstack([legacy_vector(), real_vector()]))
    metadata = [{"text": "legacy"}, {"text": "real"}]
    faiss.write_index(index, str(path))
    metadata_path.write_text(json.dumps(metadata))
    fixed, clean = repair_faiss(index, metadata, path, metadata_path)
    assert fixed.ntotal == 1 and clean == [{"text": "real"}]
    np.testing.assert_array_equal(fixed.reconstruct(0), real_vector())
    assert faiss.read_index(str(path)).ntotal == 1
    assert json.loads(metadata_path.read_text()) == clean
    backup = tmp_path / "index.faiss.legacy-synthetic-metadata.json"
    before = backup.read_bytes()
    assert json.loads(before)[0]["metadata"] == {"text": "legacy"}
    monkeypatch.setattr(
        "mnemoai.utils.embedding_integrity.is_legacy_synthetic",
        lambda vector: pytest.fail("a checked store must not scan vectors again"),
    )
    again, _ = repair_faiss(fixed, clean, path, metadata_path)
    assert again is fixed
    assert backup.read_bytes() == before


def test_chroma_repair_preserves_real_rows_and_is_idempotent(tmp_path, monkeypatch):
    import chromadb

    collection = chromadb.PersistentClient(path=str(tmp_path)).create_collection(
        "test_vectors"
    )
    collection.add(
        ids=["legacy", "real"],
        embeddings=np.vstack([legacy_vector(), real_vector()]).tolist(),
        metadatas=[{"text": "legacy"}, {"text": "real"}],
    )
    assert repair_chroma(collection, tmp_path) == 1
    assert collection.get()["ids"] == ["real"]
    backup = tmp_path / "legacy-synthetic-metadata.json"
    before = backup.read_bytes()
    assert json.loads(before)[0]["id"] == "legacy"
    monkeypatch.setattr(
        type(collection),
        "get",
        lambda *args, **kwargs: pytest.fail(
            "a checked collection must not fetch vectors"
        ),
    )
    assert repair_chroma(collection, tmp_path) == 0
    assert backup.read_bytes() == before


@pytest.mark.parametrize("backend", ["faiss", "chroma"])
def test_an_old_writer_invalidates_the_completion_marker(tmp_path, backend):
    if backend == "chroma":
        import chromadb

        collection = chromadb.PersistentClient(path=str(tmp_path)).create_collection(
            "old_writer"
        )
        collection.add(
            ids=["real"],
            embeddings=[real_vector().tolist()],
            metadatas=[{"text": "real"}],
        )
        assert repair_chroma(collection, tmp_path) == 0
        # Older releases append without refreshing the new version/count marker.
        collection.add(
            ids=["legacy"],
            embeddings=[legacy_vector().tolist()],
            metadatas=[{"text": "legacy"}],
        )
        assert repair_chroma(collection, tmp_path) == 1
        assert collection.get()["ids"] == ["real"]
    else:
        import faiss

        index_path, metadata_path = tmp_path / "index", tmp_path / "metadata.json"
        index = faiss.IndexFlatIP(128)
        index.add(real_vector()[None, :])
        metadata = [{"text": "real"}]
        faiss.write_index(index, str(index_path))
        metadata_path.write_text(json.dumps(metadata))
        repair_faiss(index, metadata, index_path, metadata_path)
        index.add(legacy_vector()[None, :])
        metadata.append({"text": "legacy"})
        faiss.write_index(index, str(index_path))
        metadata_path.write_text(json.dumps(metadata))
        repaired, remaining = repair_faiss(index, metadata, index_path, metadata_path)
        assert repaired.ntotal == 1 and remaining == [{"text": "real"}]


def test_faiss_interrupted_repair_finishes_before_metadata_is_loaded(
    tmp_path, monkeypatch
):
    import faiss

    import mnemoai.utils.embedding_integrity as integrity

    path, meta_path = tmp_path / "index.faiss", tmp_path / "metadata.json"
    index = faiss.IndexFlatIP(128)
    index.add(np.vstack([legacy_vector(), real_vector()]))
    metadata = [{"text": "legacy"}, {"text": "real"}]
    faiss.write_index(index, str(path))
    meta_path.write_text(json.dumps(metadata))
    write = integrity.atomic_write_json

    def fail_metadata(target, value):
        if target == str(meta_path):
            raise OSError("interrupted after replacing index")
        write(target, value)

    monkeypatch.setattr(integrity, "atomic_write_json", fail_metadata)
    with pytest.raises(OSError, match="interrupted"):
        repair_faiss(index, metadata, path, meta_path)
    assert faiss.read_index(str(path)).ntotal == 1
    assert len(json.loads(meta_path.read_text())) == 2
    monkeypatch.setattr(integrity, "atomic_write_json", write)
    recover_faiss_repair(faiss.read_index(str(path)), path, meta_path)
    assert json.loads(meta_path.read_text()) == [{"text": "real"}]
    assert not (tmp_path / "index.faiss.synthetic-repair.json").exists()


@pytest.mark.parametrize("backend", ["chroma", "faiss"])
def test_backup_failure_never_deletes_vectors(tmp_path, monkeypatch, backend):
    import mnemoai.utils.embedding_integrity as integrity

    def fail_backup(*args):
        raise OSError("backup unavailable")

    monkeypatch.setattr(integrity, "_backup", fail_backup)
    if backend == "chroma":
        import chromadb

        collection = chromadb.PersistentClient(path=str(tmp_path)).create_collection(
            "test_vectors"
        )
        collection.add(ids=["legacy"], embeddings=[legacy_vector().tolist()])
        with pytest.raises(OSError, match="backup unavailable"):
            repair_chroma(collection, tmp_path)
        assert collection.count() == 1
    else:
        import faiss

        index = faiss.IndexFlatIP(128)
        index.add(legacy_vector()[None, :])
        with pytest.raises(OSError, match="backup unavailable"):
            repair_faiss(
                index, [{"text": "legacy"}], tmp_path / "index", tmp_path / "meta"
            )
        assert index.ntotal == 1


class _Embeddings:
    dim = 128

    def fingerprint(self):
        return "real-model-128"

    def embed(self, texts):
        return np.vstack(
            [legacy_vector() if text == "legacy" else real_vector() for text in texts]
        )


@pytest.fixture(
    params=["rag-faiss", "rag-chromadb", "episode-faiss", "episode-chromadb"]
)
def store_case(request, tmp_path, monkeypatch):
    from mnemoai.client.memory.chroma_store import ChromaEpisodicStore
    from mnemoai.client.memory.faiss_store import FAISSEpisodicStore
    from mnemoai.server.tools.rag.session import SessionRAG
    from mnemoai.utils.config import config

    kind = request.param
    backend = kind.split("-", 1)[1]
    monkeypatch.setitem(config._config_data, "RAG", {"VECTOR_STORE": {"TYPE": backend}})

    def open_store():
        if kind.startswith("rag-"):
            rag = SessionRAG(
                embed_model_config={"NAME": "fake", "TYPE": "openai", "DIMENSION": 128},
                session_id="integrity",
                rag_dir=str(tmp_path),
            )
            rag.embeddings_controller = _Embeddings()
            return rag
        cls = FAISSEpisodicStore if backend == "faiss" else ChromaEpisodicStore
        return cls(str(tmp_path), _Embeddings())

    def add(store, text):
        if kind.startswith("rag-"):
            store.ingest(text + ".md", text)
        else:
            store.add(text, {"task": text}, episode_id=text)

    def underlying(store):
        return store.store.store if kind.startswith("rag-") else store

    def recall(store):
        return (
            store.query("postgres")[1]
            if kind.startswith("rag-")
            else store.search("postgres")
        )

    return kind, open_store, add, underlying, recall


def test_corrupt_backup_preserves_keyword_reads_and_blocks_writes(store_case):
    kind, open_store, add, underlying, recall = store_case
    original = open_store()
    add(original, "postgres")
    add(original, "legacy")
    backend = underlying(original)
    forget_integrity_stamp(backend)
    if hasattr(backend, "collection"):
        backup = Path(backend.persist_path) / "legacy-synthetic-metadata.json"
    else:
        index_path = getattr(backend, "index_path", backend.persist_path)
        backup = Path(str(index_path) + ".legacy-synthetic-metadata.json")
    backup.write_text("invalid json")

    reopened = open_store()
    assert underlying(reopened).integrity_error
    hits = recall(reopened)
    assert len(hits) == 1 and hits[0]["retrieval_method"] == "bm25"
    with pytest.raises(RuntimeError, match="needs repair"):
        add(reopened, "new")
    if kind.startswith("episode-"):
        reopened.cleanup(max_episodes=0)
        assert len(recall(reopened)) == 1
    assert backup.read_text() == "invalid json"
    if hasattr(backend, "collection"):
        assert backend.collection.count() == 2
    else:
        import faiss

        assert faiss.read_index(str(index_path)).ntotal == 2


def test_normal_writes_refresh_the_check_marker_without_a_rescan(
    store_case, monkeypatch
):
    _, open_store, add, underlying, _ = store_case
    store = open_store()
    add(store, "postgres")
    monkeypatch.setattr(
        "mnemoai.utils.embedding_integrity.is_legacy_synthetic",
        lambda vector: pytest.fail(
            "a normally updated store must not rescan embeddings"
        ),
    )
    reopened = open_store()
    assert not underlying(reopened).integrity_error
    add(reopened, "bread")
    assert not underlying(open_store()).integrity_error


@pytest.mark.parametrize("failure", ["misalignment", "bad-journal", "broken-index"])
@pytest.mark.parametrize("store_case", ["rag-faiss", "episode-faiss"], indirect=True)
def test_damaged_faiss_pair_keeps_readable_text_without_overwriting_files(
    store_case, failure
):
    _, open_store, add, underlying, recall = store_case
    store = open_store()
    add(store, "postgres")
    backend = underlying(store)
    index_path = Path(getattr(backend, "index_path", backend.persist_path))
    metadata_path = Path(backend.metadata_path)
    if failure == "misalignment":
        values = json.loads(metadata_path.read_text())
        metadata_path.write_text(
            json.dumps(values + [{"text": "orphan", "task": "orphan"}])
        )
    elif failure == "bad-journal":
        Path(str(index_path) + ".synthetic-repair.json").write_text("invalid json")
    else:
        index_path.write_bytes(b"broken")
    before = (index_path.read_bytes(), metadata_path.read_bytes())
    reopened = open_store()
    assert underlying(reopened).integrity_error
    hits = recall(reopened)
    assert len(hits) == 1 and hits[0]["retrieval_method"] == "bm25"
    with pytest.raises(RuntimeError, match="needs repair"):
        add(reopened, "new")
    assert (index_path.read_bytes(), metadata_path.read_bytes()) == before
