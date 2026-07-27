"""RAG store persistence: JSON metadata, no pickle, no /tmp.

These are the first tests for ``server/tools/rag/``. They pin the two security
properties of the persistence layer:

* metadata is JSON, so loading a store can never execute code,
* the store never lands in a world-writable shared directory.

Plus the correctness property that made the pickle swap safe: a stale or
mismatched pair on disk rebuilds instead of returning wrong chunks.
"""

import json
import os

import numpy as np
import pytest

faiss_store = pytest.importorskip(
    "mnemoai.server.tools.rag.faiss_store", reason="faiss not installed"
)
FaissStore = faiss_store.FaissStore


def _vec(dim, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((1, dim), dtype=np.float32)


class TestNoPickle:
    def test_metadata_file_is_json(self, tmp_path):
        store = FaissStore(4, session_id="s1", rag_dir=str(tmp_path))
        store.add(_vec(4), [{"text": "hello", "source": "a.md"}])

        assert store.metadata_path.endswith(".meta.json")
        loaded = json.loads(open(store.metadata_path, encoding="utf-8").read())
        assert loaded == [{"text": "hello", "source": "a.md"}]

    def test_module_does_not_import_pickle(self):
        """A regression guard: reintroducing pickle here is an RCE sink."""
        source = open(faiss_store.__file__, encoding="utf-8").read()
        code_lines = [
            ln
            for ln in source.splitlines()
            if ln.strip().startswith(("import ", "from "))
        ]
        assert not any("pickle" in ln for ln in code_lines)

    def test_legacy_pickle_meta_is_ignored(self, tmp_path):
        """An old .meta pickle must not be loaded -- the store rebuilds."""
        store = FaissStore(4, session_id="legacy", rag_dir=str(tmp_path))
        store.add(_vec(4), [{"text": "current"}])

        # Simulate a leftover pickle from the previous format.
        legacy = store.persist_path + ".meta"
        with open(legacy, "wb") as f:
            f.write(b"\x80\x04]\x94.")

        reopened = FaissStore(4, session_id="legacy", rag_dir=str(tmp_path))
        assert reopened.metadatas == [{"text": "current"}]


class TestNoTmpFallback:
    def test_session_without_rag_dir_uses_profile_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(faiss_store, "profile_dir", lambda: tmp_path)
        store = FaissStore(4, session_id="s2")
        assert not store.persist_path.startswith("/tmp/")
        assert str(tmp_path) in store.persist_path

    def test_no_args_uses_profile_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(faiss_store, "profile_dir", lambda: tmp_path)
        store = FaissStore(4)
        assert not store.persist_path.startswith("/tmp/")
        assert str(tmp_path) in store.persist_path

    def test_source_has_no_tmp_literal(self):
        source = open(faiss_store.__file__, encoding="utf-8").read()
        code = "\n".join(
            ln for ln in source.splitlines() if not ln.strip().startswith("#")
        )
        assert '"/tmp/rag_store' not in code
        assert "f\"/tmp/rag_store" not in code


class TestRoundTripAndRecovery:
    def test_persists_across_reopen(self, tmp_path):
        store = FaissStore(4, session_id="s3", rag_dir=str(tmp_path))
        v = _vec(4, seed=1)
        store.add(v, [{"text": "persisted"}])

        reopened = FaissStore(4, session_id="s3", rag_dir=str(tmp_path))
        assert reopened.metadatas == [{"text": "persisted"}]
        scores, metas = reopened.search(v[0], top_k=1)
        assert metas == [{"text": "persisted"}]

    def test_length_mismatch_rebuilds(self, tmp_path):
        """A desynced index/metadata pair would return the wrong chunk."""
        store = FaissStore(4, session_id="s4", rag_dir=str(tmp_path))
        store.add(_vec(4), [{"text": "one"}])

        # Corrupt only the metadata side.
        with open(store.metadata_path, "w", encoding="utf-8") as f:
            json.dump([{"text": "one"}, {"text": "phantom"}], f)

        reopened = FaissStore(4, session_id="s4", rag_dir=str(tmp_path))
        assert reopened.metadatas == []
        assert reopened.index.ntotal == 0

    def test_corrupt_json_rebuilds(self, tmp_path):
        store = FaissStore(4, session_id="s5", rag_dir=str(tmp_path))
        store.add(_vec(4), [{"text": "one"}])

        with open(store.metadata_path, "w", encoding="utf-8") as f:
            f.write("{not json")

        reopened = FaissStore(4, session_id="s5", rag_dir=str(tmp_path))
        assert reopened.metadatas == []

    def test_non_list_metadata_rebuilds(self, tmp_path):
        store = FaissStore(4, session_id="s6", rag_dir=str(tmp_path))
        store.add(_vec(4), [{"text": "one"}])

        with open(store.metadata_path, "w", encoding="utf-8") as f:
            json.dump({"not": "a list"}, f)

        reopened = FaissStore(4, session_id="s6", rag_dir=str(tmp_path))
        assert reopened.metadatas == []

    def test_clear_empties_and_persists(self, tmp_path):
        store = FaissStore(4, session_id="s7", rag_dir=str(tmp_path))
        store.add(_vec(4), [{"text": "gone"}])
        store.clear()

        reopened = FaissStore(4, session_id="s7", rag_dir=str(tmp_path))
        assert reopened.metadatas == []

    def test_persist_creates_missing_parent_dir(self, tmp_path):
        nested = tmp_path / "does" / "not" / "exist"
        store = FaissStore(4, session_id="s8", rag_dir=str(nested))
        store.add(_vec(4), [{"text": "ok"}])
        assert os.path.exists(store.metadata_path)
