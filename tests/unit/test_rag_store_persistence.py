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
        # The bug was a hardcoded `/tmp/rag_store*` fallback, NOT "any path under
        # /tmp": pytest's own tmp_path IS /tmp/pytest-of-*/ on Linux (it's
        # /private/var/folders/ on macOS, which is why asserting on the /tmp
        # prefix passed locally and failed in CI). Assert the real property —
        # the store lands in the profile dir it was given.
        assert str(tmp_path) in store.persist_path
        assert "rag_store" in os.path.basename(store.persist_path)

    def test_no_args_uses_profile_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(faiss_store, "profile_dir", lambda: tmp_path)
        store = FaissStore(4)
        assert str(tmp_path) in store.persist_path
        assert "rag_store" in os.path.basename(store.persist_path)

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


class TestClearingTheStore:
    """``clear_documents`` assigned to ``store.metadatas`` / ``store.index`` —
    both getter-ONLY properties on ``VectorStoreController``. Every call raised
    ``AttributeError`` into a bare ``except``, so clearing never once worked and
    reported only a generic "Error clearing documents". Each backend already had
    a correct ``clear()``; the tool now calls it.
    """

    def _store(self, tmp_path):
        s = FaissStore(dim=4, persist_path=str(tmp_path / "idx"))
        s.add(
            np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype="float32"),
            [{"id": 1}, {"id": 2}],
        )
        return s

    def test_assigning_the_property_is_what_used_to_fail(self):
        from mnemoai.server.tools.rag.vector_store_controller import (
            VectorStoreController,
        )

        c = VectorStoreController.__new__(VectorStoreController)

        class _Fake:
            metadatas = [1, 2]
            index = "idx"

        c.store = _Fake()
        with pytest.raises(AttributeError):
            c.metadatas = []  # exactly what clear_documents did

    def test_clear_through_the_controller_empties_the_store(self, tmp_path):
        from mnemoai.server.tools.rag.vector_store_controller import (
            VectorStoreController,
        )

        store = self._store(tmp_path)
        assert len(store.metadatas) == 2 and store.index.ntotal == 2

        c = VectorStoreController.__new__(VectorStoreController)
        c.store = store
        c.clear()

        assert store.metadatas == []
        assert store.index.ntotal == 0

    def test_the_cleared_state_is_persisted(self, tmp_path):
        store = self._store(tmp_path)
        store.clear()
        with open(store.persist_path + ".meta.json") as f:
            assert json.load(f) == []

    def test_clearing_drops_the_bm25_index_too(self, tmp_path):
        # Vectors alone isn't "cleared": BM25 is built separately and kept a
        # tokenized copy of every document for the life of the process. Nothing
        # was served from it only because normalized_bm25_candidates discards
        # indices past the (now empty) metadata list — isolation resting on a
        # bounds check, not on the data being gone.
        from mnemoai.server.tools.rag.session import SessionRAG
        from mnemoai.utils.bm25 import BM25

        s = SessionRAG.__new__(SessionRAG)
        s.store = self._store(tmp_path)
        s.bm25 = BM25()
        s.bm25.fit(["secret plan alpha"])
        assert max(s.bm25.score("secret plan")) > 0

        s.clear()

        assert s.bm25 is None
        assert s.store.index.ntotal == 0
        assert s.store.metadatas == []

    def test_rebuilding_on_an_empty_store_drops_a_stale_index(self, tmp_path):
        from mnemoai.server.tools.rag.session import SessionRAG
        from mnemoai.utils.bm25 import BM25

        s = SessionRAG.__new__(SessionRAG)
        s.store = FaissStore(dim=4, persist_path=str(tmp_path / "empty"))
        s.bm25 = BM25()
        s.bm25.fit(["stale text from before"])
        s._rebuild_bm25()
        assert s.bm25 is None

    def test_the_tool_clears_vectors_and_keyword_index(self, tmp_path, monkeypatch):
        """Drives the registered tool end-to-end rather than grepping its source.

        A source assertion passes if the string merely appears in a comment, and it
        says nothing about whether the call actually empties anything.
        """
        from mnemoai.server.tools import rag_tool
        from mnemoai.server.tools.rag.session import SessionRAG
        from mnemoai.utils.bm25 import BM25

        session = SessionRAG.__new__(SessionRAG)
        session.store = self._store(tmp_path)
        session.bm25 = BM25()
        session.bm25.fit(["secret plan alpha"])

        # Capture the tool the register function installs.
        captured = {}

        class _MCP:
            def tool(self, *a, **k):
                def deco(fn):
                    captured[fn.__name__] = fn
                    return fn
                return deco

        monkeypatch.setattr(rag_tool, "get_rag_session", lambda: session)
        rag_tool.register_rag_tools(_MCP())
        clear = captured["clear_documents"]

        result = clear()

        assert "cleared" in result.lower()
        assert session.store.index.ntotal == 0
        assert session.store.metadatas == []
        assert session.bm25 is None

    def test_the_tool_reports_nothing_to_clear_before_any_ingest(self, monkeypatch):
        # Previously claimed the backend "does not support clearing", because the
        # store is None until the first ingest.
        from mnemoai.server.tools import rag_tool
        from mnemoai.server.tools.rag.session import SessionRAG

        session = SessionRAG.__new__(SessionRAG)
        session.store = None
        session.bm25 = None
        captured = {}

        class _MCP:
            def tool(self, *a, **k):
                def deco(fn):
                    captured[fn.__name__] = fn
                    return fn
                return deco

        monkeypatch.setattr(rag_tool, "get_rag_session", lambda: session)
        rag_tool.register_rag_tools(_MCP())
        result = captured["clear_documents"]()
        assert "no documents to clear" in result.lower()
        assert "does not support" not in result.lower()
