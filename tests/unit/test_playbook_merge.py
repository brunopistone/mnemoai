"""Regression tests for PlaybookStore dedup/merge.

Focus: the embedding-merge fallback must NOT reference a non-existent
``__wrapped__`` (which raised AttributeError), and must fall back to the
keyword dedup instead — a real crash path with no prior coverage.
"""

from mnemoai.client.memory.playbook_store import PlaybookStore


def _store():
    # __new__ so we don't touch disk/config; set only what the merge methods use.
    s = PlaybookStore.__new__(PlaybookStore)
    s.max_entries = 100
    s.similarity_threshold = 0.85
    s.embeddings = None
    return s


def _entries():
    return [
        {"strategy": "use glob before grep", "confidence": 0.9, "timestamp": "2"},
        {"strategy": "use glob before grep", "confidence": 0.5, "timestamp": "1"},
        {"strategy": "read file before editing", "confidence": 0.8, "timestamp": "3"},
    ]


class TestMergeByStrategyKey:
    def test_keyword_dedup_keeps_highest_confidence(self):
        s = _store()
        out = s._merge_by_strategy_key(_entries())
        # Two distinct strategies; the higher-confidence duplicate is kept.
        strategies = sorted(e["strategy"] for e in out)
        assert strategies == ["read file before editing", "use glob before grep"]
        glob = next(e for e in out if e["strategy"] == "use glob before grep")
        assert glob["confidence"] == 0.9

    def test_merge_similar_dispatches_to_keyword_when_no_embeddings(self):
        s = _store()  # embeddings=None
        out = s._merge_similar(_entries())
        assert len(out) == 2  # deduped by strategy key

    def test_embedding_merge_fallback_does_not_crash(self):
        # The bug: fallback called self._merge_similar.__wrapped__ (no such attr).
        # Now it must fall back to _merge_by_strategy_key without raising.
        s = _store()

        class _BrokenEmbeddings:
            def embed(self, texts):
                raise RuntimeError("embedding backend down")

        s.embeddings = _BrokenEmbeddings()
        out = s._merge_with_embeddings(_entries())  # must NOT raise
        assert len(out) == 2  # fell back to keyword dedup

    def test_single_entry_is_noop(self):
        s = _store()
        one = [{"strategy": "x", "confidence": 1.0, "timestamp": "1"}]
        assert s._merge_similar(one) == one
