"""The shared hybrid-search ranking (BM25 normalization + weighted merge).

This logic was duplicated across the episodic Chroma store, the episodic FAISS
store, and the session RAG store; it now lives in one place, so it gets tested
directly here instead of only through three backends that each need a live
vector index.
"""

import pytest

from mnemoai.utils.bm25 import BM25
from mnemoai.utils.hybrid_search import (
    candidate_count,
    merge_and_rank,
    normalized_bm25_candidates,
    rank_with_similarity,
)


class _FakeBM25:
    """A BM25 stand-in returning fixed raw scores."""

    def __init__(self, scores):
        self._scores = scores
        self.corpus_size = len(scores)

    def score(self, query):
        return list(self._scores)


def _metas(n):
    return [{"task": f"t{i}", "idx": i} for i in range(n)]


class TestNormalizedBm25Candidates:
    def test_scores_are_normalized_against_the_best_hit(self):
        out = normalized_bm25_candidates(_FakeBM25([2.0, 4.0, 1.0]), "q", 3, _metas(3))
        assert out[1][0] == 1.0  # best hit always 1.0
        assert out[0][0] == 0.5
        assert out[2][0] == 0.25

    def test_zero_and_negative_scores_dropped(self):
        out = normalized_bm25_candidates(
            _FakeBM25([4.0, 0.0, -1.0]), "q", 3, _metas(3)
        )
        assert set(out) == {0}

    def test_respects_candidate_k(self):
        out = normalized_bm25_candidates(
            _FakeBM25([1.0, 2.0, 3.0, 4.0, 5.0]), "q", 2, _metas(5)
        )
        assert set(out) == {4, 3}  # the two best

    def test_all_zero_scores_yields_nothing(self):
        assert normalized_bm25_candidates(_FakeBM25([0.0, 0.0]), "q", 5, _metas(2)) == {}

    def test_none_or_empty_bm25_yields_nothing(self):
        assert normalized_bm25_candidates(None, "q", 5, _metas(2)) == {}
        assert normalized_bm25_candidates(_FakeBM25([]), "q", 5, []) == {}

    def test_index_past_metadata_is_skipped_not_raised(self):
        """The corpus and the metadata list can briefly disagree.

        Three corpus entries, two metadata entries: index 2 is the top-scoring
        hit but has no metadata, so it is dropped instead of raising IndexError.
        """
        out = normalized_bm25_candidates(_FakeBM25([1.0, 3.0, 5.0]), "q", 3, _metas(2))
        assert set(out) == {0, 1}
        assert out[1][0] == pytest.approx(3.0 / 5.0)  # still scaled by the max

    def test_default_key_is_the_corpus_index(self):
        out = normalized_bm25_candidates(_FakeBM25([1.0]), "q", 1, _metas(1))
        assert list(out) == [0]

    def test_key_fn_overrides_the_key(self):
        out = normalized_bm25_candidates(
            _FakeBM25([1.0]), "q", 1, _metas(1), key_fn=lambda m: m["task"]
        )
        assert list(out) == ["t0"]

    def test_carries_the_metadata_through(self):
        metas = _metas(1)
        out = normalized_bm25_candidates(_FakeBM25([1.0]), "q", 1, metas)
        assert out[0][1] is metas[0]

    def test_real_bm25_end_to_end(self):
        bm25 = BM25()
        bm25.fit(["deploy the lambda function", "train a sagemaker model"])
        out = normalized_bm25_candidates(bm25, "lambda deploy", 2, _metas(2))
        assert 0 in out
        assert out[0][0] == 1.0


class TestMergeAndRank:
    def test_weighted_sum_of_both_components(self):
        sem = {"a": (1.0, {"id": "a"})}
        kw = {"a": (0.5, {"id": "a"})}
        ranked = merge_and_rank(sem, kw, 0.7, 0.3, 5)
        assert ranked[0][0] == pytest.approx(0.7 * 1.0 + 0.3 * 0.5)

    def test_semantic_only_candidate_scores_zero_keyword(self):
        """Not dropped — winning on one signal alone is the point."""
        ranked = merge_and_rank({"a": (1.0, {"id": "a"})}, {}, 0.7, 0.3, 5)
        assert ranked[0][0] == pytest.approx(0.7)

    def test_keyword_only_candidate_is_kept(self):
        ranked = merge_and_rank({}, {"b": (1.0, {"id": "b"})}, 0.7, 0.3, 5)
        assert [m["id"] for _, m in ranked] == ["b"]

    def test_union_of_both_sets(self):
        sem = {"a": (1.0, {"id": "a"})}
        kw = {"b": (1.0, {"id": "b"})}
        ranked = merge_and_rank(sem, kw, 0.5, 0.5, 10)
        assert {m["id"] for _, m in ranked} == {"a", "b"}

    def test_sorted_best_first(self):
        sem = {"lo": (0.1, {"id": "lo"}), "hi": (0.9, {"id": "hi"})}
        ranked = merge_and_rank(sem, {}, 1.0, 0.0, 10)
        assert [m["id"] for _, m in ranked] == ["hi", "lo"]

    def test_truncates_to_top_k(self):
        sem = {f"k{i}": (i / 10, {"id": i}) for i in range(10)}
        assert len(merge_and_rank(sem, {}, 1.0, 0.0, 3)) == 3

    def test_weights_actually_change_the_order(self):
        sem = {"s": (1.0, {"id": "s"}), "k": (0.0, {"id": "k"})}
        kw = {"s": (0.0, {"id": "s"}), "k": (1.0, {"id": "k"})}
        semantic_first = merge_and_rank(sem, kw, 0.9, 0.1, 2)
        keyword_first = merge_and_rank(sem, kw, 0.1, 0.9, 2)
        assert semantic_first[0][1]["id"] == "s"
        assert keyword_first[0][1]["id"] == "k"

    def test_ties_are_deterministic(self):
        """The old set-iteration made equal scores order-unstable run to run."""
        sem = {f"k{i}": (0.5, {"id": i}) for i in range(8)}
        first = [m["id"] for _, m in merge_and_rank(sem, {}, 1.0, 0.0, 8)]
        for _ in range(5):
            assert [m["id"] for _, m in merge_and_rank(sem, {}, 1.0, 0.0, 8)] == first

    def test_metadata_is_not_copied(self):
        """merge_and_rank hands back the caller's objects untouched."""
        meta = {"id": "a"}
        ranked = merge_and_rank({"a": (1.0, meta)}, {}, 1.0, 0.0, 1)
        assert ranked[0][1] is meta
        assert "similarity" not in meta

    def test_empty_inputs(self):
        assert merge_and_rank({}, {}, 0.7, 0.3, 5) == []

    def test_integer_and_string_keys_both_work(self):
        assert len(merge_and_rank({0: (1.0, {"i": 0})}, {}, 1.0, 0.0, 5)) == 1
        assert len(merge_and_rank({"a": (1.0, {"i": "a"})}, {}, 1.0, 0.0, 5)) == 1


class TestRankWithSimilarity:
    def test_stamps_the_score_into_a_copy(self):
        meta = {"task": "x"}
        out = rank_with_similarity({"a": (1.0, meta)}, {}, 0.7, 0.3, 5)
        assert out[0]["similarity"] == pytest.approx(0.7)
        assert out[0] is not meta
        assert "similarity" not in meta  # the store's own list stays clean

    def test_returns_plain_metadata_dicts(self):
        out = rank_with_similarity({"a": (1.0, {"task": "x"})}, {}, 1.0, 0.0, 5)
        assert out == [{"task": "x", "similarity": 1.0}]

    def test_empty(self):
        assert rank_with_similarity({}, {}, 0.7, 0.3, 5) == []


class TestCandidateCount:
    def test_over_fetches_three_times_top_k(self):
        assert candidate_count(5) == 15

    def test_capped_by_corpus_size(self):
        assert candidate_count(5, corpus_size=4) == 4

    def test_zero_and_negative_top_k(self):
        assert candidate_count(0) == 0
        assert candidate_count(-3) == 0


class TestAllThreeStoresShareTheHelper:
    """The point of the refactor: one implementation, three call sites."""

    @pytest.mark.parametrize(
        "module",
        [
            "mnemoai.client.memory.chroma_store",
            "mnemoai.client.memory.faiss_store",
            "mnemoai.server.tools.rag.session",
        ],
    )
    def test_store_imports_the_shared_bm25_helper(self, module):
        source = open(
            __import__(module, fromlist=["_"]).__file__, encoding="utf-8"
        ).read()
        assert "normalized_bm25_candidates" in source
        # The hand-rolled normalization must be gone from all three.
        assert "max_bm25" not in source


class TestKeyFnArity:
    """``key_fn`` accepts ``(meta, idx)`` as well as the legacy ``(meta)``.

    The index is what lets a store key candidates by a unique id held in a
    positionally-aligned list — the Chroma episodic store needs it, because its
    only metadata-derivable key (``task + solution + tools``) is NOT unique.
    """

    def test_two_arg_key_fn_receives_the_index(self):
        seen = []

        def key_fn(meta, idx):
            seen.append(idx)
            return f"id-{idx}"

        got = normalized_bm25_candidates(
            _FakeBM25([1.0, 0.5]), "q", 2, _metas(2), key_fn=key_fn
        )
        assert sorted(got) == ["id-0", "id-1"]
        assert sorted(seen) == [0, 1]

    def test_one_arg_key_fn_still_works(self):
        got = normalized_bm25_candidates(
            _FakeBM25([1.0]), "q", 1, _metas(1), key_fn=lambda m: m["task"]
        )
        assert list(got) == [_metas(1)[0]["task"]]

    def test_type_error_inside_a_two_arg_key_fn_is_not_swallowed(self):
        # Arity is decided by inspection, not by catching TypeError from the
        # call — otherwise a real bug inside the key function would be silently
        # retried with the wrong arity and produce a wrong key.
        def key_fn(meta, idx):
            raise TypeError("a real bug inside the key function")

        with pytest.raises(TypeError, match="a real bug"):
            normalized_bm25_candidates(
                _FakeBM25([1.0]), "q", 1, _metas(1), key_fn=key_fn
            )


class TestChromaKeysByUniqueId:
    """Regression: distinct episodes must not collapse into one result.

    The Chroma store keyed hybrid candidates by ``_get_searchable_text``
    (``task + solution + tools``). Two runs of the same task produce the SAME
    text, so they collided in the candidate map and only one survived the merge —
    asking the same thing twice with different outcomes silently lost one of them.
    """

    def _store(self):
        from mnemoai.client.memory.chroma_store import ChromaEpisodicStore

        s = ChromaEpisodicStore.__new__(ChromaEpisodicStore)
        s.metadatas = [
            {"task": "fix the bug", "tools": "[]", "outcome": "success"},
            {"task": "fix the bug", "tools": "[]", "outcome": "failure"},
            {"task": "write docs", "tools": "[]", "outcome": "success"},
        ]
        s.ids = ["ep-1", "ep-2", "ep-3"]
        return s

    def test_colliding_tasks_get_distinct_keys(self):
        s = self._store()
        keys = {s._bm25_key(m, i) for i, m in enumerate(s.metadatas)}
        assert keys == {"ep-1", "ep-2", "ep-3"}

    def test_searchable_text_would_have_collided(self):
        # Proves the premise: the OLD key really was ambiguous.
        s = self._store()
        texts = {s._get_searchable_text(m) for m in s.metadatas}
        assert len(texts) == 2 < len(s.metadatas)

    def test_falls_back_to_text_when_ids_are_short(self):
        # A partial load must degrade to the old behavior, not raise mid-query.
        s = self._store()
        s.ids = ["ep-1"]
        assert s._bm25_key(s.metadatas[2], 2) == s._get_searchable_text(s.metadatas[2])
