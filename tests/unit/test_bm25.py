"""Unit tests for the lightweight BM25 scorer (utils/bm25.py)."""

from mnemoai.utils.bm25 import BM25, tokenize


class TestTokenize:
    def test_lowercases_and_splits_on_punctuation(self):
        assert tokenize("Hello, World! foo_bar") == ["hello", "world", "foo_bar"]

    def test_empty_string(self):
        assert tokenize("") == []

    def test_numbers_are_tokens(self):
        assert tokenize("python 3.11") == ["python", "3", "11"]


class TestBM25:
    def test_ranks_relevant_document_highest(self):
        docs = [
            "the cat sat on the mat",
            "dogs are great pets and loyal companions",
            "the cat chased the mouse",
        ]
        bm25 = BM25()
        bm25.fit(docs)
        scores = bm25.score("cat")
        # Both cat docs score > 0, the dog doc scores 0.
        assert scores[0] > 0
        assert scores[2] > 0
        assert scores[1] == 0.0

    def test_query_term_absent_from_corpus_scores_zero(self):
        bm25 = BM25()
        bm25.fit(["alpha beta", "gamma delta"])
        assert bm25.score("zebra") == [0.0, 0.0]

    def test_empty_corpus_returns_empty_scores(self):
        bm25 = BM25()
        bm25.fit([])
        assert bm25.score("anything") == []

    def test_score_length_matches_corpus_size(self):
        docs = ["a b c", "d e f", "g h i"]
        bm25 = BM25()
        bm25.fit(docs)
        assert len(bm25.score("a d g")) == 3

    def test_idf_favors_rarer_term(self):
        # 'rare' appears in one doc, 'common' in all three.
        docs = ["common rare", "common word", "common thing"]
        bm25 = BM25()
        bm25.fit(docs)
        rare_scores = bm25.score("rare")
        common_scores = bm25.score("common")
        # The rare term should give its single matching doc a higher score
        # than the ubiquitous term gives any doc.
        assert max(rare_scores) > max(common_scores)
