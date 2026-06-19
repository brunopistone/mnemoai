"""Lightweight Okapi BM25 scorer (no external dependencies)."""

import math
import re
from typing import Dict, List


def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class BM25:
    """Okapi BM25 ranking function for keyword-based document scoring.

    Default parameters (k1=1.2, b=0.75) match Elasticsearch/Lucene defaults,
    optimized for short to medium-length text chunks.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_dl = 0.0
        self.doc_lengths: List[int] = []
        self.doc_tokens: List[List[str]] = []
        self.df: Dict[str, int] = {}

    def fit(self, documents: List[str]) -> None:
        """Build the BM25 index from a list of document strings."""
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.doc_lengths = [len(t) for t in self.doc_tokens]
        self.corpus_size = len(documents)
        self.avg_dl = (
            sum(self.doc_lengths) / self.corpus_size if self.corpus_size else 1.0
        )

        self.df = {}
        for tokens in self.doc_tokens:
            seen = set(tokens)
            for token in seen:
                self.df[token] = self.df.get(token, 0) + 1

    def score(self, query: str) -> List[float]:
        """Return BM25 scores for every document given a query string."""
        query_tokens = tokenize(query)
        scores = [0.0] * self.corpus_size

        for qt in query_tokens:
            if qt not in self.df:
                continue
            idf = math.log(
                (self.corpus_size - self.df[qt] + 0.5) / (self.df[qt] + 0.5) + 1.0
            )
            for i, doc_toks in enumerate(self.doc_tokens):
                tf = doc_toks.count(qt)
                if tf == 0:
                    continue
                dl = self.doc_lengths[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                scores[i] += idf * numerator / denominator

        return scores
