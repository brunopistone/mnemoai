"""Hybrid (semantic + BM25) candidate scoring, shared by every store.

The same two steps were written three times — in the episodic Chroma store, the
episodic FAISS store, and the session-scoped RAG store:

1. normalize raw BM25 scores by the best score in the corpus and keep the top
   candidates, and
2. merge the semantic and keyword candidate sets and re-rank by a weighted sum.

Three copies meant a scoring change had to land three times, and they had already
drifted in small ways. The functions here take plain data (no store, no config
read) so all three call the same code and the tests can exercise the ranking
directly.

**What stays with the caller:** producing the semantic candidates. That step is
genuinely backend-specific — Chroma returns squared-L2 distances, FAISS returns
inner products, the RAG store may fall back to a pure-numpy cosine — and the
per-backend conversion onto the shared [0,1] cosine scale lives in
``client/memory/similarity.py``. Likewise the *shape* of the return value
(annotated copies vs. parallel score/metadata lists) stays with the caller.
"""

import inspect
from typing import Any, Dict, Hashable, List, Optional, Tuple

# A candidate set: key -> (score, metadata). The key identifies one item within
# one store; its type is the store's business (a chunk id, a corpus index, the
# searchable text) and is only ever compared for equality.
CandidateMap = Dict[Hashable, Tuple[float, Dict[str, Any]]]


def _apply_key_fn(key_fn, meta: Dict[str, Any], idx: int) -> Hashable:
    """Call ``key_fn`` with ``(meta, idx)`` or ``(meta)`` depending on its arity.

    Decided by inspecting the signature ONCE per call rather than by catching
    ``TypeError`` from the call: a ``TypeError`` raised *inside* a two-argument
    key function would otherwise be swallowed and silently retried with the wrong
    arity, turning a real bug into a wrong key.
    """
    try:
        params = inspect.signature(key_fn).parameters
        takes_index = len(params) >= 2 or any(
            p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()
        )
    except (TypeError, ValueError):
        takes_index = False  # builtins/C callables: assume the legacy 1-arg form
    return key_fn(meta, idx) if takes_index else key_fn(meta)


def normalized_bm25_candidates(
    bm25,
    query: str,
    candidate_k: int,
    metadatas: List[Dict[str, Any]],
    key_fn=None,
) -> CandidateMap:
    """Top-``candidate_k`` BM25 hits, scored 0..1 against the best hit.

    Dividing by the max score is what puts BM25 (an unbounded relevance score)
    on the same 0..1 scale as the cosine similarity it gets averaged with. Note
    this is a *within-query* normalization: the best hit for a query always
    scores 1.0, so a keyword score is only meaningful relative to the other
    candidates for that same query.

    Args:
        bm25: A fitted ``BM25`` (or None / unfitted — yields no candidates).
        query: The raw query text.
        candidate_k: How many top hits to keep.
        metadatas: The store's metadata list, positionally aligned with the BM25
            corpus. An index past the end is skipped rather than raising, since
            the two can briefly disagree if the corpus was refit concurrently.
        key_fn: ``(meta, idx) -> key``, or the legacy ``meta -> key``. Defaults to
            the corpus index, which is what a store keyed by position wants. The
            index is passed so a store can key by a per-item id held in a
            positionally-aligned list (Chroma keys by episode id, which is not
            derivable from the metadata alone).

    Returns:
        ``key -> (normalized_score, metadata)``, non-positive scores dropped
        (a zero BM25 score means no query term matched at all).
    """
    if bm25 is None or getattr(bm25, "corpus_size", 0) <= 0:
        return {}

    raw_scores = bm25.score(query)
    if not raw_scores:
        return {}

    max_score = max(raw_scores)
    if max_score <= 0:
        return {}

    ranked = sorted(enumerate(raw_scores), key=lambda pair: pair[1], reverse=True)
    candidates: CandidateMap = {}
    for idx, score in ranked[:candidate_k]:
        if score <= 0 or idx >= len(metadatas):
            continue
        meta = metadatas[idx]
        key = idx if key_fn is None else _apply_key_fn(key_fn, meta, idx)
        candidates[key] = (score / max_score, meta)
    return candidates


def merge_and_rank(
    semantic: CandidateMap,
    keyword: CandidateMap,
    semantic_weight: float,
    keyword_weight: float,
    top_k: int,
) -> List[Tuple[float, Dict[str, Any]]]:
    """Merge two candidate sets and re-rank them by a weighted sum.

    A key present in only one set scores 0.0 for the other component — it is NOT
    dropped. That is the point of hybrid search: a chunk that no query term
    matches can still win on semantic similarity, and an exact keyword match can
    still win on a term the embedding missed.

    Ordering is deterministic: candidates are visited in a stable key order and
    Python's sort is stable, so two items with equal scores always come back in
    the same order. (The previous per-store copies iterated a ``set``, so ties
    could come back in a different order run to run.)

    Args:
        semantic: ``key -> (score, metadata)`` from the vector search.
        keyword: ``key -> (score, metadata)`` from :func:`normalized_bm25_candidates`.
        semantic_weight: Weight for the semantic component.
        keyword_weight: Weight for the keyword component.
        top_k: How many results to return.

    Returns:
        Up to ``top_k`` ``(hybrid_score, metadata)`` pairs, best first. The
        metadata objects are the SAME objects passed in — the caller decides
        whether to copy or annotate them.
    """
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for key in sorted(set(semantic) | set(keyword), key=repr):
        sem_score, sem_meta = semantic.get(key, (0.0, None))
        kw_score, kw_meta = keyword.get(key, (0.0, None))
        meta = sem_meta if sem_meta is not None else kw_meta
        if meta is None:
            continue
        scored.append((semantic_weight * sem_score + keyword_weight * kw_score, meta))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]


def rank_with_similarity(
    semantic: CandidateMap,
    keyword: CandidateMap,
    semantic_weight: float,
    keyword_weight: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    """:func:`merge_and_rank`, returning metadata copies stamped with the score.

    The episodic stores return "episodes" — metadata dicts carrying their own
    ``similarity`` — so the score has to travel inside the dict. Copies, because
    the store's own metadata list must not gain a per-query field.
    """
    ranked = merge_and_rank(semantic, keyword, semantic_weight, keyword_weight, top_k)
    results = []
    for score, meta in ranked:
        episode = meta.copy()
        episode["similarity"] = score
        results.append(episode)
    return results


def candidate_count(top_k: int, corpus_size: Optional[int] = None) -> int:
    """How many candidates to pull per branch for a ``top_k`` result set.

    3x the requested count: each branch has to over-fetch, because an item that
    ranks poorly on one signal can still win the merged ranking, and a branch
    that returns exactly ``top_k`` can never contribute such an item.
    """
    wanted = max(top_k, 0) * 3
    if corpus_size is None:
        return wanted
    return min(wanted, corpus_size)
