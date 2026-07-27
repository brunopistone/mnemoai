import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import faiss

from mnemoai.client.memory.similarity import (
    SCORING_VERSION,
    cosine_to_unit,
    l2_normalize,
)
from mnemoai.utils.bm25 import BM25
from mnemoai.utils.config import config
from mnemoai.utils.logger import logger


class FAISSEpisodicStore:
    """FAISS-backed vector store for episodic memory."""

    def __init__(self, persist_path: str, embeddings_controller):
        """Initialize FAISS episodic memory store.

        Args:
            persist_path: Path to persist FAISS index and metadata
            embeddings_controller: Controller for generating embeddings
        """
        self.persist_path = persist_path
        os.makedirs(self.persist_path, exist_ok=True)

        self.embeddings = embeddings_controller
        self.index_path = os.path.join(persist_path, "episodic.index")
        self.metadata_path = os.path.join(persist_path, "episodic_metadata.json")

        # Load hybrid search weights from config
        episodic_config = config.get("EPISODIC_MEMORY", {})
        self.semantic_weight = episodic_config.get("SEMANTIC_WEIGHT", 0.7)
        self.keyword_weight = episodic_config.get("KEYWORD_WEIGHT", 0.3)

        # Load or create index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            logger.info(f"Loaded existing FAISS episodic index from {self.index_path}")
        else:
            # Will be initialized on first add
            self.index = None
            logger.info("FAISS episodic index will be created on first add")

        # Load metadata
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

        # The index is only comparable to vectors from the SAME embedding model:
        # a different model — even at the same dimension (e.g. qwen3-embedding@1024
        # → Cohere v4@1024) — yields incompatible vectors, and a different
        # dimension makes add/search raise (FAISS asserts on dim). We stamp a
        # sidecar fingerprint file and RESET the index+metadata when the current
        # model's fingerprint differs — episodic memory is model-scoped,
        # re-learnable scratch, so a reset is the safe migration, not a crash.
        self.fingerprint_path = os.path.join(persist_path, "episodic_fingerprint.txt")
        self._migrate_if_model_changed()

        self.bm25: Optional[BM25] = None
        self._rebuild_bm25()

    def _embed_fingerprint(self) -> str:
        """Current embedding model's fingerprint (falls back to a dim string).

        Includes ``SCORING_VERSION``: vectors are stored L2-normalized for the
        shared cosine scale, so an index written under the old raw-inner-product
        scoring is incompatible and must go through the same reset path as an
        embedding-model change.
        """
        fp = getattr(self.embeddings, "fingerprint", None)
        if callable(fp):
            base = fp()
        else:
            base = f"dim={getattr(self.embeddings, 'dim', '?')}"
        return f"{base}|score={SCORING_VERSION}"

    def _migrate_if_model_changed(self) -> None:
        """Reset the index+metadata if the embedding-model fingerprint changed."""
        current_fp = self._embed_fingerprint()
        stored_fp = None
        try:
            if os.path.exists(self.fingerprint_path):
                with open(self.fingerprint_path, "r") as f:
                    stored_fp = f.read().strip()
        except OSError:
            pass

        # Nothing indexed yet: just (re)stamp so the next build is attributed.
        if self.index is None and not self.metadata:
            self._write_fingerprint(current_fp)
            return

        # A pre-fingerprint (legacy) store: only force a reset if the dimension is
        # actually incompatible; otherwise adopt it and stamp going forward.
        if stored_fp is None:
            if self._legacy_dimension_mismatch(current_fp):
                self._reset(current_fp, "dimension changed")
            else:
                self._write_fingerprint(current_fp)
            return

        if stored_fp != current_fp:
            self._reset(current_fp, f"embedding model changed ({stored_fp} → {current_fp})")

    def _legacy_dimension_mismatch(self, current_fp: str) -> bool:
        """True if an unstamped index's dim differs from the current model's dim."""
        stored = getattr(self.index, "d", None) if self.index is not None else None
        if not stored:
            return False
        current = None
        rd = getattr(self.embeddings, "runtime_dimension", None)
        if callable(rd):
            current = rd()
        return current is not None and current != stored

    def _write_fingerprint(self, fp: str) -> None:
        try:
            with open(self.fingerprint_path, "w") as f:
                f.write(fp)
        except OSError as e:
            logger.debug(f"Failed to write episodic fingerprint: {e}")

    def _reset(self, current_fp: str, reason: str) -> None:
        """Drop the index+metadata and re-stamp the fingerprint, logging why."""
        logger.warning(
            "Resetting episodic memory (%s). Past episodes are dropped; the store "
            "will re-learn with the new embedding model.",
            reason,
        )
        self.index = None
        self.metadata = []
        for p in (self.index_path, self.metadata_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError as e:
                logger.debug(f"Failed to remove {p} during reset: {e}")
        self._write_fingerprint(current_fp)

    def _get_searchable_text(self, metadata: Dict[str, Any]) -> str:
        """Build searchable text from episode metadata for BM25 indexing."""
        parts = [metadata.get("task", ""), metadata.get("solution", "")]
        tools_str = metadata.get("tools", "")
        if isinstance(tools_str, str) and tools_str:
            parts.append(tools_str)
        return " ".join(p for p in parts if p)

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from all stored episode metadata."""
        if not self.metadata:
            return
        texts = [self._get_searchable_text(m) for m in self.metadata]
        self.bm25 = BM25()
        self.bm25.fit(texts)
        logger.debug(f"Episodic BM25 index built with {len(texts)} episodes")

    def add(self, text: str, metadata: Dict[str, Any], episode_id: str = None) -> None:
        """Add episode to FAISS index.

        Args:
            text: Searchable text representation
            metadata: Episode metadata
            episode_id: Optional unique ID
        """
        if not episode_id:
            episode_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Generate embedding. Normalized before indexing so IndexFlatIP's inner
        # product IS the cosine similarity (see memory/similarity.py).
        embedding = l2_normalize(self.embeddings.embed([text]))

        # Initialize index if needed
        if self.index is None:
            dim = embedding.shape[1]
            self.index = faiss.IndexFlatIP(dim)

        # Add to index
        self.index.add(embedding)

        # Store metadata
        metadata["episode_id"] = episode_id
        self.metadata.append(metadata)
        self._rebuild_bm25()

        self._persist()
        logger.debug(f"Stored episode in FAISS: {episode_id}")

    def _persist(self) -> None:
        """Write the index + metadata to disk, self-healing a moved/deleted dir.

        FAISS keeps the index in memory (no open DB handle like ChromaDB), so the
        equivalent failure is the persist DIR being moved/removed under us (a
        backup/sync/restore, or the app-home relocating) — ``write_index`` /
        ``open`` then raises. We recreate the dir and retry ONCE so a transient
        move self-heals instead of crashing; the in-memory index is intact, so a
        retry fully recovers. The caller (chat_interface) also treats any residual
        failure as non-fatal, so a turn's answer is never lost to this."""

        def _write() -> None:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
            # Stamp the model fingerprint alongside so a later model change is
            # detected and migrated (see _migrate_if_model_changed).
            self._write_fingerprint(self._embed_fingerprint())

        try:
            _write()
        except (OSError, RuntimeError) as e:
            # faiss.write_index wraps a file-open failure in RuntimeError, not
            # OSError, so catch both.
            logger.warning(
                f"FAISS persist failed ({e}); recreating dir and retrying once"
            )
            os.makedirs(self.persist_path, exist_ok=True)
            _write()  # retry; if it still fails the caller handles it non-fatally

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar episodes using hybrid search (semantic + BM25).

        Retrieves candidates independently from semantic search and BM25,
        merges both sets, then re-ranks with a hybrid score.

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            List of episodes with metadata
        """
        if self.index is None or len(self.metadata) == 0:
            return []

        candidate_k = min(top_k * 3, len(self.metadata))

        # --- Semantic candidates ---
        query_embedding = l2_normalize(self.embeddings.embed([query]))
        scores, indices = self.index.search(query_embedding, candidate_k)

        # idx -> (semantic_score, metadata)
        sem_candidates: Dict[int, Tuple] = {}
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            # Normalized vectors -> the inner product is a cosine; rescale it to
            # [0,1] so it shares one scale with the Chroma backend and with the
            # BM25 component below.
            sem_candidates[idx] = (
                cosine_to_unit(scores[0][i]),
                self.metadata[idx],
            )

        # --- BM25 candidates ---
        bm25_candidates: Dict[int, float] = {}
        if self.bm25 and self.bm25.corpus_size > 0:
            raw_bm25 = self.bm25.score(query)
            max_bm25 = max(raw_bm25) if raw_bm25 else 0.0

            if max_bm25 > 0:
                indexed_scores = sorted(
                    enumerate(raw_bm25), key=lambda x: x[1], reverse=True
                )[:candidate_k]

                for idx, score in indexed_scores:
                    if score <= 0 or idx >= len(self.metadata):
                        continue
                    bm25_candidates[idx] = score / max_bm25

        # --- Merge and re-rank ---
        all_indices = set(sem_candidates.keys()) | set(bm25_candidates.keys())
        hybrid_results = []

        for idx in all_indices:
            sem_score = sem_candidates[idx][0] if idx in sem_candidates else 0.0
            bm25_val = bm25_candidates.get(idx, 0.0)
            meta = (
                sem_candidates[idx][1] if idx in sem_candidates else self.metadata[idx]
            ).copy()

            hybrid_score = (
                self.semantic_weight * sem_score + self.keyword_weight * bm25_val
            )

            meta["similarity"] = hybrid_score
            hybrid_results.append((hybrid_score, meta))

        hybrid_results.sort(key=lambda x: x[0], reverse=True)
        return [meta for _, meta in hybrid_results[:top_k]]

    def cleanup(self, max_episodes: int = 1000, max_age_days: int = 90) -> None:
        """Remove old episodes and enforce size limit.

        Args:
            max_episodes: Maximum number of episodes to keep
            max_age_days: Maximum age in days
        """
        if len(self.metadata) == 0:
            return

        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        # Filter by age
        valid_indices = []
        for i, meta in enumerate(self.metadata):
            timestamp_str = meta.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp > cutoff_date:
                    valid_indices.append(i)
            except:
                valid_indices.append(i)  # Keep if can't parse

        # Enforce size limit (keep most recent)
        if len(valid_indices) > max_episodes:
            valid_indices = valid_indices[-max_episodes:]

        # Rebuild index and metadata if needed
        if len(valid_indices) < len(self.metadata):
            old_count = len(self.metadata)

            # Create new index with valid vectors
            new_index = faiss.IndexFlatIP(self.index.d)
            new_metadata = []

            for idx in valid_indices:
                vector = self.index.reconstruct(idx)
                new_index.add(vector.reshape(1, -1))
                new_metadata.append(self.metadata[idx])

            self.index = new_index
            self.metadata = new_metadata
            self._rebuild_bm25()

            self._persist()
            logger.info(
                f"Cleaned up episodic memory: {old_count} → {len(self.metadata)} episodes"
            )

    def clear(self) -> None:
        """Clear all episodes."""
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        self.index = None
        self.metadata = []
        self.bm25 = None
        logger.info("Cleared FAISS episodic memory")
