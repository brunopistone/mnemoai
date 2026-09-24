import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import faiss

from mnemoai.client.memory.similarity import (
    SCORING_VERSION,
    cosine_to_unit,
    l2_normalize,
)
from mnemoai.utils.bm25 import BM25
from mnemoai.utils.config import config
from mnemoai.utils.embedding_integrity import (
    discard_faiss_repair_state,
    load_faiss_pair,
    mark_faiss_clean,
    require_clean_vectors,
)
from mnemoai.utils.hybrid_search import (
    candidate_count,
    normalized_bm25_candidates,
    rank_with_similarity,
)
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

        self.index, self.metadata, self.integrity_error = load_faiss_pair(
            self.index_path, self.metadata_path
        )

        # The index is only comparable to vectors from the SAME embedding model:
        # a different model — even at the same dimension (e.g. qwen3-embedding@1024
        # → Cohere v4@1024) — yields incompatible vectors, and a different
        # dimension makes add/search raise (FAISS asserts on dim). We stamp a
        # sidecar fingerprint file and RESET the index+metadata when the current
        # model's fingerprint differs — episodic memory is model-scoped,
        # re-learnable scratch, so a reset is the safe migration, not a crash.
        self.fingerprint_path = os.path.join(persist_path, "episodic_fingerprint.txt")
        self._identity_pending = False
        try:
            require_clean_vectors(self)
            self._migrate_if_model_changed()
        except Exception as e:
            if not self.metadata and not self.integrity_error:
                raise
            self._identity_pending = True
            if not self.integrity_error:
                logger.warning("Embedding identity unavailable; keeping stored memory for BM25: %s", e)

        self.bm25: Optional[BM25] = None
        self._rebuild_bm25()

    def _ensure_identity(self) -> None:
        """Retry a deferred provider check before using or writing vectors."""
        require_clean_vectors(self)
        if getattr(self, "_identity_pending", False):
            self._migrate_if_model_changed()
            self._identity_pending = False
            self.bm25 = None
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
        self._ensure_identity()
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
            mark_faiss_clean(self.index_path, self.metadata_path)
            # Stamp the model fingerprint alongside so a later model change is
            # detected and migrated (see _migrate_if_model_changed).
            if not getattr(self, "_identity_pending", False):
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
        if not self.metadata or top_k <= 0:
            return []

        candidate_k = candidate_count(top_k, len(self.metadata))

        # --- Semantic candidates (backend-specific: inner product -> cosine) ---
        sem_candidates = {}
        degraded = False
        try:
            self._ensure_identity()
            query_embedding = l2_normalize(self.embeddings.embed([query]))
            scores, indices = self.index.search(query_embedding, candidate_k)
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.metadata):
                    sem_candidates[int(idx)] = (
                        cosine_to_unit(scores[0][i]), self.metadata[idx],
                    )
        except Exception as e:
            degraded = True
            logger.warning("Semantic memory search unavailable; using BM25 only: %s", e)

        # --- Keyword candidates + merge (shared with the Chroma/RAG stores).
        # Keyed by corpus index, which is exactly the BM25 helper's default. ---
        bm25_candidates = normalized_bm25_candidates(
            self.bm25, query, candidate_k, self.metadata
        )
        ranked = rank_with_similarity(
            sem_candidates,
            bm25_candidates,
            0.0 if degraded else self.semantic_weight,
            1.0 if degraded else self.keyword_weight,
            top_k,
        )
        if degraded:
            for episode in ranked:
                episode["retrieval_method"] = "bm25"
        return ranked

    def cleanup(self, max_episodes: int = 1000, max_age_days: int = 90) -> None:
        """Remove old episodes and enforce size limit.

        Args:
            max_episodes: Maximum number of episodes to keep
            max_age_days: Maximum age in days
        """
        if getattr(self, "integrity_error", None) or len(self.metadata) == 0:
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
        discard_faiss_repair_state(self.index_path)
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        self.index = None
        self.metadata = []
        self.bm25 = None
        self.integrity_error = None
        logger.info("Cleared FAISS episodic memory")
