import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import chromadb

from mnemoai.utils.bm25 import BM25
from mnemoai.utils.config import config
from mnemoai.utils.logger import logger


class ChromaEpisodicStore:
    """Vector store for episodic memory using ChromaDB."""

    def __init__(self, persist_path: str, embeddings_controller):
        """Initialize episodic memory vector store.

        Args:
            persist_path: Path to persist ChromaDB data
            embeddings_controller: Controller for generating embeddings
        """
        self.persist_path = persist_path
        os.makedirs(self.persist_path, exist_ok=True)

        self.embeddings = embeddings_controller

        # Load hybrid search weights from config
        episodic_config = config.get("EPISODIC_MEMORY", {})
        self.semantic_weight = episodic_config.get("SEMANTIC_WEIGHT", 0.7)
        self.keyword_weight = episodic_config.get("KEYWORD_WEIGHT", 0.3)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_path)

        # Get or create the collection, migrating it if the embedding model
        # changed. An existing collection's vectors are only comparable to new
        # ones from the SAME embedding model: a different model — even at the same
        # dimension (e.g. Ollama qwen3-embedding@1024 → Cohere v4@1024) — produces
        # semantically incompatible vectors, and a different dimension makes every
        # query/add raise outright ("expecting embedding with dimension of X, got
        # Y"). We stamp the collection with a model fingerprint and RESET it when
        # the current model's fingerprint differs — episodic memory is
        # model-scoped, re-learnable scratch (stores live under models/{model}/),
        # so a reset is the safe migration (old vectors can't be reused, and
        # re-embedding the whole history on every switch would be slow).
        self._open_or_migrate_collection()

        # Track metadata separately
        self.metadatas = []
        self.bm25: Optional[BM25] = None
        self._load_metadatas()
        self._rebuild_bm25()

    def _embed_fingerprint(self) -> str:
        """Current embedding model's fingerprint (falls back to a dim string)."""
        fp = getattr(self.embeddings, "fingerprint", None)
        if callable(fp):
            return fp()
        return f"dim={getattr(self.embeddings, 'dim', '?')}"

    def _create_collection(self):
        """Create the collection stamped with the current model fingerprint."""
        return self.client.create_collection(
            name="episodic_memory",
            metadata={
                "description": "Task solutions with tool usage patterns",
                "embed_fingerprint": self._embed_fingerprint(),
            },
        )

    def _open_or_migrate_collection(self) -> None:
        """Load the collection, or reset it if the embedding model changed."""
        current_fp = self._embed_fingerprint()
        try:
            self.collection = self.client.get_collection(name="episodic_memory")
        except Exception:
            self.collection = self._create_collection()
            logger.info(
                f"Created new episodic memory collection at {self.persist_path}"
            )
            return

        stored_fp = (self.collection.metadata or {}).get("embed_fingerprint")
        # An unstamped legacy collection: if it holds data at a DIFFERENT dimension
        # than the current model, it can't be queried — reset. If empty or same
        # dimension, adopt it (re-stamp on next create isn't needed — it works).
        if stored_fp is None:
            if self._legacy_dimension_mismatch():
                self._reset_collection(current_fp, reason="dimension changed")
            else:
                logger.info(
                    f"Loaded existing episodic memory collection from "
                    f"{self.persist_path}"
                )
            return

        if stored_fp == current_fp:
            logger.info(
                f"Loaded existing episodic memory collection from {self.persist_path}"
            )
            return

        self._reset_collection(
            current_fp, reason=f"embedding model changed ({stored_fp} → {current_fp})"
        )

    def _legacy_dimension_mismatch(self) -> bool:
        """For an unstamped collection: True if its stored dim differs from the
        current embedding dim (so queries would crash). None/empty → False."""
        try:
            if self.collection.count() == 0:
                return False
            peek = self.collection.peek(1)
            emb = peek.get("embeddings")
            stored = len(emb[0]) if emb is not None and len(emb) > 0 else None
        except Exception:
            return False
        if stored is None:
            return False
        current = None
        rd = getattr(self.embeddings, "runtime_dimension", None)
        if callable(rd):
            current = rd()
        return current is not None and current != stored

    def _reset_collection(self, current_fp: str, reason: str) -> None:
        """Drop and recreate the collection (re-stamped), logging why."""
        logger.warning(
            "Resetting episodic memory (%s). Past episodes are dropped; the store "
            "will re-learn with the new embedding model.",
            reason,
        )
        try:
            self.client.delete_collection(name="episodic_memory")
        except Exception as e:
            logger.debug(f"delete_collection during reset failed: {e}")
        self.collection = self._create_collection()

    def _load_metadatas(self) -> None:
        """Load existing metadatas from collection."""
        try:
            results = self.collection.get()
            if results and results["metadatas"]:
                self.metadatas = results["metadatas"]
        except Exception as e:
            logger.warning(f"Failed to load metadatas: {e}")

    def _get_searchable_text(self, metadata: Dict[str, Any]) -> str:
        """Build searchable text from episode metadata for BM25 indexing."""
        parts = [metadata.get("task", ""), metadata.get("solution", "")]
        tools_str = metadata.get("tools", "")
        if isinstance(tools_str, str) and tools_str:
            parts.append(tools_str)
        return " ".join(p for p in parts if p)

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from all stored episode metadata."""
        if not self.metadatas:
            return
        texts = [self._get_searchable_text(m) for m in self.metadatas]
        self.bm25 = BM25()
        self.bm25.fit(texts)
        logger.debug(f"Episodic BM25 index built with {len(texts)} episodes")

    def add(self, text: str, metadata: Dict[str, Any], episode_id: str = None) -> None:
        """Add episode to vector store.

        Args:
            text: Searchable text representation
            metadata: Episode metadata (task, solution, tools, outcome, timestamp)
            episode_id: Optional unique ID for episode
        """
        if not episode_id:
            episode_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Generate embedding using configured model
        embedding = self.embeddings.embed([text])

        # Add to ChromaDB with pre-computed embedding. If the on-disk DB was moved
        # or replaced under our open handle (SQLite code 1032 "readonly database
        # moved" — e.g. a backup/sync tool or a restore touched the dir), reopen
        # the client once and retry; a stale handle then self-heals.
        try:
            self.collection.add(
                embeddings=embedding.tolist(), metadatas=[metadata], ids=[episode_id]
            )
        except Exception as e:
            if not self._is_db_moved_error(e) or not self._reconnect():
                raise
            self.collection.add(
                embeddings=embedding.tolist(), metadatas=[metadata], ids=[episode_id]
            )

        # Update local metadata list
        self.metadatas.append(metadata)
        self._rebuild_bm25()

        logger.debug(f"Stored episode: {episode_id}")

    @staticmethod
    def _is_db_moved_error(exc: Exception) -> bool:
        """True for the SQLite 'database moved / readonly' family (code 1032),
        which ChromaDB surfaces when its dir was moved/replaced under an open
        connection — recoverable by reopening the client."""
        msg = str(exc).lower()
        return "readonly database" in msg or "1032" in msg or "database moved" in msg

    def _reconnect(self) -> bool:
        """Reopen the ChromaDB client + collection against the persist path.

        Returns True on success. Best-effort — a failure to reconnect returns
        False so the caller re-raises the original error rather than masking it."""
        try:
            os.makedirs(self.persist_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_path)
            try:
                self.collection = self.client.get_collection(name="episodic_memory")
            except Exception:
                self.collection = self.client.create_collection(
                    name="episodic_memory",
                    metadata={"description": "Task solutions with tool usage patterns"},
                )
            logger.info("Reconnected episodic ChromaDB after a moved-DB error")
            return True
        except Exception as e:
            logger.warning(f"Episodic ChromaDB reconnect failed: {e}")
            return False

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar episodes using hybrid search (semantic + BM25).

        Retrieves candidates independently from semantic search and BM25,
        merges both sets, then re-ranks with a hybrid score.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of episodes with metadata
        """
        if len(self.metadatas) == 0:
            return []

        candidate_k = min(top_k * 3, len(self.metadatas))

        # --- Semantic candidates ---
        query_embedding = self.embeddings.embed([query])
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(), n_results=candidate_k
        )

        # key -> (semantic_score, metadata)
        sem_candidates: Dict[str, Tuple] = {}
        if results["metadatas"] and results["metadatas"][0]:
            for i, metadata in enumerate(results["metadatas"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                sem_score = 1.0 / (1.0 + distance)
                key = self._get_searchable_text(metadata)
                sem_candidates[key] = (sem_score, metadata)

        # --- BM25 candidates ---
        bm25_candidates: Dict[str, Tuple] = {}
        if self.bm25 and self.bm25.corpus_size > 0:
            raw_bm25 = self.bm25.score(query)
            max_bm25 = max(raw_bm25) if raw_bm25 else 0.0

            if max_bm25 > 0:
                indexed_scores = sorted(
                    enumerate(raw_bm25), key=lambda x: x[1], reverse=True
                )[:candidate_k]

                for idx, score in indexed_scores:
                    if score <= 0 or idx >= len(self.metadatas):
                        continue
                    norm_score = score / max_bm25
                    meta = self.metadatas[idx]
                    key = self._get_searchable_text(meta)
                    bm25_candidates[key] = (norm_score, meta)

        # --- Merge and re-rank ---
        all_keys = set(sem_candidates.keys()) | set(bm25_candidates.keys())
        hybrid_results = []

        for key in all_keys:
            sem_score = sem_candidates[key][0] if key in sem_candidates else 0.0
            meta = (
                sem_candidates[key][1]
                if key in sem_candidates
                else bm25_candidates[key][1]
            )
            bm25_val = bm25_candidates[key][0] if key in bm25_candidates else 0.0

            hybrid_score = (
                self.semantic_weight * sem_score + self.keyword_weight * bm25_val
            )

            episode = meta.copy()
            episode["similarity"] = hybrid_score
            hybrid_results.append((hybrid_score, episode))

        hybrid_results.sort(key=lambda x: x[0], reverse=True)
        return [meta for _, meta in hybrid_results[:top_k]]

    def cleanup(self, max_episodes: int = 1000, max_age_days: int = 90) -> None:
        """Remove old episodes and enforce size limit.

        Args:
            max_episodes: Maximum number of episodes to keep
            max_age_days: Maximum age in days
        """
        if len(self.metadatas) == 0:
            return

        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        # Get all episodes with IDs
        results = self.collection.get()
        if not results or not results["ids"]:
            return

        # Filter by age and size
        valid_ids = []
        valid_metadatas = []

        for i, (id, metadata) in enumerate(zip(results["ids"], results["metadatas"])):
            timestamp_str = metadata.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp > cutoff_date:
                    valid_ids.append(id)
                    valid_metadatas.append(metadata)
            except:
                valid_ids.append(id)  # Keep if can't parse
                valid_metadatas.append(metadata)

        # Enforce size limit (keep most recent)
        if len(valid_ids) > max_episodes:
            valid_ids = valid_ids[-max_episodes:]
            valid_metadatas = valid_metadatas[-max_episodes:]

        # Delete old episodes if needed
        if len(valid_ids) < len(results["ids"]):
            old_count = len(results["ids"])

            # Get IDs to delete
            ids_to_delete = [id for id in results["ids"] if id not in valid_ids]

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                self.metadatas = valid_metadatas
                self._rebuild_bm25()
                logger.info(
                    f"Cleaned up episodic memory: {old_count} → {len(valid_ids)} episodes"
                )

    def clear(self) -> None:
        """Clear all episodes."""
        self.client.delete_collection("episodic_memory")
        self.collection = self.client.create_collection(
            name="episodic_memory",
            metadata={"description": "Task solutions with tool usage patterns"},
        )
        self.metadatas = []
        self.bm25 = None
        logger.info("Cleared episodic memory collection")
