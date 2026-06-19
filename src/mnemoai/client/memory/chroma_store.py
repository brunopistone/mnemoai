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

        # Get or create episodic memory collection
        try:
            self.collection = self.client.get_collection(name="episodic_memory")
            logger.info(
                f"Loaded existing episodic memory collection from {self.persist_path}"
            )
        except:
            self.collection = self.client.create_collection(
                name="episodic_memory",
                metadata={"description": "Task solutions with tool usage patterns"},
            )
            logger.info(
                f"Created new episodic memory collection at {self.persist_path}"
            )

        # Track metadata separately
        self.metadatas = []
        self.bm25: Optional[BM25] = None
        self._load_metadatas()
        self._rebuild_bm25()

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

        # Add to ChromaDB with pre-computed embedding
        self.collection.add(
            embeddings=embedding.tolist(), metadatas=[metadata], ids=[episode_id]
        )

        # Update local metadata list
        self.metadatas.append(metadata)
        self._rebuild_bm25()

        logger.debug(f"Stored episode: {episode_id}")

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
