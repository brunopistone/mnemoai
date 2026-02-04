import ast
from datetime import datetime, timedelta
import faiss
import json
import os
from typing import Any, Dict, List
from utils.config import config
from utils.logger import logger


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

    def add(self, text: str, metadata: Dict[str, Any], episode_id: str = None) -> None:
        """Add episode to FAISS index.

        Args:
            text: Searchable text representation
            metadata: Episode metadata
            episode_id: Optional unique ID
        """
        if not episode_id:
            episode_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Generate embedding
        embedding = self.embeddings.embed([text])

        # Initialize index if needed
        if self.index is None:
            dim = embedding.shape[1]
            self.index = faiss.IndexFlatIP(dim)

        # Add to index
        self.index.add(embedding)

        # Store metadata
        metadata["episode_id"] = episode_id
        self.metadata.append(metadata)

        # Persist
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.debug(f"Stored episode in FAISS: {episode_id}")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar episodes using hybrid search (70% semantic + 30% keyword).

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            List of episodes with metadata
        """
        if self.index is None or len(self.metadata) == 0:
            return []

        # Generate query embedding
        query_embedding = self.embeddings.embed([query])

        # Optimize: retrieve only top_k * 3 for hybrid re-ranking (not all episodes)
        # This reduces O(n) to O(k log n) where k is much smaller than n
        retrieval_k = min(top_k * 3, len(self.metadata))
        logger.debug(
            f"FAISS search: retrieving top {retrieval_k} of {len(self.metadata)} episodes for hybrid ranking"
        )

        scores, indices = self.index.search(query_embedding, retrieval_k)

        # Hybrid search: combine semantic + keyword matching
        query_lower = query.lower()
        hybrid_results = []

        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx].copy()
                semantic_score = float(scores[0][i])

                # Keyword matching: check task text and tool names
                task = meta.get("task", "").lower()
                tools_str = meta.get("tools", "")

                # Extract tool names from tools string
                tool_names = []
                if isinstance(tools_str, str):
                    try:
                        tools_list = ast.literal_eval(tools_str)
                        tool_names = [
                            t.get("name", "").lower()
                            for t in tools_list
                            if isinstance(t, dict)
                        ]
                    except:
                        pass

                # Keyword score: boost if query terms appear in task or tool names
                keyword_score = 0.0
                query_terms = query_lower.split()

                for term in query_terms:
                    if len(term) > 2:  # Skip short words
                        if term in task:
                            keyword_score += 0.5
                        if any(term in tool for tool in tool_names):
                            keyword_score += 0.5

                keyword_score = min(keyword_score, 1.0)  # Cap at 1.0

                # Hybrid: configurable semantic + keyword weights
                hybrid_score = (
                    self.semantic_weight * semantic_score
                    + self.keyword_weight * keyword_score
                )

                meta["similarity"] = hybrid_score
                hybrid_results.append((hybrid_score, meta))

        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x[0], reverse=True)

        # Return top_k
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

            # Persist
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)

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
        logger.info("Cleared FAISS episodic memory")
