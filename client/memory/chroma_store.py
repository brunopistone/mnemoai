import chromadb
from datetime import datetime
import os
from typing import Any, Dict, List
from utils.config import config
from utils.logger import logger


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
        self._load_metadatas()

    def _load_metadatas(self) -> None:
        """Load existing metadatas from collection."""
        try:
            results = self.collection.get()
            if results and results["metadatas"]:
                self.metadatas = results["metadatas"]
        except Exception as e:
            logger.warning(f"Failed to load metadatas: {e}")

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

        logger.debug(f"Stored episode: {episode_id}")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar episodes using hybrid search (70% semantic + 30% keyword).

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of episodes with metadata
        """
        if len(self.metadatas) == 0:
            return []

        # Generate query embedding
        query_embedding = self.embeddings.embed([query])

        # Optimize: retrieve only top_k * 3 for hybrid re-ranking (not all episodes)
        # This reduces O(n) to O(k log n) where k is much smaller than n
        retrieval_k = min(top_k * 3, len(self.metadatas))
        logger.debug(
            f"ChromaDB search: retrieving top {retrieval_k} of {len(self.metadatas)} episodes for hybrid ranking"
        )

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(), n_results=retrieval_k
        )

        if not results["metadatas"] or not results["metadatas"][0]:
            return []

        # Hybrid search: combine semantic + keyword matching
        query_lower = query.lower()
        hybrid_results = []

        for i, metadata in enumerate(results["metadatas"][0]):
            # Semantic score (convert distance to similarity)
            distance = results["distances"][0][i] if results["distances"] else 0.0
            semantic_score = 1.0 / (1.0 + distance)

            # Keyword matching: check task text and tool names
            task = metadata.get("task", "").lower()
            tools_str = metadata.get("tools", "")

            # Extract tool names from tools string
            tool_names = []
            if isinstance(tools_str, str):
                import ast

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

            episode = metadata.copy()
            episode["similarity"] = hybrid_score
            hybrid_results.append((hybrid_score, episode))

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
        from datetime import timedelta

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
        logger.info("Cleared episodic memory collection")
