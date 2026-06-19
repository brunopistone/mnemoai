"""ChromaDB-backed vector store for session-scoped RAG."""

import logging

logging.getLogger("chromadb").setLevel(logging.WARNING)

import os
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np

from mnemoai.utils.logger import logger


class ChromaStore:
    """A ChromaDB wrapper that stores vectors and metadata with file persistence."""

    def __init__(self, dim: int, session_id: str = None, rag_dir: str = None) -> None:
        """Initialize ChromaDB vector store.

        Args:
            dim: Embedding dimension
            session_id: Optional session ID for persistence
            rag_dir: Optional RAG directory path
        """
        self.dim = dim
        self.session_id = session_id

        # Set persistence path
        if rag_dir and session_id:
            self.persist_path = os.path.join(rag_dir, f"rag_store_{session_id}")
        elif session_id:
            self.persist_path = f"/tmp/rag_store_{session_id}"
        else:
            self.persist_path = "/tmp/rag_store"

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_path)

        # Get or create collection
        collection_name = f"rag_{session_id}" if session_id else "rag_default"
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.debug(f"Loaded existing ChromaDB collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name, metadata={"dimension": dim}
            )
            logger.debug(f"Created new ChromaDB collection: {collection_name}")

        # Track metadata separately for compatibility
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

    def add(self, vectors: np.ndarray, metadatas: List[Dict]) -> None:
        """Add vectors and metadata to ChromaDB.

        Args:
            vectors: NumPy array of vectors with shape (n, dim)
            metadatas: List of metadata dictionaries, one per vector
        """
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # Generate IDs
        start_idx = len(self.metadatas)
        ids = [f"doc_{start_idx + i}" for i in range(len(vectors))]

        # Add to ChromaDB
        self.collection.add(embeddings=vectors.tolist(), metadatas=metadatas, ids=ids)

        # Update local metadata list
        self.metadatas.extend(metadatas)
        logger.debug(f"Added {len(vectors)} vectors to ChromaDB")

    def search(self, q: np.ndarray, top_k: int = 5) -> Tuple[List[float], List[Dict]]:
        """Search using query vector.

        Args:
            q: Query vector with shape (dim,)
            top_k: Number of results to return (default: 5)

        Returns:
            Tuple of (scores, metadatas) where scores are similarity scores
        """
        if q.dtype != np.float32:
            q = q.astype(np.float32)

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[q.tolist()], n_results=min(top_k, len(self.metadatas))
        )

        if not results["distances"] or not results["distances"][0]:
            return [], []

        # Convert distances to similarity scores (ChromaDB uses L2 distance)
        # Similarity = 1 / (1 + distance)
        distances = results["distances"][0]
        scores = [1.0 / (1.0 + d) for d in distances]
        metas = results["metadatas"][0]

        return scores, metas

    def clear(self) -> None:
        """Clear all data from the store."""
        collection_name = self.collection.name
        self.client.delete_collection(collection_name)
        self.collection = self.client.create_collection(
            name=collection_name, metadata={"dimension": self.dim}
        )
        self.metadatas = []
        logger.debug(f"Cleared ChromaDB collection: {collection_name}")

    @property
    def index(self) -> None:
        """Compatibility property (ChromaDB doesn't expose raw index)."""
        return None
