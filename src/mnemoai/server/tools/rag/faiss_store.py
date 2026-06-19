"""Simple FAISS-backed vector store with file persistence for session-scoped RAG."""

import logging

logging.getLogger("faiss").setLevel(logging.WARNING)

import faiss
import numpy as np
import os
import pickle
import sys
import threading
import traceback
from typing import List, Dict, Tuple, Optional, Any

from mnemoai.utils.logger import logger


class FaissStore:
    """A minimal FAISS wrapper that stores vectors and metadata with file persistence.

    Persists to disk to survive process restarts (needed for MCP subprocess architecture).
    """

    def __init__(
        self,
        dim: int,
        persist_path: str = None,
        session_id: str = None,
        rag_dir: str = None,
    ) -> None:
        """Initialize FAISS vector store.

        Args:
            dim: Embedding dimension
            persist_path: Optional persistence path
            session_id: Optional session ID for persistence
            rag_dir: Optional RAG directory path
        """
        self.dim = dim

        # Use profile-specific path
        if rag_dir and session_id:
            self.persist_path = os.path.join(rag_dir, f"rag_store_{session_id}.faiss")
        elif session_id:
            self.persist_path = f"/tmp/rag_store_{session_id}.faiss"
        else:
            self.persist_path = persist_path or "/tmp/rag_store.faiss"

        self.metadata_path = self.persist_path + ".meta"
        self.lock = threading.Lock()

        # Try to load existing index
        if os.path.exists(self.persist_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.persist_path)
                with open(self.metadata_path, "rb") as f:
                    self.metadatas = pickle.load(f)
            except Exception:
                # If load fails, create new
                self.index = faiss.IndexFlatIP(dim)
                self.metadatas = []
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.metadatas = []

    def add(self, vectors: np.ndarray, metadatas: list[dict]) -> None:
        """Add vectors and metadata to FAISS index.

        Args:
            vectors: NumPy array of vectors with shape (n, dim)
            metadatas: List of metadata dictionaries, one per vector
        """

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        with self.lock:
            self.index.add(vectors)
            self.metadatas.extend(metadatas)
            self._persist()

    def search(self, q: np.ndarray, top_k: int = 5) -> Tuple[List[float], List[Dict]]:
        """Search using query vector.

        Args:
            q: Query vector with shape (dim,)
            top_k: Number of results to return (default: 5)

        Returns:
            Tuple of (scores, metadatas) where scores are cosine similarity scores
        """
        if q.dtype != np.float32:
            q = q.astype(np.float32)

        # normalize
        norm = np.linalg.norm(q)
        if norm == 0:
            norm = 1.0
        q = q / norm

        with self.lock:
            D, I = self.index.search(np.expand_dims(q, axis=0), top_k)

        indices = I[0].tolist()
        scores = D[0].tolist()

        results = []
        metas = []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            metas.append(self.metadatas[idx])
            results.append(score)

        return results, metas

    def clear(self) -> None:
        """Clear all vectors and metadata from the store."""
        with self.lock:
            self.index.reset()
            self.metadatas = []
            self._persist()

    def _persist(self) -> None:
        """Save index and metadata to disk."""
        try:
            faiss.write_index(self.index, self.persist_path)
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadatas, f)
        except Exception as e:
            # Log the error so we can debug
            logger.error(f"ERROR: Failed to persist RAG store: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def create_store(dim: int) -> "FaissStore":
    """Create a new FAISS store with specified dimension.

    Args:
        dim: Embedding dimension

    Returns:
        New FaissStore instance
    """
    return FaissStore(dim)
