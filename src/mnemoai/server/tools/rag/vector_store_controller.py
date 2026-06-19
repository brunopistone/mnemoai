import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mnemoai.utils.config import config
from mnemoai.utils.logger import logger


class VectorStoreController:
    """Controller for vector store operations across different providers."""

    def __init__(self, dim: int, session_id: str = None, rag_dir: str = None) -> None:
        """Initialize vector store controller.

        Args:
            dim: Embedding dimension
            session_id: Optional session ID
            rag_dir: Optional RAG directory path
        """
        self.dim = dim
        self.session_id = session_id
        self.rag_dir = rag_dir

        # Get store type from config
        store_config = config.get("RAG", {}).get("VECTOR_STORE", {})
        self.store_type = store_config.get("TYPE", "faiss")

        self.store = self._initialize_store()

    @staticmethod
    def detect_existing_store(session_id: str, rag_dir: str) -> Optional[int]:
        """Detect if a store exists and return its dimension.

        Args:
            session_id: Session identifier for the store
            rag_dir: Directory where stores are persisted

        Returns:
            Dimension if store exists, None otherwise
        """
        if not session_id or not rag_dir:
            return None

        store_config = config.get("RAG", {}).get("VECTOR_STORE", {})
        store_type = store_config.get("TYPE", "faiss")

        if store_type == "faiss":
            faiss_path = os.path.join(rag_dir, f"rag_store_{session_id}.faiss")
            if os.path.exists(faiss_path):
                try:
                    import faiss

                    index = faiss.read_index(faiss_path)
                    logger.debug(f"Detected existing FAISS store with dim={index.d}")
                    return index.d
                except Exception as e:
                    logger.warning(f"Failed to read FAISS store: {e}")
                    return None

        elif store_type == "chromadb":
            chroma_path = os.path.join(rag_dir, f"rag_store_{session_id}")
            if os.path.exists(chroma_path):
                try:
                    from .chroma_store import ChromaStore

                    temp_store = ChromaStore(
                        dim=1024, session_id=session_id, rag_dir=rag_dir
                    )
                    logger.debug(
                        f"Detected existing ChromaDB store with dim={temp_store.dim}"
                    )
                    return temp_store.dim
                except Exception as e:
                    logger.warning(f"Failed to read ChromaDB store: {e}")
                    return None

        return None

    def _initialize_store(self) -> Any:
        """Initialize the appropriate vector store.

        Returns:
            Vector store instance (FaissStore or ChromaStore)
        """
        if self.store_type == "faiss":
            from .faiss_store import FaissStore

            logger.debug(f"Initializing FAISS store (dim={self.dim})")
            return FaissStore(
                self.dim, session_id=self.session_id, rag_dir=self.rag_dir
            )
        elif self.store_type == "chromadb":
            from .chroma_store import ChromaStore

            logger.debug(f"Initializing ChromaDB store (dim={self.dim})")
            return ChromaStore(
                self.dim, session_id=self.session_id, rag_dir=self.rag_dir
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")

    def add(self, vectors: np.ndarray, metadatas: List[Dict]) -> None:
        """Add vectors and metadata to the store.

        Args:
            vectors: NumPy array of vectors with shape (n, dim)
            metadatas: List of metadata dictionaries, one per vector
        """
        return self.store.add(vectors, metadatas)

    def search(self, q: np.ndarray, top_k: int = 5) -> Tuple[List[float], List[Dict]]:
        """Search for similar vectors.

        Args:
            q: Query vector with shape (dim,)
            top_k: Number of results to return (default: 5)

        Returns:
            Tuple of (scores, metadatas) where scores are similarity scores
        """
        return self.store.search(q, top_k)

    def clear(self) -> None:
        """Clear all data from the store."""
        return self.store.clear()

    @property
    def index(self) -> Any:
        """Access underlying index (for compatibility).

        Returns:
            Underlying vector index or None if not available
        """
        return getattr(self.store, "index", None)

    @property
    def metadatas(self) -> List[Dict]:
        """Access metadata list.

        Returns:
            List of metadata dictionaries for all stored vectors
        """
        return self.store.metadatas
