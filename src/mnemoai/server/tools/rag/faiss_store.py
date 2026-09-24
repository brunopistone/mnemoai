"""Simple FAISS-backed vector store with file persistence for session-scoped RAG.

Metadata is persisted as **JSON, not pickle**. Unpickling executes arbitrary
code by design, and this file is read from a predictable path at startup, so a
planted ``.meta`` was remote code execution in the app's own process. The
metadata is plain chunk dicts (text + source + offsets), so JSON loses nothing.

The persist path is always inside the app profile directory. It previously fell
back to ``/tmp/rag_store*``, which is world-writable and shared between users --
the exact place an attacker can pre-plant a file for the load path to pick up.
"""

import logging

logging.getLogger("faiss").setLevel(logging.WARNING)

import os  # noqa: E402  (see logging.setLevel above: it must precede faiss)
import threading  # noqa: E402
from typing import Dict, List, Tuple  # noqa: E402

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from mnemoai.utils.atomic_write import atomic_write_json  # noqa: E402
from mnemoai.utils.embedding_integrity import (  # noqa: E402
    discard_faiss_repair_state,
    load_faiss_pair,
    mark_faiss_clean,
    require_clean_vectors,
)
from mnemoai.utils.logger import logger  # noqa: E402
from mnemoai.utils.paths import profile_dir  # noqa: E402


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
            persist_path: Optional explicit persistence path
            session_id: Optional session ID for persistence
            rag_dir: Optional RAG directory path (defaults to the app profile dir)
        """
        self.dim = dim

        # Always land inside the app profile dir. No /tmp fallback: a shared,
        # world-writable location is where a hostile metadata file would be
        # planted, and it also leaks indexed document text to other users.
        base_dir = rag_dir or str(profile_dir())
        if persist_path:
            self.persist_path = persist_path
        elif session_id:
            self.persist_path = os.path.join(base_dir, f"rag_store_{session_id}.faiss")
        else:
            self.persist_path = os.path.join(base_dir, "rag_store.faiss")

        # ".meta.json" (not the old ".meta") so a pickle written by a previous
        # version is never interpreted. An index without readable JSON metadata
        # stays disabled until explicitly cleared/re-ingested.
        self.metadata_path = self.persist_path + ".meta.json"
        self.lock = threading.Lock()

        self.index, self.metadatas, self.integrity_error = load_faiss_pair(
            self.persist_path, self.metadata_path
        )
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.dim = self.index.d

    def add(self, vectors: np.ndarray, metadatas: list[dict]) -> None:
        """Add vectors and metadata to FAISS index.

        Args:
            vectors: NumPy array of vectors with shape (n, dim)
            metadatas: List of metadata dictionaries, one per vector
        """

        require_clean_vectors(self)
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
        require_clean_vectors(self)
        if q.dtype != np.float32:
            q = q.astype(np.float32)

        # normalize
        norm = np.linalg.norm(q)
        if norm == 0:
            norm = 1.0
        q = q / norm

        with self.lock:
            D, I = self.index.search(np.expand_dims(q, axis=0), top_k)
            metadata_snapshot = list(self.metadatas)

        indices = I[0].tolist()
        scores = D[0].tolist()

        results = []
        metas = []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(metadata_snapshot):
                continue
            metas.append(metadata_snapshot[idx])
            results.append(score)

        return results, metas

    def replace_document(self, doc_id: str, vectors: np.ndarray, metadatas: list[dict]) -> None:
        """Build the replacement index before swapping it into the live store."""
        require_clean_vectors(self)
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms == 0, 1, norms)
        with self.lock:
            retained = [i for i, meta in enumerate(self.metadatas) if meta.get("doc_id") != doc_id]
            index = faiss.IndexFlatIP(self.dim)
            if retained:
                index.add(np.vstack([self.index.reconstruct(i) for i in retained]))
            index.add(vectors)
            self.index = index
            self.metadatas = [self.metadatas[i] for i in retained] + list(metadatas)
            self._persist()

    def clear(self, dim: int = None) -> None:
        """Clear all vectors and metadata from the store."""
        with self.lock:
            discard_faiss_repair_state(self.persist_path)
            if dim is not None:
                self.dim = dim
                self.index = faiss.IndexFlatIP(dim)
            else:
                self.index.reset()
            self.metadatas = []
            self.integrity_error = None
            self._persist()

    def _persist(self) -> None:
        """Save index and metadata to disk.

        The metadata write is atomic so a crash can't leave a half-written file
        that the next start would reject (and rebuild from scratch).
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.persist_path)), exist_ok=True)
            faiss.write_index(self.index, self.persist_path)
            atomic_write_json(self.metadata_path, self.metadatas)
            mark_faiss_clean(self.persist_path, self.metadata_path)
        except Exception as e:
            # exc_info, not file= : logger.error() takes no `file` kwarg, so the
            # previous version raised TypeError from inside its own handler.
            logger.error(f"Failed to persist RAG store: {e}", exc_info=True)


def create_store(dim: int) -> "FaissStore":
    """Create a new FAISS store with specified dimension.

    Args:
        dim: Embedding dimension

    Returns:
        New FaissStore instance
    """
    return FaissStore(dim)
