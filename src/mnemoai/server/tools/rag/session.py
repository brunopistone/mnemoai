"""Session-scoped document ingestion and hybrid semantic/BM25 search."""

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mnemoai.models.controllers.embeddings_controller import EmbeddingsController
from mnemoai.utils.bm25 import BM25
from mnemoai.utils.config import config
from mnemoai.utils.embedding_integrity import require_clean_vectors
from mnemoai.utils.hybrid_search import (
    candidate_count,
    merge_and_rank,
    normalized_bm25_candidates,
)
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import profile_dir, rag_session_pointer_path

from ..readers.chunking_helper import __split_into_chunks as split_into_chunks
from .vector_store_controller import VectorStoreController


def _chunk_key(meta: Dict[str, Any]) -> str:
    """Identity of one chunk within a session: which doc, which slice of it."""
    return f"{meta.get('doc_id', '')}:{meta.get('chunk_idx', '')}"


# Global reference to the RAG session (set by chat_interface)
_rag_session = None
_rag_session_lock = threading.RLock()


def set_rag_session(session: Optional[Any]) -> None:
    """Set the RAG session instance (called by chat_interface).

    Args:
        session: SessionRAG instance or None to clear
    """
    global _rag_session
    _rag_session = session
    logger.debug(f"RAG session set: {session.session_id if session else 'None'}")


def get_rag_session() -> Optional[Any]:
    """Get the current RAG session instance.

    In MCP subprocess, reads session_id from file and creates session if needed.

    Returns:
        SessionRAG instance or None
    """
    global _rag_session

    # If already set (same process), return it
    if _rag_session is not None:
        pointer = rag_session_pointer_path()
        try:
            wanted = pointer.read_text().strip() if pointer.is_file() else ""
        except OSError:
            wanted = ""
        if not wanted or wanted == _rag_session.session_id:
            return _rag_session
        _rag_session = None

    # MCP subprocess: read session_id from file and create session. The pointer
    # file is per-instance (namespaced by MNEMOAI_INSTANCE_ID, inherited from the
    # parent) so this subprocess reads ITS OWN parent's session, not another tab's.
    try:
        # Profile-specific directory under the app home
        rag_dir = str(profile_dir())
        session_file = rag_session_pointer_path()

        if session_file.exists():
            session_id = session_file.read_text().strip()

            embed_model_config = config.get("RAG", {}).get("EMBED_MODEL_ID", {})
            with _rag_session_lock:
                if _rag_session is None or _rag_session.session_id != session_id:
                    _rag_session = SessionRAG(
                        embed_model_config=embed_model_config,
                        session_id=session_id,
                        rag_dir=rag_dir,
                    )
            logger.debug(f"RAG session created in subprocess: {session_id}")
            return _rag_session
    except Exception as e:
        logger.warning(f"Failed to create RAG session in subprocess: {e}")

    return None


def reset_session_rag() -> None:
    """Reset the session RAG instance (called on /clear or app exit)."""
    global _rag_session
    if _rag_session is not None:
        logger.debug(f"Closing RAG session: {_rag_session.session_id}")
        _rag_session = None

    # Also remove THIS instance's pointer file (never another tab's).
    session_file = rag_session_pointer_path()
    if session_file.exists():
        session_file.unlink()


def _fallback_chunker(content: str, chunk_size: int = 1024 * 8) -> List[str]:
    """Fallback text chunker when advanced chunking is unavailable.

    Args:
        content: Text content to chunk
        chunk_size: Maximum chunk size in characters

    Returns:
        List of text chunks
    """
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 > chunk_size:
            if current:
                chunks.append(current.strip())
            current = p
        else:
            current = current + "\n\n" + p if current else p
    if current:
        chunks.append(current.strip())
    return chunks


class SessionRAG:
    def __init__(
        self,
        embed_model_config: dict = None,
        dim: int = None,
        session_id: str = None,
        rag_dir: str = None,
    ) -> None:
        """Initialize RAG session.

        Args:
            embed_model_config: Optional embedding model configuration
            dim: Optional embedding dimension
            session_id: Optional session ID
            rag_dir: Optional RAG directory path
        """
        self.embeddings_controller = EmbeddingsController(embed_model_config)
        self.dim = dim  # Will be set from first embedding if None
        self.session_id = session_id or self._generate_session_id()
        self.rag_dir = rag_dir
        self._ingest_lock = threading.RLock()

        # Load hybrid search weights from config
        rag_search_config = config.get("RAG", {}).get("SEARCH", {})
        self.semantic_weight = rag_search_config.get("SEMANTIC_WEIGHT", 0.5)
        self.keyword_weight = rag_search_config.get("KEYWORD_WEIGHT", 0.5)

        self.bm25: Optional[BM25] = None

        # Try to load existing store using controller
        if rag_dir and session_id:
            detected_dim = VectorStoreController.detect_existing_store(
                session_id, rag_dir
            )
            if detected_dim:
                self.dim = detected_dim
                self.embeddings_controller.dim = self.dim
                self.store = VectorStoreController(
                    self.dim, session_id=session_id, rag_dir=rag_dir
                )
                logger.debug(f"Loaded existing vector store with dim={self.dim}")
                self._rebuild_bm25()
            else:
                self.store = None  # Will be created on first ingest
        else:
            self.store = None  # Will be created on first ingest

    def _generate_session_id(self) -> str:
        """Generate a unique session ID with profile name.

        Returns:
            Session ID string
        """
        profile_name = config.get("PROFILE", {}).get("NAME", "default")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{profile_name}_{timestamp}"

    def _rebuild_bm25(self) -> None:
        """Rebuild the BM25 index from the current store's metadata.

        An EMPTY corpus drops the index rather than leaving the previous one in
        place — a stale BM25 outlives the vectors it was built from, and its
        tokenized copy of every document is then still searchable.
        """
        if self.store is None or not hasattr(self.store, "metadatas"):
            return
        texts = [m.get("text", "") for m in self.store.metadatas]
        if not texts:
            self.bm25 = None
            return
        self.bm25 = BM25()
        self.bm25.fit(texts)
        logger.debug(f"BM25 index built with {len(texts)} documents")

    def clear(self) -> None:
        """Drop every indexed document: vectors AND the keyword index.

        Clearing only the vector store left the BM25 corpus holding the tokenized
        text of every "cleared" document for the life of the process. Nothing was
        served from it only because ``normalized_bm25_candidates`` discards indices
        past the (now-empty) metadata list — i.e. the isolation rested on a bounds
        check, not on the data being gone. Both halves are dropped here.
        """
        if self.store is not None:
            clear = getattr(self.store, "clear", None)
            if callable(clear):
                clear()
        self.bm25 = None
        logger.debug("Session RAG cleared (vectors + BM25)")

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed texts using the embeddings controller.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings
        """
        return self.embeddings_controller.embed(texts)

    def ingest(self, doc_id: str, content: str, chunk_size_tokens: int = 2048) -> int:
        """Serialize document replacement, including first-store initialization."""
        with self._ingest_lock:
            return self._ingest(doc_id, content, chunk_size_tokens)

    def _ingest(self, doc_id: str, content: str, chunk_size_tokens: int = 2048) -> int:
        """Ingest document content into the RAG system.

        Args:
            doc_id: Unique identifier for the document
            content: Full text content to index
            chunk_size_tokens: Size of chunks in tokens (default: 2048)

        Returns:
            Number of chunks created and indexed
        """
        if self.store is not None:
            require_clean_vectors(self.store)
        logger.debug(
            f"RAG ingest: doc_id={doc_id}, content_len={len(content)}, chunk_size_tokens={chunk_size_tokens}"
        )
        try:
            chunks = split_into_chunks(content, chunk_size_tokens)
            logger.debug(f"RAG ingest: split into {len(chunks)} chunks")
            for i, c in enumerate(chunks[:3]):
                logger.debug(f"  Chunk {i}: {len(c)} chars")
        except Exception as e:
            logger.warning("Document chunking failed (%s); using paragraph chunking", e)
            chunks = _fallback_chunker(content, chunk_size_tokens)

        logger.debug("Ingesting doc %s with %d chunks", doc_id, len(chunks))

        batch_size = 16
        vectors = []
        metas: List[Dict] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vecs = self._embed_batch(batch)
            for j, chunk in enumerate(batch):
                metas.append(
                    {
                        "doc_id": doc_id,
                        "chunk_idx": i + j,
                        "text": chunk,
                        "session_id": self.session_id,
                    }
                )
            vectors.append(vecs)

        if vectors:
            all_vecs = np.vstack(vectors)
            batch_dim = int(all_vecs.shape[1])

            # Initialize store on first use with correct dimension
            if self.store is None:
                self.dim = batch_dim
                self.store = VectorStoreController(
                    batch_dim, session_id=self.session_id, rag_dir=self.rag_dir
                )
                logger.debug(
                    f"Created vector store with dim={batch_dim}, session_id={self.session_id}"
                )

            store_dim = self.store.dim
            if store_dim != batch_dim:
                if self.store.metadatas:
                    raise ValueError(
                        f"Embedding dimension changed from {store_dim} to {batch_dim}. "
                        "Clear the document index before re-indexing with a different model."
                    )
                self.store.clear(dim=batch_dim)
                self.dim = batch_dim

            self.store.replace_document(doc_id, all_vecs, metas)

            # Rebuild BM25 index with all chunks
            self._rebuild_bm25()

        return len(chunks)

    def query(self, query_text: str, top_k: int = 6) -> Tuple[List[float], List[Dict]]:
        """Search indexed documents using hybrid search (semantic + BM25).

        Retrieves candidates independently from both semantic search and BM25,
        merges the two candidate sets, then re-ranks with a hybrid score.
        This ensures keyword-strong matches surface even when semantic similarity
        is low (e.g. exact name lookups).

        Args:
            query_text: Search query text
            top_k: Number of top results to return (default: 6)

        Returns:
            Tuple of (scores, metadatas) where scores are hybrid scores and metadatas contain chunk info
        """
        if self.store is None or not self.store.metadatas or top_k <= 0:
            return [], []

        if not query_text or not query_text.strip():
            logger.warning("Empty query text provided")
            return [], []

        logger.debug(f"Querying RAG with text: '{query_text}'")
        candidate_k = candidate_count(top_k)
        sem_candidates: Dict[str, Tuple[float, Dict]] = {}
        degraded = False
        try:
            if getattr(self.store, "integrity_error", None):
                raise RuntimeError(self.store.integrity_error)
            embeddings = self._embed_batch([query_text])
            if not len(embeddings):
                raise ValueError("Embedding provider returned no query vector")
            vec = embeddings[0]
            sem_scores, sem_metas = self.store.search(
                vec, top_k=min(candidate_k, len(self.store.metadatas))
            )
            for score, meta in zip(sem_scores, sem_metas):
                sem_candidates[_chunk_key(meta)] = (score, meta)
        except Exception as e:
            degraded = True
            logger.warning("Semantic document search unavailable; using BM25 only: %s", e)

        # --- Keyword candidates + merge (shared with the episodic stores) ---
        bm25_candidates = normalized_bm25_candidates(
            self.bm25,
            query_text,
            candidate_k,
            self.store.metadatas,
            key_fn=_chunk_key,
        )
        ranked = merge_and_rank(
            sem_candidates,
            bm25_candidates,
            0.0 if degraded else self.semantic_weight,
            1.0 if degraded else self.keyword_weight,
            top_k,
        )
        metas = [
            {**meta, "retrieval_method": "bm25"} if degraded else meta
            for _, meta in ranked
        ]
        return [score for score, _ in ranked], metas
