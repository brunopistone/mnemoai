"""Session RAG helpers: embed with Ollama when available and store in FAISS (if present).

This is a compact, defensive implementation to avoid previous merge/indent issues.
"""

from .vector_store_controller import VectorStoreController
from ..readers.chunking_helper import __split_into_chunks as split_into_chunks
from datetime import datetime
import numpy as np
import os
from typing import List, Tuple, Dict, Optional, Any

from mnemoai.utils.bm25 import BM25
from mnemoai.utils.config import config
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import profile_dir
from mnemoai.models.controllers.embeddings_controller import EmbeddingsController


# Global reference to the RAG session (set by chat_interface)
_rag_session = None


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
        return _rag_session

    # MCP subprocess: read session_id from file and create session
    try:
        # Profile-specific directory under the app home
        rag_dir = str(profile_dir())
        session_file = os.path.join(rag_dir, "rag_session_id.txt")

        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                session_id = f.read().strip()

            embed_model_config = config.get("RAG", {}).get("EMBED_MODEL_ID", {})
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

    # Also remove session file
    rag_dir = str(profile_dir())
    session_file = os.path.join(rag_dir, "rag_session_id.txt")

    if os.path.exists(session_file):
        os.remove(session_file)


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
        """Rebuild the BM25 index from the current store's metadata."""
        if self.store is None or not hasattr(self.store, "metadatas"):
            return
        texts = [m.get("text", "") for m in self.store.metadatas]
        if texts:
            self.bm25 = BM25()
            self.bm25.fit(texts)
            logger.debug(f"BM25 index built with {len(texts)} documents")

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed texts using the embeddings controller.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings
        """
        return self.embeddings_controller.embed(texts)

    def ingest(self, doc_id: str, content: str, chunk_size_tokens: int = 2048) -> int:
        """Ingest document content into the RAG system.

        Args:
            doc_id: Unique identifier for the document
            content: Full text content to index
            chunk_size_tokens: Size of chunks in tokens (default: 2048)

        Returns:
            Number of chunks created and indexed
        """
        logger.debug(
            f"RAG ingest: doc_id={doc_id}, content_len={len(content)}, chunk_size_tokens={chunk_size_tokens}"
        )
        try:
            chunks = split_into_chunks(content, chunk_size_tokens)
            logger.debug(f"RAG ingest: split into {len(chunks)} chunks")
            for i, c in enumerate(chunks[:3]):
                logger.debug(f"  Chunk {i}: {len(c)} chars")
        except Exception as e:
            logger.warning(f"Failed to import chunking_helper: {e}, using fallback")
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

        if vectors and np is not None:
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

            if hasattr(self.store, "add"):
                # Check for dimension mismatch
                store_dim = getattr(self.store, "dim", None)

                if store_dim is not None and store_dim != batch_dim:
                    logger.warning(
                        "Dimension mismatch: store dim=%s, batch dim=%s. Recreating store and clearing old data.",
                        store_dim,
                        batch_dim,
                    )
                    # Recreate store with correct dim, clearing old data
                    self.store = VectorStoreController(
                        batch_dim, session_id=self.session_id, rag_dir=self.rag_dir
                    )
                    self.dim = batch_dim

                self.store.add(all_vecs, metas)
            else:
                for v, m in zip(all_vecs, metas):
                    self.store["vectors"].append(v)
                    self.store["metadatas"].append(m)

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
        if self.store is None:
            return [], []

        if not query_text or not query_text.strip():
            logger.warning("Empty query text provided")
            return [], []

        logger.debug(f"Querying RAG with text: '{query_text}'")
        embeddings = self._embed_batch([query_text])
        if embeddings.shape[0] == 0:
            logger.error(f"Embedding returned empty array for query: '{query_text}'")
            return [], []

        vec = embeddings[0]
        candidate_k = top_k * 3

        # --- Semantic candidates ---
        sem_candidates: Dict[str, Tuple[float, Dict]] = {}
        if hasattr(self.store, "search"):
            sem_scores, sem_metas = self.store.search(
                vec, top_k=min(candidate_k, len(self.store.metadatas))
            )
        else:
            scores_metas: List[Tuple[float, Dict]] = []
            for v, m in zip(self.store["vectors"], self.store["metadatas"]):
                score = float(
                    np.dot(vec, v) / (np.linalg.norm(vec) * np.linalg.norm(v) + 1e-12)
                )
                scores_metas.append((score, m))
            scores_metas.sort(key=lambda x: x[0], reverse=True)
            sem_scores = [s for s, _ in scores_metas[:candidate_k]]
            sem_metas = [m for _, m in scores_metas[:candidate_k]]

        for score, meta in zip(sem_scores, sem_metas):
            key = f"{meta.get('doc_id', '')}:{meta.get('chunk_idx', '')}"
            sem_candidates[key] = (score, meta)

        # --- BM25 candidates ---
        bm25_candidates: Dict[str, Tuple[float, Dict]] = {}
        if self.bm25 and self.bm25.corpus_size > 0:
            raw_bm25 = self.bm25.score(query_text)
            max_bm25 = max(raw_bm25) if raw_bm25 else 0.0

            if max_bm25 > 0:
                # Get top candidate_k BM25 results by index
                indexed_scores = sorted(
                    enumerate(raw_bm25), key=lambda x: x[1], reverse=True
                )[:candidate_k]

                all_metas = self.store.metadatas
                for idx, score in indexed_scores:
                    if score <= 0 or idx >= len(all_metas):
                        continue
                    norm_score = score / max_bm25
                    meta = all_metas[idx]
                    key = f"{meta.get('doc_id', '')}:{meta.get('chunk_idx', '')}"
                    bm25_candidates[key] = (norm_score, meta)

        # --- Merge and re-rank ---
        all_keys = set(sem_candidates.keys()) | set(bm25_candidates.keys())
        hybrid_results = []

        for key in all_keys:
            sem_score = sem_candidates[key][0] if key in sem_candidates else 0.0
            bm25_score = bm25_candidates[key][0] if key in bm25_candidates else 0.0
            meta = (
                sem_candidates[key][1]
                if key in sem_candidates
                else bm25_candidates[key][1]
            )

            hybrid_score = (
                self.semantic_weight * sem_score + self.keyword_weight * bm25_score
            )
            hybrid_results.append((hybrid_score, meta))

        hybrid_results.sort(key=lambda x: x[0], reverse=True)

        top_results = hybrid_results[:top_k]
        return [r[0] for r in top_results], [r[1] for r in top_results]
