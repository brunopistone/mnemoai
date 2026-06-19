"""RAG tools for managing and querying indexed documents."""

from .rag.session import get_rag_session
from mcp.server.fastmcp import FastMCP

from mnemoai.utils.logger import logger


def register_rag_tools(mcp: FastMCP) -> None:
    """Register RAG tools (list_documents, search_in_documents, clear_documents) with MCP server.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    def list_documents() -> str:
        """List all documents currently indexed in the RAG system.

        Use this to check what documents are available before searching.

        Returns:
            List of indexed documents with their chunk counts
        """
        logger.debug("Tool list_documents called")

        try:
            rag_instance = get_rag_session()
            if rag_instance is None or rag_instance.store is None:
                return "No documents have been indexed yet."

            if hasattr(rag_instance.store, "metadatas"):
                metas = rag_instance.store.metadatas
            else:
                metas = rag_instance.store.get("metadatas", [])

            if not metas:
                return "No documents have been indexed yet."

            # Count chunks per document
            doc_counts = {}
            for meta in metas:
                doc_id = meta.get("doc_id", "unknown")
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

            result = f"Indexed documents ({len(doc_counts)} total):\n\n"
            for doc_id, count in doc_counts.items():
                result += f"• {doc_id} ({count} chunks)\n"

            return result.strip()

        except Exception as e:
            logger.exception("Error listing documents")
            return f"Error listing documents: {e}"

    @mcp.tool()
    def search_in_documents(query: str, top_k: int = 8) -> str:
        """Search for information in indexed documents using hybrid semantic + BM25 keyword search.

        Use short, focused queries (3-6 keywords) for best results. Extract only the key terms from the user's question. Use top_k of at least 5 for accurate results.

        Examples:
        - User asks "How do I format JSON for SFT with reasoning?"
          → Good query: "SFT JSON reasoning"
          → Bad query: "how to format JSON for SFT with reasoning content"

        Args:
            query: SHORT search query with key terms (3-6 keywords)
            top_k: Number of relevant chunks to retrieve (default 8)

        Returns:
            Relevant text chunks from indexed documents with similarity scores
        """
        logger.debug(
            f"Tool search_in_documents called with query: {query}, top_k: {top_k}"
        )

        try:
            if not query:
                return "Error: query parameter is required."

            rag_instance = get_rag_session()
            if rag_instance is None:
                return "No documents have been indexed yet."

            scores, metas = rag_instance.query(query, top_k=top_k)

            if not metas:
                return "No relevant information found in indexed documents."

            result = f"Found {len(metas)} relevant chunks:\n\n"
            for i, (score, meta) in enumerate(zip(scores, metas), 1):
                doc_id = meta.get("doc_id", "unknown")
                chunk_idx = meta.get("chunk_idx", "?")
                text = meta.get("text", "")
                result += f"[{i}] Score: {score:.3f} | Doc: {doc_id} | Chunk: {chunk_idx}\n{text}\n\n"

            return result.strip()

        except Exception as e:
            logger.exception("Error searching documents")
            return f"Error searching documents: {e}"

    @mcp.tool()
    def clear_documents() -> str:
        """Clear all indexed documents from the RAG system.

        This removes all documents and resets the RAG database.
        Use this when you want to start fresh with new documents.

        Returns:
            Confirmation message
        """
        logger.debug("Tool clear_documents called")

        try:
            rag_instance = get_rag_session()
            if rag_instance is None:
                return "No documents to clear."

            # Reset the store
            if hasattr(rag_instance.store, "metadatas"):
                rag_instance.store.metadatas = []
                if hasattr(rag_instance.store, "index"):
                    import faiss

                    rag_instance.store.index = faiss.IndexFlatIP(rag_instance.dim)
                rag_instance.store._persist()
            else:
                rag_instance.store["vectors"] = []
                rag_instance.store["metadatas"] = []

            return "All documents have been cleared from the RAG system."

        except Exception as e:
            logger.exception("Error clearing documents")
            return f"Error clearing documents: {e}"
