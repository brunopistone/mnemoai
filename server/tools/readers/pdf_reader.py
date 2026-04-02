"""PDF reading functionality."""

from .. import validate_file_path
import json
import PyPDF2
import sys
import os

# Add parent directory to path to allow imports from root
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from utils.config import config
from utils.logger import logger
from .chunking_helper import __count_tokens as count_tokens, process_large_content

try:
    from ..rag.session import get_rag_session

    _rag_available = True
except ImportError:
    _rag_available = False
    logger.debug("RAG module not available")


async def read_pdf(file_path: str) -> str:
    """Read PDF content with automatic chunking and summarization for large files.

    Args:
        file_path: Path to the PDF file (use EXACTLY as provided - do NOT escape spaces with backslashes)

    Returns:
        JSON with PDF content, truncated if exceeds MAX_TOKENS (20480).
    """
    try:
        # Validate and normalize path
        is_valid, normalized_path, error_dict = validate_file_path(file_path)
        if not is_valid:
            return json.dumps(error_dict)

        # Read PDF
        with open(normalized_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)

            full_text = ""
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text.strip():
                    full_text += f"--- Page {page_num} ---\n{page_text.strip()}\n\n"

            if not full_text.strip():
                return json.dumps(
                    {
                        "content": "",
                        "message": "PDF appears to be empty or contains only images",
                        "file_path": normalized_path,
                        "total_pages": total_pages,
                    }
                )

            # If RAG enabled, ingest into session store instead of summarizing everything
            if config.get("ENABLE_RAG", False) and _rag_available:
                tokens = count_tokens(full_text)
                if tokens > config.get("RAG", {}).get("MAX_TOKENS", 1024 * 8):
                    try:
                        rag = get_rag_session()
                        if rag is None:
                            logger.warning(
                                "RAG session not initialized - falling back to summarization"
                            )
                        else:
                            num_chunks = rag.ingest(
                                os.path.basename(normalized_path),
                                full_text,
                                chunk_size_tokens=int(
                                    config.get("RAG", {}).get("CHUNK_TOKENS", 1024)
                                ),
                            )

                            return json.dumps(
                                {
                                    "content": "",
                                    "message": "Document ingested into RAG store",
                                    "file_path": normalized_path,
                                    "total_pages": total_pages,
                                    "chunks_indexed": num_chunks,
                                }
                            )
                    except Exception as e:
                        logger.exception("RAG ingestion failed: %s", e)
                        # fallback to summarization

            # Process with chunking if needed (fallback or RAG not enabled)
            processed_content, metadata = await process_large_content(full_text)

            return json.dumps(
                {
                    "content": processed_content.strip(),
                    "file_path": normalized_path,
                    "total_pages": total_pages,
                    "processing_metadata": metadata,
                }
            )

    except Exception as e:
        logger.error(f"Error during PDF read: {str(e)}", exc_info=True)

        return json.dumps(
            {
                "error": True,
                "message": f"Error reading PDF: {str(e)}",
                "file_path": file_path,
            }
        )
