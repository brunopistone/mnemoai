"""Universal chunking and summarization for large files.

Improvements added:
- simple on-disk SQLite cache to store per-chunk summaries keyed by a sha256 hash
- concurrent summarization of chunks using an asyncio Semaphore limited by
  `CHUNKING_CONCURRENCY` in config (defaults to 3)
"""

import sys
import os
import textwrap
import tiktoken
import hashlib
import sqlite3
import asyncio
from datetime import datetime

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from models.llm_controller import LangChainLLMController
from utils.config import config
from utils.logger import logger

MODEL_ID = "gpt-4"


def _get_cache_db_path() -> str:
    """Get cache DB path from session file (same pattern as RAG).

    Returns:
        Path to cache database
    """

    # Read session_id from file (written by client)
    user_home = os.path.expanduser("~")
    profile_name = config.get("PROFILE", {}).get("NAME", "default")
    rag_dir = os.path.join(user_home, "agent-conversations", profile_name)
    session_file = os.path.join(rag_dir, "chunk_session_id.txt")

    if os.path.exists(session_file):
        try:
            with open(session_file, "r") as f:
                session_id = f.read().strip()
            _cache_db_path = os.path.join(rag_dir, f"chunk_cache_{session_id}.db")
            return _cache_db_path
        except Exception:
            pass

    # Fallback to profile-level cache
    _cache_db_path = os.path.join(rag_dir, "chunk_cache.db")
    return _cache_db_path


def reset_session_chunk_cache() -> None:
    """Reset the session Chunk cache instance (called on /clear or app exit)."""

    user_home = os.path.expanduser("~")
    profile_name = config.get("PROFILE", {}).get("NAME", "default")
    rag_dir = os.path.join(user_home, "agent-conversations", profile_name)
    session_file = os.path.join(rag_dir, "chunk_session_id.txt")

    if os.path.exists(session_file):
        os.remove(session_file)


def _get_cached_summary(key: str) -> str | None:
    """Get cached summary from SQLite database.

    Args:
        key: Cache key (hash of content)

    Returns:
        Cached summary or None if not found
    """
    try:
        db_path = _get_cache_db_path()

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT summary FROM chunk_cache WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _set_cached_summary(key: str, summary: str) -> None:
    """Store summary in SQLite cache.

    Args:
        key: Cache key (hash of content)
        summary: Summary text to cache

    Returns:
        None
    """
    try:
        db_path = _get_cache_db_path()

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "REPLACE INTO chunk_cache (key, summary, updated_at) VALUES (?, ?, ?)",
            (key, summary, datetime.utcnow().isoformat()),
        )
        conn.commit()
    except Exception:
        logger.exception("Failed to write summary cache")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def __count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for

    Returns:
        Token count
    """
    try:
        encoding = tiktoken.get_encoding(MODEL_ID)
        return len(encoding.encode(text))
    except:
        return len(text) // 4  # Rough estimate


def __split_into_chunks(content: str, chunk_size: int = 512) -> list[str]:
    """Split content into overlapping chunks by token count.

    Args:
        content: Text content to split
        chunk_size: Target tokens per chunk

    Returns:
        List of text chunks with 10% overlap
    """
    separators = ["\n\n", "\n", ". ", "! ", "? ", " "]

    def hard_split(text: str, max_tokens: int) -> list[str]:
        """Hard split by words when no separators work."""
        words = text.split()
        chunks = []
        current_words = []
        for word in words:
            current_words.append(word)
            if __count_tokens(" ".join(current_words)) >= max_tokens:
                current_words.pop()
                if current_words:
                    chunks.append(" ".join(current_words))
                current_words = [word]
        if current_words:
            chunks.append(" ".join(current_words))
        return chunks

    def recursive_split(text: str, sep_index: int = 0) -> list[str]:
        """Recursively split text using hierarchical separators."""
        if __count_tokens(text) <= chunk_size:
            return [text] if text.strip() else []

        if sep_index >= len(separators):
            return hard_split(text, chunk_size)

        separator = separators[sep_index]
        parts = text.split(separator)
        chunks = []
        current = ""

        for part in parts:
            if not part.strip():
                continue

            test = current + separator + part if current else part

            if __count_tokens(test) > chunk_size:
                if current:
                    chunks.append(current.strip())
                # Recursively split part if too large
                if __count_tokens(part) > chunk_size:
                    chunks.extend(recursive_split(part, sep_index + 1))
                    current = ""
                else:
                    current = part
            else:
                current = test

        if current.strip():
            if __count_tokens(current) > chunk_size:
                chunks.extend(recursive_split(current, sep_index + 1))
            else:
                chunks.append(current.strip())

        return chunks

    base_chunks = recursive_split(content)

    if len(base_chunks) <= 1:
        return base_chunks

    # Add 10% overlap
    overlap_tokens = int(chunk_size * 0.1)
    overlapped = [base_chunks[0]]
    for i in range(1, len(base_chunks)):
        prev = base_chunks[i - 1]
        sentences = prev.replace("! ", ". ").replace("? ", ". ").split(". ")
        overlap_text = ""
        for sentence in reversed(sentences):
            test = sentence + ". " + overlap_text if overlap_text else sentence
            if __count_tokens(test) > overlap_tokens:
                break
            overlap_text = test
        overlapped.append(
            f"{overlap_text}\n\n{base_chunks[i]}" if overlap_text else base_chunks[i]
        )

    logger.debug(
        f"Split {len(content)} chars into {len(overlapped)} chunks of max {chunk_size} tokens"
    )
    return overlapped


async def __summarize_with_model(text: str, context: str = "") -> str:
    """Summarize text using the configured model.

    Args:
        text: Text to summarize
        context: Additional context (e.g., "chunk 1 of 5")

    Returns:
        Summarized text
    """
    try:
        llm_controller = LangChainLLMController()
        llm_controller.initialize_model()
        model = llm_controller.get_model()

        if not model:
            return text[:2000] + "...[model initialization failed]"

        prompt = f"""
        <instructions>
        You are creating a detailed summary of a technical document. Your summary will be used to answer specific questions.

        CRITICAL - Preserve ALL:
        - Exact names, IDs, paths, URLs, container names, model names
        - All configuration parameters, settings, and their values
        - Code snippets, commands, and technical syntax
        - Version numbers and specifications
        - Step-by-step procedures and instructions
        - Examples and sample configurations

        Do NOT generalize or simplify technical details.
        {f'Note: This is {context}' if context else ''}
        </instructions>

        <text>
        {text}
        </text>

        Provide a comprehensive summary retaining all technical details:
        """

        prompt = textwrap.dedent(prompt).strip()

        # Use LangChain model's invoke method
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content=prompt)]
        response = model.invoke(messages)

        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        return (
            response_text.strip() if response_text else text[:2000] + "...[no response]"
        )

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return text[:2000] + "...[summarization failed]"


async def process_large_content(
    content: str, chunk_size: int = 1024 * 8
) -> tuple[str, dict]:
    """Process large content with chunking and summarization.

    Args:
        content: Full content to process
        chunk_size: Target output size (default: 8000 tokens)

    Returns:
        Tuple of (processed_content, metadata)
    """

    total_tokens = __count_tokens(content)
    # If small enough, return as-is
    if total_tokens <= chunk_size:
        return content, {
            "chunked": False,
            "total_tokens": total_tokens,
            "chunks_processed": 0,
        }

    print(f"\033[38;5;98m[CHUNKING]\033[0m Total tokens: {total_tokens}")

    chunks = __split_into_chunks(content, chunk_size)

    print(f"\033[38;5;98m[CHUNKING]\033[0m Total chunks: {len(chunks)}")
    # Summarize each chunk concurrently with a bounded semaphore and local cache
    concurrency = int(config.get("CHUNKING_CONCURRENCY", 3))
    semaphore = asyncio.Semaphore(concurrency)

    async def _summarize_chunk(index: int, chunk_text: str) -> str:
        """Summarize a single chunk with caching and concurrency control.

        Args:
            index: Chunk index
            chunk_text: Text content to summarize

        Returns:
            Summary text
        """
        # Compute a stable key for this chunk
        key = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

        # Try cache first
        cached = _get_cached_summary(key)
        if cached:
            logger.debug(f"[CHUNK_CACHE] hit for chunk {index+1}")
            return f"=== Part {index+1}/{len(chunks)} ===\n{cached}"

        async with semaphore:
            print(f"\033[38;5;98m[CHUNKING]\033[0m Processing chunk {index+1}")
            context = f"Part {index+1} of {len(chunks)}"
            summary = await __summarize_with_model(chunk_text, context)

        # Cache summary (best-effort)
        try:
            _set_cached_summary(key, summary)
        except Exception:
            logger.debug("Failed to cache chunk summary, continuing")

        return f"=== Part {index+1}/{len(chunks)} ===\n{summary}"

    # Launch concurrent summarization tasks
    tasks = [asyncio.create_task(_summarize_chunk(i, c)) for i, c in enumerate(chunks)]
    summaries = await asyncio.gather(*tasks)
    combined = "\n\n".join(summaries)
    combined_tokens = __count_tokens(combined)

    # If combined summaries still too large, summarize again
    if combined_tokens > chunk_size:
        final_summary = await __summarize_with_model(
            combined, f"Final summary of {len(chunks)} parts"
        )
        final_tokens = __count_tokens(final_summary)
        compression = f"{total_tokens / final_tokens:.1f}x"

        return final_summary, {
            "chunked": True,
            "original_tokens": total_tokens,
            "chunks_processed": len(chunks),
            "final_tokens": final_tokens,
            "compression_ratio": compression,
        }

    compression = f"{total_tokens / combined_tokens:.1f}x"

    return combined, {
        "chunked": True,
        "original_tokens": total_tokens,
        "chunks_processed": len(chunks),
        "final_tokens": combined_tokens,
        "compression_ratio": compression,
    }
