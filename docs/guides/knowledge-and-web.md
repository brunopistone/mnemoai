# Knowledge & web tools

## Web Search Configuration

This tool uses the Brave Search API. Obtain an API key from [Brave Search Developer Portal](https://brave.com/search/api/).

```yaml
BRAVE_API_KEY: your-api-key-here # For web search
```

## Web Crawler Configuration

Enable web page content extraction with automatic RAG integration:

```yaml
ENABLE_WEB_CRAWL: true
```

When enabled, the `web_crawler` tool:

- Extracts content from web pages as markdown
- Automatically ingests large pages (>8K tokens) into RAG (if enabled)
- Uses the same chunking configuration as PDF/DOCX readers

> **Browser dependency.** Crawling uses a headless Chromium via Playwright,
> whose browser binary is a separate ~260MB download not pulled in by
> `pip` / `uv tool install`. The tool installs it automatically on the first
> crawl after a fresh install/upgrade. If that auto-install fails (e.g.
> offline), run it manually in the same environment:
> `python -m playwright install chromium` (for an installed CLI:
> `~/.local/share/uv/tools/mnemoai/bin/python -m playwright install chromium`).

## RAG (Retrieval-Augmented Generation)

The RAG system automatically indexes documents for semantic search with **hybrid search** (semantic embeddings + BM25 keyword scoring).

**How it works:**

1. Read a PDF/DOCX file → Automatically chunked and indexed
2. Ask questions → Assistant searches indexed documents first using hybrid search
3. Session-scoped → Cleared on `/clear` or exit

**RAG Tools:**

- `list_documents()`: Show indexed documents
- `search_in_documents(query, top_k)`: Hybrid semantic + BM25 search
- `clear_documents()`: Clear RAG index

**Search internals:**

- Recursive chunking with 10% overlap
- Hybrid search: BM25 (Okapi BM25 with TF-IDF, term saturation, length normalization) + semantic similarity
- Independent candidate retrieval from both BM25 and embeddings, merged and re-ranked

For config keys — chunk size, vector store choice (ChromaDB/FAISS), and
hybrid search weights — see [RAG Configuration](../configuration.md#rag-configuration).
