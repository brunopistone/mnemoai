# Knowledge & web tools

## Web Search Configuration

Choose **Brave Search** or **Tavily** with **`/config search`**, or during
first-run setup. This choice is independent of the chat-model provider.
Changing it restarts MnemoAI with the existing conversation preserved.
`/features` enables or disables web search.

Existing configurations use Brave unless you explicitly select Tavily. Only
the selected provider receives queries; a failure does not fall back to the
other service. You can retain both keys and switch providers at any time.

```yaml
ENABLE_WEB_SEARCH: true
WEB_SEARCH_PROVIDER: brave # brave (default) or tavily
BRAVE_API_KEY: your_brave_api_key
TAVILY_API_KEY: your_tavily_api_key
```

Get keys from [Brave](https://brave.com/search/api/) or
[Tavily](https://app.tavily.com/). Keys may also be supplied through environment
variables of the same names or the YAML `ENV:` block. A real top-level key
takes precedence over the environment; empty/template values do not.
`/doctor` checks the selected provider's key. Keep real keys out of Git.

Both providers expose **`web_search`** with citation-ready titles, links and
descriptions, plus provider metadata. The tool's parameters reflect the selected
adapter's capabilities.

### Brave Search

Brave keeps its existing parameters: `query`, `search_lang` (default `en`),
`num_results` (default 10, clamped to 1–20), `freshness`, `country`, and `ui_lang`.
Freshness accepts `pd`, `pw`, `pm`, `py`, or `YYYY-MM-DDtoYYYY-MM-DD`. Optional
filters are omitted when empty. Safe search remains off.

### Tavily advanced search

Tavily uses the same basic search fields plus:

| Parameters                                                        | Behavior                                                                                                                                                                                |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `search_depth`                                                    | `basic` (default), `advanced`, `fast`, or `ultra-fast`. Basic/fast/ultra-fast use 1 search credit; advanced uses 2.                                                                     |
| `topic`                                                           | `general` (default), `news`, or `finance`.                                                                                                                                              |
| `freshness`, `start_date`, `end_date`                             | Recency shortcuts map to day/week/month/year. Use freshness **or** explicit ISO dates, including open-ended ranges.                                                                     |
| `include_domains`, `exclude_domains`                              | Limit to up to 300 included domains; exclude up to 150. `include_domains_mode: prefer` boosts included domains instead of restricting results to them.                                  |
| `search_lang`, `filter_by_language`                               | Boost results in a language, or strictly filter by it.                                                                                                                                  |
| `country`                                                         | Full English country name or a common Brave country code such as `US`, `GB`, `IT`. Only supported with `topic: general`. `ui_lang` is Brave-only; Tavily reports a warning if supplied. |
| `include_answer`                                                  | `false` (default), `true`/`basic`, or `advanced`: optionally request Tavily's generated summary. The agent must still check and cite the sources.                                       |
| `include_raw_content`                                             | `false` (default), `true`/`markdown`, or `text`: request page content alongside snippets.                                                                                               |
| `chunks_per_source`                                               | 1–3 snippets per source, default 3; omitted for ultra-fast search.                                                                                                                      |
| `include_images`, `include_image_descriptions`, `include_favicon` | Optional image and icon metadata. Image descriptions require images enabled.                                                                                                            |
| `exact_match`                                                     | Require exact quoted phrases in results.                                                                                                                                                |
| `max_age_hours`                                                   | Maximum age of cached content: `-1` cache only, `0` live, or 1–720 hours.                                                                                                               |
| `auto_parameters`                                                 | Off by default. Tavily may select remaining parameters; explicit `search_depth` still controls search depth.                                                                            |
| `timeout`                                                         | 1–120 seconds, default 60.                                                                                                                                                              |

For example, ask MnemoAI to “Use advanced search for recent Python release notes,
restricted to python.org, and cite the sources.” Search settings are tool
arguments the model chooses for each request. Basic search is the default;
more expensive modes are available when useful.

Results retain relevance scores, publication dates when supplied, optional
content, request ID and credit usage. Returned page/answer text has a shared
100,000-character cap with explicit truncation markers; source URLs remain intact.

### Tavily Extract, Crawl and Map

These additional tools are available when Tavily is selected and **both**
`ENABLE_WEB_SEARCH` and `ENABLE_WEB_CRAWL` are enabled:

| Tool             | Use                                                                                                                          | Defaults and limits                                                                                                                                  |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tavily_extract` | Read 1–20 known URLs. Optional query reranks passages; basic/advanced extraction; Markdown/text; images and favicons.        | Basic extraction, 30-second timeout (1–60). With a query, select 1–5 chunks per source. Partial successes retain `failed_results`.                   |
| `tavily_crawl`   | Explore a website and extract multiple relevant pages. Supports natural-language instructions and regex path/domain filters. | Depth 1, breadth 20, total 10 pages. MnemoAI permits depth 1–5, breadth/limit 1–100, timeout 10–150 seconds. External links are disabled by default. |
| `tavily_map`     | Discover relevant website URLs before selectively reading them. Supports instructions and the same regex filters.            | Depth 1, breadth 20, total 20 URLs. Same traversal/time limits as Crawl; external links disabled by default.                                         |

Crawl also supports advanced extraction, Markdown/text, images, favicons and
1–5 chunks per source when instructions are supplied. Crawl and extraction
costs depend on pages and extraction depth; keep limits appropriate to the task.
Responses include credit usage. Page text is capped as described above.

Supplied URLs pass the existing public-HTTP(S) preflight policy before being
sent to Tavily. Tavily performs remote fetching; its tools do not write local
files or automatically ingest results into RAG. The existing `web_crawler`
remains available with its local browser and RAG behavior.

Implementation: `WebSearchController` selects an adapter implementing
`WebSearchProvider`. Brave and Tavily own their SDK calls and concrete tool
schemas; only the selected SDK is lazily imported on first use. Another
provider can implement the same contract and register in the controller and
configuration registries.

See the [Tavily Python SDK reference](https://docs.tavily.com/sdk/python/reference)
for upstream API semantics and [pricing](https://docs.tavily.com/documentation/api-credits).

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
> `~/.local/share/uv/tools/mnemoai-assistant/bin/python -m playwright install chromium`).

## RAG (Retrieval-Augmented Generation)

The RAG system automatically indexes documents for semantic search with **hybrid search** (semantic embeddings + BM25 keyword scoring).

During an embedding outage, existing documents remain searchable with
**BM25-only keyword ranking**, explicitly labeled in results. New ingestion fails
without adding fabricated vectors. Episodic recall and storage follow the same
read/write split. The [embedding configuration](../configuration/embeddings.md#embeddings-configuration)
describes legacy config compatibility and recovery backups for old synthetic vectors.

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
hybrid search weights — see [RAG Configuration](../configuration/retrieval.md#rag-configuration).
