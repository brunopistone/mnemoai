# Advanced Features

## 📚 Advanced Features

### Query Routing

When enabled, the assistant classifies each query before processing it and routes it to a specialized tool subset. This reduces noise for the model and improves response quality.

**Categories:**

| Route       | Description                                 | Tools Available                                      |
| ----------- | ------------------------------------------- | ---------------------------------------------------- |
| `simple_qa` | Greetings, explanations, general knowledge  | None (direct LLM answer)                             |
| `code`      | File ops, code editing, git, shell commands | fs_read, fs_write, file_edit, bash, git, search, etc |
| `research`  | Web search, URL fetching                    | web_search, web_crawler                              |
| `knowledge` | Document reading, indexing, RAG queries     | pdf/csv/docx/json readers, RAG tools, fs_read        |
| `full`      | Multi-category or ambiguous tasks           | All tools (fallback)                                 |

**How it works:**

1. A lightweight LLM call classifies the query into one of the categories above
2. The agent node binds only the tools for that category
3. If a query spans multiple categories, it routes to `full` (all tools)
4. The classifier prompt is customizable via `ROUTING_PROMPT` in `config.yaml`

**Configuration:**

```yaml
ENABLE_ROUTING: true
ROUTING_PROMPT: |
  # Custom classifier prompt (optional, has a sensible default)
  ...
```

### Orchestrator-Workers

When enabled alongside routing, tasks classified as `full` (spanning multiple categories) are automatically decomposed into focused subtasks executed by specialized workers.

**How it works:**

1. **Orchestrator**: An LLM call decomposes the complex query into ordered subtasks, each assigned a category (code, research, knowledge, etc.)
2. **Workers**: Each subtask is executed by a worker agent with only the tools for its category. Workers run sequentially — each receives context from previously completed subtasks.
3. **Aggregator**: If there were multiple subtasks, a final LLM call synthesizes all worker results into a single coherent response.

**Example flow for "Read this PDF and write a summary to a file":**

```
Orchestrator decomposes into:
  [Step 1/2: Read and summarize the PDF document]        → knowledge worker
  [Step 2/2: Write the summary to summary.md]            → code worker
  [Synthesizing results...]                               → aggregator
```

**Configuration:**

```yaml
ENABLE_ROUTING: true # Required
ENABLE_ORCHESTRATION: true # Activates orchestrator for 'full' route
# ORCHESTRATOR_PROMPT: |      # Optional: customize decomposition prompt
# AGGREGATOR_PROMPT: |        # Optional: customize synthesis prompt
```

**When orchestration is disabled**, `full` routes use all tools in a single agent loop (the previous behavior). No regression.

### Web Search Configuration

This tool uses the Brave Search API. Obtain an API key from [Brave Search Developer Portal](https://brave.com/search/api/).

```yaml
BRAVE_API_KEY: your-api-key-here # For web search
```

### Web Crawler Configuration

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

### External MCP Servers

mnemoai always runs its own built-in MCP server (file ops, bash, git, web, RAG,
vision, planning). You can add **more** MCP servers by creating
`~/.mnemoai/mcp/mcp.json` with the standard `mcpServers` schema (an
`mcp.json.example` is seeded there on first run). Their tools are merged with the
built-in ones and made available to the agent.

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": { "BRAVE_API_KEY": "your_brave_api_key" }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
      "disabled": true
    }
  }
}
```

Per-server fields: `command` (required), `args` (optional list), `env`
(optional; merged over the process environment), and `disabled` (optional;
`true` skips the server). A template ships at
`~/.mnemoai/mcp/mcp.json.example` (seeded on first run from the bundled
`src/mnemoai/utils/mcp.json.example`).

Behavior:

- **Additive** — the built-in server is always on; external servers run
  alongside it. Tools from all servers are merged into one list.
- **Resilient** — if an external server fails to start (bad command, missing
  binary, crash), it's logged in red and skipped; the app still runs with the
  built-in server and any others that connected.
- **No shadowing** — if an external tool's name collides with a built-in one,
  the external tool is exposed as `servername__tool` so core tools are never
  overridden (the server is still called with the original tool name).
- **Works with routing & orchestration** — external tools are appended to every
  non-empty query route, and when orchestration is enabled the task decomposer
  is told which external tools exist and steers subtasks that need them to the
  `full` category (which binds every tool). So external tools stay reachable
  whether routing/orchestration is on or off.
- Run **`/mcp`** in the chat to see configured servers, status, and tool counts.

### RAG (Retrieval-Augmented Generation)

The RAG system automatically indexes documents for semantic search with **hybrid search** (semantic embeddings + BM25 keyword scoring).

**How it works:**

1. Read a PDF/DOCX file → Automatically chunked and indexed
2. Ask questions → Assistant searches indexed documents first using hybrid search
3. Session-scoped → Cleared on `/clear` or exit

**RAG Tools:**

- `list_documents()`: Show indexed documents
- `search_in_documents(query, top_k)`: Hybrid semantic + BM25 search
- `clear_documents()`: Clear RAG index

**Configuration:**

- `RAG.CHUNK_TOKENS`: Chunk size (recommended: 512-2048)
- `RAG.VECTOR_STORE.TYPE`: Choose between `faiss` or `chromadb`
- `RAG.SEARCH.SEMANTIC_WEIGHT` / `RAG.SEARCH.KEYWORD_WEIGHT`: Configurable hybrid weights
- Recursive chunking with 10% overlap
- Hybrid search: BM25 (Okapi BM25 with TF-IDF, term saturation, length normalization) + semantic similarity
- Independent candidate retrieval from both BM25 and embeddings, merged and re-ranked

**Vector Store Options:**

- **ChromaDB**: Persistent vector database with metadata support (default)
- **FAISS**: Fast in-memory search with disk persistence

The system uses a **VectorStoreController** for easy switching between stores. All functionality (indexing, searching, clearing) works identically regardless of the chosen store.

### User Profile Learning

After 5+ interactions, the assistant builds a profile:

- **Cognitive style**: Analytical, creative, pragmatic, systematic
- **Domain expertise**: Python, AWS, DevOps, ML, etc.
- **Learning style**: Visual, hands-on, theoretical
- **Communication patterns**: Tone, complexity, question styles
- **Code preferences**: Testing, documentation, type hints

Profile is automatically injected into system prompt for personalization.

### 🧠 Persistent Memory (MEMORY.md)

A small, agent-curated markdown file the assistant maintains itself to remember durable facts across sessions — user/environment details, conventions, lessons learned, tool quirks, and completed work. It lives at `~/.mnemoai/{profile}/MEMORY.md` (profile-scoped, **shared across models**, unlike episodic memory and the playbook).

**How it works:**

1. **Always injected**: The entire file is loaded into the system prompt at the start of every session — a "frozen snapshot". Writes made during a session take effect on the **next** session, not the current one.
2. **Agent-curated**: The assistant edits its own memory via the `memory` MCP tool (`add` / `replace` / `remove` actions over a `§`-delimited entry list), deciding what is worth remembering.
3. **Bounded**: A hard character cap (`MEMORY.MAX_CHARS`, default 2200) forces the agent to **consolidate** — merging or removing stale entries instead of growing unbounded.

**How it differs from Episodic Memory:** persistent memory is a curated set of facts that is **always** in context, whereas episodic memory is a store of past task completions **retrieved by similarity** per query. The two complement each other (and the ACE Playbook, which stores tool strategies).

**Command:** Run `/memory` to view the current memory, or `/memory clear` to wipe it (with a y/N confirm).

**Configuration:**

```yaml
ENABLE_MEMORY: true # Master toggle for the memory tool + injection
REQUIRE_MEMORY_CONFIRMATION: false # Auto-saves like Hermes; set true to require y/N before each memory write
MEMORY:
  MAX_CHARS: 2200 # Hard cap — forces consolidation when exceeded
```

`REQUIRE_MEMORY_CONFIRMATION` defaults to `false` (the agent auto-saves). Set it to `true` to gate each memory write behind a y/N prompt, reusing the same client-side confirmation gate as bash/file writes.

**Storage Location:** `~/.mnemoai/{profile}/MEMORY.md`

### Episodic Memory

The episodic memory system learns from successful task completions and retrieves similar solutions for future queries.

**How it works:**

1. **Automatic Storage**: After each successful interaction, stores:
   - Initial user query
   - Full conversation context
   - Tools used with arguments
   - Final solution
   - Timestamp

2. **Hybrid Search**: Retrieves similar episodes using:
   - 70% semantic similarity (task intent)
   - 30% BM25 keyword scoring (tool names, action verbs)

3. **Context Injection**: Before processing queries, injects compact context:

   ```
   [Episodic Memory - Similar Past Tasks]
   1. "read DOCX about ML" → fs_read → success (similarity: 0.85)
   2. "analyze PDF report" → fs_read, web_search → success (similarity: 0.78)
   ```

4. **Automatic Cleanup**: Maintains bounded memory:
   - Max 1000 episodes
   - Removes entries older than 90 days
   - Runs on startup

**Success Detection:**

- User feedback: "thanks", "perfect", "great", "worked"
- No error markers in response
- All tools executed successfully
- Filters out greetings and simple acknowledgments (<300 chars, no tools)

**Storage Location:**

- FAISS: `~/.mnemoai/{profile}/models/{model}/episodic_memory/episodic.index`
- ChromaDB: `~/.mnemoai/{profile}/models/{model}/episodic_memory/`

**Configuration:**

```yaml
ENABLE_EPISODIC_MEMORY: true
EPISODIC_MEMORY:
  STORE_TYPE: chromadb # or faiss
RAG:
  EMBED_MODEL_ID: # Required for both stores
    NAME: mxbai-embed-large
    TYPE: ollama
```

### ACE Playbook (Agentic Context Engineering)

The ACE Playbook learns strategies from both successes AND failures, implementing the Agentic Context Engineering framework for continuous improvement.

**How it works:**

1. **Reflector**: After each interaction, analyzes tool executions:
   - Detects failure patterns (file not found, string not found, permission denied, etc.)
   - Identifies successful strategies for specific tools (file_edit, execute_bash)
   - Extracts specific, actionable insights (not generic summaries)
   - Tracks metrics (success/failure rates, failure types) in `metrics.json`

2. **Playbook Store**: Maintains structured strategy entries:

   ```json
   {
     "context": "editing python files",
     "strategy": "Read the file first to get exact string including whitespace before using str_replace",
     "source": "Failed file_edit on 2026-02-01: string_not_found",
     "outcome": "failure",
     "tools": ["file_edit"],
     "confidence": 0.9
   }
   ```

3. **Context Injection**: Injects relevant strategies into the system prompt at startup:

   ```
   [Playbook - Learned Strategies]
   Avoid these patterns:
     ✗ [editing files]: Read the file first to get exact string before str_replace
   Effective strategies:
     ✓ [searching files]: Use glob_search instead of find for better performance
   ```

4. **Lazy Refinement**: Only deduplicates when hitting token limits, using semantic similarity if embeddings are configured.

**What gets stored:**

- **Failures**: Specific patterns like `string_not_found`, `file_not_found`, `permission_denied`, `command_failed`, etc.
- **Successes**: Only for tools with reusable patterns (file_edit, execute_bash with specific commands)
- **Not stored**: Generic successes without actionable strategies

**Key Differences from Episodic Memory:**

| Feature     | Episodic Memory       | ACE Playbook            |
| ----------- | --------------------- | ----------------------- |
| Stores      | Full task completions | Granular strategies     |
| Learns from | Successes only        | Successes AND failures  |
| Format      | Conversation context  | Structured rules        |
| Retrieval   | Semantic similarity   | Context + tool matching |

**Configuration:**

```yaml
ENABLE_PLAYBOOK: true
PLAYBOOK:
  MAX_ENTRIES: 500 # Maximum entries before refinement
  SIMILARITY_THRESHOLD: 0.85 # Threshold for merging similar strategies
  MAX_INJECT: 10 # Maximum entries to inject per query
```

**Storage Location:**

- Strategies: `~/.mnemoai/{profile}/models/{model}/playbook/playbook.json`
- Metrics: `~/.mnemoai/{profile}/models/{model}/playbook/metrics.json`
