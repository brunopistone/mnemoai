# Advanced Features

## 📚 Advanced Features

### Query Routing

When enabled, the assistant classifies each query before processing it and routes it to a specialized tool subset. This reduces noise for the model and improves response quality.

**Categories:**

| Route       | Description                                 | Tools Available                                              |
| ----------- | ------------------------------------------- | ------------------------------------------------------------ |
| `simple_qa` | Greetings, explanations, general knowledge  | None (direct LLM answer)                                     |
| `code`      | File ops, code editing, git, shell commands | fs_read, fs_write, file_edit, execute_bash, git, search, etc |
| `research`  | Web search, URL fetching                    | web_search, web_crawler                                      |
| `knowledge` | Document reading, indexing, RAG queries     | pdf/csv/docx/json readers, RAG tools, fs_read                |
| `full`      | Multi-category or ambiguous tasks           | All tools (fallback)                                         |

**How it works:**

1. A lightweight LLM call classifies the query into one of the categories above
2. The agent node binds only the tools for that category
3. If a query spans multiple categories, it routes to `full` (all tools)
4. The classifier prompt is customizable via `ROUTING_PROMPT` in `prompts.yaml`
   (all prompts live there, separate from `config.yaml`)

**Configuration:**

```yaml
# config.yaml — toggle
ENABLE_ROUTING: true

# prompts.yaml — the classifier prompt
ROUTING_PROMPT: |
  # Custom classifier prompt (optional, has a sensible default)
  ...
```

### Orchestrator-Workers

When enabled alongside routing, tasks classified as `full` (spanning multiple categories) are automatically decomposed into focused subtasks executed by specialized workers.

**How it works:**

1. **Orchestrator**: An LLM call decomposes the complex query into subtasks, each assigned a category (code, research, knowledge, etc.) and an optional `depends_on` (which earlier subtasks it needs).
2. **Workers**: Each subtask runs with only the tools for its category. Subtasks are **scheduled by their dependencies** — independent subtasks run **in parallel** (bounded by `SUBAGENT_MAX_CONCURRENCY`), while a subtask that declared a `depends_on` waits for exactly those results (threaded into its prompt). A purely linear plan still runs in order.
3. **Aggregator**: If there were multiple subtasks, a final LLM call synthesizes all worker results into a single coherent response.

**Example flow for "Read this PDF and write a summary to a file" (a dependency chain):**

```
Orchestrator decomposes into:
  [Step 1/2: Read and summarize the PDF document]        → knowledge worker
  [Step 2/2: Write the summary to summary.md]            → code worker (depends_on step 1)
  [Synthesizing results...]                               → aggregator
```

**Example flow for "Investigate X and Y independently" (parallel):**

```
Orchestrator decomposes into (no dependencies between them):
  [Running 2 steps in parallel]
  [Step 1/2: Investigate X]   → worker  ┐ run
  [Step 2/2: Investigate Y]   → worker  ┘ together
  [Synthesizing results...]                               → aggregator
```

**Configuration:**

```yaml
# config.yaml — toggles
ENABLE_ROUTING: true # Required
ENABLE_ORCHESTRATION: true # Activates orchestrator for 'full' route

# prompts.yaml — customize the prompts (optional; sensible defaults bundled)
# ORCHESTRATOR_PROMPT: |      # decomposition prompt (teaches the depends_on field)
# AGGREGATOR_PROMPT: |        # synthesis prompt

# LLM — bounds how many independent subtasks run at once (shared with sub-agents)
LLM:
  SUBAGENT_MAX_CONCURRENCY: 4 # 1 = force sequential
```

**When orchestration is disabled**, `full` routes use all tools in a single agent loop (the previous behavior). No regression. Distinct from model-initiated **sub-agents** (`spawn_agent`): the orchestrator is framework-driven (it decomposes complex `full` queries for you), while sub-agents are the model's own on-demand delegation — both now share the same bounded concurrency engine.

### Sub-agents (`spawn_agent`)

Distinct from the orchestrator (which the framework drives), sub-agents are
**model-initiated**: the assistant can call the `spawn_agent` tool to hand a
self-contained task to a fresh sub-agent that runs in its **own isolated
context** and returns **only its final report**. The sub-agent's intermediate
tool calls never enter the main conversation, so the assistant's context stays
clean during search-heavy or multi-step work.

**How it works:**

1. The assistant calls `spawn_agent(agent_type, prompt, description)` with a
   complete, self-contained brief (the sub-agent doesn't see the conversation).
2. The sub-agent runs its own model↔tool loop with its type's tool allowlist and
   system prompt, then returns a concise report.
3. The assistant summarizes that report for you (the raw result isn't shown
   directly).

The sub-agent runs **quietly**: its internal steps (searches, file reads,
reasoning) don't stream to your terminal — you see a compact status line
(`… N sub-agents running`) while it works, and only its final report comes back.
For **independent** investigations the assistant can spawn several sub-agents in
one turn and they run **in parallel** (bounded by `LLM.SUBAGENT_MAX_CONCURRENCY`,
default 4; set it to 1 to force sequential). A destructive tool used inside a
sub-agent still asks for confirmation the same way the main assistant does.

**Background sub-agents.** For a long task you don't want to wait on, the
assistant can run a sub-agent **in the background**: the call returns right away
and you keep working. When it finishes, its report **surfaces automatically** —
if you're idle, the assistant speaks up on its own to deliver it; if you're
mid-conversation, it's folded into your next turn. Because a background sub-agent
has no terminal to prompt on, it **automatically skips any destructive tool that
isn't already approved** — so background work is safe by default; use it for
read-only investigation or when the needed actions were pre-approved. The
assistant can also **resume** a finished sub-agent with a follow-up ("now also
check the tests"), which continues it with its prior work as context — this
works even after restarting the app or loading a saved conversation, because
each run's brief and report are saved to disk.

**Built-in agent types:**

| Type              | Tools                   | Use for                                                   |
| ----------------- | ----------------------- | --------------------------------------------------------- |
| `general-purpose` | All tools               | Multi-step tasks that may research **and** make changes   |
| `explore`         | Read-only (search/read) | Locating code, tracing how things work, gathering context |
| `plan`            | Read-only (search/read) | Investigating, then returning a step-by-step plan         |

`explore` and `plan` cannot edit files or run commands. Nested spawning is
blocked (a sub-agent can't spawn its own sub-agents). Each type's system prompt
lives in `prompts.yaml` (`SUBAGENT_GENERAL_PROMPT`, `SUBAGENT_EXPLORE_PROMPT`,
`SUBAGENT_PLAN_PROMPT`) and can be customized.

**Custom agent types.** Define your own by dropping a markdown file in
`~/.mnemoai/agents/` (one file per agent, e.g. `reviewer.md`):

```markdown
---
name: reviewer # optional; defaults to the filename (reviewer)
description: Reviews a diff for bugs and style issues. Read-only.
tools: fs_read, grep_search, glob_search # optional allowlist; omit or "*" = all tools
---

You are a meticulous code reviewer. Given a diff or set of files, identify
bugs, edge cases, and style issues, then return a concise prioritized report.
```

- `description` is required (it's what the assistant reads to decide when to
  spawn the type); `name` and `tools` are optional.
- `tools` may be a YAML list or a comma-separated string; omit it (or use `*`)
  for the full toolset. `fs_read`/`describe_image` are always available and
  `spawn_agent` is always removed (no nested spawning), regardless of the list.
- The markdown body is the sub-agent's system prompt.
- A custom type **overrides** a built-in of the same name.
- Bad or incomplete files are skipped (reported, never fatal).

There's no `/agents` command: agents are authored as files and discovered
automatically. All available types (built-in + custom) are listed to the
assistant, which decides when to spawn one — you don't invoke them directly.

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

**Search internals:**

- Recursive chunking with 10% overlap
- Hybrid search: BM25 (Okapi BM25 with TF-IDF, term saturation, length normalization) + semantic similarity
- Independent candidate retrieval from both BM25 and embeddings, merged and re-ranked

For config keys — chunk size, vector store choice (ChromaDB/FAISS), and
hybrid search weights — see [RAG Configuration](configuration.md#rag-configuration).

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
2. **Agent-curated**: The assistant edits its own memory via the `memory` MCP tool (`add` / `replace` / `remove` actions over an entry list separated by a Markdown `---` rule), deciding what is worth remembering.
3. **Tagged by kind**: Each entry is tagged with one of four kinds — `[user]` (who you are), `[feedback]` (how you want it to work, with the why), `[project]` (ongoing work and constraints), `[reference]` (external pointers) — so the memory stays structured and high-signal, and the assistant is guided to skip anything the repo/git already records.
4. **Bounded**: A hard character cap (`MEMORY.MAX_CHARS`, default 2200) forces the agent to **consolidate** — merging or removing stale entries instead of growing unbounded.

**How it differs from Episodic Memory:** persistent memory is a curated set of facts that is **always** in context, whereas episodic memory is a store of past task completions **retrieved by similarity** per query. The two complement each other (and the ACE Playbook, which stores tool strategies).

**Auto-extraction (optional).** By default the assistant only writes to `MEMORY.md` when it decides to call the `memory` tool mid-turn. Enable `ENABLE_MEMORY_AUTO_EXTRACTION` to also run a **background pass after every turn** that distills durable facts from the exchange and writes them automatically — the auto-learning counterpart to the tool. It's **off by default** because, unlike the tool, it writes **without** a confirmation prompt (it can only add/consolidate entries in `MEMORY.md`, nothing else), and it costs one extra background model call per turn. The extraction runs on a daemon thread, so it never blocks your turn.

**Command:** Run `/memory` to view the current memory, or `/memory clear` to wipe it (with a y/N confirm).

**Configuration:**

```yaml
ENABLE_MEMORY: true # Master toggle for the memory tool + injection
REQUIRE_MEMORY_CONFIRMATION: false # Auto-saves; set true to require y/N before each memory write
ENABLE_MEMORY_AUTO_EXTRACTION: false # Background turn-end auto-save (writes without a prompt); off by default
MEMORY:
  MAX_CHARS: 2200 # Hard cap — forces consolidation when exceeded
```

`REQUIRE_MEMORY_CONFIRMATION` defaults to `false` (the agent auto-saves when it calls the tool). Set it to `true` to gate each tool-driven memory write behind a y/N prompt, reusing the same client-side confirmation gate as bash/file writes. `ENABLE_MEMORY_AUTO_EXTRACTION` is independent: it governs the background turn-end distiller, which writes directly (no prompt) and only into `MEMORY.md`.

**Storage Location:** `~/.mnemoai/{profile}/MEMORY.md`

### 🧭 Steering (STEERING.md)

`STEERING.md` is where **you** write always-on instructions for the assistant — conventions, commands, and "always do X" rules it should follow every turn. It's the user-authored counterpart to `MEMORY.md`: the assistant maintains `MEMORY.md` itself, but never writes `STEERING.md` — that file is yours.

**Where it lives (two levels, both optional):**

- **Global:** `~/.mnemoai/STEERING.md` — applies in every session, everywhere.
- **Project:** `./STEERING.md` — discovered by walking up from your working directory to the repository root (the first ancestor containing `.git`). Put project-specific conventions here and check it into the repo so your whole team shares them.

When both exist they're **combined**, global first then project (so project instructions take precedence by appearing last). Nothing above the repo root is picked up.

**How it's applied:**

- The content is **prepended to every prompt** as an authoritative instruction block, framed so it overrides default behavior.
- It's **re-read from disk on every turn**, so editing `STEERING.md` takes effect immediately — no restart needed.
- It is **never summarized by compaction**: unlike the conversation, the steering block is re-injected verbatim each turn, so long sessions never dilute or lose your instructions.

**What to put in it:** build/test commands, code-style rules, project layout notes, a commit-message format, "prefer X over Y" preferences — anything you'd tell a new collaborator. Keep it focused (a couple hundred lines at most); it's in context every turn, so brevity helps adherence.

**Let the assistant write it for you.** You don't have to author `STEERING.md` by hand. A bundled **`steering-creator`** skill ships out of the box: ask the assistant to _"create a STEERING.md for this project"_ (or _"document how to work in this repo"_) and it investigates the codebase — README, build/test config, layout — and writes a well-formed file following best practices (specific rules, scannable structure, only durable always-on facts). Ask it to _"improve my STEERING.md"_ and it refines the existing one.

There's no config toggle: the file's presence is the switch. If neither `STEERING.md` exists, nothing is injected. It's distinct from `MEMORY.md` (facts the agent learns), skills (on-demand procedures), and the base system prompt.

**Storage Location:** `~/.mnemoai/STEERING.md` (global) and/or `./STEERING.md` (per project)

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

For the full config reference (all thresholds, hybrid search weights, and
cleanup limits) see [Episodic Memory Configuration](configuration.md#episodic-memory-configuration).

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
