# Memory & learning

## User Profile Learning

After 5+ interactions, the assistant builds a profile:

- **Cognitive style**: Analytical, creative, pragmatic, systematic
- **Domain expertise**: Python, AWS, DevOps, ML, etc.
- **Learning style**: Visual, hands-on, theoretical
- **Communication patterns**: Tone, complexity, question styles
- **Code preferences**: Testing, documentation, type hints

Profile is automatically injected into system prompt for personalization.

## 🧠 Persistent Memory (MEMORY.md)

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

## 🧭 Steering (STEERING.md)

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

## Episodic Memory

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
cleanup limits) see [Episodic Memory Configuration](../configuration.md#episodic-memory-configuration).

## ACE Playbook (Agentic Context Engineering)

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
