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

## Embedding provenance

Persistent document indexes bind vectors to the embedding model/endpoint
fingerprint, not just their dimension. A changed or unknown identity disables
semantic operations and new writes; stored text remains available for explicitly
labelled BM25 keyword search. Unstamped legacy episodic vectors are likewise not
silently adopted. Rebuild a document index by explicitly clearing and re-ingesting
it; no automatic migration fabricates vectors or guesses the old model.

FAISS document writes stage the index and journal the corresponding metadata
before publication. A failed write is reported, not counted as successful
ingestion; an interrupted publication can be recovered when the store next
opens. Clearing documents is serialized with ingestion and search.

Curated `MEMORY.md` add/replace/remove operations use a cross-process transaction
lock so simultaneous agents do not silently overwrite each other's additions.
Optional MEMORY.md auto-extraction is separate from local playbook learning:
its model calls are included in `/usage`.

## 🧭 Steering (STEERING.md)

`STEERING.md` is where **you** write always-on instructions for the assistant — conventions, commands, and "always do X" rules it should follow every turn. It's the user-authored counterpart to `MEMORY.md`: the assistant maintains `MEMORY.md` itself, but never writes `STEERING.md` — that file is yours.

**Where it lives (two levels, both optional):**

- **Global:** `~/.mnemoai/STEERING.md` — applies in every session, everywhere.
- **Project:** `./STEERING.md` — discovered by walking up from your working directory to the repository root (the first ancestor containing `.git`). Put project-specific conventions here and check it into the repo so your whole team shares them.

When both exist they're **combined**, global first then project (so project instructions take precedence by appearing last). Nothing above the repo root is picked up. If a directory isn't in a repository at all, the search stops just below your home directory rather than continuing to `/` — so a stray instructions file in your home folder never becomes always-on for everything. (A repository checked out directly at your home directory — a dotfiles repo — is a deliberate choice and still counts as a project root.)

**`CLAUDE.md` works too.** At every level the file may be named either `STEERING.md` or `CLAUDE.md`, so a repo that already keeps its agent instructions under that name is picked up with no extra file and nothing to configure. The two names are read identically — only the precedence differs:

- **Within one directory, `STEERING.md` wins** and a `CLAUDE.md` sitting beside it is ignored. That's what lets a project keep both: shared instructions in `CLAUDE.md`, and the parts meant for this assistant in `STEERING.md`, without the two being concatenated or contradicting each other.
- **The choice is per directory, not global.** A global `CLAUDE.md` still applies alongside a project `STEERING.md`, and a subdirectory's `CLAUDE.md` still applies if it has no `STEERING.md` beside it. Finding one name in one directory never switches the other name off elsewhere.

Only the app home is searched for the global file — `~/.mnemoai/STEERING.md`, else `~/.mnemoai/CLAUDE.md`. Files belonging to other tools outside that directory are never read.

**How it's applied:**

- The content is **prepended to every prompt** as an authoritative instruction block, framed so it overrides default behavior.
- Instructions are loaded into a **session snapshot** at startup. Before a new
  prompt or command, MnemoAI checks for changed sources, references, or limits and
  refreshes the snapshot when needed. Edits apply on the next interaction without
  a restart; unchanged files are not repeatedly read.
- It is **never summarized by compaction**: unlike the conversation, the steering block is re-injected verbatim each turn, so long sessions never dilute or lose your instructions.

The request, footer estimate, and context breakdown use the same snapshot.
Reading or measuring it never prints warnings. Truncation and read problems get
one compact grey notice per condition, with details in `/context` and `/doctor`.
The notice is shown again only if the condition changes or returns after recovery.

**What to put in it:** build/test commands, code-style rules, project layout notes, a commit-message format, "prefer X over Y" preferences — anything you'd tell a new collaborator. Keep it focused (a couple hundred lines at most); it's in context every turn, so brevity helps adherence.

**Pull in another file with `@path`.** Rather than one long file, reference others and they're injected alongside it:

```markdown
Follow the conventions in @docs/style-guide.md.
Release steps live in @~/.mnemoai/release-checklist.md.
```

A path is resolved relative to the file that mentions it (so a project file means its own neighbours, wherever you launched the app); `~` and absolute paths work too. The reference stays in the sentence and the referenced file is appended under its own `Contents of …` header, so the assistant knows exactly what it's reading. Details worth knowing:

- **Only real files are inlined.** A reference to something missing — or to a directory — is left as plain text, which is also why an `@staticmethod` in a code example or an `@handle` in prose is harmless.
- **Chains are followed** (a referenced file may reference more), up to three levels deep and 20 files per turn, and each file is injected once no matter how often it's named — a cycle is safe.
- **References share the referencing file's `STEERING.MAX_CHARS` budget**, so splitting a large ruleset across files doesn't sidestep the cap. Anything dropped is reported in the block rather than silently omitted.

**Let the assistant write it for you.** You don't have to author `STEERING.md` by hand. A bundled **`steering-creator`** skill ships out of the box: ask the assistant to _"create a STEERING.md for this project"_ (or _"document how to work in this repo"_) and it investigates the codebase — README, build/test config, layout — and writes a well-formed file following best practices (specific rules, scannable structure, only durable always-on facts). Ask it to _"improve my STEERING.md"_ and it refines the existing one.

There's no config toggle: the file's presence is the switch. If no instruction file exists at either level, nothing is injected. It's distinct from `MEMORY.md` (facts the agent learns), skills (on-demand procedures), and the base system prompt.

The snapshot avoids repeated file reads, not context usage: instructions still
accompany model requests. Each file's contribution is capped at
`STEERING.MAX_CHARS`. When the setting is absent, the code default is **45000
characters per file**; `0` disables the cap. Configuration examples may choose
a different explicit limit, which takes precedence over the fallback:

```yaml
STEERING:
  MAX_CHARS: 45000 # Per file; content past this is dropped with a visible note
```

The injected block still tells the model what was truncated and names the file,
so it can read the rest with its file tools. No instruction file is modified.
`/context` and `/doctor` show how much source text was loaded. A compact,
always-on ruleset with longer reference material read on demand can reduce
context use; increasing the cap is a separate choice from suppressing duplicate
terminal notices.

**Storage Location:** `~/.mnemoai/STEERING.md` (global) and/or `./STEERING.md` (per project) — `CLAUDE.md` accepted at either level

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

The playbook records scoped tool observations locally and preserves existing
model-inferred and legacy notes. These are historical evidence, not proven rules
or permissions. Local learning is enabled by default and makes **no extra LLM
or embedding calls**. There is no model-reflection option.

**How it works:**

1. **Local observation:** after a tool-using turn, record up to three failures,
   later successes on the same tool/target, or successful check commands.
   Ordinary successful reads do not generate generic advice. Blocked/declined
   calls and calls without results are excluded. No model is built or called.
   Credential-like fields are redacted; file bodies and arbitrary stdout are not
   stored as advice. Redaction is not a comprehensive secret detector.

2. **Playbook Store**: Maintains structured strategy entries:

   ```json
   {
     "context": "editing python files",
     "strategy": "Observed file_edit failed (string_not_found). A later edit of the same target succeeded. Historical outcome, not proof of cause.",
     "id": "mem-example",
     "revision": 1,
     "status": "active",
     "provenance": "observed",
     "scope": "/path/to/project",
     "source_refs": [
       { "session_id": "session-id", "turn": 4, "tool_call_id": "call-id" }
     ],
     "outcome": "failure",
     "tools": ["file_edit"],
     "confidence": 0.5
   }
   ```

3. **Context injection:** ranks eligible notes against the current task, before
   using confidence/recency as small tie-breakers. High-confidence generic legacy
   notes cannot displace more relevant newer observations. Notes refresh each
   turn and after `/learned` changes. New observations apply to the directory
   where they were recorded; legacy scope is not invented.

   ```
   [Tool-use notes from past sessions]
   Noted after past errors:
     · [editing files]: Read the current target before retrying an edit
   ```

4. **Preservation:** old files are backed up before adding record metadata.
   Repeated observations retain source references without increasing confidence.
   Capacity refinement archives entries instead of deleting them. Edit history
   stays attached to the same ID, with revision checks and cross-process locking.

### Inspect and control learning

The feature is not YAML-only:

| Change                                                   | Interactive path               | When it applies                                  |
| -------------------------------------------------------- | ------------------------------ | ------------------------------------------------ |
| Initial setup                                            | `/config`                      | Full configuration restart                       |
| Enable/disable learning                                  | `/features`                    | In place when only local learning changes |
| Enable/disable, capacity and candidate count             | `/config playbook`             | In place; no model rebuild or conversation reset |
| Inspect/edit/disable notes and give feedback             | `/learned`                     | Next memory injection; revisions persist         |
| Suppress injection for this session                      | `/learned off` / `/learned on` | Immediate; local recording still runs |

`/config playbook` changes only `ENABLE_PLAYBOOK` and `PLAYBOOK`, preserves other configuration, and
supports cancelling or stepping back without a partial save. Capacity changes
take effect at the next refinement. The candidate-count setting is still subject
to the two-notes-per-outcome injection cap. Deduplication is local and exact;
no embedding service is used. YAML remains an alternative, not a requirement.

`/learned` shows the current model's store path, injection state, entries, and
any local storage/learning error. Optional-learning failures stay in logs and
on-demand diagnostics, not routine yellow chat warnings. These commands are user-only:

- `/learned inspect <id>`: evidence references, scope, counters, and edit history.
- `/learned edit <id>`: edit `context` and `strategy` as JSON in `$EDITOR`, then
  confirm the change.
- `/learned disable <id>` / `/learned restore <id>`: control future injection
  without erasing the record.
- `/learned helpful <id>` / `/learned unhelpful <id>`: explicit feedback.
- `/learned off` / `/learned on`: suppress/enable injection for this session.
  This does not stop extraction; use `/features` to disable the whole playbook.
- `/learned clear`: confirm removal of all entries, retaining a backup beside
  `playbook.json`.

An unambiguous ID prefix works. Lists show at most 30 entries; inspect accepts
any ID. Disabled and archived entries remain inspectable. Negative feedback
can make an active entry dormant; restoring its status does not reset feedback.

`injection_count` counts turns supplied with the note. Observed tool outcomes
are associations, not proof the note helped or the user's task succeeded.
Only explicit helpful/unhelpful feedback changes the feedback-based confidence;
below 0.2 the note is not injected. Legacy scores are retained, with provenance
labeled `legacy`. No automatic causal usefulness claim is made.

### Retract an incorrect note

Disabling a note is a preference about injection. Retraction records a reason
for withdrawing it and blocks reuse of its known evidence for automatic learning:

```text
/learned preview mem-<id>
/learned retract mem-<id> That failure was temporary, not a permanent restriction.
/learned retracted
/learned restore mem-<id> New verification shows that the withdrawal was mistaken.
```

Retraction and restoration always show an impact preview and require confirmation.
The plan is checked again before writing: changes to notes or their source
relationships require a fresh preview. Original data is backed up, and the
reason, actor, time, prior state, and revisions remain inspectable.

The first stage has deliberately narrow semantics:

- The selected record stops being injected. Known observation IDs are quarantined
  from automatic local learning in that model/project scope. Filtering happens before
  local extraction and again when committing observations, including concurrent learning.
- Automatic learning cannot re-enable the exact withdrawn record by recording it again.
  Unknown evidence is not treated as independent evidence when the scope has
  an active quarantine. Fresh, independently identified observations can support
  new lessons; this is not a semantic ban on every future paraphrase.
- Other notes sharing evidence are listed for manual review and left unchanged.
  Sharing a tool result is **not** proof that one conclusion depends on another.
  There are no automatic dependency cascades in this stage.
- Restoring a retraction requires a reason and returns the note to its prior state
  (which may have been disabled or archived). Another retracted note may keep
  shared evidence quarantined; the preview shows that. Ordinary disable/restore
  commands continue to work as before on notes without an active retraction.
- Legacy or incomplete provenance is reported explicitly. Retraction can withdraw
  the record, but cannot reliably trace unknown sources. It does not infer links
  from similar wording, transcript text read under new observation IDs, or age.

Current generated playbook blocks are refreshed at the model-send boundary for
foreground and delegated work. User messages, historical summaries, old tool
results, and already-sent requests are not rewritten or recalled. A new process
or a restored session observes the same persisted retractions. No tool permission
is granted or revoked, and the model has no retraction tool it can approve itself.

`/learned clear` removes all records, including retractions, after a warning and
with a backup. That intentionally removes their learning quarantines too; original
transcripts and other memory subsystems are not erased. Retraction is not a
privacy-purge command. Before downgrading to a version without retraction support,
reverse active retractions or restore an appropriate pre-retraction backup;
older versions may refuse the new lifecycle state rather than silently trust it.

Source references may outlive session transcripts; their IDs are not a promise
of perpetual replay. Use `/config playbook` and `/features` as before:
retraction adds no configuration toggle and makes no extra model call.

**Key Differences from Episodic Memory:**

| Feature     | Episodic Memory       | ACE Playbook            |
| ----------- | --------------------- | ----------------------- |
| Stores      | Full task completions | Tool observations and retained notes |
| Learns from | Successes only        | Successes AND failures  |
| Format      | Conversation context  | Structured, evidence-linked records |
| Retrieval   | Semantic similarity   | Context + tool matching |

**Configuration:**

```yaml
ENABLE_PLAYBOOK: true
PLAYBOOK:
  MAX_ENTRIES: 500 # Active entries before archival/refinement
  MAX_INJECT: 10 # Maximum entries to inject per query
```

Set `ENABLE_PLAYBOOK: false` to disable local learning and playbook injection.
An absent flag defaults to `true`; explicit user choices are preserved.
Old `AREA_MODELS.REFLECTOR`, `PLAYBOOK.REFLECTION_TIMEOUT` and
`PLAYBOOK.SIMILARITY_THRESHOLD` keys are ignored and may be removed. They cannot
enable model reflection. Existing notes and their feedback/retractions remain
available; this change does not clear them. Other features (chat, optional peer
review, compaction, episodic embeddings, and MEMORY.md auto-extraction) retain
their own independent model usage and controls.

**Storage Location:**

- Strategies: `~/.mnemoai/{profile}/models/{model}/playbook/playbook.json`
- Metrics: `~/.mnemoai/{profile}/models/{model}/playbook/metrics.json`
