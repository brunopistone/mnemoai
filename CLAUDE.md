# CLAUDE.md

## Project Overview

Local agentic AI assistant built on LangGraph + MCP (Model Context Protocol). The client spawns an MCP server as a subprocess, routes queries through a StateGraph (classify → orchestrate/call_model ↔ execute_tools), and persists episodic memory, learned strategies (ACE Playbook), and user profiles across sessions. Supports Ollama, AWS Bedrock, OpenAI, Anthropic, SageMaker, and LiteLLM as LLM providers.

## Quick Commands

```bash
# Run from a checkout (src layout: package under src/, run as a module)
PYTHONPATH=src python -m mnemoai            # verbose (shows thinking)
PYTHONPATH=src python -m mnemoai --no-verbose

# Or install, then use the console command
uv tool install .        # or: pip install -e .
mnemoai

# Install dependencies (checkout dev)
pip install -r requirements.txt

# System-wide access (symlink once)
chmod +x bash/system-command-app/mnemoai-wrapper.sh
ln -sf $(pwd)/bash/system-command-app/mnemoai-wrapper.sh /usr/local/bin/mnemoai
```

## Architecture

```
main.py → ChatInterface → LangGraphClient.query()
  → inject episodic memory context
  → LangGraphAgent (StateGraph):
      classifier → [route] → call_model ↔ execute_tools (MCP)
                → [full]  → orchestrator → worker loops → aggregator
  → AgentConversationManager (summarize if over token limit)
  → UserProfileManager (learn preferences)
  → Reflector + PlaybookStore (learn from tool successes/failures)
```

**Client-server split:** The MCP server (`server/server.py`) runs as a stdio subprocess. The client maintains a persistent connection via a background asyncio thread in `client/mcp_tool_wrapper.py`. All tool calls route through MCP protocol.

## Directory Structure

**src layout:** the single package `mnemoai` lives under `src/`.
Paths below are relative to `src/mnemoai/` (e.g. `client/` is
`src/mnemoai/client/`). `main.py` is the package entry (`cli()`),
also runnable as `python -m mnemoai`. `tests/`, `docs/`, `bash/`
stay at the repo root.

| Directory               | Role                                                                                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `client/`               | LangGraphClient facade, MCP bridge                                                                                                                                 |
| `client/agent/`         | Agent loop: StateGraph agent, query router, orchestrator, reasoning utils, and pure collaborators (message_codec, message_sanitizer, plan_policy, tool_formatting) |
| `client/memory/`        | Episodic memory (ChromaDB/FAISS), ACE Playbook, Reflector                                                                                                          |
| `client/managers/`      | Conversation token management, user profile learning                                                                                                               |
| `client/ui/`            | Chat REPL (`chat_interface`), pinned-input prompt_toolkit UI + dialogs (`tui`), styled reasoning/tool blocks (`turn_view`), spinner                                |
| `server/`               | FastMCP server entry point (run as a subprocess)                                                                                                                   |
| `server/tools/`         | Tool implementations (bash, file ops, git, web, RAG, vision, planning)                                                                                             |
| `server/tools/rag/`     | Session-scoped vector store, hybrid search engine                                                                                                                  |
| `server/tools/readers/` | File format readers (PDF, DOCX, CSV, JSON, directory, line, search)                                                                                                |
| `server/tools/safety/`  | Server-side catastrophic-command + write-path policies (shared by shell/file tools)                                                                                |
| `models/`               | `provider_params` registry + `mantle_factory`                                                                                                                      |
| `models/controllers/`   | Provider-dispatching LLM/vision/embeddings controllers                                                                                                             |
| `models/chat_models/`   | Concrete LangChain ChatModel subclasses (ChatOllamaWrapper, ChatSageMaker)                                                                                         |
| `utils/`                | Config singleton, configurator, paths, logger, BM25, text formatting                                                                                               |
| `bash/` (repo root)     | Shell scripts (system command wrapper, Ollama VRAM management)                                                                                                     |

## Detailed File Map

The full per-file reference (every module, its key classes/functions, and
what it does) lives in [`ARCHITECTURE.md`](ARCHITECTURE.md) to keep
this file lean. Consult it when you need to locate or understand a specific
file; the sections below cover the high-level architecture and conventions.

## Key Patterns

> Each pattern below is a one-paragraph summary. The **full implementation
> detail, edge cases, and gotchas** for the starred (★) subsystems live in
> [`ARCHITECTURE.md`](ARCHITECTURE.md) under **Key Subsystems — Implementation
> Detail**; consult it when changing one of them.

### Config singleton (`utils/config.py`) ★

All configuration flows through `Config()` (`Config().get("SECTION.KEY", default)`); `utils/config.yaml` is gitignored. **Prompts are separate**: model-facing prompts live in **`prompts.yaml`** (accessed via `Config().prompt("KEY")`), never in `config.yaml`. `_load_prompts` layers the bundled package `prompts.yaml` under the user's file (user keys win, missing keys fall back), so a new prompt key reaches existing installs.

### MCP tool registration (`server/tools/tools_manager.py`)

`ToolManager.register_tools(mcp)` conditionally registers tool groups based on config toggles. Each tool file defines functions decorated with `@mcp.tool()`.

### Server-side safety policies (`server/tools/safety/`) ★

Two pure classifiers enforce, **inside the MCP server**, a hard floor that holds even if the server is driven directly. Narrow: they block only **catastrophic, irreversible** actions (root/home `rm -rf`, `mkfs`, raw-device `dd`, power-state changes, fork bomb; system-dir writes), NOT ordinary scoped mutations (those stay gated by the client `_confirm_tool` gate + plan mode). `bash_policy.classify_shell_command` (in `execute_bash` + `start_background_task`) and `path_policy.classify_write_path` (in `fs_write` + `file_edit`); config-independent, unit-tested.

### Read-before-write / staleness gate (`server/tools/read_state.py`) ★

A third server-side floor, distinct from the safety classifiers: an in-process registry that records the on-disk mtime of every file the model reads via `fs_read` (`record_read`), and blocks `fs_write`/`file_edit` from modifying an **existing** file that was never read or that **changed on disk since it was last read** (`check_write_allowed` → an `error`/`must_read_first`/`stale_read` payload). Creating a brand-new file is exempt; a successful write re-baselines the file (chained same-turn edits are fine). Keyed by `normcase(realpath(path))` so different spellings/case map to one entry. It never prompts (returns a normal tool error, so non-TTY / directly-driven servers can't deadlock) and is purely additive to the client `_confirm_tool` gate and the `path_policy` floor.

### External MCP servers (`client/mcp_config.py`, `MultiMCPClient`) ★

Built-in server always launched; extra stdio servers declared in `~/.mnemoai/mcp/mcp.json` (`load_external_servers()`, tolerant). `MultiMCPClient` merges tools — a colliding external tool is namespaced `servername__tool` (built-in names win). External tools are appended to **every** route (incl. `simple_qa`) so routing never hides them, and injected into the decomposition prompt when orchestration is on. `/mcp` lists status.

### Hybrid search (semantic + BM25)

Used in both episodic memory and RAG. Pattern: get top-N candidates from vector store, get top-N from BM25, merge with configurable weights (`utils/bm25.py`).

### Multi-provider LLM abstraction (`models/controllers/llm_controller.py`) ★

`LangChainLLMController.initialize_model()` dispatches on `MODEL_ID.TYPE` (bedrock/mantle/ollama/openai/anthropic/sagemaker/litellm). Supported keys/kwarg mapping per provider live in `models/provider_params.py` (via `build_kwargs`); an `EXTRA_PARAMS` dict passes anything else verbatim into the request body. `anthropic` is the direct Anthropic API — distinct from Mantle's `anthropic` _protocol_ (Claude via Bedrock).

### Bedrock Mantle (`models/mantle_factory.py`) ★

`TYPE: mantle` reaches Bedrock Mantle via a bearer token minted from AWS creds. `API_PROTOCOL` picks the wire protocol: `chat_completions` (`/v1`), `responses` (`/openai/v1`), `anthropic` (`/anthropic`). Shared by LLM + vision controllers; availability varies by region.

### Conversation compaction (`client/managers/agent_conversation_manager.py`) ★

Keeps history under `MAX_CONVERSATION_TOKENS` by summarizing older turns into the system prompt while keeping recent ones verbatim (auto over-budget, or `/compact`). The kept window is bounded by message count AND a token budget; the split point is tool-pair-safe (`_safe_tool_boundary`). **Layered context-overflow protection** (all auto-derive from the window): (1) cap each tool result at the source (`MAX_TOOL_RESULT_CHARS`); (2) pre-flight compaction — cheap tool-result eviction first (`evict_old_tool_results`, no model call), full LLM summary only if still over; (3) a 400 overflow backstop that compacts instead of re-looping. **The high-water trigger reads the provider's exact `input_tokens` (ground truth, = `[Context: N]`) when a turn has run, not the ~2×-inflated estimate**, so it doesn't fire early. **Eviction runs first on EVERY compaction path** (manual/auto/backstop), not just the proactive check. The **summary uses a reasoning-disabled variant of the same model** (`llm_controller.build_non_reasoning_model`; provider-agnostic — thinking off, `verbose` off for Ollama; `LLM.SUMMARIZATION_THINK: true` keeps it on) and is a **parallel map + single reduce** (`generate_summary`: batches summarized concurrently under `SUBAGENT_MAX_CONCURRENCY`, then one reduce folds the ordered partials + any prior summary). Separately, **output-token truncation auto-continues** (`_continue_truncated_turn`, `MAX_OUTPUT_CONTINUE_RETRIES`) and a **stalled-stream watchdog** (`_iter_stream_with_idle_timeout`, `STREAM_IDLE_TIMEOUT`) re-runs a dead-socket turn on a fresh connection. See ARCHITECTURE.md for the full behavior of each.

### Query routing (`client/agent/router.py`)

`QueryRouter.classify()` uses the LLM to categorize queries. Routes map to tool subsets in `ROUTE_TOOLS` — only relevant tools are bound per query.

### Orchestrator-workers (`client/agent/orchestrator.py`, `client/agent/agent.py`) ★

For "full" tasks: decompose → parse subtasks (JSON, with optional `depends_on`) → run a worker loop per subtask with category tools → aggregate. `_run_subtasks_scheduled` runs each dependency-satisfied **wave** concurrently on the bounded `ThreadPoolExecutor` (`LLM.SUBAGENT_MAX_CONCURRENCY`); no `depends_on` → sequential (backward-compatible). Parallel-wave workers run **headless** (`_headless_tl`) so an untrusted destructive tool auto-denies instead of stacking confirm prompts.

### Sub-agents (`spawn_agent`) (`client/agent/subagents.py`, `server/tools/subagent_tool.py`) ★

Model-initiated (distinct from the orchestrator): `spawn_agent(agent_type, prompt, description)` hands a task to a fresh sub-agent on its **own isolated context**, returning **only its final report**. Thin-stub + `_handle_spawn_agent` interception at both chokepoints; reuses `_run_worker_loop` in `quiet=True` mode (still streams via `_stream_once_quiet`, suppresses display). Built-in types `general-purpose`/`explore`/`plan` (+ custom `~/.mnemoai/agents/*.md`); no nested spawning; discovered via `available_subagents_block()`, no `/agents` command. Bounded parallel (`_run_spawn_batch`); **`run_in_background` defaults to true** (daemon thread + `BackgroundAgentRegistry`, auto-delivered when idle) — the model passes `false` only when it needs the report to continue the turn or a `general-purpose` sub-agent must edit un-approved things. `resume_agent` also defaults background (`load_from_disk` across restart). Background sub-agents run headless (auto-deny untrusted destructive tools). See ARCHITECTURE.md for the complete four-phase behavior.

### Agent pure collaborators (`client/agent/{message_codec,message_sanitizer,plan_policy,tool_formatting}.py`) ★

`LangGraphAgent` is the coordinator; its **stateless** logic lives in siblings it delegates to: `message_codec` (Strands↔LangChain), `message_sanitizer` (`sanitize_tool_pairs`), `plan_policy` (plan-block decision + read-only-bash heuristic + tables), `tool_formatting` (marker rendering, arg normalization, error translation). Thin `_…` methods on the agent delegate in, preserving the class surface the unit tests build against. Put new pure tool/plan/message logic in the collaborator, not `agent.py`.

### ACE Playbook learning (`client/memory/reflector.py`, `client/memory/playbook_store.py`)

After each interaction, the Reflector analyzes tool trajectories, detects failure patterns, and extracts reusable strategies stored in the PlaybookStore. Relevant strategies are injected into the system prompt for future queries.

### Episodic memory (`client/memory/episodic_memory.py`)

Stores successful task completions with tool usage patterns. Retrieved via hybrid search before each query and injected as context.

### Curated memory (MEMORY.md) (`client/memory/memory_store.py`, `server/tools/memory_tool.py`) ★

A small, bounded, profile-scoped `~/.mnemoai/{profile}/MEMORY.md` of durable facts the agent **curates itself** via the MCP `memory` tool (`add`/`replace`/`remove`; entries `---`-separated). Injected **whole** into the system prompt at session start (frozen snapshot). Hard char cap (`MEMORY.MAX_CHARS`, default 2200) forces consolidation. Entries tagged `[user]`/`[feedback]`/`[project]`/`[reference]`. Gated by `ENABLE_MEMORY`; writes confirmation-gated by `REQUIRE_MEMORY_CONFIRMATION` (default false). **Auto-extraction** (opt-in `ENABLE_MEMORY_AUTO_EXTRACTION`) distills facts at turn end on a daemon thread, writing without a prompt. Distinct from episodic memory + the playbook.

### Steering (STEERING.md) (`client/memory/steering_store.py`, `utils/paths.steering_files`) ★

**User-authored, always-on instructions** — the read-only counterpart to `MEMORY.md`. Discovered hierarchically (global `~/.mnemoai/STEERING.md` + project `./STEERING.md` walked up to the repo root), concatenated broadest→specific. Injected as a leading `<steering>` block **every turn** by `client._steering_reminder()`, then **stripped before storage** by `_strip_ephemeral` — so it's re-read from disk each turn (edits apply immediately) and never summarized by compaction (always verbatim). No config toggle (presence is the switch), no slash command.

### Agent Skills (`client/memory/skill_store.py`, `server/tools/skill_tool.py`) ★

Authored, on-demand instruction packs, one per dir: `~/.mnemoai/skills/{name}/SKILL.md` (frontmatter `name`+`description`, optional `argument_hint`, markdown body). **Three-tier progressive disclosure**: (1) name+description injected at session start (`<available_skills>`, size-bounded); (2) full body loaded only when the model calls `use_skill` (return value _is_ the body); (3) bundled resources read on demand. Directory name is the id. `SkillStore` is tolerant pure file logic shared by the server tool + client injection + `/skills`. `use_skill` is always-available. Gated by `ENABLE_SKILLS`; bundled example seeded on first run.

### Terminal UI (`client/ui/chat_interface.py`, `client/ui/tui.py`, `client/ui/turn_view.py`) ★

Pinned-input REPL on a TTY (`PinnedPromptReader`): a non-full-screen prompt*toolkit `Application` keeps `>` at the bottom while output streams above via `patch_stdout(raw=True)` (scrollback/copy-paste preserved). Query runs on a worker thread (`client.query()` calls `asyncio.run()` internally). `Ctrl+J` newline / Enter submit; Esc/Ctrl+C cancels a turn; **a message submitted mid-turn is QUEUED FIFO and runs as its own turn after the current one ends** (shown as a dim `> … (queued)` line, never concurrent, never folded into the running turn). The agent-side mid-turn \_steering* machinery (`agent.steer`/`_drain_steering`, drained in `_execute_tools`/`_orchestrate`) still exists but is **not** fed by the UI — it was retired as the default because a message steered during the final tool-call-free model call was never drained at turn end and leaked into the next turn. (Distinct from background sub-agent auto-delivery, which uses its own `drain_background_completions`, not the steer queue.) Styled turn view: collapsed "Thought for Ns…" blocks + `ToolName`/`↳ arg` blocks; loaded conversations replay through the **same** `CodeFormatter` (`turn_view.render_markdown`) so reloaded answers render identically. Confirmations + dialog commands (`/load`, `/config`, …) run via in-app hooks; degrades to a plain `input()` loop off-TTY. Streamed answers render through `CodeFormatter` (markdown-it-py re-parse-per-chunk + Pygments). See ARCHITECTURE.md for the full behavior.

**UI-thread ↔ worker-thread invariants (a violation here HANGS the whole app):**

- **Never drop a `run_in_terminal` awaitable.** It suspends the app to write to scrollback and chains on `app._running_in_terminal_f`, so an un-awaited call both hides its own failure (it surfaces only as an unretrieved-task warning) and can stall a *later* in-terminal write. Always `await` it inside a task with its own `except` — `_notice()` is the ready-made helper for a one-line dim message from the UI thread; the confirm echo in `_await_confirm` does the same and repaints in a `finally`.
- **A scrollback write must never gate a prompt.** The pinned confirm/approval prompt is painted by the app while the *worker* blocks on an `Event`; if the echo can block the paint, the user gets a bare cursor with nothing to press.
- **Never block a worker thread on an unbounded `Event.wait()`.** Esc/Ctrl+C cannot reach a thread parked in `wait()`, so an unanswerable prompt becomes an unrecoverable freeze. Poll with a timeout and bail on `_cancel_requested()` (`_cancelled`, reset per turn). **A cancelled/failed wait must return the DENY answer, never the caller's `default`** — plan approval passes `default="approve"`, and a prompt nobody saw must never count as approval.
- **`run_dialog` is the deliberate exception:** its `done.set()` lives in a `finally` on the event loop, so it releases the worker even when the dialog raises. Don't add a timeout there — bailing out with `_pending_dialog` still armed leaves the app half-torn-down. Note `_open_detail` runs on the UI thread and so must NOT block on it (it pre-renders and stashes a `_pending_dialog` instead).
- **Thread affinity:** key handlers run on the event loop (so `loop.create_task` is fine there); anything called from the worker thread must marshal via `call_soon_threadsafe`. Check which thread a caller is on before picking one.

## Configuration

`utils/config.yaml` (gitignored). Copy from one of the provided templates:
`utils/config.yaml.example` (Ollama/local), `utils/config.yaml.bedrock.example` (standard Bedrock), or `utils/config.yaml.bedrock.mantle.example` (Bedrock Mantle). Each is a complete drop-in config for that provider — keep them in sync when adding shared config keys.

| Section           | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `MODEL_ID`        | LLM provider/`TYPE` (bedrock, mantle, ollama, openai, anthropic, sagemaker, litellm), model name, inference params. Mantle adds `API_PROTOCOL` (chat_completions/responses/anthropic) and optional `ENDPOINT_URL`. Anthropic uses `API_KEY`/`ANTHROPIC_API_KEY` + optional `ENDPOINT_URL` base URL. `openai` accepts `API_BASE` (alias `ENDPOINT_URL`) + optional `API_KEY` to target any OpenAI-compatible server — local `llama-server`/LM Studio/vLLM — as an Ollama alternative. |
| `VISION_MODEL_ID` | Vision model for image description (same provider types as `MODEL_ID`)                                                                                                                                                                                                                                                                                                                                                                                                               |
| `RAG`             | Embedding model, chunk size, hybrid weights, vector store type                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `EPISODIC_MEMORY` | Thresholds, search weights, success/error markers                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `PLAYBOOK`        | Max entries, similarity threshold, injection limit                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `LLM`             | Retry config (incl. `MAX_OUTPUT_CONTINUE_RETRIES`, `STREAM_IDLE_TIMEOUT`), thinking toggle, agent `RECURSION_LIMIT`, `SUBAGENT_MAX_CONCURRENCY`, `MCP_CALL_TIMEOUT`, and compaction (`KEEP_RECENT_MESSAGES`, `MANUAL_COMPACT_KEEP_RECENT`, `KEEP_RECENT_TOKEN_BUDGET`, `COMPACT_HIGH_WATER_TOKENS`, `MAX_TOOL_RESULT_CHARS`, `TOOL_EVICTION_KEEP_RECENT`, `EVICTED_TOOL_RESULT_CHARS`)                                                                                               |
| `PROFILE`         | User name (data isolation), profiling toggle                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `BRAVE_API_KEY`   | Web search API key                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

**Prompts** live in **`prompts.yaml`** (NOT `config.yaml`), accessed via `Config().prompt("KEY")`:

| Prompt key              | Purpose                                |
| ----------------------- | -------------------------------------- |
| `SYSTEM_PROMPT`         | Full system prompt (XML-structured)    |
| `ROUTING_PROMPT`        | Query classifier prompt                |
| `ORCHESTRATOR_PROMPT`   | Task decomposition prompt              |
| `AGGREGATOR_PROMPT`     | Result synthesis prompt                |
| `SUMMARY_SYSTEM_PROMPT` | System framing for compaction          |
| `SUMMARY_TASK_PROMPT`   | Structured compaction/summary template |

**Feature toggles** (all boolean in config root):
`ENABLE_RAG`, `ENABLE_EPISODIC_MEMORY`, `ENABLE_PLAYBOOK`, `ENABLE_WEB_SEARCH`, `ENABLE_WEB_CRAWL`, `ENABLE_ROUTING`, `ENABLE_ORCHESTRATION`, `REQUIRE_BASH_CONFIRMATION` (default true), `REQUIRE_WRITE_CONFIRMATION` (default true), `ENABLE_MEMORY` (default true), `REQUIRE_MEMORY_CONFIRMATION` (default false), `ENABLE_MEMORY_AUTO_EXTRACTION` (default false), `ENABLE_SKILLS` (default true)

**Environment variables:**

- `OPENAI_API_KEY` — for OpenAI provider
- `LOG_LEVEL` — logging verbosity (default: INFO)
- AWS credentials via `aws configure` for Bedrock/SageMaker/Mantle (Mantle mints a bearer token from these via `aws-bedrock-token-generator`)
- Config `ENV` section sets additional env vars at startup

**Runtime data:** All state lives under a single app home, `~/.mnemoai/` (override with `$MNEMOAI_HOME`), resolved centrally in `utils/paths.py`. Layout: `config/` (config.yaml + **prompts.yaml** + bundled `*.example` copies), `mcp/` (optional mcp.json + mcp.json.example), `skills/` (one `{name}/SKILL.md` per agent skill), `plans/`, `tasks/`, and per-profile `{profile_name}/` (conversations, todos, RAG indexes, chunk caches, user profile, and `MEMORY.md` — the curated persistent memory). On first run (and every startup) `seed_example_files()` copies the package's bundled examples into `config/` and `mcp/`, seeds the live `config/prompts.yaml` from the package copy, and seeds the bundled example skills into `skills/` **per-skill** (each bundled skill whose directory is absent is copied — so a newly-bundled skill also reaches an already-populated `skills/` on upgrade; idempotent, never overwrites a user's own skill). Three refresh paths keep EXISTING installs current without clobbering user files: the read-only `*.example` reference files are re-copied from the bundle when they **differ** (`_refresh_example`); an already-installed bundled skill's `SKILL.md` is refreshed in place when the installed copy is **pristine** — its hash is in `_PRISTINE_BUNDLED_SKILL_HASHES` (a version we shipped, unmodified), via `_refresh_pristine_skill` — while a user-edited skill is left untouched; and the live **`prompts.yaml`** is refreshed in place the same way when pristine (hash in `_PRISTINE_BUNDLED_PROMPTS_HASHES`, via `_refresh_pristine_prompts`), so **prompt improvements (edits to existing keys, which the bundled-fallback loader can't deliver — it only fills MISSING keys) reach existing installs**, while a user-customized `prompts.yaml` is untouched. When a bundled `SKILL.md` or `prompts.yaml` changes, append its PREVIOUS shipped hash to the corresponding set so the prior version is still recognized as pristine on the next upgrade. Config resolves `config/config.yaml` → legacy flat `config.yaml` → package fallback; prompts resolve `$MNEMOAI_PROMPTS` → `config/prompts.yaml` → package `utils/prompts.yaml`. **Episodic memory and the ACE playbook are model-scoped** under `{profile_name}/models/{sanitized_model_name}/` so switching the chat model doesn't contaminate memory built with a different one. A user-authored `STEERING.md` may live at the app-home root (`~/.mnemoai/STEERING.md`) and/or per-project (`./STEERING.md`, walked up to the repo root); see the Steering section. All path construction goes through `utils/paths.py` (`app_home`, `config_dir`, `config_path`, `prompts_path`, `mcp_dir`, `mcp_config_path`, `skills_dir`, `plans_dir`, `tasks_dir`, `profile_dir`, `model_dir`, `memory_file_path`, `global_steering_path`, `steering_files`).

## Code Conventions

- **Comments** — concise and focused. One-line docstring per method; inline comments only for non-obvious "why", never restating the code. No verbose multi-line explanations or essays. Keep the surrounding file's comment density.
- **New features must reach EXISTING installs, not just fresh ones** — when a feature adds anything to the app home (`~/.mnemoai/`: a bundled skill, an `*.example`, a `prompts.yaml` key, a new config file, seeded data), it MUST be delivered so a user who **already installed the app** gets it on the next upgrade — never gated on a first-run-only / empty-dir condition. Use the `seed_example_files()` **per-item "seed if absent"** pattern (each bundled item copied when its own destination doesn't exist), not "seed only when the whole dir is empty". To push an **update to an already-seeded item**, refresh it too — but only when safe: `*.example` reference files (never loaded as config) are refreshed whenever they differ (`_refresh_example`); a bundled skill's `SKILL.md` AND the live `prompts.yaml` are refreshed in place only when the installed copy is **pristine** (hash in `_PRISTINE_BUNDLED_SKILL_HASHES` / `_PRISTINE_BUNDLED_PROMPTS_HASHES` — a version we shipped, unmodified; `_refresh_pristine_skill` / `_refresh_pristine_prompts`), never over a user's edits — this is how a change to an EXISTING prompt key reaches installed users (the bundled-fallback loader only fills MISSING keys). Prefer **presence-based / code-driven** activation over a new config toggle so nothing depends on the user editing a pre-existing `config.yaml` (e.g. STEERING.md: the file's presence is the switch; new `config.yaml`/`prompts.yaml` keys fall back to a bundled default in code — the live `config.yaml` is never rewritten). Always add a test that the item reaches an **already-populated** app home (see `tests/unit/test_paths.py::test_seed_skills_per_skill_reaches_populated_dir` and `test_pristine_installed_skill_is_refreshed`), and never clobber a user's own edits.
- **Tests** — pytest unit suite in `tests/` covers pure-logic modules (no LLM/Ollama needed). Run with `python -m pytest`. See the Testing section below
- **Error handling in tools** — `@tool_error_handler` decorator (`server/error_handler.py`) for standardized responses
- **Action confirmation** — destructive tools (`execute_bash`; `fs_write`/`file_edit`; `memory` writes) are hard-gated by `LangGraphAgent._confirm_tool()` in BOTH `_execute_tools` and `_run_worker_loop`, before `tool.invoke()`. It must live client-side: the MCP server is a piped subprocess and can't prompt the terminal. Toggles `REQUIRE_BASH_CONFIRMATION` / `REQUIRE_WRITE_CONFIRMATION` (default true), `REQUIRE_MEMORY_CONFIRMATION` (default false); non-TTY auto-proceeds. The prompt is `Proceed? (y/N/a)` — `a` trusts that whole category (`bash`/`write`/`memory`, tracked in `_trusted_confirm_categories`) for the rest of the session so a multi-step task doesn't re-prompt; default-deny otherwise. **This client gate is a confirmation layer, not the last line of defense**: a separate server-side floor (`server/tools/safety/`, above) hard-blocks catastrophic commands and system-path writes regardless of confirmation, since the MCP server can be driven directly
- **Plan mode (enforced, user-toggled)** — `/plan` flips `client.plan_mode_active`; the agent reads it via `plan_mode_provider` and `_is_blocked_by_plan_mode` HARD-BLOCKS the mutating/exec tools (`_PLAN_BLOCKED_TOOLS` = execute*bash, fs_write, file_edit, git_safe, git_commit_safe, start_background_task) at both chokepoints, **above** `_confirm_tool` (blocked → never even prompts). Conditionally allowed: `execute_bash` when read-only (`_is_readonly_bash`), and `fs_write`/`file_edit` only for the plan file (`.md` under `plans_dir()`, `_is_plan_file`). Read-only tools + `memory` always pass. `client.query()` prepends a read-only reminder each turn (`_plan_mode_reminder`). **Approval → execute handoff:** the model calls `exit_plan_mode(plan)` (thin stub in `_ALWAYS_AVAILABLE_TOOLS`); `_handle_exit_plan_mode` intercepts client-side at both chokepoints, drives `_plan_approval_ui` (y=approve&run / e=edit in `$EDITOR` / n=keep planning), then `client._approve_plan` flips plan mode off, persists to `plans/plan*<ts>.md`, and hands the plan back so the model executes **in the same turn** (non-TTY auto-approves). `exit_plan_mode(plan, allowed_bash=[…])`pre-declares commands stored on`agent.\_preapproved_bash`so`\_confirm_tool`auto-confirms them (equal/prefix match,`\_is_preapproved_bash`). Approval also sets `agent.\_execute_plan_route = True`so`\_effective_route`forces the **full** toolset for the rest of the task (else a read-only re-classification would leave the write tools unbound);`\_route_after_classify`sends those turns straight to the agent, not the orchestrator. Pre-approved bash + route pin are plan-scoped (cleared on`/clear` and plan re-entry). This enforced path is the only plan mode (the legacy JSON-workflow tools were retired)
- **Async/sync bridge** — MCP client uses a background thread with `asyncio.new_event_loop()` in `client/mcp_tool_wrapper.py`; sync callers use `run_coroutine_threadsafe`
- **Imports** — absolute (`from mnemoai.…`), at module top level, grouped stdlib → third-party → first-party and **alphabetized within each group** (enforced by `ruff check --select I .`, the CI gate). **Lazy/function-local imports are the exception, NOT the default** — add one only for a real circular import or a heavy/optional dependency you must defer; convenience/habit are not reasons. **Prove the cycle before going lazy:** cold-import the module that would close the loop with the import at the top (e.g. `PYTHONPATH=src python -c "import mnemoai.client.agent.agent"`, and worst case the new module first); if both resolve, keep it at the top. A cycle needs A→B _and_ B→A both at top level; a lazy back-reference makes a top-level forward import safe. **Test-isolation consequence:** a top-level `from x import name` binds `name` into the importing module, so patch it where it's _looked up_ (`monkeypatch.setattr(subagents, "agents_dir", …)`), NOT at its source — the latter silently no-ops.
- **Type hints** — used for LangChain/LangGraph state schemas (`TypedDict`), model classes; not enforced everywhere
- **Naming** — snake_case functions/variables, PascalCase classes, UPPER_CASE config keys
- **File I/O** — JSON for persistence (playbook, todos, profile, episodic metadata), SQLite for chunk cache
- **Token counting** — tiktoken for OpenAI/Bedrock, character-based approximation (÷4) for Ollama

## Testing

```bash
pip install -r requirements-dev.txt         # installs pytest
python -m pytest                             # everything (integration auto-skips)
python -m pytest tests/unit                  # unit tier only
python -m pytest tests/unit/test_bm25.py     # run one file
python -m pytest -m integration              # integration tier (needs Ollama + config.yaml)
python -m pytest -m "not integration"        # explicitly exclude integration
ruff check --select I .                      # import-sort gate (same as CI); --fix to apply
```

- Layout: `tests/unit/` (pure-logic) and `tests/integration/` (live agent). Configured via `pytest.ini` (testpaths=tests). `tests/conftest.py` puts the repo root on `sys.path` so `utils`/`client`/`server` import cleanly.
- **Unit tier (default):** deterministic, pure-logic tests — no LLM, Ollama, or network needed, runs in seconds. Covers `utils/bm25.py`, `client/agent/reasoning_utils.py`, `utils/formatting/` (response_parser, url_formatter, code_formatter), `client/agent/orchestrator.parse_subtasks`, `server/error_handler.py`, `server/tools/git_safety.py` (command-danger classification), `server/tools/file_edit.py` + `glob_search`, `execute_bash` timeout/process-group behavior + output cap, `grep_search` modes/total-cap/context/non-UTF-8/regex-error, the `read_state` read-before-write gate, background-task group-cancel, `file_encoding` BOM/CRLF round-trips + encoding-preserving edits, streamed line reads, skill token-budget/memoization/argument-substitution, `client/memory/episodic_memory` heuristics, Bedrock/Mantle model wiring (`test_bedrock_endpoint.py`), vision content normalization (`test_vision_content.py`), and conversation compaction incl. token-aware retention, parallel map-reduce summarization, eviction-first, and the ground-truth trigger + reasoning-disabled summary model (`test_conversation_compaction.py`, `test_preflight_compaction.py`).
- Unit tests must not require a `config.yaml` — modules degrade gracefully without one. Keep import-time side effects config-independent so new code stays unit-testable.
- **Integration tier (`tests/integration/`, marked `@pytest.mark.integration`):** drives the real `LangGraphClient` + Ollama + MCP subprocess (greeting/routing, tool calls, bash timeout, no-silent-empty-turn). Auto-skipped unless a runtime `config.yaml` exists AND the configured Ollama host is reachable (see `tests/integration/conftest.py`). The shared client is session-scoped; an autouse fixture calls `clear_context()` between tests for isolation.

## Adding a New MCP Tool

1. Create `server/tools/my_tool.py`
2. Define function with `@mcp.tool()` decorator (receives `mcp` from registration)
3. Add `@tool_error_handler` for standardized error responses
4. Register in `server/tools/tools_manager.py` → `register_tools()` method
5. If conditionally enabled, gate behind a config toggle
6. Add route mapping in `client/agent/router.py` → `ROUTE_TOOLS` if it belongs to a specific category

## Adding a New LLM Provider

1. Add provider case in `models/controllers/llm_controller.py` → `initialize_model()`
2. Register the provider's supported config keys in `models/provider_params.py` (the registry consumed via `build_kwargs` and used by `/model` pruning)
3. If custom LangChain model needed, create class in `models/chat_models/`
4. Add embedding support in `models/controllers/embeddings_controller.py` if provider offers embeddings
5. Add vision support in `models/controllers/vision_model_controller.py` if applicable
6. Document config structure in all `utils/config.yaml*.example` templates

## Stability & Versioning

Semver. The **public contract** (what a major bump protects) is: `config.yaml` keys (model sections, `ENABLE_*`/`REQUIRE_*` toggles, documented sections), the `mcp.json` `mcpServers` schema, the CLI slash-commands + `mnemoai` console command, and the `mnemoai-assistant` dist / `mnemoai` import name. Everything under `client/`/`server/`/`models/`/`utils/` not in that list is internal and may change freely. All changes go in `CHANGELOG.md`; releases follow the checklist in `docs/development/index.md`. CI (`.github/workflows/tests.yml`) runs the unit tier + import-sort on push/PR; the integration tier is run locally before releases.

## Known Limitations

- Unit tests cover pure logic only — agent/LLM integration paths still need manual verification (run the integration tier + the release checklist before releasing)
- MCP server is a subprocess — debugging requires attaching to child process or reading logs
- No Docker/containerization — runs directly on host with system Python/conda/venv
- No database — all persistence is file-based (JSON, FAISS index files, SQLite chunk cache)
- Single-user — profile name in config isolates data but no auth/multi-tenancy
- `ripgrep` (rg) required for `grep_search` tool — install separately
