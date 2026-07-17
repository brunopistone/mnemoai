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

### Config singleton (`utils/config.py`)

All configuration flows through `Config()` which loads `utils/config.yaml` (gitignored). Access via `Config().get("SECTION.KEY", default)`. **Prompts are separate**: every model-facing prompt (`SYSTEM_PROMPT`, `ROUTING_PROMPT`, `ORCHESTRATOR_PROMPT`, `AGGREGATOR_PROMPT`, `SUMMARY_SYSTEM_PROMPT`, `SUMMARY_TASK_PROMPT`) lives in **`prompts.yaml`** (sibling of `config.yaml`, same resolution + seeding; `$MNEMOAI_PROMPTS` overrides), accessed via `Config().prompt("KEY")`. `config.yaml` holds _only_ configuration — prompt keys left there are ignored with a one-time migration warning. There are no hardcoded prompts in the code paths (a bundled package `prompts.yaml` provides the defaults; tiny computed crash-guards exist only for a totally-missing file). `_load_prompts` **layers the bundled package `prompts.yaml` underneath the user's file** (user keys win, missing keys fall back to the bundle) — so a NEW prompt shipped in a release resolves on an EXISTING install whose seeded `prompts.yaml` is never overwritten, while user customizations still take precedence. The `.system_prompt` property reads `SYSTEM_PROMPT` from prompts.yaml.

### MCP tool registration (`server/tools/tools_manager.py`)

`ToolManager.register_tools(mcp)` conditionally registers tool groups based on config toggles. Each tool file defines functions decorated with `@mcp.tool()`.

### Server-side safety policies (`server/tools/safety/`)

Two pure classifiers enforce, **inside the MCP server**, the hard limits the tool docstrings advertise — a floor that holds even if the server subprocess is driven directly (not via the client's `_confirm_tool` gate). Deliberately narrow: they block only **catastrophic, irreversible** actions, NOT ordinary scoped mutations (those stay gated by the client confirmation + plan mode). `bash_policy.classify_shell_command(cmd) -> BashPolicyResult` blocks root/home recursive force-deletes (`rm -rf /|~|/*`), `mkfs*`, raw-device `dd`/`shred`/`wipefs`, power-state changes (`shutdown`/`reboot`/`halt`/`poweroff`, `init 0/6`), and the fork bomb — enforced at the top of `execute_bash` **and** `start_background_task` (one shared policy, so shell safety isn't duplicated). `path_policy.classify_write_path(path) -> PathPolicyResult` normalizes `..`/`~` then blocks writes into system dirs (`/`, `/etc`, `/bin`, `/usr`, `/boot`, `/dev`, `/System`, `/Library`, …; NOT the macOS `/private/var/folders` temp dir) — enforced in `fs_write` and `file_edit`. Both are config-independent and unit-tested (`tests/unit/test_safety_policies.py`). This is the server-side complement to the client-side `_confirm_tool` gate and plan mode.

### External MCP servers (`client/mcp_config.py`, `MultiMCPClient`)

The built-in server is always launched; additional stdio MCP servers can be declared in `~/.mnemoai/mcp/mcp.json` (standard `mcpServers` schema; legacy flat `~/.mnemoai/mcp.json` still read). `load_external_servers()` parses them (tolerant: missing/bad file or entry → skip, don't crash). `MultiMCPClient` (in `mcp_tool_wrapper.py`) owns the built-in wrapper + one per external server, connects them together, and merges tools — namespacing a colliding external tool as `servername__tool` (built-in names always win; the server is still called with the original name). External tools are appended to **every** route in `agent.py` — including the no-tools `simple_qa` route — so routing never hides them (a short factual question like "what time is it?" classifies as `simple_qa`, so an external server such as `time` must be reachable there too). When orchestration is enabled, `_external_tools_prompt_block()` injects the external tool names/descriptions into the decomposition prompt and instructs the decomposer to route subtasks needing them to the `full` category (which binds every tool) — otherwise the decomposer, unaware they exist, can't target them. `/mcp` lists status.

### Hybrid search (semantic + BM25)

Used in both episodic memory and RAG. Pattern: get top-N candidates from vector store, get top-N from BM25, merge with configurable weights (`utils/bm25.py`).

### Multi-provider LLM abstraction (`models/controllers/llm_controller.py`)

`LangChainLLMController.initialize_model()` dispatches on `MODEL_ID.TYPE` (bedrock/mantle/ollama/openai/anthropic/sagemaker/litellm). Each provider's supported config keys / client-kwarg mapping live in `models/provider_params.py` (consumed via `build_kwargs`). For anything the registry doesn't model, every section accepts an `EXTRA_PARAMS` dict — a generic passthrough merged verbatim into the model's request body (`provider_params.extra_params`), used e.g. for `reasoning_effort` (OpenAI/Mantle responses) or `thinking` (Anthropic/Mantle anthropic). Note: `anthropic` is the direct Anthropic API (`ChatAnthropic`, `ANTHROPIC_API_KEY`), distinct from Mantle's `anthropic` _protocol_ (Claude via Bedrock).

### Bedrock Mantle (`models/mantle_factory.py`)

`TYPE: mantle` reaches AWS Bedrock Mantle via a bearer token minted from standard AWS (SigV4) credentials. `API_PROTOCOL` selects the wire protocol: `chat_completions` (OpenAI `/v1`), `responses` (OpenAI Responses `/openai/v1`, e.g. GPT-5.4), `anthropic` (Anthropic Messages `/anthropic`, Claude). The factory is shared by the LLM and vision controllers. Model availability varies by region (e.g. GPT-5.4 is in us-west-2).

### Conversation compaction (`client/managers/agent_conversation_manager.py`)

Keeps the conversation under `MAX_CONVERSATION_TOKENS` by summarizing older messages into the system prompt while keeping recent turns verbatim. Triggers automatically when over budget, or manually via `/compact`. The kept window is bounded by message count AND a token budget so an oversized recent message (e.g. a pasted document) is summarized, not kept. Tool calls/results are preserved in the summary. The summary uses a structured prompt (a "summarizing conversations" system framing + a 9-section `<analysis>`-then-summary task template, with `/compact <focus>` injected as compact instructions); the `<analysis>` scratchpad is stripped, and the injected block carries a continuation instruction so the model resumes seamlessly. **The split point is tool-pair-safe** (`_safe_tool_boundary`): it never starts the kept window with an orphaned tool result nor cuts an assistant tool-call turn from its results — which would otherwise make the OpenAI Responses API reject the next turn with "No tool call found for function call output".

**Context-overflow protection (layered, prevents the runaway 400-loop). Size knobs auto-derive from `MAX_CONVERSATION_TOKENS` (the model's context window) so they scale per-model, not a fixed number:** (1) **Cap tool results at the source** — `tool_formatting.truncate_tool_result` bounds each `ToolMessage` to `LLM.MAX_TOOL_RESULT_CHARS` (defaults to 10% of the window in chars, ~4 chars/token; head+tail kept with a truncation note; 0 disables) at both tool-exec chokepoints (`_execute_tools`, `_run_worker_loop`), so one runaway result (e.g. `grep_search(max_results=4000)`) can't alone overflow the window. (2) **Pre-flight compaction, layered** — `client._compact_now` (wired into the agent as `_compact_provider`, called at the top of `invoke()`) proactively compacts when history exceeds `LLM.COMPACT_HIGH_WATER_TOKENS` (defaults to 80% of `MAX_CONVERSATION_TOKENS`, 0 disables), the "count before send" analog. It tries the **cheapest layer first**: `AgentConversationManager.evict_old_tool_results` shrinks the bodies of _old_ tool results (outside `LLM.TOOL_EVICTION_KEEP_RECENT`, default 8) to a short head + a marker (`LLM.EVICTED_TOOL_RESULT_CHARS`, default 500; 0 disables), with **no model call** and **without dropping any message** (only content is trimmed, so tool-call/result pairing is preserved); if that alone gets back under the high-water mark the expensive full summary is skipped. Otherwise it falls through to `_compact` (the map-reduce LLM summary). (3) **Overflow backstop** — `_is_context_overflow_error` classifies a "prompt is too long"/context-window 400 in `_stream_once`; instead of re-invoking the same oversized prompt (the old loop), it force-compacts for the next turn and returns a terminal AIMessage so the turn ends. All are guarded with `getattr(self, "_compact_provider", None)` so bare test objects still work.

**Output-token truncation → auto-continue (not a dead-end).** Distinct from the *input* overflow above: when the model's OUTPUT hits `MODEL_ID.MAX_TOKENS` mid-turn (common with `REASONING_EFFORT: max` on a large context — reasoning + a partial answer/tool call exhaust the budget), the turn finishes with `stop_reason: max_tokens` and no completed tool call. `_call_model` detects this (`_was_truncated_by_tokens`) and **auto-continues** via `_continue_truncated_turn`: it feeds the partial turn back with a "continue where you left off" nudge and re-streams, accumulating visible text, up to `LLM.MAX_OUTPUT_CONTINUE_RETRIES` (default 3). Parts are glued directly (each is `_extract_visible`-stripped, so a resumed stream can't gain a spurious space mid-word). It stops early when a continuation finishes cleanly (returns the assembled answer) or emits a tool call (returned so the graph runs it, partial text preserved). Only if retries exhaust with nothing usable does it surface the "increase MAX_TOKENS" message — otherwise the user never has to type "continue". This replaced the earlier dead-end that ended the turn with a warning.

**Stalled-stream recovery (dead socket / laptop sleep).** A streaming read (`model.stream()`) is a blocking C-level socket read; if the connection dies silently mid-response (the laptop sleeps and the TCP socket dies), it would park the single worker thread forever — the turn looks stopped AND new input freezes behind it (FIFO worker). The provider's own request timeout only covers getting response HEADERS, not a stalled body, so it can't help. mnemoai consumes the stream through a per-chunk idle-timeout watchdog: `_iter_stream_with_idle_timeout` runs `model.stream()` on a daemon reader thread feeding a `queue.Queue` and times each `get(timeout=LLM.STREAM_IDLE_TIMEOUT)` (default 120s; 0 disables → iterate directly). No chunk within the window → raise `_StreamIdleTimeout` and ABANDON the reader (a daemon; can't force-kill a C socket read, so it unwinds when its socket finally errors). `_stream_response` then treats `_StreamIdleTimeout` + transient network errors (`_is_transient_network_error` — reset/broken-pipe/timeout/5xx/overload, provider-agnostic substrings) as retryable and re-runs the turn on a fresh connection with exponential backoff (reusing `LLM.RETRY_DELAY`/`RETRY_BACKOFF`, capped 30s), discarding the partial (a dropped stream can't resume mid-generation). Exhausted retries surface a terminal "lost the connection — send your message again" AIMessage (never a crash). `_ContextOverflow` is excluded from this retry (the caller compacts instead). Lives in the stream layer, so it's provider-agnostic. This is mnemoai's analog of Claude Code's stream-idle watchdog + retry (CC re-runs the turn too; neither resumes mid-generation).

### Query routing (`client/agent/router.py`)

`QueryRouter.classify()` uses the LLM to categorize queries. Routes map to tool subsets in `ROUTE_TOOLS` dict — only relevant tools are bound per query.

### Orchestrator-workers (`client/agent/orchestrator.py`, `client/agent/agent.py`)

For "full" complexity tasks: decompose → parse subtasks (JSON) → run worker loop per subtask with category-specific tools → aggregate results.

### Agent pure collaborators (`client/agent/{message_codec,message_sanitizer,plan_policy,tool_formatting}.py`)

`LangGraphAgent` is the coordinator; its **stateless** logic lives in sibling modules so the class stays focused on the graph/loop. `message_codec` = Strands↔LangChain message conversion; `message_sanitizer` = orphaned tool-pair repair (`sanitize_tool_pairs`); `plan_policy` = the plan-mode block decision + read-only-bash heuristic + data tables (`PLAN_BLOCKED_TOOLS`, `READONLY_BASH_CMDS`, …); `tool_formatting` = the `[⚙ …]` marker rendering (`format_tool_call`/`elide_middle`), `normalize_tool_args`, and `tool_error_message`. The agent keeps thin `_sanitize_tool_pairs`/`_is_blocked_by_plan_mode`/`_is_readonly_bash`/`_format_tool_call`/… methods that **delegate** into these, preserving the historical class surface the unit tests build against (`LangGraphAgent.__new__(...)` + `LangGraphAgent._…`) and the `getattr(agent, "_sanitize_tool_pairs", …)` call in `AgentConversationManager`. When adding pure tool/plan/message logic, put it in the collaborator and delegate — don't grow `agent.py`.

### ACE Playbook learning (`client/memory/reflector.py`, `client/memory/playbook_store.py`)

After each interaction, the Reflector analyzes tool execution trajectories, detects failure patterns, and extracts reusable strategies stored in the PlaybookStore. Relevant strategies are injected into the system prompt for future queries.

### Episodic memory (`client/memory/episodic_memory.py`)

Stores successful task completions with tool usage patterns. Retrieved via hybrid search before each query and injected as context.

### Curated memory (MEMORY.md) (`client/memory/memory_store.py`, `server/tools/memory_tool.py`)

A small, bounded, profile-scoped (shared across models) `~/.mnemoai/{profile}/MEMORY.md` of durable facts (user/environment details, conventions, lessons, tool quirks, completed work) that the agent **curates itself** via the MCP `memory` tool (`add`/`replace`/`remove` over a list of entries separated by a Markdown `---` rule — legacy `§`-delimited files still parse and migrate to `---` on next write; logic in `MemoryStore`). It is injected **whole** into the system prompt at session start by `_build_system_prompt`/`_inject_memory_context` in `client.py` (a frozen snapshot — writes during a session apply next session). A hard char cap (`MEMORY.MAX_CHARS`, default 2200) forces the agent to consolidate (merge/remove) instead of growing unbounded. The tool prompt guides the model to tag each entry with one of four kinds — `[user]`/`[feedback]`/`[project]`/`[reference]` — and to skip anything the repo/git/CLAUDE.md already records (a tagging convention only; storage stays a single flat file). Distinct from episodic memory (similarity-retrieved per query) and the ACE playbook (tool strategies); it complements both. Gated by `ENABLE_MEMORY` (default true). The `/memory` command views it (`/memory clear` wipes it); writes are confirmation-gated when `REQUIRE_MEMORY_CONFIRMATION` is true (default false — auto-saves), via the same client-side `_confirm_tool()` gate as bash/file writes. **Auto-extraction (opt-in, `ENABLE_MEMORY_AUTO_EXTRACTION`, default false):** the auto-learning counterpart to the tool — `client.auto_extract_memory(query, response)` runs at turn end (wired in `chat_interface` after `reflect_and_learn`, getattr-guarded) on a daemon thread, distilling durable facts from the exchange via the `MEMORY_EXTRACTION_PROMPT` (a one-shot `_invoke_model_once` call that doesn't touch agent state) and applying the returned JSON `add`/`replace` ops through `MemoryStore`. Unlike the tool path it writes **without** a confirmation prompt (hence its own opt-in toggle) but is confined to `MEMORY.md`; `_parse_memory_ops` is tolerant (bad output → no-op) and the worker never raises.

### Steering (STEERING.md) (`client/memory/steering_store.py`, `utils/paths.steering_files`)

**User-authored, always-on instructions** — the counterpart to the agent-curated `MEMORY.md`. Discovered hierarchically (`steering_files`): a global `~/.mnemoai/STEERING.md` plus a project `./STEERING.md` found by walking from the CWD **up to the repo root** (first ancestor with `.git`, else the fs root), concatenated broadest→most-specific, each under a `Contents of <path>:` header (`SteeringStore.read`, read-only — the agent never writes it, the key difference from `MemoryStore`). Injected as a leading **`<steering>`** block prepended to the prompt **every turn** by `client._steering_reminder()` in `query()` (with an "these instructions OVERRIDE any default behavior" framing), then **stripped before storage** by `_strip_ephemeral` (the shared ephemeral-block regex, same rail as the plan-mode banner). Consequences: it's **re-read from disk each turn** (edits apply immediately, no restart) and **never enters history**, so compaction can never summarize it into a lossy paraphrase — it always reaches the model verbatim. This is the same guarantee an always-on instructions file gets by living outside the summarized conversation. **No config toggle** — its presence _is_ the switch (no file → nothing injected), so there's nothing to enable. No dedicated slash command (edit the files directly); distinct from `MEMORY.md` (agent-curated facts), skills (on-demand procedures), and `prompts.yaml` (the base system prompt).

### Agent Skills (`client/memory/skill_store.py`, `server/tools/skill_tool.py`)

Authored, on-demand instruction packs, one per directory: `~/.mnemoai/skills/{name}/SKILL.md` (YAML frontmatter with `name`+`description` required and an optional `argument_hint`, then a markdown body; may bundle `reference.md`/`scripts/`). **Three-tier progressive disclosure:** (1) only each skill's name+description (and `argument_hint`, rendered as `(expects: …)`) is injected into the system prompt at session start — the `<available_skills>` block, cheap and always-on; the whole listing is bounded by `_MAX_LISTING_CHARS` (overflow collapses into a `+N more` line) so a large skills library can't dominate the prompt; (2) the full body is loaded into context **only when the model calls the `use_skill` tool** (its return value _is_ the body, landing as a `ToolMessage`); (3) bundled resources are read/run on demand via `fs_read`/`execute_bash`. The **directory name is the canonical id** (`use_skill(name)`); frontmatter `name` is display only. `SkillStore` is pure file logic (tolerant scan — bad/incomplete skill skipped, never fatal — mirroring `load_external_servers`), shared by the server `use_skill` tool and the client's tier-1 injection (`client._inject_skills_context`) + `/skills` command — the same shared-store pattern as `MemoryStore`. `use_skill` is in `_ALWAYS_AVAILABLE_TOOLS` so a skill-matching query that classifies as `simple_qa` (e.g. "write a commit message") can still trigger it. **Compaction caveat:** `_build_system_with_summary` re-fetches the base prompt fresh (dropping all session-start injections), so it re-injects the `<available_skills>` block via `format_available_skills` (a loaded body lives in the conversation as a tool result and is summarized like any other). Gated by `ENABLE_SKILLS` (default true); a bundled example (`utils/skills_example/`) is seeded on first run. Distinct from MEMORY.md (always-on facts), the playbook (learned tactics), and prompts.yaml (always-on framing): skills are _authored, on-demand procedures_. `/skills` lists them; `/skills <name>` previews a body. Distinct from the enforced plan mode (`server/tools/plan_mode_exit.py` + `plan_policy.py`) and the curated memory tool.

### Terminal UI (`client/ui/chat_interface.py`, `client/ui/tui.py`, `client/ui/turn_view.py`)

The chat REPL is a **pinned-input** UI on a TTY (`PinnedPromptReader` in `tui.py`): a persistent, **non-full-screen** prompt*toolkit `Application` keeps the `>` input fixed at the bottom of the terminal while a query runs, and its output — the answer, the styled reasoning/tool blocks, the `[Context: N]` line — streams **above** it via `patch_stdout(raw=True)` into native scrollback, so **wrapping, scrollback, and copy/paste are preserved**. The query runs on a **worker thread** (required: `client.query()` calls `asyncio.run()` internally, which raises on a thread that already owns a running loop) via `asyncio.to_thread`; the app's event loop stays live to service keystrokes and animate the status line. Key behaviors: **`Ctrl+J`** newline / **Enter** submit; **Esc or Ctrl+C during a turn cancels** it (`_request_cancel` injects `KeyboardInterrupt` into the worker thread → `client.query` returns "Operation was cancelled." — delivered at the next stream-chunk boundary), while Ctrl+C/Ctrl+D twice when idle exits; an **input queue** — a line submitted mid-turn shows a dim `> … (queued)` line in the pinned region and runs FIFO after (never a concurrent query), echoed to scrollback (paired with its own answer) only when the worker starts it. The animated status line shows `⠙ Thinking… (esc to cancel)` with cycling dots, driven by the app's `refresh_interval` (no `\r` writes — a `SpinnerStatus` **sink** flips state and the pinned line renders it; the `\r` stdout `Spinner` mode is used only by the non-TTY plain loop). Slash completion (`SlashCommandCompleter`) pops a menu whose rows the input window reserves space for via a dynamic height (`_input_height`, mirroring `PromptSession`'s reserve-space-for-menu) so it renders even with the input at the terminal bottom. **Styled turn view** (pinned mode, `agent.styled_turn_view`): reasoning is buffered and shown as a collapsed green-bordered **"Thought for Ns…"** block (`client/ui/turn_view.py`), and tool calls render as **`ToolName` + `↳ arg=value`** blocks instead of the `[⚙ …]` marker (that marker is still used by the non-TTY plain loop). **Confirmations** (`Proceed?`): the worker-thread `_confirm_tool` gate calls an in-app `_confirm_ui` hook that shows the prompt in the pinned region and captures a `y`/`n`/`a` keypress. **Dialog commands** (`/load`, `/config`, `/model`, `/params`, `/features`, `/memory clear`): a nested full-screen app can't run inside the pinned one, so `reader.run_dialog` **exits the pinned app** (terminal returns to cooked mode), runs the command with the normal full-screen dialogs (`tui.select_from_list`, `configurator`'s `\_dialog*\*`), then relaunches the pinned app; the configurator suppresses its scrollback banner/overview on a TTY (dialogs carry that), leaving only the final outcome line. **Everything degrades to a plain `input()` loop when not a TTY** (`\_plain_loop`), so pipes / CI / the unit suite never block on a modal.

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
| `LLM`             | Retry config (incl. `MAX_OUTPUT_CONTINUE_RETRIES`, `STREAM_IDLE_TIMEOUT`), thinking toggle, agent `RECURSION_LIMIT`, `MCP_CALL_TIMEOUT`, and compaction (`KEEP_RECENT_MESSAGES`, `MANUAL_COMPACT_KEEP_RECENT`, `KEEP_RECENT_TOKEN_BUDGET`, `COMPACT_HIGH_WATER_TOKENS`, `MAX_TOOL_RESULT_CHARS`, `TOOL_EVICTION_KEEP_RECENT`, `EVICTED_TOOL_RESULT_CHARS`)                                                                                                                                                                                        |
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

**Runtime data:** All state lives under a single app home, `~/.mnemoai/` (override with `$MNEMOAI_HOME`), resolved centrally in `utils/paths.py`. Layout: `config/` (config.yaml + **prompts.yaml** + bundled `*.example` copies), `mcp/` (optional mcp.json + mcp.json.example), `skills/` (one `{name}/SKILL.md` per agent skill), `plans/`, `tasks/`, and per-profile `{profile_name}/` (conversations, todos, RAG indexes, chunk caches, user profile, and `MEMORY.md` — the curated persistent memory). On first run (and every startup) `seed_example_files()` copies the package's bundled examples into `config/` and `mcp/`, seeds the live `config/prompts.yaml` from the package copy, and seeds the bundled example skills into `skills/` **per-skill** (each bundled skill whose directory is absent is copied — so a newly-bundled skill also reaches an already-populated `skills/` on upgrade; idempotent, never overwrites a user's own skill). Two refresh paths keep EXISTING installs current without clobbering user files: the read-only `*.example` reference files are re-copied from the bundle when they **differ** (`_refresh_example`), and an already-installed bundled skill's `SKILL.md` is refreshed in place when the installed copy is **pristine** — its hash is in `_PRISTINE_BUNDLED_SKILL_HASHES` (a version we shipped, unmodified), via `_refresh_pristine_skill` — while a user-edited skill is left untouched. When a bundled `SKILL.md` changes, append its PREVIOUS shipped hash to that set so the prior version is still recognized as pristine on the next upgrade. Config resolves `config/config.yaml` → legacy flat `config.yaml` → package fallback; prompts resolve `$MNEMOAI_PROMPTS` → `config/prompts.yaml` → package `utils/prompts.yaml`. **Episodic memory and the ACE playbook are model-scoped** under `{profile_name}/models/{sanitized_model_name}/` so switching the chat model doesn't contaminate memory built with a different one. A user-authored `STEERING.md` may live at the app-home root (`~/.mnemoai/STEERING.md`) and/or per-project (`./STEERING.md`, walked up to the repo root); see the Steering section. All path construction goes through `utils/paths.py` (`app_home`, `config_dir`, `config_path`, `prompts_path`, `mcp_dir`, `mcp_config_path`, `skills_dir`, `plans_dir`, `tasks_dir`, `profile_dir`, `model_dir`, `memory_file_path`, `global_steering_path`, `steering_files`).

## Code Conventions

- **Comments** — concise and focused. One-line docstring per method; inline comments only for non-obvious "why", never restating the code. No verbose multi-line explanations or essays. Keep the surrounding file's comment density.
- **New features must reach EXISTING installs, not just fresh ones** — when a feature adds anything to the app home (`~/.mnemoai/`: a bundled skill, an `*.example`, a `prompts.yaml` key, a new config file, seeded data), it MUST be delivered so a user who **already installed the app** gets it on the next upgrade — never gated on a first-run-only / empty-dir condition. Use the `seed_example_files()` **per-item "seed if absent"** pattern (each bundled item copied when its own destination doesn't exist), not "seed only when the whole dir is empty". To push an **update to an already-seeded item**, refresh it too — but only when safe: `*.example` reference files (never loaded as config) are refreshed whenever they differ (`_refresh_example`), and a bundled skill's `SKILL.md` is refreshed in place only when the installed copy is **pristine** (hash in `_PRISTINE_BUNDLED_SKILL_HASHES` — a version we shipped, unmodified; `_refresh_pristine_skill`), never over a user's edits. Prefer **presence-based / code-driven** activation over a new config toggle so nothing depends on the user editing a pre-existing `config.yaml` (e.g. STEERING.md: the file's presence is the switch; new `config.yaml`/`prompts.yaml` keys fall back to a bundled default in code — the live `config.yaml` is never rewritten). Always add a test that the item reaches an **already-populated** app home (see `tests/unit/test_paths.py::test_seed_skills_per_skill_reaches_populated_dir` and `test_pristine_installed_skill_is_refreshed`), and never clobber a user's own edits.
- **Tests** — pytest unit suite in `tests/` covers pure-logic modules (no LLM/Ollama needed). Run with `python -m pytest`. See the Testing section below
- **Error handling in tools** — `@tool_error_handler` decorator (`server/error_handler.py`) for standardized responses
- **Action confirmation** — destructive tools (`execute_bash`; `fs_write`/`file_edit`; `memory` writes) are hard-gated by `LangGraphAgent._confirm_tool()` in BOTH `_execute_tools` and `_run_worker_loop`, before `tool.invoke()`. It must live client-side: the MCP server is a piped subprocess and can't prompt the terminal. Toggles `REQUIRE_BASH_CONFIRMATION` / `REQUIRE_WRITE_CONFIRMATION` (default true), `REQUIRE_MEMORY_CONFIRMATION` (default false); non-TTY auto-proceeds. The prompt is `Proceed? (y/N/a)` — `a` trusts that whole category (`bash`/`write`/`memory`, tracked in `_trusted_confirm_categories`) for the rest of the session so a multi-step task doesn't re-prompt; default-deny otherwise. **This client gate is a confirmation layer, not the last line of defense**: a separate server-side floor (`server/tools/safety/`, above) hard-blocks catastrophic commands and system-path writes regardless of confirmation, since the MCP server can be driven directly
- **Plan mode (enforced, user-toggled)** — the `/plan` command flips `client.plan_mode_active`; the agent reads it via a `plan_mode_provider` callback and `_is_blocked_by_plan_mode(tool_name, tool_args)` HARD-BLOCKS the mutating/exec tools (`_PLAN_BLOCKED_TOOLS` = execute*bash, fs*write, file_edit, git_safe, git_commit_safe, start_background_task) at both chokepoints, above `_confirm_tool` (so a blocked tool never even prompts). Three of those are **conditionally** allowed: `execute_bash` runs when the command is read-only (`_is_readonly_bash` — leading program in `_READONLY_BASH_CMDS`, no mutation operators in `_BASH_MUTATION_OPS`, git limited to `_READONLY_GIT_SUBCMDS`), and `fs_write`/`file_edit` are allowed only when writing the plan file (a `.md` under `paths.plans_dir()`, via `_is_plan_file`). Read-only tools + the `memory` notebook always pass. `client.query()` prepends a firm read-only reminder each turn (`_plan_mode_reminder` — "supersedes any other instructions", tells the model to call `exit_plan_mode` when ready). **Approval → execute handoff:** when the plan is ready the model calls the `exit_plan_mode(plan)` tool (`server/tools/plan_mode_exit.py`, a thin stub — in `_ALWAYS_AVAILABLE_TOOLS`); the agent intercepts it **client-side** at both chokepoints (`_handle_exit_plan_mode`) and drives an in-app approval prompt via `_plan_approval_ui` (`reader.plan_approval_ui`, modeled on `confirm_ui`: **y** = approve & run, **e** = edit in `$EDITOR`, **n** = keep planning). On approve, `_exit_plan_mode_provider` (`client._approve_plan`) flips plan mode off and persists the approved plan to `plans/plan*<ts>.md`, and a ToolMessage hands the approved plan back so the model executes it **in the same turn** (no manual `/plan`-off + re-prompt). Non-TTY/tests (hooks unset) auto-approve. **Pre-approved bash:** `exit_plan_mode(plan, allowed_bash=[…])` lets the plan pre-declare the shell commands it will run during execution; on approval `_handle_exit_plan_mode` stores them on `agent._preapproved_bash` and `_confirm_tool`'s bash branch auto-confirms a command that equals or prefix-matches one (`_is_preapproved_bash`), so an approved plan's expected commands don't re-prompt one-by-one (plan mode is off during execution, so they'd otherwise hit the `Proceed?` gate). Cleared on `/clear` and when plan mode is re-entered. **Execution route pin:** because routing re-classifies every turn independently and each route binds only its tool subset (`code`/`full` bind the write/exec tools; `knowledge`/`research`/`simple_qa` do NOT), an approved implementation plan whose execution turns re-classify as a read-only route would find `file_edit`/`fs_write`/`execute_bash` **unbound** and be unable to apply itself. So plan approval also sets `agent._execute_plan_route = True`, and `_effective_route` (consulted by `_get_route_model`/`_get_route_tools`) forces the **full** toolset regardless of the classifier for the rest of the task; `_route_after_classify` likewise sends those turns straight to the agent (not the orchestrator — the approved plan already IS the decomposition). Same plan-scoped lifetime as the pre-approved bash (cleared on `/clear` and plan re-entry). The legacy advisory JSON-workflow plan tools (`enter_plan_mode`/`add_plan_step`/…) were retired — this enforced `/plan` path is the only plan mode
- **Async/sync bridge** — MCP client uses a background thread with `asyncio.new_event_loop()` in `client/mcp_tool_wrapper.py`; sync callers use `run_coroutine_threadsafe`
- **Imports** — absolute (`from mnemoai.…`), at module top level, grouped stdlib → third-party → first-party and **alphabetized within each group** (enforced by `ruff check --select I .` — the CI import-sort gate). Do **not** add lazy/function-local imports unless genuinely necessary to break a real circular import; prove it first by importing the module at top level (`python -c "import mnemoai.client.agent.agent"`) — if it resolves, keep the import at the top. When adding a symbol from a module already imported (e.g. `mnemoai.utils.paths`), extend that existing top-level import group rather than re-importing inside a function
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
- **Unit tier (default):** deterministic, pure-logic tests — no LLM, Ollama, or network needed, runs in seconds. Covers `utils/bm25.py`, `client/agent/reasoning_utils.py`, `utils/formatting/` (response_parser, url_formatter, code_formatter), `client/agent/orchestrator.parse_subtasks`, `server/error_handler.py`, `server/tools/git_safety.py` (command-danger classification), `server/tools/file_edit.py` + `glob_search`, `execute_bash` timeout/process-group behavior, `client/memory/episodic_memory` heuristics, Bedrock/Mantle model wiring (`test_bedrock_endpoint.py`), vision content normalization (`test_vision_content.py`), and conversation compaction incl. token-aware retention (`test_conversation_compaction.py`).
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

Semver. The **public contract** (what a major bump protects) is: `config.yaml` keys (model sections, `ENABLE_*`/`REQUIRE_*` toggles, documented sections), the `mcp.json` `mcpServers` schema, the CLI slash-commands + `mnemoai` console command, and the `mnemoai-assistant` dist / `mnemoai` import name. Everything under `client/`/`server/`/`models/`/`utils/` not in that list is internal and may change freely. All changes go in `CHANGELOG.md`; releases follow the checklist in `docs/development.md`. CI (`.github/workflows/tests.yml`) runs the unit tier + import-sort on push/PR; the integration tier is run locally before releases.

## Known Limitations

- Unit tests cover pure logic only — agent/LLM integration paths still need manual verification (run the integration tier + the release checklist before releasing)
- MCP server is a subprocess — debugging requires attaching to child process or reading logs
- No Docker/containerization — runs directly on host with system Python/conda/venv
- No database — all persistence is file-based (JSON, FAISS index files, SQLite chunk cache)
- Single-user — profile name in config isolates data but no auth/multi-tenancy
- `ripgrep` (rg) required for `grep_search` tool — install separately
