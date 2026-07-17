# Changelog

All notable changes to **Mnemo AI** (PyPI: `mnemoai-assistant`) are documented
here. The format follows [Keep a Changelog](https://keepachangelog.com/), and
the project aims to follow [Semantic Versioning](https://semver.org/): until
1.0.0, minor versions may introduce features and occasional breaking changes;
from 1.0.0 on, breaking changes to the public surface (config keys, the
`mcp.json` schema, CLI commands, the package/CLI name) bump the major version.

## [Unreleased]

## [1.4.3] — 2026-07-16

### Fixed

- **A dropped/stalled stream connection no longer hangs the app.** When the
  laptop sleeps (or the network drops) mid-response, the TCP socket to the model
  dies silently; the streaming read had no idle bound, so on resume it blocked the
  single worker thread forever — the conversation looked stopped AND new messages
  froze behind it. The stream is now consumed through a **per-chunk idle-timeout
  watchdog** (`_iter_stream_with_idle_timeout`, `LLM.STREAM_IDLE_TIMEOUT`, default
  120s; 0 disables): if no chunk arrives within the window the wedged read is
  abandoned so the worker is never parked. The retry wrapper (`_stream_response`)
  then classifies idle-timeout + transient network errors (reset/broken-pipe/
  timeout/5xx/overload — `_is_transient_network_error`, provider-agnostic) as
  retryable and **re-runs the turn on a fresh connection with exponential
  backoff**; if all retries fail it ends with a clear "lost the connection — send
  your message again" message instead of crashing. The partial (including partial
  reasoning) is discarded — a dropped stream can't be resumed mid-generation — but
  the conversation continues, matching how Claude Code recovers. This lives in the
  agent stream layer, so it applies to **every** provider, not one client.

### Changed

- **`agent.py` housekeeping:** the class-level constants that were interleaved
  between methods (streaming error/network markers, self-reporting tools,
  confirmation categories, the ephemeral-block regex) are consolidated into one
  labeled block at the top of `LangGraphAgent`. Pure reorganization — no behavior
  change.

## [1.4.2] — 2026-07-16

### Fixed

- **A turn cut off by the output-token limit now auto-continues instead of
  dead-ending.** With `REASONING_EFFORT: max` on a large context, the model can
  spend its whole `MODEL_ID.MAX_TOKENS` output budget on reasoning plus a partial
  answer/tool call and stop mid-turn (`stop_reason: max_tokens`) with no completed
  tool call — the turn just ended and the user had to type "continue". `_call_model`
  now detects this (`_was_truncated_by_tokens`) and **auto-continues** via
  `_continue_truncated_turn`: it feeds the partial turn back with a "continue where
  you left off" nudge and re-streams, accumulating text, up to
  `LLM.MAX_OUTPUT_CONTINUE_RETRIES` (default 3). It stops early when a continuation
  finishes cleanly (returns the assembled answer) or emits a tool call (returned so
  the graph runs it, partial text preserved); resumed parts glue directly so a
  split word isn't corrupted by a spurious space. Only if retries exhaust with
  nothing usable does it surface the "increase MAX_TOKENS" message. This replaces
  the earlier behavior that ended the turn with a warning. Diagnosed from a real
  session (Claude Opus via Mantle, `REASONING_EFFORT: max`, ~666k-token context).

### Changed

- Quieter compaction: dropped the green "Token limit exceeded… compacting" and
  "evicted old tool output" status lines that printed mid-task — compaction now
  runs without narrating itself.

## [1.4.1] — 2026-07-16

### Fixed

- **An approved plan could not edit files when its execution turns re-classified
  as a read-only route.** Query routing re-classifies **every** turn, and each
  route binds only its tool subset — `code`/`full` bind `file_edit`/`fs_write`/
  `execute_bash`, but `knowledge`/`research`/`simple_qa` do not. So after
  approving a plan (e.g. from a docs-review question that classified as
  `knowledge`), the execution turns could land on a read-only route with the
  mutating tools unbound — the model would truthfully report it "had no write
  tools," then find them again on a turn that happened to classify as `code`,
  flip-flopping mid-task. Plan approval now sets `agent._execute_plan_route`,
  which forces the **full** toolset (via `_effective_route`, consulted by
  `_get_route_model`/`_get_route_tools`) and routes straight to the agent (not the
  orchestrator) for the rest of the task — so an approved plan can always apply
  itself. Same plan-scoped lifetime as the pre-approved bash (cleared on `/clear`
  and when plan mode is re-entered).

## [1.4.0] — 2026-07-16

### Added

- **Background memory auto-extraction (opt-in).** After each turn, if
  `ENABLE_MEMORY_AUTO_EXTRACTION` is on (default **off**), a background pass
  distills durable facts from the exchange and writes them to `MEMORY.md`
  automatically — the auto-learning counterpart to the model calling the `memory`
  tool itself. It runs on a daemon thread (never blocks the turn), makes one
  isolated model call that doesn't touch the conversation state
  (`client._invoke_model_once`), and applies the returned JSON `add`/`replace`
  ops through `MemoryStore` (so the char cap + dedup still apply). It's gated
  behind its own toggle because, unlike the tool path, it writes **without** a
  confirmation prompt — though it can only add/consolidate entries in `MEMORY.md`.
  New `MEMORY_EXTRACTION_PROMPT` (four-kind taxonomy + "what not to save"), and
  `client.auto_extract_memory` wired at turn end. Storage is unchanged (still one
  bounded flat file).

### Changed

- **New prompts reach existing installs.** `Config._load_prompts` now layers the
  bundled package `prompts.yaml` underneath the user's file (user keys win,
  missing keys fall back to the bundle). A user's `prompts.yaml` is seeded once
  and never overwritten, so previously a prompt added in a release wouldn't
  resolve on an existing install; now it does, while any prompt the user
  customized still takes precedence. (This is what lets the new
  `MEMORY_EXTRACTION_PROMPT` work on upgrade without re-copying the file.)

## [1.3.0] — 2026-07-16

### Added

- **Tool-result eviction — a cheaper compaction layer that runs before the LLM
  summary.** When history crosses the high-water mark, the agent now first shrinks
  the bodies of **old** tool results (the grep/read/web dumps outside the recent
  window that carry most of the context but are rarely needed verbatim after the
  model has acted on them) to a short head plus an eviction marker — with **no
  model call**. If that alone brings the estimate back under budget, the expensive
  full summarization is skipped entirely. Recent turns stay verbatim and **no
  message is dropped** (only content is trimmed), so tool-call/result pairing is
  preserved and no provider rejects the next turn. New method
  `AgentConversationManager.evict_old_tool_results`, wired as the first pass in
  `client._compact_now`. Config: `LLM.TOOL_EVICTION_KEEP_RECENT` (default 8) and
  `LLM.EVICTED_TOOL_RESULT_CHARS` (default 500; 0 disables the layer). This makes
  compaction layered — cheap eviction first, full summary only when eviction isn't
  enough — the biggest structural improvement to the context handling reworked
  across 1.2.x.
- **Skills: optional `argument_hint` frontmatter + a bounded tier-1 listing.** A
  skill's `SKILL.md` frontmatter may now carry an `argument_hint` (a short phrase
  naming what the skill expects, e.g. "a PR number"); it renders in the always-on
  `<available_skills>` listing as `(expects: …)` so the model knows what to gather
  before calling `use_skill`. The whole listing is also now bounded in total size
  (`_MAX_LISTING_CHARS`, ~4000 chars): with a large skills library the overflow
  collapses into a `… (+N more — see /skills)` line, so many installed skills can't
  dominate the system prompt injected every turn. The bundled skill-creator skill
  documents the new key.
- **Plan mode: pre-approved shell commands.** `exit_plan_mode` now takes an
  optional `allowed_bash` list — the commands the plan will run during execution
  (tests, builds, installs). Approving the plan pre-approves them, so during
  execution they run **without** the per-command `Proceed?` confirmation (a command
  auto-confirms when it equals or prefix-matches an entry, e.g. `pytest` pre-approves
  `pytest tests/unit`); anything else still prompts. The pre-approvals are scoped to
  the approved plan — cleared on `/clear` and whenever plan mode is re-entered. The
  plan-mode reminder now tells the model to declare these commands.

### Changed

- **Bundled examples reach existing installs on upgrade.** The `*.example`
  reference files (`config.yaml*.example`, `mcp.json.example`) are now refreshed
  from the package when they differ (they're read-only reference, never loaded as
  config), so a newly-documented key — e.g. the eviction knobs above — appears in
  an already-installed user's examples. Bundled example skills additionally get an
  in-place `SKILL.md` refresh when the installed copy is **pristine** (byte-identical
  to a version we shipped, tracked by hash), so doc/frontmatter updates (e.g. the
  new `argument_hint`) reach existing installs — while a user-edited skill is never
  touched. The live `config.yaml` / `mcp.json` / `prompts.yaml` and user-authored
  skills are still only created when absent and never overwritten.
- **Curated memory: a four-kind tagging convention.** The `memory` tool now guides
  the model to tag each entry with its kind — `[user]` (who the user is),
  `[feedback]` (how to work, with the why), `[project]` (ongoing work/constraints,
  absolute dates), `[reference]` (external pointers) — and sharpens the "what not to
  save" rule to exclude anything the repo/git/CLAUDE.md already records. Storage is
  unchanged (still one bounded `MEMORY.md`); this is prompt guidance that yields
  more structured, higher-signal, less redundant entries.

## [1.2.2] — 2026-07-16

### Fixed

- **Context-window overflow root cause: the token counter undercounted non-OpenAI
  models.** A single tiktoken (OpenAI) encoder was used for every provider, which
  undercounts Claude ~1.5x on code/JSON history — so the compaction high-water
  mark never fired (the agent believed it was under the window when it was ~15%
  over) and the prompt overflowed the API. Two fixes:
  - **Provider-aware, never-undercount estimate** (`utils.tokenization`): an
    o200k_base basis scaled by a conservative per-provider multiplier
    (anthropic/mantle ×1.5, bedrock/litellm/sagemaker ×1.35, openai exact ×1.0;
    overridable via `LLM.TOKEN_COUNTING.<TYPE>_MULTIPLIER`). Ollama uses a
    chars/ratio (`OLLAMA_CHARS_PER_TOKEN`, default 3.0).
  - **Provider's exact count as ground truth**: the agent now captures
    `usage_metadata.input_tokens` from each response and the compaction high-water
    check + the `[Context: N]` display use `max(estimate, actual)`, so the real
    context size drives the decision (reset on `/clear` and after compaction).

### Changed

- Consolidated token counting onto the single shared `utils.tokenization.count_tokens`
  (removed the conversation manager's duplicate tiktoken encoder) and added a
  small per-message overhead for role/formatting wrappers.

## [1.2.1] — 2026-07-15

### Fixed

- **Compaction no longer overflows the summarization call on a near-full, large
  context window.** The batched summarizer (added in 1.0.2) set the per-call
  budget to 50% of the window — on a 1M-token model that meant a ~500k-token
  batch, which together with the rolling summary + prompt + reasoning output
  exceeded the window and 400'd ("could not compact"), so a full history couldn't
  be shrunk. The per-call budget is now a conservative fraction of the window
  (`_SUMMARY_CALL_FRACTION`, 0.15), the rolling summary carried between batches is
  capped, and a single message larger than the budget is truncated (head+tail) so
  it can't overflow its own batch. Verified against a real 539-message / ~750k-token
  conversation: batches drop from one 498k-token call to five ≤150k-token calls.

## [1.2.0] — 2026-07-15

### Added

- **File edits and writes render as styled diff/content blocks** (pinned UI),
  instead of a flattened `↳ file_path=… ↳ old_string=…` line:
  - `file_edit` and `fs_write` `str_replace` show an `Update(path)` header with a
    red `-` / green `+` diff of the change.
  - `fs_write` `create` shows a `Create file` header with the new content as
    numbered lines; `insert`/`append` show the added text as green `+` lines.
  - Home paths are shortened to `~/…`. Both the live turn and the `/load` replay
    use the same renderer, so restored conversations look identical.

## [1.1.6] — 2026-07-15

### Fixed

- **A large crawled page is now retrievable on the research route.** When
  `web_crawler` fetches a page over `RAG.MAX_TOKENS`, it ingests the content into
  the document store instead of returning it inline — but the `research` route
  bound no RAG retrieval tool, so the model got a "content ingested" pointer it
  couldn't follow and fell back to answering from memory. The route now also
  binds `search_in_documents` + `list_documents`, and the crawler's ingest result
  explicitly instructs the model to retrieve via `search_in_documents(query=...)`
  rather than answer from memory.
- **Approved plan files no longer accumulate.** `plans/plan_<ts>.md` (written on
  each plan approval) were never cleaned up. A startup sweep (`sweep_old_plans`)
  now prunes `plan_*.md` older than 7 days (`PLAN_MAX_AGE_DAYS`, 0 disables) and
  removes the stale `current_plan.json` left by the retired legacy plan tools.
  Recent plans are kept (they survive compaction and stay re-readable).

## [1.1.5] — 2026-07-14

### Fixed

- **Launch-banner tagline is centered under the wordmark again.** The
  "local agentic AI assistant · learns & remembers" line was centered to the
  command-box width (which widens to fit the longest command row), so it drifted
  right of the ASCII wordmark whenever the box was wider than the banner. It now
  centers to the banner's own width.

## [1.1.4] — 2026-07-13

### Fixed

- **Background-task log files no longer accumulate forever.** Each background
  task writes a `~/.mnemoai/tasks/{id}.log`; the only cleanup
  (`clear_completed_tasks`) iterated the in-memory task registry, which is empty
  on every restart — so `.log` files from prior sessions were unreachable and
  piled up indefinitely. A startup sweep (`sweep_old_task_logs`) now prunes task
  `.log` files older than 7 days (`TASK_LOG_MAX_AGE_DAYS`, 0 disables). It only
  touches `*.log` files and is best-effort (never blocks startup); recent logs
  are kept so `get_task_output` still works for active tasks.

## [1.1.3] — 2026-07-13

### Fixed

- **The MCP stderr log (`~/.mnemoai/logs/mcp.log`, added in 1.1.2) no longer
  grows without bound.** It's now size-rotated: when it reaches ~1 MB
  (`MCP_LOG_MAX_BYTES`) it's rotated to `mcp.log.1` (one backup generation,
  replaced not stacked) and a fresh log starts, bounding on-disk use to ~2 MB.
  Rotation is best-effort (never blocks startup). New `paths.open_mcp_log()`
  helper owns the rotation policy.

## [1.1.2] — 2026-07-13

### Fixed

- **MCP server subprocess stderr no longer leaks into the terminal.** External
  MCP servers (and the built-in one) are spawned via `stdio_client`, whose
  `errlog` defaulted to the parent's stderr — so a noisy server (e.g. the
  `npm notice` lines from an `npx`-launched server) wrote straight to the
  terminal, bypassing the pinned UI. Their stderr is now routed to
  `~/.mnemoai/logs/mcp.log`: the terminal stays clean while the output remains
  available for debugging a server that fails to start. New `paths.logs_dir()`
  and `paths.mcp_log_path()` helpers.

## [1.1.1] — 2026-07-13

### Fixed

- **Malformed extended-thinking blocks no longer 400 the whole request.** An
  Anthropic (Claude) request rejects wholesale with
  `messages.N.content.M.thinking.thinking: Field required` when a `thinking`
  content block reaches the API with empty/missing inner text — which can enter
  history via a cut-short stream or a mid-session model switch. The message
  sanitizer (already run before every model call) now drops **only** such
  provably-invalid thinking blocks, leaving healthy content and tool pairs
  intact. Scoped precisely to the malformed-Anthropic shape: non-Anthropic
  histories (Bedrock `reasoning_content`, OpenAI Responses `reasoning`, Ollama /
  LiteLLM string reasoning, plain text) pass through as the same object,
  untouched.

## [1.1.0] — 2026-07-10

### Added

- **`STEERING.md` — user-authored always-on instructions.** The user's
  counterpart to the agent-curated `MEMORY.md`: write build/test commands, code
  conventions, and "always do X" rules that the assistant follows every turn.
  Discovered hierarchically — a global `~/.mnemoai/STEERING.md` plus a project
  `./STEERING.md` found by walking up to the repo root, combined broadest→most
  specific. Injected as a leading instruction block **re-read from disk each
  turn** (edits apply immediately) and **never summarized by compaction** (it's
  re-injected verbatim, so long sessions never dilute it). No config toggle — the
  file's presence is the switch; no file → nothing injected.
- **`steering-creator` bundled skill.** Ask the assistant to "create a
  STEERING.md for this project" (or "document how to work in this repo") and it
  investigates the codebase and writes a well-formed file following best
  practices; "improve my STEERING.md" refines an existing one.

### Changed

- **Bundled example skills now seed per-skill (reaches existing installs on
  upgrade).** `seed_example_files()` previously seeded skills only when the
  `skills/` dir was entirely empty, so a newly-bundled skill (like
  `steering-creator`) never reached a user who already had skills. It now copies
  each bundled skill whose own directory is absent — matching how the config
  `*.example` files already seed — so new bundled content reaches existing
  installs, without ever clobbering a user's own skills. Codified as a standing
  convention in `CLAUDE.md` (new features must reach existing installs, not just
  fresh ones).

## [1.0.2] — 2026-07-10

### Fixed

- **A context-window overflow no longer abandons the in-flight task.** When a
  turn exceeded the model's context (e.g. after a large paste or file read), the
  backstop compacted history but returned a terminal "I stopped this turn — try
  again" message, dropping the work in progress (you had to re-ask). Now
  `_stream_once` raises a typed overflow, and `_call_model` compacts and
  **re-invokes on the shrunken prompt in the same turn**, so the task continues.
  A terminal message is returned only if compaction can't shrink history further
  or the prompt still overflows after compacting (no retry-storm). The worker
  loop and aggregator degrade gracefully on the same signal.
- **Summarization no longer silently loses history when it overflows.** The
  compaction summary was generated in a single model call; if the older history
  itself exceeded the context window the call 400'd and the summary degraded to
  a content-free placeholder ("Previous conversation covered multiple topics"),
  discarding hundreds of messages. Summarization is now **batched (map-reduce):**
  older messages are split into window-sized batches folded into a rolling
  summary, so the summary call never itself overflows. If every batch still
  fails, it keeps a **bounded excerpt** of the real history rather than a
  placeholder.

## [1.0.1] — 2026-07-09

### Fixed

- **Mid-turn log warnings no longer corrupt the pinned prompt.** The log handler
  captured the original `sys.stderr` at import, bypassing the pinned UI's
  `patch_stdout` (which swaps `sys.stderr` for a proxy that renders above the
  input). A warning fired during a turn (e.g. an embed retry) printed on top of
  the `(esc to cancel)` status line. The handler now resolves `sys.stderr` at
  emit time, so logs render cleanly in scrollback during a turn and to real
  stderr otherwise.
- **Embedding a very long input no longer spams retries or silently degrades.**
  Oversized text (e.g. a large paste) could exceed the embed model's context and
  fail. Fixes, all provider-agnostic (applied at the shared truncation point, so
  Ollama/OpenAI/Bedrock/SageMaker/LiteLLM all benefit):
  - a **token-aware input cap** resolved from `RAG.EMBED_MODEL_ID.MAX_INPUT_TOKENS`
    (new, optional) → else the model's own context window (Ollama probe, with a
    0.9 safety margin) → else a conservative default (8192);
  - `truncate=True` on the Ollama embed call as a runner-side backstop;
  - **deterministic context-overflow errors skip the retry loop** (retrying the
    identical input can't help) while genuinely transient EOFs still retry.

## [1.0.0] — 2026-07-09

First stable release. The public contract is now frozen under semver: breaking
changes to `config.yaml` keys, the `mcp.json` `mcpServers` schema, the CLI
slash-commands + `mnemoai` console command, and the `mnemoai-assistant` dist /
`mnemoai` import name will bump the major version.

### Changed

- **README demo is now a link instead of an inline GIF.** GitHub serves the
  10.5 MB `.gif` as `application/octet-stream`, which PyPI's image proxy refuses,
  so the demo rendered as a broken image on the PyPI project page. It now links
  to the GIF (renders inline on GitHub, resolves cleanly on PyPI).

### Fixed

- **Integration-test config gate now mirrors the real loader.** The integration
  tier keyed its skip guard off a single hardcoded `src/mnemoai/utils/config.yaml`
  path, out of sync with the app's actual resolution (`$MNEMOAI_CONFIG` →
  `~/.mnemoai/config/config.yaml` → legacy → package-relative). It now reuses
  `Config._resolve_config_path()`, so `python -m pytest -m integration` runs
  from the installed config with no env-var or file-copy workaround.

## [0.19.1] — 2026-07-08

### Changed

- Recompressed the demo GIF (`images/assistant-demo.gif`, ~19 MB → ~10 MB) so the
  README loads faster on PyPI and GitHub. No code changes.

## [0.19.0] — 2026-07-07

### Added

- **Plan mode style refactor → execute handoff.** When its
  plan is ready the model calls a new `exit_plan_mode(plan)` tool
  (`server/tools/plan_mode_exit.py`, a thin stub in `_ALWAYS_AVAILABLE_TOOLS`);
  the agent intercepts it client-side (`_handle_exit_plan_mode`) at both tool
  chokepoints and shows an in-app approval prompt (`reader.plan_approval_ui`,
  modeled on the confirmation gate): **y** approves and runs, **e** opens the
  plan in `$EDITOR` to tweak before re-review, **n** keeps planning (stays
  read-only). Approving flips plan mode off, persists the approved plan to
  `plans/plan_<ts>.md`, and hands it back so the model executes it **in the same
  turn** — no more manual `/plan`-off + "now do it" re-prompt. Non-TTY/tests
  auto-approve. The plan is shown as a **bordered, word-wrapped, markdown-aware
  block** (`turn_view.render_plan` — bold headings, cyan code spans, hanging
  list indents) above the approval prompt, instead of a flattened
  `↳ plan=…` tool-arg line.

### Fixed

- **Confirmation & plan-approval prompts cleaned up.** The pinned
  bottom-of-screen confirm line embedded the full (often multi-line) command
  detail, which prompt_toolkit clipped to its first line — rendering a confusing
  truncated `▶ Run shell command?  python3 -c "` under a duplicated options hint.
  Now the pinned line is compact (`▶ <question>   [y = yes · n = no · a = allow
all]`), the full command/plan echoes to scrollback above it just once, and the
  pinned line styles the question in the accent color with the `[y · n · a]`
  keys dimmed — so the eye separates the prompt from the actionable keys. Applies
  to both the shell/file/memory confirmation and the plan-approval prompt.

### Removed

- **Retired the legacy advisory plan-mode JSON-workflow tools**
  (`enter_plan_mode`, `add_plan_step`, `add_plan_file`, `add_plan_risk`,
  `present_plan`, `approve_plan`, `get_plan_status`, and the old
  `exit_plan_mode(cancel)`). They were bookkeeping only — never wired to
  enforcement — and the old `exit_plan_mode` collided with the new tool
  (`Tool names must be unique` → a 400 loop). The enforced `/plan` path is now
  the only plan mode.

## [0.18.3] — 2026-07-04

### Fixed

- **Force-quit now takes two Ctrl+C presses (not one), and restores the
  terminal.** The 0.18.2 escalation fired on the first Ctrl+C after an Esc
  cancel (it keyed off `_cancelled`, which Esc had already set), and `os._exit`
  left the tty in raw/no-echo mode so the shell looked frozen after exit. Now a
  dedicated per-turn flag requires a genuine second Ctrl+C (with a
  `(press Ctrl+C again to force-quit)` hint on the first), and `_force_quit`
  restores cooked+echo mode via termios (with an `stty sane` fallback) before
  exiting.

## [0.18.2] — 2026-07-04

### Fixed

- **A second Ctrl+C now force-quits when a cancel is stuck.** Pressing Esc
  cancels a turn by injecting `KeyboardInterrupt` into the worker thread, but
  that only lands at the next Python bytecode boundary — a worker blocked inside
  a long native tool call (e.g. a broad `glob_search`) stayed on `(cancelling…)`
  and Ctrl+C was a no-op (it just re-requested the already-pending cancel). A
  second Ctrl+C while cancelling now force-quits via `os._exit` (restoring the
  terminal first), bypassing the `asyncio` executor join that would otherwise
  hang on the wedged thread.

## [0.18.1] — 2026-07-04

### Changed

- **`fs_write` no longer takes `dry_run`/`confirmed` — it writes directly.** The
  old two-step dance (preview with `dry_run=True`, then re-call with
  `dry_run=False, confirmed=True`) is redundant now that the client-side
  confirmation gate (`REQUIRE_WRITE_CONFIRMATION`) asks the user before every
  write. Dropping it removes a whole extra round-trip (and the preview's tokens)
  per file operation. The server-side system-path safety floor is unchanged.

## [0.18.0] — 2026-07-04

### Added

- **Context-overflow protection (3 layers) — a runaway tool result no longer
  loops the agent on a "prompt is too long" 400.** A single oversized tool
  result (e.g. `grep_search` with a huge `max_results`) used to enter context
  whole, and every subsequent model call 400'd while the graph retried the same
  oversized prompt to the recursion limit. Now:
  1. **Tool results are capped at the source** (`tool_formatting.truncate_tool_result`)
     — each result is bounded head+tail with a truncation note at both tool-exec
     chokepoints, so one result can't alone overflow the window.
  2. **Pre-flight compaction** — before a turn, history over the high-water mark
     is compacted first (reusing the existing summarize-and-keep-recent manager).
  3. **Overflow backstop** — a context-window 400 is caught, force-compacted, and
     the turn ends with a clear message instead of re-sending the oversized prompt.
     Both size knobs **auto-derive from `MAX_CONVERSATION_TOKENS`** (the model's
     context window) so they scale per-model: `MAX_TOOL_RESULT_CHARS` → 10% of the
     window in chars, `COMPACT_HIGH_WATER_TOKENS` → 80% of the window. Either can be
     set explicitly (0 disables). Documented in the `config.yaml.*` examples and docs.

### Changed

- Removed references from code comments/docstrings and config
  example comments (retained only in `CLAUDE.md`/`ARCHITECTURE.md`).

## [0.17.5] — 2026-07-03

### Fixed

- **Ollama embeddings no longer silently degrade to deterministic fallback on a
  transient runner EOF.** The llama.cpp embedding runner can EOF intermittently
  (first call after a (re)load / under memory pressure) even for input it embeds
  fine on the next try — e.g. right after pasting a large document. `_embed_ollama`
  now retries (default 3×, `RAG.EMBEDDINGS.RETRIES` / `RETRY_DELAY`) before falling
  back, and caps per-text input as hygiene (`RAG.EMBEDDINGS.MAX_INPUT_CHARS`,
  default 200000, 0 disables) so a genuinely huge text can't OOM the runner.

## [0.17.4] — 2026-07-03

### Fixed

- **Orchestrated ("full"-route) turns now render tool calls with the styled
  block, not the old `[⚙ …]` marker.** `_run_worker_loop` printed the plain
  marker unconditionally, so a single conversation could mix the new
  `ToolName ↳ arg=value` blocks with the old style. Both tool-exec chokepoints
  now share one `_print_tool_marker` helper that honors `styled_turn_view`.

## [0.17.3] — 2026-07-03

### Fixed

- **Option+←/→ no longer cancels the running turn while typing a queued message.**
  The bare-Esc cancel binding was `eager`, so it fired on the `ESC` prefix of
  Option+←/→ (`ESC b` / `ESC f`) before the letter arrived — cancelling instead
  of moving the cursor. It's now non-eager, so word-motion works while typing
  (idle or mid-turn) and a lone Esc still cancels.
- **A cancelled turn now shows `⊘ Stopped`.** The transient `(cancelling…)` line
  previously never resolved; the cancelled response is now surfaced as a final
  stopped line.

## [0.17.2] — 2026-07-03

### Fixed

- **macOS Option+←/→ now jump by word instead of typing `b`/`f`.** Option+←/→
  send `ESC b` / `ESC f` (backward/forward-word), but the pinned input's
  `eager=True` bare-Esc binding consumed the `ESC` prefix, leaving the `b`/`f` to
  self-insert. The eager Esc-to-cancel is now gated to while a turn runs; at the
  idle prompt Esc stays a prefix so the default word-motion bindings fire.

## [0.17.1] — 2026-07-03

### Changed

- **The `/load` picker now labels each conversation with an auto-derived title**
  (the first real user message, skipping injected episodic-memory / plan-mode
  context) instead of the raw `conversation_<timestamp>.json` filename.
  No LLM call: the title is read verbatim from the saved file's first user message
  (filename shown as fallback when there's none).

## [0.17.0] — 2026-07-03

### Added

- **`/load` now replays the conversation to the screen.**.
  Previously a loaded conversation was restored into context but
  nothing was shown. It now prints the full transcript to scrollback in the
  styled view — user prompts, collapsed `Thought…` reasoning blocks,
  `ToolName ↳ arg=value` tool-call blocks, and `●`-marked answers — before the
  `[Context: N tokens]` line. New pure `turn_view.render_conversation` builder;
  tool _results_ are omitted (the call block is enough), matching the live UX.

### Changed

- **Documentation refresh.** Reorganized the docs site (`docs/`, `mkdocs.yml`)
  and moved the per-file reference `ARCHITECTURE.md` to the repo root (with the
  matching link update in `CLAUDE.md`). Docs-only — no code or behavior change.

## [0.16.1] — 2026-07-03

### Fixed

- **Spinner no longer disappears during the silent stretch before a tool batch.**
  On the Responses/Bedrock block protocols the streaming callback's `token` is the
  repr of a content-block list, so a reasoning block (`{'id':'rs_…'}`) or `[]` is a
  truthy string that `on_llm_new_token` mistook for visible answer text and stopped
  the spinner — leaving a dead pause until the next turn. The handler now inspects
  the chunk's actual blocks and stops only on a non-empty `text` block.

## [0.16.0] — 2026-07-03

### Added

- **Back navigation in the config wizards (`/config`, `/model`, `/params`).** Each
  step now shows a **Back** button (and **Ctrl+B**) that returns to the previous
  field, keeping what you already entered; Esc still cancels the whole flow. A
  small step-runner (`_run_steps`) drives the sequences and rewinds cleanly on a
  Back (restoring the pre-step text so a re-answer replaces, not stacks). Non-TTY
  runs are unchanged.

### Changed

- **Configurator no longer prints mid-flow noise on a TTY.** Section headers and
  inline notes (`-- Chat model --`, "Vision disabled", "Configuring for…", the
  inference-reset note, …) were wiped by the next full-screen dialog anyway; they
  are dropped on a TTY. Only what's useful after the in-place restart survives to
  scrollback — the final "Config written / Updated / No changes / Cancelled"
  outcome and the provider's env-based auth note — plus the unchanged non-TTY
  fallback banners.

### Fixed

- **Reasoning now surfaces on the `bedrock` provider with newer Claude (Sonnet 5,
  Opus 4.6+).** Three bugs in the Bedrock path, all only reachable on the newer
  models: (1) `_initialize_bedrock_model` gated thinking on `REASONING` alone, so
  `REASONING_EFFORT` by itself sent no thinking directive and no reasoning came
  back; (2) it hardcoded the old `thinking={"type":"enabled",…}` form, which these
  models reject with `"thinking.type.enabled" is not supported … use adaptive`
  (a hard `ValidationException`) — it now reuses the version-aware
  `_anthropic_thinking_kwargs` (adaptive + `output_config.effort`), matching the
  Mantle/Anthropic paths; (3) the reasoning extractor didn't recognize the Bedrock
  Converse `reasoning_content` block shape (`{"type":"reasoning_content",
"reasoning_content":{"text":…}}`), so even when text was returned it was dropped.

## [0.15.0] — 2026-07-02

### Added

- **Live "stream-then-collapse" reasoning in the pinned UI.** Reasoning now
  streams live inside a green-bordered **"Thinking… (Ns)"** block in the
  transient region, then commits to scrollback as the collapsed **"Thought for
  Ns…"** block once the answer begins. Implemented with a thread-safe
  `ReasoningStatus` sink (`client/ui/turn_view.py`) the agent's worker thread
  appends to and the reader renders; long lines wrap to the terminal width.

### Fixed

- **Spinner no longer freezes during a large `fs_write` (or any big tool
  argument).** Tool-call argument tokens stream as many non-empty tokens with no
  visible content; the streaming callback used to stop the spinner on them and
  never restart it, so the UI looked frozen for the whole write. It now ignores
  tool-argument tokens (detected via the chunk's `tool_call_chunks`) and stops
  only on real answer text.
- **No more "dead pause" while a reasoning model thinks.** In the styled UI
  reasoning is buffered, so the spinner must keep running through it; it was
  stopping on the first reasoning chunk and leaving the terminal blank.
- **The `●` answer marker and streamed answer no longer disappear.**
  `patch_stdout` only commits a line to scrollback on a newline; the streamed
  answer had no trailing newline and was erased by a pinned-UI repaint before
  `[Context: …]` printed. The answer line is now newline-terminated immediately,
  and the `●` marker is prepended to the first answer chunk (one committed line)
  rather than a separate write that could be erased on its own.
- **Ctrl+C on a non-empty input clears the line** instead of leaving the typed
  text; an empty line still needs a second Ctrl+C/Ctrl+D to exit.

### Changed

- Condensed docstrings and inline comments across `client/`, `models/`, and
  `utils/configurator.py` — one-line docstrings, comments only where intent
  isn't obvious; no behavior change.

## [0.14.0] — 2026-07-02

### Changed

- **Pinned-input terminal UI is now the default on a
  TTY.** The `>` input stays fixed at the bottom of the terminal while a query
  runs; the answer, a green-bordered **"Thought for Ns…"** reasoning block, and
  styled **`ToolName` + `↳ arg=value`** tool blocks stream _above_ it in native
  scrollback (wrapping, scrollback, and copy/paste preserved). Built on a
  persistent, non-full-screen prompt_toolkit `Application` with the query on a
  worker thread and output routed via `patch_stdout`. New behaviors:
  - **Esc / Ctrl+C cancels the in-flight turn** (interrupts the worker; the turn
    ends with "Operation was cancelled."); when idle, Ctrl+C/Ctrl+D twice exits.
  - **Input queue:** typing during a running turn shows a dim `> … (queued)` line
    and runs FIFO after the current turn — never a concurrent query. Each queued
    line is echoed to scrollback (paired with its own answer) only when it starts.
  - **Confirmations** (`Proceed?` for bash/writes) are answered with an in-app
    `y`/`n`/`a` keypress.
  - **Dialog commands** (`/load`, `/config`, `/model`, `/params`, `/memory clear`)
    briefly drop the pinned app to show their normal full-screen dialogs, then
    relaunch — and no longer print the redundant banner / "Current setup"
    overview to scrollback (only the final outcome line remains; `/config`'s
    overwrite caveat now rides in the confirmation prompt).
  - The animated status shows `⠙ Thinking… (esc to cancel)` with cycling dots.
  - **Non-TTY sessions** (pipes / CI / tests) keep the plain `input()` loop.

### Removed

- The previous **app-per-prompt** input reader (`PromptReader`) and its `Ctrl+O`
  tool-call panel — the styled tool blocks now show full arguments inline, so the
  panel is redundant. Also removed the associated `VDISCARD`/termios handling and
  the now-unused `last_turn_tool_calls` capture.

## [0.13.1] — 2026-07-01

### Fixed

- **`/model` and `/params` now show the "Current setup" overview inside the
  selection dialog.** The chat/vision/embeddings summary was printed to
  scrollback and then hidden by the full-screen model-picker dialog, so it was
  invisible exactly when you needed it. It's now rendered in the dialog body
  above the choices (and still printed to scrollback for non-TTY runs).
- **Spinner no longer looks frozen while the model streams a large tool-call
  argument.** A tool call (e.g. `fs_write` with a big document as `file_text`)
  streams its arguments as many content-less chunks; after reasoning had already
  stopped the "Thinking" spinner, that long silent stretch presented a frozen
  terminal. The stream loop now re-raises a "Preparing tool call" spinner while
  tool-call-argument chunks are flowing (stopped as soon as the tool marker
  prints), so a large write never looks stuck.

## [0.13.0] — 2026-07-01

### Added

- **Server-side safety policies (`server/tools/safety/`).** The catastrophic-command
  and system-path limits the tool docstrings advertise are now _enforced inside the
  MCP server_, not only by the client's confirmation gate — so they hold even if the
  server is driven directly.
  - `bash_policy.classify_shell_command` blocks a small, curated set of
    irreversible, system-destroying commands: recursive force-deletes of a
    root/home target (`rm -rf /`, `rm -rf ~`, `rm -rf /*`), filesystem creation
    (`mkfs*`), raw-device overwrite (`dd of=/dev/…`, `shred`/`wipefs` on a device),
    power-state changes (`shutdown`/`reboot`/`halt`/`poweroff`, `init 0/6`), and the
    classic fork bomb. Enforced in **`execute_bash`** and **`start_background_task`**
    (one shared policy). Ordinary scoped mutations (`rm -rf build/`,
    `git reset --hard`) are intentionally **not** blocked here — they remain gated
    by the client confirmation prompt.
  - `path_policy.classify_write_path` blocks writes into critical system
    directories (`/`, `/etc`, `/bin`, `/usr`, `/boot`, `/dev`, `/System`,
    `/Library`, …; `..` escapes are normalized first). Enforced in **`fs_write`**
    and **`file_edit`**. Home, project, temp (incl. the macOS `/private/var/folders`
    temp dir), and app paths remain writable.
  - Both are pure, config-independent functions with direct unit tests
    (`tests/unit/test_safety_policies.py`).

### Changed

- **Token counting lives in `utils/tokenization.py`.** `count_tokens` moved to a
  neutral module that both the client and the server import, removing the
  `client.py → server.tools` import that crossed the MCP layer boundary.
  `ToolManager.count_tokens` now delegates to it (single implementation). The
  encoder is created lazily, so importing the module has no side effects.

### Fixed

- **Duplicate RAG/chunk-cache flush** in `LangGraphClient.clear_context()` — the
  flush block was executed twice; removed the redundant copy.

### Internal

- **`agent.py` decomposition (pure helpers).** Extracted the stateless logic out of
  `client/agent/agent.py` (2120 → ~1685 lines) into four focused, independently
  testable modules, with `LangGraphAgent` keeping thin delegating methods so its
  public surface is unchanged:
  - `client/agent/message_codec.py` — Strands↔LangChain message conversion.
  - `client/agent/message_sanitizer.py` — orphaned tool-call/result pair repair.
  - `client/agent/plan_policy.py` — plan-mode block decision, read-only-bash
    heuristic, and the associated data tables.
  - `client/agent/tool_formatting.py` — tool-call marker rendering, ctrl+o capture,
    arg-normalization, and tool-error translation.
    No behavior change; all existing unit tests pass unchanged. The heavier
    streaming / tool-execution / orchestration paths intentionally stay in `agent.py`
    until they have unit coverage.
- `requirements.txt`: collapsed the redundant `mcp` / `mcp[cli]` pins into a single
  `mcp[cli]>=1.26.0`, and dropped `uv` from runtime deps (it's an install/dev tool,
  not imported at runtime).

## [0.12.0] — 2026-07-01

### Added

- **Richer terminal UI (prompt_toolkit).** The input prompt is now a
  prompt_toolkit reader with slash-command completion, history, a plan-mode tag,
  and **`Ctrl+O`** — a toggle panel that shows the last turn's tool calls with
  their **full, un-elided arguments** (the inline `[⚙ …]` marker still elides
  long commands/paths to keep the stream readable; `Ctrl+O` reveals them in
  full). The panel is ephemeral (press again to hide) and leaves nothing in
  scrollback. The design is **app-per-prompt**: prompt_toolkit runs only while
  reading a line, and the answer then streams with ordinary `print()` — so
  markdown, the `●` marker, line-wrapping, native scrollback, and copy/paste all
  work exactly as before. Non-TTY runs (pipes / CI) transparently fall back to
  plain `input()`.
- **Dialog-driven configuration.** `/load` now offers an arrow-key picker of
  saved conversations, and `/config`, `/model`, `/params` present their prompts
  as full-screen dialogs (text, yes/no, and single-choice lists). Consistent
  keys throughout: **Enter confirms, Esc cancels** (cancelling aborts the flow
  with nothing written). Text fields prefill the current value when you're
  editing an existing setting; a fresh model name is offered as a _suggestion_
  (empty field) rather than a prefilled default; **mandatory fields won't advance
  while empty**. Everything degrades to the previous `input()` prompts when not a
  TTY.

### Changed

- **`DOC_MAX_TOKENS` is now derived as 25% of `MAX_CONVERSATION_TOKENS`**
  automatically in `/config`, `/model`, and `/params`, so the document read cap
  scales with the context window instead of being tuned by hand.

## [0.11.3] — 2026-06-30

### Fixed

- **Ollama embeddings now honor the configured `HOST`/`PORT`** instead of
  silently following `$OLLAMA_HOST`. `_embed_ollama` called the bare
  `ollama.embed()`, whose host comes from the `OLLAMA_HOST` env var (or the lib
  default) — so a stray `OLLAMA_HOST` pointing at another server (e.g. a
  `llama-swap` port left over from a local-engine experiment) hijacked the embed
  request, which 400'd and degraded _silently_ to the sha256 fallback vectors
  ("semantic search will be DEGRADED"). It now builds an explicit
  `ollama.Client(host=…)` from `RAG.EMBED_MODEL_ID.HOST`/`PORT` (default
  `localhost:11434`), matching how the LLM and vision controllers resolve their
  Ollama base URL. The host is resolved inside the Ollama path only, so a
  non-Ollama embeddings provider (Bedrock/OpenAI/SageMaker/LiteLLM) is
  unaffected. Episodic-memory and RAG retrieval are unchanged in shape — they
  just stop falling back to degraded embeddings when an unrelated `OLLAMA_HOST`
  is set.

## [0.11.2] — 2026-06-30

### Fixed

- **Plan mode no longer "sticks" to a saved-then-reloaded conversation.** When
  plan mode is active, the client prepends an ephemeral `<plan-mode-active>`
  reminder to the turn's prompt. That banner was being baked into the stored
  `HumanMessage`, so saving the conversation and reloading it (with `/plan` since
  toggled off) re-fed the banner to the model, which then behaved as if still in
  plan mode. The banner is now stripped before the turn is stored (the model
  still receives it that turn, so enforcement is unchanged) and stripped again
  from any banner-bearing messages on load, so reloaded chats start clean.
- **Clearer tool arg-validation errors.** When the model called a tool with a
  required argument missing (notably `file_edit` without `new_string`, which a
  capable model still did ~80× in one session while trying to DELETE text), the
  raw pydantic "Field required" text was handed back and retried verbatim in a
  loop. Such failures are now translated into plain guidance naming the missing
  argument(s) (e.g. _"the call to `file_edit` is missing required argument(s):
  new_string … pass \"\" to delete text"_), and `file_edit`'s description now
  states all three args are required and documents `new_string=""` for deletion
  (with an example), steering wholesale rewrites toward `fs_write`.

## [0.11.1] — 2026-06-30

### Fixed

- **`fs_read` (and RAG/memory/compaction token counting) no longer crash on
  files containing special-token text** like `<|endoftext|>`. tiktoken's
  `encode()` raises on such text by default; the token-counting helpers now pass
  `disallowed_special=()` so the text is COUNTED as ordinary content (these are
  length estimates, not encodings sent to the model). Fixed across all four call
  sites: `ToolManager.count_tokens` (fs_read's size check), the RAG chunking
  helper, `EpisodicMemoryManager`, and `AgentConversationManager`. Reading a file
  that mentions `<|endoftext|>` (e.g. a config `STOP` list or a prompts file) now
  works instead of erroring with "disallowed special token".

## [0.11.0] — 2026-06-30

### Added

- **`TYPE: openai` can point at any OpenAI-compatible server** via `API_BASE`
  (alias `ENDPOINT_URL`) + optional `API_KEY` — making a local
  [`llama-server`](https://github.com/ggml-org/llama.cpp) (llama.cpp), LM Studio,
  vLLM, or `llama-swap` a drop-in alternative to Ollama with no extra provider.
  Local servers usually ignore auth, so a placeholder key is sent automatically
  when `API_BASE` is set and no `API_KEY` is given. Applies to **all three model
  roles** — `MODEL_ID` (chat), `VISION_MODEL_ID`, and `RAG.EMBED_MODEL_ID`
  (embeddings) — so a single local endpoint (e.g. `llama-swap` hot-swapping
  several `llama-server` instances) can serve everything. The openai embeddings
  path previously **ignored** `API_BASE`/`API_KEY` (always hit api.openai.com);
  it now honors them, matching the chat/vision controllers. Ollama
  (`TYPE: ollama`) remains fully supported — this is an alternative, not a
  replacement. See the new "Local OpenAI-compatible servers" recipe in
  `docs/configuration.md`.
- **Configurable embedding dimension** via `RAG.EMBED_MODEL_ID.DIMENSION`. The
  vector size was hardcoded (name lookup → 1024 default); it's used for the
  SHA256/zeros fallback and empty-result shape (real embeddings pass through at
  the provider's native size). Set it to your embedder's real dimension so the
  fallback stays index-consistent if the provider ever flaps. Settable via
  **`/params`** (the embeddings model is now a `/params` target) and **`/model`**
  (an "Embedding dimension" prompt), or by hand.
- **`/model` and `/params` now expose the local-server keys.** `/model`'s openai
  connection flow prompts for an optional OpenAI-compatible base URL + key (chat,
  vision, embeddings), and the embeddings flow prompts for `DIMENSION` — so the
  whole local-engine switch is UI-driven, no hand-editing required.

## [0.10.5] — 2026-06-29

### Fixed

- **Newer Claude models (Opus 4.7+) no longer 400 on the Anthropic protocol with
  "thinking.type.enabled is not supported for this model".** The factory and the
  direct-Anthropic controller always sent the legacy
  `thinking={"type":"enabled","budget_tokens":…}` form, which Opus 4.7+ rejects —
  they require `thinking={"type":"adaptive","display":"summarized"}` plus
  `output_config={"effort":…}`. The thinking request form is now chosen by model
  version (parsed from the id): **4.7+** → adaptive + summarized display,
  **4.6** → adaptive (no display), **≤4.5 / 3.x** → enabled + budget (these
  reject adaptive). An unknown version assumes the current adaptive API;
  `EXTRA_PARAMS` (e.g. your own `thinking`/`output_config`) overrides if version
  detection is wrong. Verified live against a Mantle Opus 4.8 endpoint — both
  the no-crash path and a returned `thinking` summary block. Applies to both the
  Mantle `anthropic` protocol and direct `TYPE: anthropic`.
- **Thinking on the Mantle `anthropic` protocol is now opt-in**, matching direct
  `TYPE: anthropic`. It was always-requested (a default `thinking_tokens` made
  the gate always true), so a Claude model that doesn't support extended
  thinking would 400 on every call with no way to disable it. Thinking is now
  enabled only when `REASONING_EFFORT` or `REASONING: true` is set; a
  non-reasoning Claude on Mantle works again. (Note: the OpenAI Responses
  reasoning _summary_ is requested correctly on Mantle GPT-5, but the Mantle
  gateway returns an empty summary — the reasoning happens and is billed, the
  text just isn't forwarded; direct `TYPE: openai` on the responses protocol
  does return it. This is upstream, not a client issue.)

## [0.10.4] — 2026-06-29

### Fixed

- **Regression from 0.10.3: every query failed on the Responses protocol with
  "Responses.create() got an unexpected keyword argument 'reasoning_effort'".**
  0.10.3 builds responses-protocol models with a `reasoning` OBJECT
  (`{"effort": …, "summary": "auto"}`), but `disable_reasoning` (used for the
  auxiliary classify/decompose calls) still set the scalar `model.reasoning_effort
= "none"`. The model then carried BOTH, and the Responses API rejects them
  together. `disable_reasoning`/`restore_reasoning` now operate on the
  `reasoning` object when present (setting its effort to `"none"` in place,
  preserving the summary), and only fall back to the scalar `reasoning_effort`
  for the legacy shape. Also fixed the Ollama-branch guard so a `reasoning`
  _dict_ is no longer mistaken for the Ollama boolean toggle. Verified at the
  request-payload level (no `reasoning_effort` reaches the Responses API).

## [0.10.3] — 2026-06-29

### Fixed

- **Reasoning is now visible for OpenAI-style reasoning models, not just
  Anthropic.** Two halves were missing:
  - **Request side:** the OpenAI Responses API only returns a reasoning
    _summary_ if you ask for it (`reasoning={"effort": …, "summary": "auto"}`);
    effort alone reasons invisibly. mnemoai sent only the bare effort, so
    Mantle GPT-5/Grok (and direct OpenAI) produced no visible reasoning. Now,
    on the `responses` protocol, `REASONING_EFFORT` is sent as a `reasoning`
    object that requests the summary. Direct `TYPE: openai` gains an
    `API_PROTOCOL` key (`chat_completions` default | `responses`) to opt into
    the Responses API and get the same treatment.
  - **Extraction side:** the agent's reasoning extractor now recognizes the
    OpenAI Responses summary block shape
    (`{"type":"reasoning","summary":[{"type":"summary_text","text":…}]}`)
    alongside the existing Bedrock `thinking` blocks, `reasoning_content`, and
    `<think>` tags. The summary streams as gray reasoning and never leaks into
    the visible answer.
- **Anthropic thinking enables on `REASONING_EFFORT` alone.** Direct
  `TYPE: anthropic` previously needed `REASONING: true` to turn on extended
  thinking; `REASONING_EFFORT` alone now enables it too (matching the OpenAI
  behavior). Anthropic thinking was already extracted/displayed once enabled —
  this just removes the extra flag.
- `EXTRA_PARAMS` still wins everywhere: set your own `reasoning`/
  `reasoning_effort` (OpenAI/responses) or `thinking` (Anthropic) to override
  the derived defaults — e.g. `reasoning: {effort: high, summary: detailed}` or
  `summary: none`.

## [0.10.2] — 2026-06-29

### Fixed

- **Shell-escaped / quoted file paths now resolve in tools.** A path the user
  copied from a terminal — `/Users/me/Screenshot\ 2026.png` (backslash-escaped
  spaces) or `"…/My File.png"` (quoted) — was passed verbatim to tools like
  `describe_image`, `fs_read`, `file_edit`, `glob_search`/`grep_search`, which
  aren't shells, so the literal backslashes/quotes made the file "not found"
  (often sending a smaller model into a retry/guess spiral). New
  `utils/path_utils.normalize_path` resolves these without breaking legitimate
  paths: it prefers the path exactly as given and only falls back to a
  de-escaped/de-quoted variant when the literal one doesn't exist on disk. Write
  targets (which may not exist yet) use `clean_path_syntax`, a syntactic-only
  cleanup. Applied across the file/image/search tools.

## [0.10.1] — 2026-06-29

### Fixed

- **`/save` now writes back to the open conversation file.** After `/load`-ing a
  conversation (or saving one earlier in the session), a bare `/save` overwrites
  that same file instead of creating a new `conversation_<timestamp>.json`. The
  open file is tracked in `current_conversation_path`, set on load and on first
  save; `/clear` resets it (a fresh conversation saves to a new file), and
  `/save <path>` still targets an explicit file/dir (and becomes the new open
  file).

### Changed

- **Removed the model name from the input prompt.** The status indicator added
  in 0.10.0 crowded the prompt line with long model names (e.g.
  `brnpistone/Qwen3.5-4B-AgentCoder-q6-k:latest`). The prompt is back to a clean
  `>`, keeping only the compact `🔒 plan` tag while plan mode is active. Use
  `/model` to see or change the current model.

## [0.10.0] — 2026-06-29

### Added

- **Lightweight Markdown rendering in streamed answers.** The model's Markdown is
  now rendered in the terminal instead of printed as literal syntax: headers
  (`##`) show as bold without the hashes, `-`/`*` bullets become `•`, numbered
  lists are kept, and `**bold**`/`*italic*` are styled. Inline `code` (bold cyan)
  and fenced code blocks (Pygments syntax highlighting) are unchanged, and
  clickable URLs still work. Implemented in `CodeFormatter` with a line-buffered
  pass — **no new dependency** (no `rich`). Emphasis is conservative: a spaced
  expression like `a * b * c` (e.g. `cost = instance_count * price_per_hour`) is
  never mis-italicized, and `**x**` inside backticks stays literal.
- **"Allow for this session" at confirmation prompts.** The destructive-tool gate
  now offers `Proceed? (y/N/a)` — answering `a` trusts that whole category
  (shell / file-write / memory) for the rest of the session, so a multi-step task
  no longer re-prompts for every command. Default-deny is unchanged; trust is
  per-category and resets on restart.
- **Status indicators on the input prompt.** The prompt line now shows a
  `🔒 plan` tag while plan mode is active and a dim chat-model name, so it's
  always clear which model is answering and whether mutations are blocked.

### Changed

- **Tool-call markers keep both ends of long values.** `[⚙ execute_bash(command=…)]`
  now middle-elides (`head…tail`) instead of cutting the tail, so a command's
  meaningful end (subcommand, flags) and trailing args (e.g. `timeout=30`) stay
  visible.

## [0.9.3] — 2026-06-27

### Fixed

- **Web-crawler progress no longer collides with the loading spinner.** The
  per-tool progress spinner added in 0.8.19 animated (carriage-return redraws)
  _during_ a `web_crawler` call, while crawl4ai writes its own live
  `[INIT]/[FETCH]/[SCRAPE]/[COMPLETE]` progress to the terminal — the two
  overwrote each other on the same lines (e.g. `⠏ Running web_crawler…[FETCH]…`).
  Tools that report their own progress are now listed in
  `_SELF_REPORTING_TOOLS`, and `_invoke_tool` keeps the spinner **stopped** for
  them so their output shows cleanly (as it did before 0.8.19), while every other
  tool still gets the progress spinner. Also fixed a stale progress-label key
  (`web_crawl` → the real tool name is `web_crawler`).

## [0.9.2] — 2026-06-26

### Fixed

- **Orphaned tool call/result pairs no longer wedge the conversation.** A turn
  cut short (recursion limit, stream error, interrupt) or a history slice could
  leave an assistant `tool_call` with no matching tool result (or vice-versa)
  persisted in the conversation. Strict providers — the OpenAI/Mantle Responses
  API — then rejected **every** subsequent turn with "No tool output found for
  function call …" (or the mirror, "No tool call found for function call
  output …"), with retries and the non-streaming fallback all re-sending the
  same broken history and failing identically. Added `_sanitize_tool_pairs`: an
  id-based repair that drops orphaned tool calls (keeping any visible text on
  that turn) and orphaned tool results before each model call, and on the kept
  window during compaction so the persisted history is repaired too. Extends the
  existing `_safe_tool_boundary` guard (which only covered the orphaned-_result_
  case at the compaction split) to both orphan directions, anywhere in history.

## [0.9.1] — 2026-06-26

### Fixed

- **Trivial queries no longer produce a blank answer.** A short conversational
  prompt ("can you do it?", "please do it") could be classified as `full` and
  sent through the orchestrator, which decomposed it into a single subtask run by
  the worker loop. That loop — unlike the normal `call_model` path — had **no
  empty-turn salvage**, so when a reasoning model streamed only hidden thinking
  (no visible text), nothing was printed and the turn ended silently after the
  `[Step 1/1]` / `[Context: …]` lines. Two fixes:
  - **Worker-loop salvage:** `_run_worker_loop` now mirrors `_call_model`'s
    guarantee — when a worker finishes with no visible content, it retries once
    with reasoning disabled (streamed, so the answer prints) and falls back to a
    visible message if still empty. The orchestrator path can no longer surface a
    blank answer.
  - **Smarter orchestrator gating:** a trivial, signal-free query classified as
    `full` now goes to the normal `call_model` path (which binds the same tools
    and already has the empty-turn safety net) instead of being decomposed. Only
    substantive tasks are orchestrated (`router.is_trivial_query`).

## [0.9.0] — 2026-06-26

### Added

- **Agent Skills** — authored, on-demand instruction packs.
  A skill is a directory `~/.mnemoai/skills/<name>/SKILL.md`
  with YAML frontmatter (`name` + `description`) and a markdown body, optionally
  bundling `reference.md` / `scripts/`. Skills fill the gap between always-on
  context (system prompt, `MEMORY.md`) and learned tactics (the ACE playbook):
  _authored procedures the model follows when a task matches_.
  - **Three-tier progressive disclosure:** (1) only each skill's name+description
    is injected into the system prompt at session start (cheap, always-on — the
    `<available_skills>` block); (2) the full body is loaded into context **only
    when the model calls the new `use_skill` tool** (its return value becomes the
    body); (3) bundled `reference.md`/`scripts/` are read/run on demand via the
    existing `fs_read`/`execute_bash` tools. Installing many skills stays cheap.
  - `use_skill` is bound on **every** route (it's in `_ALWAYS_AVAILABLE_TOOLS`),
    so a skill-matching request that classifies as `simple_qa` (e.g. "write a
    commit message") can still trigger it.
  - The metadata block is **re-injected after conversation compaction** (which
    rebuilds the system prompt fresh), so skills don't vanish mid-session.
  - **`/skills`** lists installed skills; a malformed skill is shown under
    _Skipped_ with the reason (missing `description`, bad YAML, over-long
    description) instead of being silently ignored — the authoring-feedback loop.
    `/skills <name>` previews a skill's full body.
  - Two skills are **seeded on first run**: a `commit-message` example, and a
    **`skill-creator`** meta-skill — ask the assistant to "create a skill for X"
    and it loads that guidance and authors a well-formed `SKILL.md` for you.
  - New module `client/memory/skill_store.py` (`SkillStore`, shared by the
    server tool and the client like `MemoryStore`), `server/tools/skill_tool.py`,
    `utils/paths.py:skills_dir()` + seeding. Gated by `ENABLE_SKILLS` (default
    true). See the Skills section in `docs/usage.md`.

## [0.8.21] — 2026-06-25

### Fixed

- **External MCP tools are now reachable on the `simple_qa` route.** A short
  factual question ("what time is in Singapore?") classifies as `simple_qa` —
  the no-tools route, which bound only the built-in meta tools (`memory`,
  `describe_image`, `fs_read`). External (`mcp.json`) tools were appended only to
  _non-empty_ routes, so a configured server like `time` was invisible on the
  very route such questions land in, and the model answered from its own
  knowledge ("I don't have access to the live clock") instead of calling the
  tool. External tools are user-configured capabilities, so they're now bound on
  **every** route, including `simple_qa`. (Built-in tool pruning per route is
  unchanged — only the handful of explicitly configured external tools ride
  along everywhere.)

## [0.8.20] — 2026-06-25

### Fixed

- **Closed the last "stuck"-looking gap: the final answer after the last tool.**
  0.8.19 added a spinner _during_ tool execution, but the model call that turns
  the tool results into the final reply (`_call_model`) relied on the spinner
  being left running by the preceding tool node — which no longer held once each
  tool call stopped its own spinner on completion. The result was a blank pause
  between the last tool and the model's answer. `_call_model` now starts the
  spinner at entry itself (idempotent; stopped as soon as visible text/reasoning
  streams or the next tool starts), mirroring `_aggregate`, so the wait for the
  model's first token always shows progress.

## [0.8.19] — 2026-06-25

### Fixed

- **No more "stuck"-looking terminal while a tool runs.** Previously the spinner
  stopped when a tool was about to execute and only restarted _after_ all tools
  finished, so a slow `tool.invoke()` (executing Python, a long shell command, a
  web fetch, a large file write) — especially right after the user confirmed it
  — showed a frozen, blank terminal with no sign of progress. The agent now
  animates a spinner with a per-tool label (e.g. `Running: python run.py`,
  `Searching the web`, `Writing /path/to/file`) for the full duration of each
  tool call, at both execution chokepoints (main loop and orchestrator worker
  loop). The spinner is always stopped afterward, even if the tool errors.

## [0.8.18] — 2026-06-25

### Changed

- **Plan mode is now "read-only except…" rather than a blanket block.** While `/plan` is active:
  - **Read-only shell commands run.** `execute_bash` is allowed when the command
    is read-only (leading program in an allowlist — `ls`, `cat`, `grep`, `rg`,
    `find`, `git status/log/diff/show`, etc. — with no redirection/chaining
    operators, and `git` limited to read-only subcommands). Mutating commands stay
    blocked. This lets the agent investigate properly while planning.
  - **The plan can be written to disk.** `fs_write`/`file_edit` are allowed only
    for a Markdown (`.md`) file under the plans directory (`paths.plans_dir()`),
    so the model can draft its plan incrementally. All other file writes remain
    blocked.
  - **The per-turn reminder is firmer.** ("Plan mode is active… you MUST NOT make any
    edits… this supersedes any other instructions"), points the model at the
    single writable plan file, and tells it to ask clarifying questions rather
    than guess.
  - Blocked-tool feedback is tailored per tool (read-only-bash hint; plan-file
    path hint) so the model pivots cleanly instead of just erroring.

## [0.8.17] — 2026-06-25

### Changed

- **Prompts moved out of `config.yaml` into a dedicated `prompts.yaml`.** All
  model-facing prompts — `SYSTEM_PROMPT`, `ROUTING_PROMPT`, `ORCHESTRATOR_PROMPT`,
  `AGGREGATOR_PROMPT`, and the compaction prompts (`SUMMARY_SYSTEM_PROMPT`,
  `SUMMARY_TASK_PROMPT`, previously hardcoded in Python) — now live in a
  `prompts.yaml` sibling of `config.yaml` (same `config/` dir, same resolution +
  first-run seeding; `$MNEMOAI_PROMPTS` overrides). `config.yaml` now holds
  configuration only (~165 lines, down from ~400). Access is via
  `Config().prompt("KEY")`.

### Breaking

- Prompts are read **only** from `prompts.yaml` — never from `config.yaml`, and
  there are no in-code prompt fallbacks. Prompt keys left in `config.yaml` are
  ignored (a one-time migration warning is logged); move customizations to
  `prompts.yaml`. The app **fails fast at startup** (`PromptError`) if a required
  prompt is missing: the mandatory prompts (`SYSTEM_PROMPT`,
  `SUMMARY_SYSTEM_PROMPT`, `SUMMARY_TASK_PROMPT`) are always required, and a
  feature's prompt is required when that feature is enabled (`ROUTING_PROMPT`
  with `ENABLE_ROUTING`; `ORCHESTRATOR_PROMPT`/`AGGREGATOR_PROMPT` with
  `ENABLE_ORCHESTRATION`). A bundled `prompts.yaml` seeds new installs and
  provides the defaults, so out-of-the-box runs are unaffected.

## [0.8.16] — 2026-06-24

### Fixed

- Image and file questions no longer fail by routing to a tool subset that
  can't handle them. `describe_image` and `fs_read` are now always-available
  meta tools (bound on every route, like `memory`), so "what's in this image?"
  or "what's in config.yaml?" — which classify as `simple_qa`/`knowledge` — can
  always reach the vision/read tools instead of falling back to reading bytes
  as text.
- `fs_read` on a binary/image file now returns a clear message steering the
  model to `describe_image` (for images) instead of dumping a raw
  `UnicodeDecodeError` stack trace. Binary files are detected up front
  (extension + content sniff) and logged calmly.
- **Route table audit & repair.** The `knowledge` route named four reader tools
  that don't exist (`read_csv/json/pdf/docx` — `fs_read` handles all formats via
  its `mode`), and six tools (`list_background_tasks`, `cancel_background_task`,
  `clear_completed_tasks`, `get_plan_status`, `todo_clear`, `clear_documents`)
  were reachable only via `full`. Routes are now re-derived from the real tool
  surface: `code` binds the complete todo/plan/background-task suites; `knowledge`
  is scoped to the RAG index (`list_documents`/`search_in_documents`/
  `clear_documents`). A regression test guards against future orphans/stale refs.

### Changed

- The query router gained a deterministic heuristic fast-path: obvious queries
  (greetings → `simple_qa`, a lone file path → `code`, a URL → `research`, a doc
  extension → `knowledge`) route instantly with **no LLM classification call**,
  while multi-signal or ambiguous queries fall back to `full`/the LLM. This cuts
  a per-query round-trip, sidesteps the empty/blank-route issues on reasoning
  models, and — by always preferring `full` on mixed signals — never under-binds
  tools. The `knowledge` category in `ROUTING_PROMPT` is re-scoped to RAG
  document search (reading a file by path works from any category).

## [0.8.15] — 2026-06-24

### Fixed

- The "Thinking" spinner no longer stops too early on models that stream hidden
  reasoning. Some providers (e.g. Anthropic via Bedrock) send redacted/secret
  reasoning chunks before the answer; the spinner used to stop on the first such
  chunk, leaving a dead pause (spinner gone, nothing printed) until the visible
  answer arrived. The spinner now stops only when something is actually about to
  be displayed — visible answer text, or reasoning shown in verbose mode — and
  keeps spinning through hidden reasoning. The streaming callback likewise
  ignores empty/whitespace-only tokens.

## [0.8.14] — 2026-06-24

### Fixed

- The interactive configurator (`/config`, `/model`, `/params`, first-run setup)
  now **re-asks on invalid input** instead of silently proceeding. Previously a
  bad menu choice kept the current value (or cancelled), and a non-numeric entry
  for `MAX_TOKENS` / `MAX_CONVERSATION_TOKENS` / the Mantle protocol was accepted
  or defaulted. Now menu prompts re-ask until a listed option is chosen, numeric
  prompts re-ask until the value parses as an int/float (`none` still clears
  optional `MAX_TOKENS`), and the `/model` and `/params` model pickers re-ask
  within the configured sections rather than cancelling on a wrong number.
- Streamed output no longer drops code at end-of-stream. The `CodeFormatter`'s
  `flush()` (now actually called when a stream ends) emits a response that ended
  inside an unclosed ` ``` ` fence, a held-back trailing backtick, and resets the
  terminal color after an unbalanced inline backtick — previously those were
  silently lost or left the prompt stuck in cyan. Bare `except:` clauses in the
  highlighter were also narrowed to `except Exception`.

### Changed

- Inline code / identifiers in streamed output are now **bold cyan**,
  instead of plain cyan, for a crisper distinction from surrounding prose.
  Fenced code blocks keep Pygments/monokai highlighting.

### Added

- A visible cancel affordance in the configurator: every interactive flow shows
  "Press Ctrl+C or Ctrl+D at any prompt to cancel — nothing is saved", and
  EOF/interrupt at any prompt now aborts cleanly (config left untouched) instead
  of half-applying an entry.

## [0.8.13] — 2026-06-24

### Fixed

- `/save` now writes to the profile's `conversations/` directory instead of the
  profile root (`~/.mnemoai/<profile>/`), where saved chats were cluttering the
  top level. New `paths.conversations_dir()` helper; existing files in the root
  are not moved (load them with an explicit `/load <path>`).

### Added

- `/save [path]` accepts an optional destination — a directory (saved there with
  the default `conversation_<ts>.json` name) or a full file path (`.json` added
  if missing). With no argument it saves to `conversations/` as before.
- `/load` with no argument now lists saved conversations (newest first, with
  relative times) and lets you pick one by number, instead of requiring a typed
  path. `/load <path>` still loads a specific file directly.

### Changed

- Compaction now shows phased progress on the spinner instead of a static
  "Generating summary …" line that looked frozen during the (single, long) LLM
  summary call. The spinner animates through `Summarizing N older messages` →
  `Applying summary` → the green "Compacted: …" result. (A true % bar isn't
  possible — one LLM generation has no measurable total — so this surfaces the
  discrete stages honestly rather than a fake bar.) The `Spinner` gained an
  optional `start(label=…)` argument and a `set_label()` method; the default
  label stays "Thinking" for the normal agent loop.

## [0.8.12] — 2026-06-24

### Changed

- The agent loop no longer hard-stops at 50 steps. That cap cut off legitimate
  long tasks mid-work with "Agent hit recursion limit". Where context compaction,
  not a step count, is the real limiter — the default `LLM.RECURSION_LIMIT`
  is now **200** (still configurable). It remains a runaway guard
  (LangGraph requires a finite bound), so hitting it now signals a
  likely stuck loop and the message says so and points at the config knob.

### Fixed

- Log lines (warnings/errors/info) no longer print inline with streamed answer
  text. The chat UI streams chunks to stdout without a trailing newline, so a
  log written to stderr afterwards landed on the same visual line. A cursor
  tracker now records whether stdout is mid-line, and the log handler prepends a
  newline when needed — so logs always start on their own line (no-op on
  piped/non-TTY output).

## [0.8.11] — 2026-06-23

### Fixed

- Compaction no longer corrupts the conversation by splitting a tool
  call/result pair. Previously the kept-verbatim window could start with an
  orphaned tool result (its originating assistant tool-call turn summarized
  away), which the OpenAI Responses API rejects on the very next turn with a
  deterministic 400 — "No tool call found for function call output with call_id
  …" — looping until the query fails. The split point is now tool-pair-safe
  (`_safe_tool_boundary`): it moves earlier as needed so a tool call and its
  result are always kept (or summarized) together.

### Changed

- Conversation compaction: a "summarizing conversations" system framing plus
  the verbatim structured task prompt (an `<analysis>` pass, then nine fixed
  sections — Primary Request, Key Technical Concepts, Files and Code Sections,
  Errors and fixes, Problem Solving, All user messages, Pending Tasks, Current Work,
  Optional Next Step). `/compact <focus>` is injected under a `## Compact Instructions`
  header; the `<analysis>` scratchpad is stripped from the result; and the injected summary
  block carries the verbatim continuation instruction so the model resumes the
  work seamlessly instead of recapping.

## [0.8.10] — 2026-06-23

### Added

- `REASONING_EFFORT` is now a first-class, `/params`-tunable knob for **every**
  provider that supports reasoning — `openai`, `anthropic`, `bedrock`, `mantle`,
  and `litellm` — translated to each provider's mechanism: forwarded as
  `reasoning_effort` on OpenAI and Mantle's `responses` protocol; mapped to a
  `thinking` token budget on Anthropic, standard Bedrock, and Mantle's
  `anthropic` protocol; passed through LiteLLM (which translates per backend).
  This gives Bedrock Mantle a real reasoning path it previously lacked. When
  thinking is enabled this way, `temperature`/`top_p`/`top_k` are dropped (the
  providers reject them).
- `EXTRA_PARAMS`: a generic per-model passthrough for the long tail. Any
  `MODEL_ID` / `VISION_MODEL_ID` may include an `EXTRA_PARAMS` dict whose
  contents are forwarded verbatim to the underlying model's request body (using
  the provider's own parameter names), so provider-specific knobs the curated
  registry doesn't model need no code change. Works for every provider (OpenAI,
  Anthropic, Bedrock, Ollama, SageMaker, LiteLLM, and all three Mantle
  protocols). `reasoning_effort` is lifted to a first-class arg on OpenAI-family
  clients; everything else merges into `model_kwargs`. A non-dict value is
  ignored. It is **config.yaml-only** — supported everywhere (never pruned by
  `/model`) but not prompted by `/model` or `/params`. `EXTRA_PARAMS` overrides
  the first-class `REASONING_EFFORT` if both set the same key.

### Changed

- `/params` only offers (and only writes) params the chosen provider actually
  supports — now covered by a regression test (e.g. Anthropic is never prompted
  for `PRESENCE_PENALTY`/`FREQUENCY_PENALTY`, but is for `REASONING_EFFORT`).

## [0.8.9] — 2026-06-23

### Changed

- The curated `MEMORY.md` now separates entries with a Markdown `---` rule
  instead of a `§` section sign, so the file renders as clean prose with
  dividers rather than showing a stray symbol. Existing `§`-delimited files are
  still read correctly and migrate to `---` automatically on the next memory
  write.

## [0.8.8] — 2026-06-22

### Changed

- `/clear` now wipes the terminal screen and scrollback and re-shows the welcome
  banner, for a true fresh start, instead of appending "Context cleared!" below
  the old conversation. No-op when stdout isn't a TTY (piped/redirected output
  stays clean).

## [0.8.7] — 2026-06-22

### Removed

- Reverted the streaming repetition-loop guard added in 0.8.6
  (`_is_degenerate_repetition`). Investigation confirmed the `<unused6226>`
  degeneration is a **non-deterministic serving-side issue** with
  `google.gemma-4-31b` on Bedrock Mantle (the identical request produces a clean
  answer on most calls and a repetition loop on a minority), not a client
  problem — so it belongs upstream, not as a heuristic in the stream loop. The
  guard added complexity and a regex scan on every chunk for a vendor bug that
  is being reported to AWS; removing it keeps the streaming path lean.

## [0.8.6] — 2026-06-22

### Fixed

- A runaway repetition loop no longer hangs the UI or burns the whole token
  budget. Some Bedrock Mantle-served models (observed with `google.gemma-4-31b`)
  intermittently degenerate into emitting a single reserved token
  (`<unused6226>`) until `MAX_TOKENS`, flooding the screen. The streaming loop
  now detects a special token repeated many times at the tail, aborts the
  stream early, and shows a clear message suggesting a different model or
  `API_PROTOCOL`. Conservative thresholds avoid tripping on legitimate
  repetition (lists, code, prose).

## [0.8.5] — 2026-06-22

### Fixed

- Arrow keys and backspace now work while typing answers in the `/config` and
  `/model` (and first-run) configurator prompts. They previously leaked raw
  escape sequences (e.g. `^[[D`) into the value because the `input()` prompts
  had no line editing; importing the stdlib `readline` module enables it.

## [0.8.4] — 2026-06-22

### Changed

- The answer marker (`●`) now precedes the answer on the **same line**
  (`● answer…`) instead of sitting on its own line above it, and it is also
  shown when reasoning is visible — printed after the gray reasoning block, on
  the answer line — so every assistant reply carries the marker.

## [0.8.3] — 2026-06-22

### Added

- A subtle cyan `●` marker is printed before a streamed answer when the model
  shows no reasoning, so the reply is visually distinct from the user's prompt
  instead of butting directly against it. Shown only on user-facing answer
  turns (main reply, retried answer, aggregated result) — worker streams already
  carry a `[Step N/N]` header.

## [0.8.2] — 2026-06-22

### Fixed

- The query classifier no longer crashes when no `ROUTING_PROMPT` is configured
  (e.g. a stripped config): `get_classifier_prompt()` now returns a built-in
  default instead of `None`, which previously raised a pydantic
  `ValidationError` while building the classification `SystemMessage`. This also
  fixes the unit suite running without a `config.yaml` (CI).

## [0.8.1] — 2026-06-22

### Changed

- `/model` now resets a section's inference parameters (temperature, top_p,
  penalties, reasoning, stop, stream — everything except the separately-prompted
  `MAX_TOKENS`) whenever the model is changed. These are model-specific, so a
  value tuned for one model is no longer silently carried into another that may
  reject it (e.g. newer Claude/GPT reject `temperature`); re-tune via `/params`.

### Fixed

- Reasoning models on the OpenAI Responses API (e.g. Bedrock Mantle Grok /
  GPT-5) no longer spam `Router returned unknown route '', falling back to
'full'` on every turn. The query classifier now disables reasoning on these
  models (`reasoning_effort="none"`) so the one-word route lands in the
  response instead of being eaten by reasoning, retries once on a transient
  empty response, and falls back to `full` quietly (debug, not a warning) when
  classification genuinely yields nothing.
- No more silent empty turn when a reasoning model is truncated by the
  output-token limit: if a turn ends with no answer and the response reports a
  token-limit cutoff (`status: incomplete` / `finish_reason: length` /
  `stop_reason: max_tokens`), the agent now surfaces a clear "increase
  `MAX_TOKENS`" message instead of an empty reply. Reasoning models spend output
  tokens reasoning before answering, so a low `MAX_TOKENS` could consume the
  whole budget before any answer was produced.
- Transient empty model responses are now retried. Some endpoints (notably
  Bedrock Mantle reasoning models on the Responses API) intermittently return a
  completely empty response — no content, reasoning, or tool call — for the same
  prompt that succeeds on a retry. Every model call (the main loop, orchestrator
  workers, and the aggregator) now retries an empty turn up to `LLM.MAX_RETRIES`
  times. This fixes blank `[Step N/N: …]` turns seen under orchestration.

## [0.8.0] — 2026-06-22

### Removed

- The `/good` command and its conversation "quality marker" plumbing. The
  markers were written into saved-conversation JSON but never consumed (no
  training/export pipeline existed), so the feature over-promised. `/save` now
  writes a plain conversation file. (Removing a command pre-1.0; noted here so
  it isn't a surprise.)

### Added

- CI: a `tests` GitHub Actions workflow runs the unit suite (and an import-sort
  check) on every push/PR across Python 3.11-3.13.
- This `CHANGELOG.md`, plus a documented Stability & Versioning contract and a
  release checklist (see `docs/development.md`).
- Expanded the integration tier: plan-mode enforcement (deterministic — asserts
  the file write is blocked) and the bash-timeout regression run end-to-end.

### Fixed

- No more silent empty turns: when the model ends a turn with no visible text
  right after a tool ran (e.g. a bash timeout) and produced no reasoning either,
  the agent now salvages the last tool result (or a fallback message) instead of
  returning an empty string.
- Streaming errors no longer lose the turn: on a mid-stream failure the agent
  now retries once non-streaming and prefers that complete result, instead of
  keeping a truncated partial chunk.
- Integration tests are provider-aware: they no longer falsely skip with
  "Ollama not reachable" when the configured model is a cloud provider
  (Bedrock/Mantle/OpenAI/Anthropic/SageMaker/LiteLLM). Verified end-to-end
  against Bedrock (Claude Sonnet).
- Welcome box auto-sizes to its widest command row instead of a hardcoded
  width, so longer command descriptions (e.g. `/plan`) no longer overflow the
  border.

## [0.7.0] — 2026-06-19

### Added

- **Enforced plan mode (`/plan`).** A user-toggled, read-only mode that
  hard-blocks mutating/exec tools (`execute_bash`, `fs_write`, `file_edit`,
  git writes, background tasks) client-side until you exit — so "analyze the
  repo" can never turn into edits, regardless of the model. Read-only tools and
  the memory notebook stay allowed. Distinct from the advisory `plan_mode.py`
  bookkeeping tools.

## [0.6.1] — 2026-06-19

### Changed

- Refreshed the PyPI project description after the README was split into a
  MkDocs site; re-exported the demo GIF smaller.

## [0.6.0] — 2026-06-19

### Added

- **Curated persistent memory (`MEMORY.md`).** A small, bounded, profile-scoped
  markdown file the agent maintains itself via a `memory` tool
  (`add`/`replace`/`remove`), injected whole into the system prompt at session
  start. Char-capped (forces consolidation); complements episodic memory and the
  ACE playbook. `ENABLE_MEMORY` / `REQUIRE_MEMORY_CONFIRMATION` toggles; `/memory`
  command. The `memory` tool is bound on every route (incl. `simple_qa`).
- Documentation split into a MkDocs Material site published to GitHub Pages.

## [0.5.2] — 2026-06-19

### Fixed

- Logger output is colored by level (errors red, warnings yellow) on a TTY;
  redirected/piped logs stay plain.

## [0.5.1] — 2026-06-19

### Fixed

- Repair a common malformed tool-args shape from smaller models
  (`{'query="…"': ''}` → `{'query': '…'}`) before validation, so the call
  succeeds instead of failing the turn.

## [0.5.0] — 2026-06-19

### Added

- **Hard confirmation gate** for destructive tools: `execute_bash`
  (`REQUIRE_BASH_CONFIRMATION`) and `fs_write`/`file_edit`
  (`REQUIRE_WRITE_CONFIRMATION`), both default on. Enforced client-side; the
  prompt always fires regardless of the model. Non-interactive runs auto-proceed.

## [0.4.0] — 2026-06-19

### Added

- **External MCP servers** via `~/.mnemoai/mcp/mcp.json`.
  Tools from external servers are merged with the built-in
  server; colliding names are namespaced `servername__tool`. `/mcp` lists them.
- Orchestrator awareness of external tools (routes subtasks needing them to the
  `full` category).
- App-home reorganized into `config/` and `mcp/` subfolders, each seeded with
  example templates on first run (legacy flat paths still read).

## [0.3.0] — 2026-06-19

### Added

- **Direct Anthropic API provider** (`TYPE: anthropic`) for chat and vision via
  `langchain-anthropic` — distinct from the Bedrock Mantle `anthropic` protocol.

## [0.2.1] — 2026-06-19

### Added

- A gray `[⚙ tool(args)]` marker between reasoning blocks so tool calls are
  visible after MCP request logs were silenced.

## [0.2.0] — 2026-06-19

### Added

- `/params` command to tune a model's inference parameters interactively.
- Numbered provider-type menu in `/model` (Mantle shown as `bedrock-mantle`).

## [0.1.0] — 2026-06

### Added

- Initial public release on PyPI as `mnemoai-assistant`.
- Local agentic assistant on LangGraph + MCP: StateGraph agent (classify →
  route/orchestrate → call model ↔ execute tools), multi-provider LLM support
  (Ollama, Amazon Bedrock, Bedrock Mantle, OpenAI, SageMaker, LiteLLM), episodic
  memory, ACE playbook, user-profile learning, RAG, web search/crawl, vision,
  and a `prompt_toolkit` chat UI with `/config` / `/model` configurators.

[Unreleased]: https://github.com/brunopistone/mnemoai/compare/v0.11.1...HEAD
[0.11.1]: https://github.com/brunopistone/mnemoai/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/brunopistone/mnemoai/compare/v0.10.5...v0.11.0
[0.10.5]: https://github.com/brunopistone/mnemoai/compare/v0.10.4...v0.10.5
[0.10.4]: https://github.com/brunopistone/mnemoai/compare/v0.10.3...v0.10.4
[0.10.3]: https://github.com/brunopistone/mnemoai/compare/v0.10.2...v0.10.3
[0.10.2]: https://github.com/brunopistone/mnemoai/compare/v0.10.1...v0.10.2
[0.10.1]: https://github.com/brunopistone/mnemoai/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/brunopistone/mnemoai/compare/v0.9.3...v0.10.0
[0.9.3]: https://github.com/brunopistone/mnemoai/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/brunopistone/mnemoai/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/brunopistone/mnemoai/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/brunopistone/mnemoai/compare/v0.8.21...v0.9.0
[0.8.21]: https://github.com/brunopistone/mnemoai/compare/v0.8.20...v0.8.21
[0.8.20]: https://github.com/brunopistone/mnemoai/compare/v0.8.19...v0.8.20
[0.8.19]: https://github.com/brunopistone/mnemoai/compare/v0.8.18...v0.8.19
[0.8.18]: https://github.com/brunopistone/mnemoai/compare/v0.8.17...v0.8.18
[0.8.17]: https://github.com/brunopistone/mnemoai/compare/v0.8.16...v0.8.17
[0.8.16]: https://github.com/brunopistone/mnemoai/compare/v0.8.15...v0.8.16
[0.8.15]: https://github.com/brunopistone/mnemoai/compare/v0.8.14...v0.8.15
[0.8.14]: https://github.com/brunopistone/mnemoai/compare/v0.8.13...v0.8.14
[0.8.13]: https://github.com/brunopistone/mnemoai/compare/v0.8.12...v0.8.13
[0.8.12]: https://github.com/brunopistone/mnemoai/compare/v0.8.11...v0.8.12
[0.8.11]: https://github.com/brunopistone/mnemoai/compare/v0.8.10...v0.8.11
[0.8.10]: https://github.com/brunopistone/mnemoai/compare/v0.8.9...v0.8.10
[0.8.9]: https://github.com/brunopistone/mnemoai/compare/v0.8.8...v0.8.9
[0.8.8]: https://github.com/brunopistone/mnemoai/compare/v0.8.7...v0.8.8
[0.8.7]: https://github.com/brunopistone/mnemoai/compare/v0.8.6...v0.8.7
[0.8.6]: https://github.com/brunopistone/mnemoai/compare/v0.8.5...v0.8.6
[0.8.5]: https://github.com/brunopistone/mnemoai/compare/v0.8.4...v0.8.5
[0.8.4]: https://github.com/brunopistone/mnemoai/compare/v0.8.3...v0.8.4
[0.8.3]: https://github.com/brunopistone/mnemoai/compare/v0.8.2...v0.8.3
[0.8.2]: https://github.com/brunopistone/mnemoai/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/brunopistone/mnemoai/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/brunopistone/mnemoai/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/brunopistone/mnemoai/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/brunopistone/mnemoai/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/brunopistone/mnemoai/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/brunopistone/mnemoai/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/brunopistone/mnemoai/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/brunopistone/mnemoai/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/brunopistone/mnemoai/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/brunopistone/mnemoai/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/brunopistone/mnemoai/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/brunopistone/mnemoai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/brunopistone/mnemoai/releases/tag/v0.1.0
