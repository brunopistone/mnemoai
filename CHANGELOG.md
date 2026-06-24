# Changelog

All notable changes to **Mnemo AI** (PyPI: `mnemoai-assistant`) are documented
here. The format follows [Keep a Changelog](https://keepachangelog.com/), and
the project aims to follow [Semantic Versioning](https://semver.org/): until
1.0.0, minor versions may introduce features and occasional breaking changes;
from 1.0.0 on, breaking changes to the public surface (config keys, the
`mcp.json` schema, CLI commands, the package/CLI name) bump the major version.

## [Unreleased]

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

- Inline code / identifiers in streamed output are now **bold cyan** (matching
  Claude Code's look and the Rich default) instead of plain cyan, for a crisper
  distinction from surrounding prose. Fenced code blocks keep Pygments/monokai
  highlighting.

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
  long tasks mid-work with "Agent hit recursion limit". Following Claude Code's
  model — where context compaction, not a step count, is the real limiter — the
  default `LLM.RECURSION_LIMIT` is now **200** (still configurable). It remains a
  runaway guard (LangGraph requires a finite bound), so hitting it now signals a
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

- Conversation compaction now uses Claude Code's compaction approach: a
  "summarizing conversations" system framing plus the verbatim structured task
  prompt (an `<analysis>` pass, then nine fixed sections — Primary Request,
  Key Technical Concepts, Files and Code Sections, Errors and fixes, Problem
  Solving, All user messages, Pending Tasks, Current Work, Optional Next Step).
  `/compact <focus>` is injected under a `## Compact Instructions` header; the
  `<analysis>` scratchpad is stripped from the result; and the injected summary
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

- **External MCP servers** via `~/.mnemoai/mcp/mcp.json` (Claude Code/kiro
  `mcpServers` schema). Tools from external servers are merged with the built-in
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

[Unreleased]: https://github.com/brunopistone/mnemoai/compare/v0.8.15...HEAD
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
