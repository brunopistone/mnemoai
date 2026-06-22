# Changelog

All notable changes to **Mnemo AI** (PyPI: `mnemoai-assistant`) are documented
here. The format follows [Keep a Changelog](https://keepachangelog.com/), and
the project aims to follow [Semantic Versioning](https://semver.org/): until
1.0.0, minor versions may introduce features and occasional breaking changes;
from 1.0.0 on, breaking changes to the public surface (config keys, the
`mcp.json` schema, CLI commands, the package/CLI name) bump the major version.

## [Unreleased]

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

[Unreleased]: https://github.com/brunopistone/mnemoai/compare/v0.8.2...HEAD
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
