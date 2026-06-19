# CLAUDE.md

## Project Overview

Local agentic AI assistant built on LangGraph + MCP (Model Context Protocol). The client spawns an MCP server as a subprocess, routes queries through a StateGraph (classify → orchestrate/call_model ↔ execute_tools), and persists episodic memory, learned strategies (ACE Playbook), and user profiles across sessions. Supports Ollama, AWS Bedrock, OpenAI, SageMaker, and LiteLLM as LLM providers.

## Quick Commands

```bash
# Run from a checkout (src layout: package under src/, run as a module)
PYTHONPATH=src python -m personal_ai_assistant            # verbose (shows thinking)
PYTHONPATH=src python -m personal_ai_assistant --no-verbose

# Or install, then use the console command
uv tool install .        # or: pip install -e .
personal-ai-assistant

# Install dependencies (checkout dev)
pip install -r requirements.txt

# System-wide access (symlink once)
chmod +x bash/system-command-app/personal-ai-assistant-wrapper.sh
ln -sf $(pwd)/bash/system-command-app/personal-ai-assistant-wrapper.sh /usr/local/bin/personal-ai-assistant
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

**src layout:** the single package `personal_ai_assistant` lives under `src/`.
Paths below are relative to `src/personal_ai_assistant/` (e.g. `client/` is
`src/personal_ai_assistant/client/`). `main.py` is the package entry (`cli()`),
also runnable as `python -m personal_ai_assistant`. `tests/`, `docs/`, `bash/`
stay at the repo root.

| Directory                | Role                                                                      |
| ------------------------ | ------------------------------------------------------------------------- |
| `client/`                | LangGraphClient facade, MCP bridge                                        |
| `client/agent/`          | Agent loop: StateGraph agent, query router, orchestrator, reasoning utils |
| `client/memory/`         | Episodic memory (ChromaDB/FAISS), ACE Playbook, Reflector                 |
| `client/managers/`       | Conversation token management, user profile learning                      |
| `client/ui/`             | prompt_toolkit chat loop, spinner                                         |
| `server/`                | FastMCP server entry point (run as a subprocess)                          |
| `server/tools/`          | Tool implementations (bash, file ops, git, web, RAG, vision, planning)    |
| `server/tools/rag/`      | Session-scoped vector store, hybrid search engine                         |
| `server/tools/readers/`  | File format readers (PDF, DOCX, CSV, JSON, directory, line, search)       |
| `models/`                | `provider_params` registry + `mantle_factory`                             |
| `models/controllers/`    | Provider-dispatching LLM/vision/embeddings controllers                    |
| `models/chat_models/`    | Concrete LangChain ChatModel subclasses (ChatOllamaWrapper, ChatSageMaker)|
| `utils/`                 | Config singleton, configurator, paths, logger, BM25, text formatting      |
| `bash/` (repo root)      | Shell scripts (system command wrapper, Ollama VRAM management)            |

## Detailed File Map

The full per-file reference (every module, its key classes/functions, and
what it does) lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) to keep
this file lean. Consult it when you need to locate or understand a specific
file; the sections below cover the high-level architecture and conventions.

## Key Patterns

### Config singleton (`utils/config.py`)

All configuration flows through `Config()` which loads `utils/config.yaml` (gitignored). Access via `Config().get("SECTION.KEY", default)`. The `.system_prompt` property injects the current date.

### MCP tool registration (`server/tools/tools_manager.py`)

`ToolManager.register_tools(mcp)` conditionally registers tool groups based on config toggles. Each tool file defines functions decorated with `@mcp.tool()`.

### Hybrid search (semantic + BM25)

Used in both episodic memory and RAG. Pattern: get top-N candidates from vector store, get top-N from BM25, merge with configurable weights (`utils/bm25.py`).

### Multi-provider LLM abstraction (`models/controllers/llm_controller.py`)

`LangChainLLMController.initialize_model()` dispatches on `MODEL_ID.TYPE` (bedrock/mantle/ollama/openai/sagemaker/litellm). Each provider's supported config keys / client-kwarg mapping live in `models/provider_params.py` (consumed via `build_kwargs`).

### Bedrock Mantle (`models/mantle_factory.py`)

`TYPE: mantle` reaches AWS Bedrock Mantle via a bearer token minted from standard AWS (SigV4) credentials. `API_PROTOCOL` selects the wire protocol: `chat_completions` (OpenAI `/v1`), `responses` (OpenAI Responses `/openai/v1`, e.g. GPT-5.4), `anthropic` (Anthropic Messages `/anthropic`, Claude). The factory is shared by the LLM and vision controllers. Model availability varies by region (e.g. GPT-5.4 is in us-west-2).

### Conversation compaction (`client/managers/agent_conversation_manager.py`)

Keeps the conversation under `MAX_CONVERSATION_TOKENS` by summarizing older messages into the system prompt while keeping recent turns verbatim. Triggers automatically when over budget, or manually via `/compact`. The kept window is bounded by message count AND a token budget so an oversized recent message (e.g. a pasted document) is summarized, not kept. Tool calls/results are preserved in the summary.

### Query routing (`client/agent/router.py`)

`QueryRouter.classify()` uses the LLM to categorize queries. Routes map to tool subsets in `ROUTE_TOOLS` dict — only relevant tools are bound per query.

### Orchestrator-workers (`client/agent/orchestrator.py`, `client/agent/agent.py`)

For "full" complexity tasks: decompose → parse subtasks (JSON) → run worker loop per subtask with category-specific tools → aggregate results.

### ACE Playbook learning (`client/memory/reflector.py`, `client/memory/playbook_store.py`)

After each interaction, the Reflector analyzes tool execution trajectories, detects failure patterns, and extracts reusable strategies stored in the PlaybookStore. Relevant strategies are injected into the system prompt for future queries.

### Episodic memory (`client/memory/episodic_memory.py`)

Stores successful task completions with tool usage patterns. Retrieved via hybrid search before each query and injected as context.

## Configuration

`utils/config.yaml` (gitignored). Copy from one of the provided templates:
`utils/config.yaml.example` (Ollama/local), `utils/config.yaml.bedrock.example` (standard Bedrock), or `utils/config.yaml.bedrock.mantle.example` (Bedrock Mantle). Each is a complete drop-in config for that provider — keep them in sync when adding shared config keys.

| Section               | Purpose                                                                                                                                                                                                 |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MODEL_ID`            | LLM provider/`TYPE` (bedrock, mantle, ollama, openai, sagemaker, litellm), model name, inference params. Mantle adds `API_PROTOCOL` (chat_completions/responses/anthropic) and optional `ENDPOINT_URL`. |
| `VISION_MODEL_ID`     | Vision model for image description (same provider types as `MODEL_ID`)                                                                                                                                  |
| `RAG`                 | Embedding model, chunk size, hybrid weights, vector store type                                                                                                                                          |
| `EPISODIC_MEMORY`     | Thresholds, search weights, success/error markers                                                                                                                                                       |
| `PLAYBOOK`            | Max entries, similarity threshold, injection limit                                                                                                                                                      |
| `LLM`                 | Retry config, thinking toggle, agent `RECURSION_LIMIT`, `MCP_CALL_TIMEOUT`, and compaction (`KEEP_RECENT_MESSAGES`, `MANUAL_COMPACT_KEEP_RECENT`, `KEEP_RECENT_TOKEN_BUDGET`)                           |
| `PROFILE`             | User name (data isolation), profiling toggle                                                                                                                                                            |
| `BRAVE_API_KEY`       | Web search API key                                                                                                                                                                                      |
| `SYSTEM_PROMPT`       | Full system prompt (XML-structured)                                                                                                                                                                     |
| `ROUTING_PROMPT`      | Query classifier prompt                                                                                                                                                                                 |
| `ORCHESTRATOR_PROMPT` | Task decomposition prompt                                                                                                                                                                               |
| `AGGREGATOR_PROMPT`   | Result synthesis prompt                                                                                                                                                                                 |

**Feature toggles** (all boolean in config root):
`ENABLE_RAG`, `ENABLE_EPISODIC_MEMORY`, `ENABLE_PLAYBOOK`, `ENABLE_WEB_SEARCH`, `ENABLE_WEB_CRAWL`, `ENABLE_ROUTING`, `ENABLE_ORCHESTRATION`

**Environment variables:**

- `OPENAI_API_KEY` — for OpenAI provider
- `LOG_LEVEL` — logging verbosity (default: INFO)
- AWS credentials via `aws configure` for Bedrock/SageMaker/Mantle (Mantle mints a bearer token from these via `aws-bedrock-token-generator`)
- Config `ENV` section sets additional env vars at startup

**Runtime data:** All state lives under a single app home, `~/.personal-ai-assistant/` (override with `$PERSONAL_AI_ASSISTANT_HOME`), resolved centrally in `utils/paths.py`. Layout: `config.yaml`, `plans/`, `tasks/`, and per-profile `{profile_name}/` (conversations, todos, RAG indexes, chunk caches, user profile). **Episodic memory and the ACE playbook are model-scoped** under `{profile_name}/models/{sanitized_model_name}/` so switching the chat model doesn't contaminate memory built with a different one. All path construction goes through `utils/paths.py` (`app_home`, `config_path`, `plans_dir`, `tasks_dir`, `profile_dir`, `model_dir`).

## Code Conventions

- **Tests** — pytest unit suite in `tests/` covers pure-logic modules (no LLM/Ollama needed). Run with `python -m pytest`. See the Testing section below
- **Error handling in tools** — `@tool_error_handler` decorator (`server/error_handler.py`) for standardized responses
- **Async/sync bridge** — MCP client uses a background thread with `asyncio.new_event_loop()` in `client/mcp_tool_wrapper.py`; sync callers use `run_coroutine_threadsafe`
- **Imports** — relative within packages, absolute across packages
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

## Known Limitations

- Unit tests cover pure logic only — agent/LLM integration paths still need manual verification
- MCP server is a subprocess — debugging requires attaching to child process or reading logs
- No Docker/containerization — runs directly on host with system Python/conda/venv
- No database — all persistence is file-based (JSON, FAISS index files, SQLite chunk cache)
- Single-user — profile name in config isolates data but no auth/multi-tenancy
- `ripgrep` (rg) required for `grep_search` tool — install separately
