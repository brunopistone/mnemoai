# CLAUDE.md

## Project Overview

Local agentic AI assistant built on the **Strands Agents SDK** + MCP (Model Context Protocol). The client spawns an MCP server as a subprocess, optionally classifies each query to a tool subset (router), and runs a Strands `Agent` that loops model ↔ tools natively. Persists episodic memory, learned strategies (ACE Playbook), and user profiles across sessions. Supports Ollama, AWS Bedrock, Bedrock Mantle, OpenAI, and SageMaker as model providers.

> **Branch note:** This is the `strands` branch. A sibling `langgraph` branch implements the same app on LangGraph + LangChain. The `server/` side (MCP server + tools) is essentially identical between branches; the difference is the agent framework in `client/` and `models/`.

## Quick Commands

```bash
# Run (verbose mode — shows thinking process)
python main.py

# Run (hide thinking)
python main.py --no-verbose

# Install dependencies
pip install -r requirements.txt

# Tests
pip install -r requirements-dev.txt
python -m pytest
```

## Architecture

```
main.py → ChatInterface → StrandsClient.query()
  → inject episodic memory context
  → [route] classify (QueryRouter) → tool subset
      simple_qa → _handle_simple_qa (model.stream, no tools)
      else      → _handle_agent_query → strands Agent(model, tools) loop
  → AgentConversationManager (compact if over token limit)
  → UserProfileManager (learn preferences)
  → Reflector + PlaybookStore (learn from tool successes/failures)
```

**Client-server split:** The MCP server (`server/server.py`) runs as a stdio subprocess. The client connects via Strands' built-in `strands.tools.mcp.MCPClient` (used as a context manager around each query). Tools are listed with `list_tools_sync()` and passed to the `Agent`.

**No StateGraph:** Unlike the langgraph branch, there is no `agent.py` / StateGraph — the Strands `Agent` runs the model↔tool loop internally. Orchestrator-workers *is* supported (see `client/orchestrator.py` + `StrandsClient._handle_orchestrated_query`), but it's a hand-rolled loop over Strands `Agent` instances rather than graph nodes.

## Directory Structure

| Directory               | Role                                                                      |
| ----------------------- | ------------------------------------------------------------------------- |
| `client/`               | StrandsClient, query routing, UI                                          |
| `client/memory/`        | Episodic memory (ChromaDB/FAISS), ACE Playbook, Reflector                 |
| `client/managers/`      | Conversation compaction, user profile learning                            |
| `client/ui/`            | prompt_toolkit chat loop, spinner                                         |
| `server/`               | FastMCP server entry point                                                |
| `server/tools/`         | Tool implementations (bash, file ops, git, web, RAG, vision, planning)    |
| `server/tools/rag/`     | Session-scoped vector store, hybrid search engine                         |
| `server/tools/readers/` | File format readers (PDF, DOCX, CSV, JSON, directory, line, search)       |
| `models/`               | Strands model controllers, multi-provider abstraction                     |
| `models/classes/`       | Custom Strands model classes (ThinkingOllamaModel, SageMakerVisionModel)  |
| `utils/`                | Config singleton, logger, text formatting                                 |

## Detailed File Map

### Entry Point

| File      | Purpose                                                                                                                                          |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `main.py` | CLI entry point. Parses `--no-verbose` (default verbose=True), creates `StrandsClient(server_path="server/server.py")`, starts it, runs `ChatInterface`. |

### `client/` — Client, Routing, UI

| File                          | Purpose                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `client/__init__.py`          | Exports `StrandsClient`                                                                                                                                                                                                                                                                                                                                                                                          |
| `client/client.py`            | **Central orchestration class `StrandsClient`.** Manages MCP connection (`strands.tools.mcp.MCPClient`), model init, the `strands.Agent`, query processing, episodic memory injection, playbook context, conversation save/load, RAG session, chunk cache, context clearing, compaction trigger. Key methods: `start()`, `query()`, `_handle_simple_qa()`, `_handle_agent_query()`, `compact_conversation()`, `clear_context()`, custom streaming callback handlers (`__minimal_callback_handler`, `__verbose_callback_handler`) for spinner + thinking-tag handling. |
| `client/router.py`            | **Query classifier.** `QueryRouter.classify()` sends query + `ROUTING_PROMPT` to the model, returns one of: `simple_qa`, `code`, `research`, `knowledge`, `full`. `ROUTE_TOOLS` dict maps each route to allowed tool names. `simple_qa` is handled without tools for speed.                                                                                                                                       |

### `client/managers/` — Conversation & Profile Management

| File                                            | Purpose                                                                                                                                                                                                                                                                                                                                                                                                              |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `client/managers/agent_conversation_manager.py` | **Conversation compaction** (token counting + model summarization). Auto-compacts when over `MAX_CONVERSATION_TOKENS`; `compact()` is the manual `/compact` path. Summarizes older messages into the system prompt while keeping recent turns verbatim — the kept window is bounded by BOTH message count (`KEEP_RECENT_MESSAGES` / `MANUAL_COMPACT_KEEP_RECENT`) and a token budget (`KEEP_RECENT_TOKEN_BUDGET`, default 25% of max), so an oversized recent message is summarized rather than kept. Operates on Strands native dict messages. Key methods: `count_tokens()`, `generate_summary()`, `manage_messages()`, `compact()`, `_compact()`, `_split_keep_recent()`. |
| `client/managers/user_profile_manager.py`       | Learns user preferences via Exponential Moving Average (EMA). Tracks verbosity, directness, technical level, abstraction preference, top domains, tool success per intent. Generates a compact profile summary for system prompt injection. Persists as JSON at `~/agent-conversations/{profile}/`.                                                                                                                  |

### `client/memory/` — Episodic Memory & Learning

| File                               | Purpose                                                                                                                                                                                                            |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `client/memory/episodic_memory.py` | High-level episodic memory manager. Stores successful task patterns with tool usage. Delegates to ChromaDB or FAISS. `EpisodicMemoryManager` with `store()` / `retrieve_similar_episodes()` / `cleanup()`.         |
| `client/memory/chroma_store.py`    | ChromaDB-backed episodic store with hybrid search (semantic + BM25 re-ranking).                                                                                                                                    |
| `client/memory/faiss_store.py`     | FAISS-backed episodic store (alternative to ChromaDB).                                                                                                                                                             |
| `client/memory/reflector.py`       | **ACE Reflector.** Analyzes tool execution trajectories, detects failure patterns, extracts reusable strategies as playbook entries.                                                                              |
| `client/memory/playbook_store.py`  | **ACE Playbook.** Append-only store for learned strategies with lazy semantic dedup. Retrieves relevant entries by task context for system prompt injection. Persists at `~/agent-conversations/{profile}/playbook/`. |

### `client/ui/` — User Interface

| File                          | Purpose                                                                                                                                                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `client/ui/chat_interface.py` | Interactive CLI using prompt_toolkit. Multiline input (Ctrl+J), commands: `/clear`, `/load`, `/save`, `/exit`, `/quit`, `/good`, `/compact [focus]` (manual context compaction). Episodic memory storage, ACE reflection triggers. |
| `client/ui/spinner.py`        | Threaded "Thinking..." spinner shown during processing.                                                                                                                                                            |

### `server/` — MCP Server & Tools

The `server/` tree is shared with the langgraph branch. `server/server.py` creates a `FastMCP` instance, calls `register_tools(mcp)`, and runs over stdio. `server/tools/` holds all tool implementations (bash, file ops, git, web, RAG, vision, planning, todos, background tasks); `server/tools/rag/` is the session-scoped hybrid-search RAG engine; `server/tools/readers/` holds file-format readers. Key safety/util files: `error_handler.py` (`@tool_error_handler`), `git_safety.py` (blocks force-push to main, flags `git branch -D`), `execute_bash.py` (process-group kill on timeout), `tools_manager.py` (`register_tools`, conditional vision init).

### `models/` — Strands Model Abstraction

| File                                    | Purpose                                                                                                                                                                                                                                                                                                                                                                                          |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `models/base_model_controller.py`       | Shared inference-parameter helpers: `_set_bedrock_inference_parameters()`, `_set_ollama_inference_parameters()`, `_set_openai_inference_parameters()`, `_set_sagemaker_inference_parameters()`.                                                                                                                                                                                                  |
| `models/llm_controller.py`              | **Primary LLM controller `LLMController`.** Dispatches on `MODEL_ID.TYPE`: `bedrock` (`BedrockModel`), `ollama` (`ThinkingOllamaModel`), `openai` (`OpenAIModel`), `sagemaker` (`SageMakerAIModel`), `mantle` (Bedrock Mantle — see below). `temperature` only sent when explicitly configured (newer Claude models reject it).                                                                  |
| `models/vision_model_controller.py`     | Vision model controller. Same provider types incl. `mantle`. `format_request()` builds the provider-appropriate image message; `get_model()` lazy-inits.                                                                                                                                                                                                                                        |
| `models/embeddings_controller.py`       | Multi-provider embeddings with caching + SHA256 fallback.                                                                                                                                                                                                                                                                                                                                       |
| `models/classes/thinking_ollama.py`     | `ThinkingOllamaModel` — Ollama model that preserves `<think>` content.                                                                                                                                                                                                                                                                                                                          |
| `models/classes/sagemaker_vision_model.py` | `SageMakerVisionModel` — SageMaker model extended with image input support.                                                                                                                                                                                                                                                                                                                |

### `utils/` — Shared Utilities

`config.py` (singleton over `utils/config.yaml`, `.system_prompt` injects date), `logger.py`, and `formatting/` (`code_formatter.py` streaming syntax highlight, `response_parser.py`, `url_formatter.py`).

## Key Patterns

### Strands Agent loop (`client/client.py`)

A `strands.Agent(model, tools, system_prompt, callback_handler)` runs the model↔tool loop natively. Streaming is surfaced via custom `callback_handler` functions that drive the spinner and strip/show `<think>` tags. There is no manual graph; tool execution is handled by the SDK.

### MCP via Strands (`strands.tools.mcp.MCPClient`)

The client wraps each query in `with self.mcp_client:` and gets tools through `list_tools_sync()`. No custom async-thread bridge (unlike the langgraph branch's `mcp_tool_wrapper.py`).

### Multi-provider model abstraction (`models/llm_controller.py`)

`LLMController.initialize_model()` dispatches on `MODEL_ID.TYPE` (bedrock/ollama/openai/sagemaker/mantle). Each provider maps inference params in `BaseModelController`.

### Bedrock Mantle (`models/llm_controller.py` / `vision_model_controller.py`)

`TYPE: mantle` reaches AWS Bedrock Mantle using a bearer token minted from standard AWS (SigV4) credentials (via `aws-bedrock-token-generator`; handled natively by Strands ≥1.43). `API_PROTOCOL` selects the wire protocol:
- `chat_completions` (default) — `OpenAIModel(bedrock_mantle_config={"region": ...})`, `/v1`. Most models.
- `responses` — `OpenAIResponsesModel` pointed at `/openai/v1` via `client_args` (base_url + token); e.g. `openai.gpt-5.4` (us-west-2). Uses `max_output_tokens`.
- `anthropic` — `AnthropicModel` pointed at `/anthropic` via `client_args`; Claude models. `max_tokens` is a top-level kwarg.

Requires `strands-agents>=1.43.0`. Model availability varies by region.

### Query routing (`client/router.py`)

`QueryRouter.classify()` categorizes queries; `ROUTE_TOOLS` maps each route to a tool subset. `simple_qa` bypasses tools entirely (`_handle_simple_qa`). When `ENABLE_ORCHESTRATION` is set, the `full` route is decomposed into worker subtasks (`_handle_orchestrated_query`); otherwise `full` binds all tools to one agent.

### Orchestrator-workers (`client/orchestrator.py`, `StrandsClient`)

When `ENABLE_ORCHESTRATION` is on, a `full`-routed query is: decomposed into subtasks via the model (`_decompose_task`, reasoning off, `parse_subtasks` tolerant of non-JSON), each subtask run by its own route-scoped `strands.Agent` (`_run_worker`, tools filtered by `ROUTE_TOOLS` via `.tool_name`, prior results passed as context), then synthesized (`_aggregate_results`). Per-worker and aggregation failures degrade gracefully. Uses `ORCHESTRATOR_PROMPT` / `AGGREGATOR_PROMPT` from config.

### Conversation compaction (`client/managers/agent_conversation_manager.py`)

Keeps the conversation under `MAX_CONVERSATION_TOKENS` by summarizing older messages into the system prompt while keeping recent turns verbatim. Auto when over budget, or manual via `/compact [focus]`. Kept window bounded by message count AND a token budget so an oversized recent message (e.g. a pasted document) is summarized, not kept. Works on Strands native dict messages (tool use/results already in content).

## Configuration

`utils/config.yaml` (gitignored). Copy from `utils/config.yaml.example`.

| Section               | Purpose                                                                                                                                                                |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MODEL_ID`            | Provider/`TYPE` (bedrock, ollama, openai, sagemaker, mantle), model name, inference params. Mantle adds `API_PROTOCOL` (chat_completions/responses/anthropic) and optional `ENDPOINT_URL`. |
| `VISION_MODEL_ID`     | Vision model for image description (same provider types)                                                                                                              |
| `EMBED_MODEL_ID`      | Embeddings model for episodic memory / RAG                                                                                                                            |
| `RAG`                 | Chunk size, hybrid weights, vector store type                                                                                                                         |
| `EPISODIC_MEMORY`     | Thresholds, search weights, success/error markers                                                                                                                     |
| `PLAYBOOK`            | Max entries, similarity threshold, injection limit                                                                                                                    |
| `LLM`                 | Thinking toggle, token counting, and compaction (`KEEP_RECENT_MESSAGES`, `MANUAL_COMPACT_KEEP_RECENT`, `KEEP_RECENT_TOKEN_BUDGET`)                                     |
| `PROFILE`             | User name (data isolation), profiling toggle                                                                                                                          |
| `SYSTEM_PROMPT` / `ROUTING_PROMPT` | System prompt and query-classifier prompt                                                                                                                |
| `ORCHESTRATOR_PROMPT` / `AGGREGATOR_PROMPT` | Task-decomposition and result-synthesis prompts (used when `ENABLE_ORCHESTRATION`)                                                          |

**Feature toggles** (boolean in config root): `ENABLE_RAG`, `ENABLE_EPISODIC_MEMORY`, `ENABLE_PLAYBOOK`, `ENABLE_WEB_SEARCH`, `ENABLE_WEB_CRAWL`, `ENABLE_ROUTING`, `ENABLE_ORCHESTRATION` (requires routing).

**Environment variables:** `OPENAI_API_KEY` (OpenAI), `LOG_LEVEL`, AWS credentials via `aws configure` for Bedrock/SageMaker/Mantle (Mantle mints a bearer token from these).

**Runtime data:** `~/agent-conversations/{profile_name}/`.

## Testing

```bash
pip install -r requirements-dev.txt    # installs pytest
python -m pytest                        # full unit suite
python -m pytest tests/unit/test_git_safety.py
```

- Tests live in `tests/unit/`, configured via `pytest.ini` (testpaths=tests). `tests/conftest.py` puts the repo root on `sys.path`.
- Deterministic, pure-logic unit tests — no live model/Ollama/network. Covers `server/tools/git_safety.py`, `error_handler.py`, `file_edit.py` + `glob_search`, `execute_bash` timeout/process-group behavior, `utils/formatting/` (response_parser, url/code formatters), and conversation compaction incl. token-aware retention.
- Unit tests must not require a `config.yaml` — modules degrade gracefully without one. Keep import-time side effects config-independent.

## Code Conventions

- **Strands messages are native dicts** — `{"role", "content": [{"text"|"toolUse"|"toolResult"}]}`. Tool calls/results live in the content blocks (no separate message objects).
- **Error handling in tools** — `@tool_error_handler` decorator returns structured JSON.
- **Streaming** — Strands emits events to a `callback_handler`; the client parses `contentBlockDelta`/`messageStart` etc. to drive the spinner and thinking-tag display.
- **Naming** — snake_case functions/variables, PascalCase classes, UPPER_CASE config keys.
- **Token counting** — tiktoken for OpenAI/Bedrock, char-based approximation for Ollama.

## Adding a New Model Provider

1. Add a `TYPE` case in `models/llm_controller.py` → `initialize_model()` using the matching Strands model class.
2. Add inference-parameter mapping in `models/base_model_controller.py`.
3. If a custom model class is needed, add it under `models/classes/`.
4. Add vision support in `models/vision_model_controller.py` if applicable.
5. Document the config structure in `utils/config.yaml.example`.

## Known Limitations

- Unit tests cover pure logic only — agent/model integration paths need manual verification.
- Orchestrator-workers is a hand-rolled loop over Strands `Agent` instances (no graph); only the `full` route triggers it, and only when `ENABLE_ORCHESTRATION` is set.
- MCP server is a subprocess — debugging requires reading logs / attaching to the child.
- No Docker — runs directly on host Python/conda/venv.
- Single-user — profile name isolates data, no auth/multi-tenancy.
- `ripgrep` (rg) required for `grep_search`.
