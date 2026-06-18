# CLAUDE.md

## Project Overview

Local agentic AI assistant built on LangGraph + MCP (Model Context Protocol). The client spawns an MCP server as a subprocess, routes queries through a StateGraph (classify → orchestrate/call_model ↔ execute_tools), and persists episodic memory, learned strategies (ACE Playbook), and user profiles across sessions. Supports Ollama, AWS Bedrock, OpenAI, SageMaker, and LiteLLM as LLM providers.

## Quick Commands

```bash
# Run (verbose mode — shows thinking process)
python main.py

# Run (hide thinking)
python main.py --no-verbose

# Install dependencies
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

| Directory               | Role                                                                      |
| ----------------------- | ------------------------------------------------------------------------- |
| `client/`               | LangGraph agent, query routing, orchestrator, MCP bridge, UI              |
| `client/memory/`        | Episodic memory (ChromaDB/FAISS), ACE Playbook, Reflector                 |
| `client/managers/`      | Conversation token management, user profile learning                      |
| `client/ui/`            | prompt_toolkit chat loop, spinner                                         |
| `server/`               | FastMCP server entry point                                                |
| `server/tools/`         | Tool implementations (bash, file ops, git, web, RAG, vision, planning)    |
| `server/tools/rag/`     | Session-scoped vector store, hybrid search engine                         |
| `server/tools/readers/` | File format readers (PDF, DOCX, CSV, JSON, directory, line, search)       |
| `models/`               | LLM/embeddings/vision controllers, multi-provider abstraction             |
| `models/classes/`       | Custom LangChain model implementations (ChatOllamaWrapper, ChatSageMaker) |
| `utils/`                | Config singleton, logger, BM25, text formatting                           |
| `bash/`                 | Shell scripts (system command wrapper, Ollama VRAM management)            |

## Detailed File Map

### Entry Point

| File      | Purpose                                                                                                                                                     |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main.py` | CLI entry point. Parses `--no-verbose` flag, creates `LangGraphClient(server_path="server/server.py")`, starts it, creates `ChatInterface`, runs chat loop. |

### `client/` — Agent, Routing, Orchestration, UI

| File                         | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `client/__init__.py`         | Exports `LangGraphClient`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `client/client.py`           | **Central orchestration class.** Manages full lifecycle: MCP connection, LLM init, agent creation, query processing, episodic memory injection, playbook context injection, conversation save/load, RAG session, chunk cache, context clearing. Key methods: `start()`, `query()`, `clear_context()`, `save_conversation()`, `load_conversation()`, `reflect_and_learn()`, `_inject_episodic_context()`, `_inject_playbook_context()`.                                                                                                                          |
| `client/agent.py`            | **LangGraph StateGraph agent.** Builds graph with nodes: `classifier`, `orchestrator`, `agent` (call_model), `tools` (execute_tools). Handles streaming output with `CodeFormatter`, thinking/reasoning extraction, retry on empty content. Key class: `LangGraphAgent` with `invoke()`, `_build_graph()`, `_call_model()`, `_execute_tools()`, `_stream_response()`, `_classify()`, `_orchestrate()`, `_run_worker_loop()`, `_aggregate_results()`. Also has `convert_strands_messages_to_langchain()` / `convert_langchain_messages_to_strands()` converters. |
| `client/mcp_tool_wrapper.py` | **MCP↔LangChain bridge.** Runs background asyncio event loop in daemon thread for persistent MCP server connection. `MCPClientWrapper` manages lifecycle; `MCPToolWrapper(BaseTool)` wraps individual MCP tools with Pydantic arg schemas for LangChain compatibility. Sync wrappers use `run_coroutine_threadsafe`.                                                                                                                                                                                                                                            |
| `client/router.py`           | **Query classifier.** `QueryRouter.classify()` sends query + `ROUTING_PROMPT` to LLM, returns one of: `simple_qa`, `code`, `research`, `knowledge`, `full`. `ROUTE_TOOLS` dict maps each route to allowed tool names.                                                                                                                                                                                                                                                                                                                                           |
| `client/orchestrator.py`     | **Task decomposition.** `get_orchestrator_prompt()` and `get_aggregator_prompt()` load prompts from config. `parse_subtasks(content, fallback_query, valid_categories)` robustly parses JSON subtask list from LLM output with multiple fallback strategies.                                                                                                                                                                                                                                                                                                    |
| `client/reasoning_utils.py`  | **Shared reasoning-model helpers.** `disable_reasoning(model)` / `restore_reasoning(model, saved)` temporarily turn off thinking for auxiliary LLM calls (router classification, task decomposition) so output lands in `response.content` instead of the reasoning field. `extract_visible_text(content)` strips `<think>` tags / Bedrock thinking blocks. Used by `agent.py` and `router.py`.                                                                                                                                                                 |

### `client/managers/` — Conversation & Profile Management

| File                                            | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `client/managers/agent_conversation_manager.py` | **Conversation compaction** (token counting + LLM summarization). Auto-compacts when over `MAX_CONVERSATION_TOKENS`; `compact()` is the manual `/compact` path. Summarizes older messages into the system prompt while keeping recent turns verbatim — the kept window is bounded by BOTH message count (`KEEP_RECENT_MESSAGES` / `MANUAL_COMPACT_KEEP_RECENT`) and a token budget (`KEEP_RECENT_TOKEN_BUDGET`, default 25% of max), so an oversized recent message is summarized rather than kept. Preserves tool calls/results in the summary. Key methods: `count_tokens()`, `generate_summary()`, `manage_messages()`, `compact()`, `_compact()`, `_split_keep_recent()`. |
| `client/managers/user_profile_manager.py`       | Learns user preferences via Exponential Moving Average (EMA). Tracks: verbosity, directness, technical level, abstraction preference, top domains, tool success per intent. Generates compact profile summary for system prompt injection. Persists as JSON at `~/.personal-ai-assistant/{profile}/`. Key class: `UserProfileManager` with `analyze_conversation()`, `classify_intent()`, `get_profile_summary()`.                                                                                                                                                                                                                                                            |

### `client/memory/` — Episodic Memory & Learning

| File                               | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `client/memory/episodic_memory.py` | High-level episodic memory manager. Stores successful task patterns with tool usage. Supports query expansion (synonyms). Delegates to ChromaDB or FAISS store. Key functions: `is_task_successful()` (heuristic checks success/correction/error markers), `extract_tools_from_messages()`. Key class: `EpisodicMemoryManager` with `store()`, `retrieve()`, `clear()`.                                                                  |
| `client/memory/chroma_store.py`    | ChromaDB-backed episodic store with hybrid search (semantic from ChromaDB + BM25 re-ranking). Key class: `ChromaEpisodicStore` with `add()`, `search()`, `cleanup()`.                                                                                                                                                                                                                                                                    |
| `client/memory/faiss_store.py`     | FAISS-backed episodic store (alternative to ChromaDB). Same hybrid search pattern. Key class: `FAISSEpisodicStore` with `add()`, `search()`, `cleanup()`, `clear()`.                                                                                                                                                                                                                                                                     |
| `client/memory/reflector.py`       | **ACE Reflector.** Analyzes tool execution trajectories after each interaction. Detects failures (string_not_found, file_not_found, permission_denied, syntax_error, timeout, api_error). Extracts reusable strategies as `PlaybookEntry` objects. Tracks metrics (total/successful/failed calls, failure types, daily stats). Key class: `Reflector` with `analyze_tool_execution()`, `reflect_on_trajectory()`.                        |
| `client/memory/playbook_store.py`  | **ACE Playbook.** Append-only store for learned strategies with lazy semantic deduplication. Retrieves relevant entries by task context for system prompt injection. Key class: `PlaybookStore` with `append()`, `append_batch()`, `get_relevant_entries()`, `format_for_prompt()`, `_refine()` (triggers when over max_entries). Persists at `~/.personal-ai-assistant/{profile}/models/{model}/playbook/playbook.json` (model-scoped). |

### `client/ui/` — User Interface

| File                          | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `client/ui/__init__.py`       | Package marker                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `client/ui/chat_interface.py` | Interactive CLI using prompt_toolkit. Multiline input (Ctrl+J), commands: `/clear`, `/load`, `/save`, `/exit`, `/quit`, `/good`, `/compact [focus]` (manual context compaction), `/config` (re-run the configurator via `run_reconfigure()`; overwrites config.yaml), `/model` (override just one model section — LLM/vision/embeddings — via `run_model_override()`). `/config` and `/model` both call `_restart_in_place()`, which re-execs the process via `os.execv` so all settings — incl. MCP tool toggles decided at subprocess boot — take effect. Handles episodic memory storage (immediate and delayed modes), ACE reflection triggers, double Ctrl+C exit. Key class: `ChatInterface` with `run_chat_loop()`, `get_multiline_input()`. |
| `client/ui/spinner.py`        | Threaded "Thinking..." spinner animation shown during LLM processing. Key class: `Spinner` with `start()`, `stop()`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

### `server/` — MCP Server & Tools

| File                 | Purpose                                                                                                                             |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `server/__init__.py` | Package marker                                                                                                                      |
| `server/server.py`   | MCP server entry point (run as subprocess). Creates `FastMCP("MCP Server")`, calls `register_tools(mcp)`, runs via stdio transport. |

### `server/tools/` — Tool Implementations

| File                               | Purpose                                                                                                                                                                                                                                                             |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `server/tools/__init__.py`         | Creates global `ToolManager` singleton. Exports `register_tools`, `validate_file_path`, `count_tokens`, `vision_model`, `vision_model_controller`.                                                                                                                  |
| `server/tools/tools_manager.py`    | Central tool management. Initializes vision model, provides token counting, file path validation. `register_tools(mcp)` conditionally registers all tool categories based on config toggles.                                                                        |
| `server/tools/error_handler.py`    | `@tool_error_handler` decorator. Catches typed exceptions (FileNotFoundError, PermissionError, etc.) and returns structured JSON with `error_type`, `message`, `next_steps`.                                                                                        |
| `server/tools/execute_bash.py`     | `execute_bash(command, timeout=30)` — safe shell execution with timeout. Blocks dangerous commands (rm, mkfs, dd, shutdown, chmod 777). Returns stdout/stderr/exit_status.                                                                                          |
| `server/tools/file_edit.py`        | `file_edit(file_path, old_string, new_string, replace_all=False)` — precise string replacement. Validates file exists, checks for unique matches.                                                                                                                   |
| `server/tools/file_search.py`      | `glob_search(pattern, path, max_results, sort_by_mtime)` — file name search. `grep_search(pattern, path, file_pattern, case_insensitive, output_mode, context_lines, max_results)` — content search via ripgrep.                                                    |
| `server/tools/fs_read.py`          | `fs_read(path, mode, start_line, end_line, pattern, context_lines, depth)` — multi-mode reader. Modes: Line, Search, Directory, CSV, JSON, JSONL, PDF, DOCX. Delegates to readers/.                                                                                 |
| `server/tools/fs_write.py`         | `fs_write(path, command, ...)` — two-step write with confirmation (dry_run=True for preview, then dry_run=False + confirmed=True). Commands: create, str_replace, insert, append.                                                                                   |
| `server/tools/git_safety.py`       | `git_safe(command, allow_dangerous, reason)`, `git_status_safe()`, `git_commit_safe(message, add_all, add_files, amend, allow_empty)` — git with safety checks. Blocks force push to main/master, warns about hard resets.                                          |
| `server/tools/describe_image.py`   | `describe_image(image_path, question)` — sends base64 image to vision model for description.                                                                                                                                                                        |
| `server/tools/plan_mode.py`        | Multi-step planning workflow. Tools: `enter_plan_mode()`, `add_plan_step()`, `add_plan_file()`, `add_plan_risk()`, `present_plan()`, `approve_plan()`, `exit_plan_mode()`, `get_plan_status()`. Persists at `~/.personal-ai-assistant/plans/current_plan.json`.     |
| `server/tools/background_tasks.py` | Background task execution in threads. Tools: `start_background_task()`, `get_task_status()`, `get_task_output()`, `list_background_tasks()`, `cancel_background_task()`, `wait_for_task()`, `clear_completed_tasks()`. Output at `~/.personal-ai-assistant/tasks/`. |
| `server/tools/todo_manager.py`     | Task tracking. Enforces one in_progress task at a time. Tools: `todo_write(todos)`, `todo_read()`, `todo_clear()`. Persists at `~/.personal-ai-assistant/{profile}/todos/current_todos.json`.                                                                       |
| `server/tools/web_crawler.py`      | `web_crawler(url)` — extracts page content as markdown via crawl4ai. Optionally ingests large pages into RAG store.                                                                                                                                                 |
| `server/tools/web_search.py`       | `web_search(query, search_lang, num_results)` — internet search via Brave Search API. Returns structured results.                                                                                                                                                   |
| `server/tools/rag_tool.py`         | MCP-exposed RAG tools: `list_documents()`, `search_in_documents(query, top_k)`, `clear_documents()`.                                                                                                                                                                |

### `server/tools/rag/` — RAG Engine

| File                                          | Purpose                                                                                                                                                                                                                                                                                                                          |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `server/tools/rag/__init__.py`                | Exports `get_rag_session`, `reset_session_rag`, `SessionRAG`, `FaissStore`, `create_store`, `register_rag_tools`.                                                                                                                                                                                                                |
| `server/tools/rag/session.py`                 | **Core RAG engine.** Session-scoped vector store with hybrid search (semantic + BM25). `SessionRAG` class with `ingest(doc_id, content, chunk_size_tokens)` and `query(query_text, top_k)`. Cross-process session sharing via file-based session_id. Functions: `get_rag_session()`, `set_rag_session()`, `reset_session_rag()`. |
| `server/tools/rag/vector_store_controller.py` | Abstraction over FAISS/ChromaDB backends. Factory pattern via `VectorStoreController` with `add()`, `search()`, `clear()`, `detect_existing_store()` (static).                                                                                                                                                                   |
| `server/tools/rag/faiss_store.py`             | FAISS IndexFlatIP (cosine similarity on L2-normalized vectors). Thread-safe with `threading.Lock()`. File persistence (faiss index + pickle metadata). Key class: `FaissStore`.                                                                                                                                                  |
| `server/tools/rag/chroma_store.py`            | ChromaDB-backed store alternative. Persistent client with automatic collection management. Key class: `ChromaStore`.                                                                                                                                                                                                             |

### `server/tools/readers/` — File Format Readers

| File                                       | Purpose                                                                                                                                                                                                                                     |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `server/tools/readers/__init__.py`         | Exports all readers                                                                                                                                                                                                                         |
| `server/tools/readers/chunking_helper.py`  | Universal chunking + LLM summarization for large files. Recursive splitting with 10% overlap. SQLite chunk cache. Concurrent summarization with asyncio semaphore. Key functions: `process_large_content()`, `reset_session_chunk_cache()`. |
| `server/tools/readers/line_reader.py`      | `read_lines(path, start_line, end_line)` — line-based file reading with token limit.                                                                                                                                                        |
| `server/tools/readers/directory_reader.py` | `read_directory(path, depth)` — recursive directory listing.                                                                                                                                                                                |
| `server/tools/readers/csv_reader.py`       | `read_csv(path)` — CSV with auto-delimiter detection, encoding fallbacks, token truncation.                                                                                                                                                 |
| `server/tools/readers/json_reader.py`      | `read_json(path, start_line, end_line)` — JSON/JSONL reading with line ranges and token limits.                                                                                                                                             |
| `server/tools/readers/pdf_reader.py`       | `read_pdf(file_path)` — PyPDF2 reader. Large PDFs auto-ingest into RAG if enabled, else chunk+summarize.                                                                                                                                    |
| `server/tools/readers/docx_reader.py`      | `read_docx(file_path)` — python-docx reader. Same RAG/chunking fallback as PDF.                                                                                                                                                             |
| `server/tools/readers/search_reader.py`    | `search_file(path, pattern, context_lines)` — regex search within a single file with context.                                                                                                                                               |

### `models/` — LLM Provider Abstraction

| File                                    | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `models/__init__.py`                    | Empty package marker                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `models/base_model_controller.py`       | Abstract base with shared inference param methods: `_set_bedrock_inference_parameters()`, `_set_ollama_inference_parameters()`, `_set_openai_inference_parameters()`, `_set_sagemaker_inference_parameters()`.                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `models/llm_controller.py`              | **Primary LLM controller.** `LangChainLLMController(BaseModelController)` reads `MODEL_ID` config, initializes the correct LangChain chat model. Methods: `initialize_model()`, `get_model()`, `get_model_type()`. Supports: `bedrock` (ChatBedrockConverse), `mantle` (Bedrock Mantle, via `mantle_factory`), `ollama` (ChatOllamaWrapper), `openai` (ChatOpenAI), `sagemaker` (ChatSageMaker), `litellm` (ChatLiteLLM). Handles extended thinking (Bedrock Claude), Ollama reasoning, OpenAI reasoning_effort. Note: `temperature` only sent when explicitly configured (newer Claude models reject it). Optional `ENDPOINT_URL` overrides the Bedrock endpoint. |
| `models/mantle_factory.py`              | **Bedrock Mantle model factory.** `build_mantle_model(model_id, ...)` returns the right LangChain model for `API_PROTOCOL`: `chat_completions` (ChatOpenAI, `/v1`), `responses` (ChatOpenAI `use_responses_api=True`, `/openai/v1`), `anthropic` (ChatAnthropic, `/anthropic`). Auth: uses a Bedrock API key when present (`MODEL_ID.API_KEY` or the `BEDROCK_API_KEY` env var), else mints a short-lived bearer token via `aws_bedrock_token_generator.provide_token()`. Used by both the LLM and vision controllers.                                                                                                                                                                                                                                                             |
| `models/embeddings_controller.py`       | Multi-provider embeddings with LRU caching. Supports Ollama, Bedrock, OpenAI, SageMaker. Falls back to SHA256-based deterministic embeddings on failure. Key class: `EmbeddingsController` with `embed(texts)` → numpy array.                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `models/vision_model_controller.py`     | Vision model controller. `VisionModelController(BaseModelController)` with `describe_image()`, `format_request()` (multimodal HumanMessage with base64), and `_content_to_text()` (normalizes string/list-of-blocks responses). Supports Bedrock, Mantle (all 3 protocols), Ollama, OpenAI.                                                                                                                                                                                                                                                                                                                                                                        |
| `models/classes/__init__.py`            | Empty                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `models/classes/chat_ollama_wrapper.py` | `ChatOllamaWrapper(ChatOllama)` — extends ChatOllama to add `presence_penalty` and `frequency_penalty` support in the options dict.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `models/classes/sagemaker_chat.py`      | `ChatSageMaker(BaseChatModel)` — full LangChain BaseChatModel for SageMaker endpoints. Supports OpenAI chat format and HuggingFace text_generation format. Implements `_generate()` and `_stream()` (SSE parsing). Handles reasoning/thinking tags. `bind_tools()` support.                                                                                                                                                                                                                                                                                                                                                                                        |

### `utils/` — Shared Utilities

| File                                  | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `utils/__init__.py`                   | Package marker                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `utils/paths.py`                      | **Central path helper — single source of truth for all runtime locations.** `app_home()` (defaults to `~/.personal-ai-assistant`, honors `$PERSONAL_AI_ASSISTANT_HOME`), `config_path()`, `plans_dir()`, `tasks_dir()`, `profile_dir(profile=None)`, `model_dir(model_name, profile=None)`, `sanitize_model_name(name)`. Every call site that touches the home dir routes through here. Lazy-imports `utils.config` to avoid a cycle.                                                                                                                                                                                                          |
| `utils/config.py`                     | **Singleton config manager.** Loads config via `_resolve_config_path()` (`$PERSONAL_AI_ASSISTANT_CONFIG` → `<app_home>/config.yaml` → repo `utils/config.yaml` fallback; prints a copy-a-template hint if none found). Exposes `.get("SECTION.KEY", default)`, `.system_prompt` property (injects current date), and `reload()` (re-reads disk into the existing singleton, used after first-run setup). Sets env vars from `ENV` section.                                                                                                                                                                                                     |
| `utils/configurator.py`               | **First-run interactive setup.** When no config resolves, `cli()` (in `main.py`) runs `run_first_run_setup()` on a TTY: picks a provider template (Ollama/Bedrock/Mantle) and prompts for chat model + connection + optional max output tokens (`MAX_TOKENS`; `none`/blank drops it) + mandatory max context window (`MAX_CONVERSATION_TOKENS`, default 65536), vision model (mirrors chat host/region, own optional max output tokens), profile, Brave key, and each feature toggle. Patches them in via line-targeted edits (`_set_in_section`/`_set_top_level`/`_set_bool`) — reading current defaults with `_get_in_section`/`_get_top_level` — so the rich prompt blocks/comments survive. Writes `<app_home>/config.yaml`, then the caller calls `config.reload()`. `config_exists()` gates the trigger. `run_model_override()` (the `/model` command) edits just one model section in place — chat/vision/embeddings (embeddings offered only when configured) — using depth-agnostic helpers (`_get_field`/`_set_field`/`_remove_field`) that reach the nested `RAG.EMBED_MODEL_ID`. Switching a section away from Ollama strips Ollama-only params (`TEMPERATURE`/penalties/`STOP`, plus stale `HOST`/`PORT`) since other providers may reject them, and reminds the user to tune `config.yaml` for the new provider. `STOP` is kept in the example template (documentation) but never written into a generated config. |
| `utils/logger.py`                     | Logger setup. Configurable via `LOG_LEVEL` env var. Suppresses noisy Brave Search logs. Exports `logger` singleton.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `utils/bm25.py`                       | Lightweight BM25 (Okapi BM25) implementation. `BM25` class with `fit(corpus)` and `score(query)`. `tokenize()` function (regex word tokenizer). Used by episodic memory and RAG hybrid search.                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `utils/formatting/__init__.py`        | Exports `make_urls_clickable` and everything from `response_parser`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `utils/formatting/code_formatter.py`  | Real-time streaming syntax highlighting. Handles triple-backtick code blocks (Pygments language detection) and inline code. `CodeFormatter` with `process_chunk()` and `flush()`.                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `utils/formatting/response_parser.py` | Extracts structured content from AI responses. Functions: `extract_answer()` (from `<answer>` tags), `extract_thinking()` (from `<think>`/`<thinking>` tags), `format_response()`.                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `utils/formatting/url_formatter.py`   | Makes URLs clickable in terminal (ANSI escapes for iTerm/VSCode, fallback to color). Handles plain URLs and markdown links. Functions: `make_urls_clickable()`, `highlight_urls()`, `format_url()`.                                                                                                                                                                                                                                                                                                                                                                                                                                            |

## Key Patterns

### Config singleton (`utils/config.py`)

All configuration flows through `Config()` which loads `utils/config.yaml` (gitignored). Access via `Config().get("SECTION.KEY", default)`. The `.system_prompt` property injects the current date.

### MCP tool registration (`server/tools/tools_manager.py`)

`ToolManager.register_tools(mcp)` conditionally registers tool groups based on config toggles. Each tool file defines functions decorated with `@mcp.tool()`.

### Hybrid search (semantic + BM25)

Used in both episodic memory and RAG. Pattern: get top-N candidates from vector store, get top-N from BM25, merge with configurable weights (`utils/bm25.py`).

### Multi-provider LLM abstraction (`models/llm_controller.py`)

`LangChainLLMController.initialize_model()` dispatches on `MODEL_ID.TYPE` (bedrock/mantle/ollama/openai/sagemaker/litellm). Each provider has inference parameter mapping in `BaseModelController`.

### Bedrock Mantle (`models/mantle_factory.py`)

`TYPE: mantle` reaches AWS Bedrock Mantle via a bearer token minted from standard AWS (SigV4) credentials. `API_PROTOCOL` selects the wire protocol: `chat_completions` (OpenAI `/v1`), `responses` (OpenAI Responses `/openai/v1`, e.g. GPT-5.4), `anthropic` (Anthropic Messages `/anthropic`, Claude). The factory is shared by the LLM and vision controllers. Model availability varies by region (e.g. GPT-5.4 is in us-west-2).

### Conversation compaction (`client/managers/agent_conversation_manager.py`)

Keeps the conversation under `MAX_CONVERSATION_TOKENS` by summarizing older messages into the system prompt while keeping recent turns verbatim. Triggers automatically when over budget, or manually via `/compact`. The kept window is bounded by message count AND a token budget so an oversized recent message (e.g. a pasted document) is summarized, not kept. Tool calls/results are preserved in the summary.

### Query routing (`client/router.py`)

`QueryRouter.classify()` uses the LLM to categorize queries. Routes map to tool subsets in `ROUTE_TOOLS` dict — only relevant tools are bound per query.

### Orchestrator-workers (`client/orchestrator.py`, `client/agent.py`)

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
- **Error handling in tools** — `@tool_error_handler` decorator (`server/tools/error_handler.py`) for standardized responses
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
- **Unit tier (default):** deterministic, pure-logic tests — no LLM, Ollama, or network needed, runs in seconds. Covers `utils/bm25.py`, `client/reasoning_utils.py`, `utils/formatting/` (response_parser, url_formatter, code_formatter), `client/orchestrator.parse_subtasks`, `server/tools/error_handler.py`, `server/tools/git_safety.py` (command-danger classification), `server/tools/file_edit.py` + `glob_search`, `execute_bash` timeout/process-group behavior, `client/memory/episodic_memory` heuristics, Bedrock/Mantle model wiring (`test_bedrock_endpoint.py`), vision content normalization (`test_vision_content.py`), and conversation compaction incl. token-aware retention (`test_conversation_compaction.py`).
- Unit tests must not require a `config.yaml` — modules degrade gracefully without one. Keep import-time side effects config-independent so new code stays unit-testable.
- **Integration tier (`tests/integration/`, marked `@pytest.mark.integration`):** drives the real `LangGraphClient` + Ollama + MCP subprocess (greeting/routing, tool calls, bash timeout, no-silent-empty-turn). Auto-skipped unless a runtime `config.yaml` exists AND the configured Ollama host is reachable (see `tests/integration/conftest.py`). The shared client is session-scoped; an autouse fixture calls `clear_context()` between tests for isolation.

## Adding a New MCP Tool

1. Create `server/tools/my_tool.py`
2. Define function with `@mcp.tool()` decorator (receives `mcp` from registration)
3. Add `@tool_error_handler` for standardized error responses
4. Register in `server/tools/tools_manager.py` → `register_tools()` method
5. If conditionally enabled, gate behind a config toggle
6. Add route mapping in `client/router.py` → `ROUTE_TOOLS` if it belongs to a specific category

## Adding a New LLM Provider

1. Add provider case in `models/llm_controller.py` → `initialize_model()`
2. Add inference parameter method in `models/base_model_controller.py`
3. If custom LangChain model needed, create class in `models/classes/`
4. Add embedding support in `models/embeddings_controller.py` if provider offers embeddings
5. Add vision support in `models/vision_model_controller.py` if applicable
6. Document config structure in all `utils/config.yaml*.example` templates

## Known Limitations

- Unit tests cover pure logic only — agent/LLM integration paths still need manual verification
- MCP server is a subprocess — debugging requires attaching to child process or reading logs
- No Docker/containerization — runs directly on host with system Python/conda/venv
- No database — all persistence is file-based (JSON, FAISS index files, SQLite chunk cache)
- Single-user — profile name in config isolates data but no auth/multi-tenancy
- `ripgrep` (rg) required for `grep_search` tool — install separately
