# Architecture

## 🏗️ Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                             │
│                    (Application Entry)                      │
└─────────────────────────────┬───────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
      ┌─────────────────┐            ┌──────────────────┐
      │ LangGraphClient │◄──────────►│  MCP Server      │
      │  (client.py)    │            │  (server.py)     │
      └────────┬────────┘            └────────┬─────────┘
               │                              │
          ┌────┴─────┐                        ▼
          │          │                   ┌──────────┐
          ▼          ▼                   │  Tools   │
      ┌────────┐ ┌──────────┐            └────┬─────┘
      │  UI    │ │ Managers │                 │
      └────────┘ └──────────┘            ┌────┴────┐
          │          │                   │         │
          └────┬─────┘                   ▼         ▼
               ▼                    ┌──────────┐ ┌─────┐
          ┌──────────┐              │ Readers  │ │ RAG │
          │LangGraph │              └──────────┘ └─────┘
          │  Agent   │
          └──────────┘
```

### Component Breakdown

#### 1. **Client Layer** (`client/`)

The client manages the conversation flow and user interaction.

- **`client.py`**: Core LangGraph client
  - Initializes MCP connection
  - Manages conversation state
  - Handles model configuration
  - Coordinates managers (profile, conversation)
- **`agent.py`**: LangGraph agent implementation
  - State graph with agent and tools nodes
  - Streaming support with reasoning display
  - Code syntax highlighting
- **`router.py`**: Query classifier and routing
  - Classifies queries into categories (simple_qa, code, research, knowledge, full)
  - Routes each category to a specialized tool subset
  - Configurable classifier prompt via `ROUTING_PROMPT` in config
- **`orchestrator.py`**: Task decomposition and worker orchestration
  - Decomposes complex tasks into ordered subtasks with category assignments
  - Configurable orchestrator and aggregator prompts via config
- **`reasoning_utils.py`**: Shared reasoning/thinking helpers
  - Temporarily disables reasoning for auxiliary LLM calls (routing, task decomposition) so output lands in the response content
  - Extracts visible text from `<think>` tags and Bedrock thinking blocks
- **Pure agent collaborators** (stateless logic extracted from `agent.py`; `LangGraphAgent` delegates to them):
  - `message_codec.py`: Strands ↔ LangChain message conversion
  - `message_sanitizer.py`: repairs orphaned tool-call/result pairs so strict providers don't reject the request
  - `plan_policy.py`: plan-mode block decision + read-only-bash heuristic + data tables
  - `tool_formatting.py`: tool-call marker rendering, arg normalization, tool-error translation
- **`mcp_tool_wrapper.py`**: MCP to LangChain adapter
  - Wraps MCP tools as LangChain BaseTool
  - Handles async/sync conversion
- **`ui/`**: User interface components
  - `chat_interface.py`: Interactive chat loop with command handling
  - `tui.py`: pinned-input prompt_toolkit UI — `>` stays at the bottom while output streams above it (via `patch_stdout`); slash completion, history, input queue, Esc/Ctrl+C cancel, in-app confirmations, full-screen dialogs; degrades to a plain `input()` loop off-TTY
  - `turn_view.py`: styled "Thought for Ns…" reasoning block + `ToolName`/`↳ arg` tool blocks shown above the pinned input
  - `spinner.py`: status/spinner animation (stdout mode + a state sink for the pinned UI)
- **`managers/`**: Business logic
  - `agent_conversation_manager.py`: Conversation state and token tracking
  - `user_profile_manager.py`: Automatic user profiling and learning

#### 2. **Server Layer** (`server/`)

MCP server that provides tools to the LLM.

- **`server.py`**: FastMCP server initialization
- **`error_handler.py`**: `@tool_error_handler` decorator (shared by all tools)
- **`tools/`**: Tool implementations
  - `tools_manager.py`: Centralized tool registration and utilities
  - `fs_read.py`: File reading (text, CSV, JSON, PDF, DOCX)
  - `fs_write.py`: File writing; writes are gated client-side by `REQUIRE_WRITE_CONFIRMATION` and blocked from system dirs by the server-side `safety/` floor
  - `file_edit.py`: Precise string replacement with validation and uniqueness checking; same server-side write-path floor as `fs_write`
  - `execute_bash.py`: Shell command execution with intelligent error handling; catastrophic commands (`rm -rf /`, `mkfs`, `dd` to a device, `shutdown`, fork bomb) are hard-blocked by the server-side `safety/` floor
  - `file_search.py`: Fast file/content search (glob patterns + ripgrep)
  - **`safety/`**: Server-side safety policies enforced below the client confirmation gate
    - `bash_policy.py`: `classify_shell_command` — blocks catastrophic shell commands (shared by `execute_bash` + `start_background_task`)
    - `path_policy.py`: `classify_write_path` — blocks writes into system directories (shared by `fs_write` + `file_edit`)
  - `todo_manager.py`: Todo list management for multi-step tasks
  - `web_search.py`: Brave Search integration
  - `web_crawler.py`: Web page content extraction with RAG integration
  - `describe_image.py`: Vision model image analysis
  - `rag_tool.py`: RAG tools registration
  - **`rag/`**: RAG system
    - `session.py`: Session-scoped RAG management with hybrid search
    - `vector_store_controller.py`: Vector store abstraction layer
    - `faiss_store.py`: FAISS vector store implementation
    - `chroma_store.py`: ChromaDB vector store implementation
  - **`readers/`**: Specialized file readers
    - `line_reader.py`, `directory_reader.py`, `search_reader.py`
    - `csv_reader.py`, `json_reader.py`
    - `pdf_reader.py`, `docx_reader.py`
    - `chunking_helper.py`: Document chunking for RAG

#### 3. **Models Layer** (`models/`)

Model controllers and custom implementations.

- `provider_params.py`: Single source of truth for the config keys each provider consumes (per modality); controllers build their client kwargs from it via `build_kwargs`, and `/model` prunes unsupported keys from it
- `mantle_factory.py`: Bedrock Mantle factory (chat_completions / responses / anthropic protocols), shared by the LLM and vision controllers
- **`controllers/`** (provider-dispatching model initialization):
  - `base_model_controller.py`: Minimal shared base type for the controllers
  - `llm_controller.py`: LLM model initialization (Bedrock, Mantle, Ollama, OpenAI, Anthropic, SageMaker AI, LiteLLM)
  - `vision_model_controller.py`: Vision model initialization
  - `embeddings_controller.py`: Embedding model initialization for RAG
- **`chat_models/`** (concrete LangChain `ChatModel` subclasses):
  - `chat_ollama_wrapper.py`: Extends ChatOllama with `presence_penalty` and `frequency_penalty` support
  - `sagemaker_chat.py`: Full LangChain `BaseChatModel` for SageMaker endpoints (streaming, tool calling, reasoning)

#### 4. **Utils Layer** (`utils/`)

Shared utilities and configuration.

- `config.py`: Configuration loader
- `configurator.py`: First-run interactive setup (when no config resolves) and the `/config` (full reconfigure), `/model` (override one model section), `/params` (tune inference params), and `/features` (toggle `ENABLE_*` subsystems) chat commands
- `paths.py`: Central path helper — single source of truth for the app home (`~/.mnemoai`, override with `$MNEMOAI_HOME`) and all runtime subdirectories (config, plans, tasks, per-profile, per-model)
- `config.yaml.example`: Configuration template (copy to `config.yaml` and add your settings; `.bedrock` and `.bedrock.mantle` variants also provided)
- `bm25.py`: Lightweight BM25 implementation for hybrid (semantic + keyword) search
- `logger.py`: Logging utilities (stderr output)
- **`formatting/`**: Text formatting
  - `code_formatter.py`: Code syntax highlighting
  - `url_formatter.py`: URL highlighting
  - `response_parser.py`: Response processing

### Data Flow

1. **User Input** → `ChatInterface` → `LangGraphClient`
2. **Client** → Invokes LangGraph agent with MCP tools
3. **Classifier** → Routes query to a category (`simple_qa`, `code`, `research`, `knowledge`, `full`) (_if routing enabled_)
4. **Orchestrator** → For `full` tasks: decomposes into subtasks, spawns workers, aggregates results (_if orchestration enabled_)
5. **LangGraph** → Executes agent node with route-specific tools, decides to use tools
6. **MCP Server** → Executes tool (e.g., fs_read, web_search, RAG)
7. **Tool Result** → Returned to agent via tools node
8. **LangGraph** → Continues agent loop until response complete
9. **Response** → Displayed to user via `ChatInterface`

### Session Management

Each chat session has a unique ID used for:

- RAG document indexing (session-scoped)
- Chunk caching for file summarization
- The session transcript that `--resume` restores (see [Resuming a session](../guides/usage.md#resuming-a-session))

Session data is stored in `~/.mnemoai/{profile_name}/`:

```
~/.mnemoai/
└── {profile_name}/
    ├── conversations/           # Saved conversations (/save — kept until you delete them)
    ├── sessions/                # Recorded sessions for --resume, per launch directory
    │   └── {directory}/         # e.g. Users-you-dev-myproject
    │       └── session_*.jsonl  # append-only, one file per session (expires: 30d)
    ├── profiles/                # User profiles
    ├── todos/                   # Todo list data
    ├── rag_session_id_*.txt     # Per-instance RAG session pointer (one per tab)
    ├── chunk_session_id_*.txt   # Per-instance chunk-cache session pointer
    ├── rag_store_*.faiss        # FAISS vector index (or ChromaDB directory)
    ├── chunk_cache_*.db         # SQLite chunk cache
    └── models/                  # Per-model memory (isolated by chat model)
        └── {sanitized_model}/   # e.g. global.anthropic.claude-fable-5
            ├── episodic_memory/ # Episodic memory store (FAISS or ChromaDB)
            └── playbook/        # ACE playbook strategies and metrics
```

> **Model-scoped memory:** episodic memory and the playbook live under `models/{model}/` so trying a different chat model doesn't contaminate the memory/strategies learned with another. Conversations, todos, RAG, and the user profile remain shared across models.

> **Session transcripts vs saved conversations:** `sessions/` is written **automatically**, one append-only file per session, grouped by the directory you launched from — that's what `--resume` reads, and old files expire (`SESSION_MAX_AGE_DAYS`, default 30). `conversations/` is written **only when you ask** (`/save`) and is never expired. The transcript is append-only rather than a snapshot of the live history because [compaction](../guides/memory.md) rewrites that history in place; recording each turn as it happens means a resumed session **shows** the whole conversation, not a summarized stub — while a compaction also records the summary and the window it kept, so resuming **restores** the compacted state rather than re-inflating everything the summary already covers. Sub-agent runs are excluded (they run on their own isolated context, so logging them would interleave a second conversation).

> **Multi-instance isolation:** several `mnemoai` instances (e.g. one per terminal tab) share the profile dir, so the per-session RAG/chunk pointer files are **namespaced per-instance** (`MNEMOAI_INSTANCE_ID`, inherited by each instance's MCP subprocess). One instance uses only its own session's `rag_store_*` / `chunk_cache_*`, and exiting cleans up only its own — never another live tab's. Stale orphans from a crashed instance are pruned at startup on a 7-day age policy.

#### Context Compaction

To keep long conversations within the model's context window, the assistant compacts history by summarizing it:

- **Automatic** — after a turn pushes the conversation past `MAX_CONVERSATION_TOKENS`, older messages are summarized into the system prompt while the most recent `LLM.KEEP_RECENT_MESSAGES` turns are kept verbatim.
- **Manual** — run `/compact` any time (optionally `/compact <focus instructions>` to steer what the summary emphasizes). Manual compaction keeps a smaller recent window (`LLM.MANUAL_COMPACT_KEEP_RECENT`).

The kept-verbatim window is bounded by **both** a message count and a token budget (`LLM.KEEP_RECENT_TOKEN_BUDGET`, default 25% of `MAX_CONVERSATION_TOKENS`). Walking newest→oldest, a message that would exceed the budget is summarized instead of kept — so a single oversized recent message (e.g. a pasted document that alone fills the context window) cannot survive compaction verbatim.

The summary preserves topics, decisions, and **tool calls/results** (which tools ran, their inputs, and outcomes), so the agent retains actionable context after compacting.

A compaction is also **recorded in the session transcript** — the messages that stayed live, plus the summary when older turns were summarized away — so `--resume`, `/load` and `/branch` come back to the context size the session ended at instead of restoring the raw history it replaced. The cheaper tool-result eviction pass is recorded the same way, even though it produces no summary. See [Resuming a session](../guides/usage.md#resuming-a-session).

For the full per-file reference, see [`ARCHITECTURE.md`](https://github.com/brunopistone/mnemoai/blob/main/ARCHITECTURE.md) in the repository (a repo-only, agent/contributor-facing reference, not part of this docs site).
