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
- **`mcp_tool_wrapper.py`**: MCP to LangChain adapter
  - Wraps MCP tools as LangChain BaseTool
  - Handles async/sync conversion
- **`ui/`**: User interface components
  - `chat_interface.py`: Interactive chat loop with command handling
  - `spinner.py`: Loading animations
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
  - `fs_write.py`: File writing (dry-run preview); writes are hard-gated client-side by `REQUIRE_WRITE_CONFIRMATION`
  - `file_edit.py`: Precise string replacement with validation and uniqueness checking
  - `execute_bash.py`: Shell command execution with intelligent error handling
  - `file_search.py`: Fast file/content search (glob patterns + ripgrep)
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
- `configurator.py`: First-run interactive setup (when no config resolves) and the `/config` (full reconfigure) and `/model` (override one model section) chat commands
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
3. **Classifier** → Routes query to a category (simple*qa, code, research, knowledge, full) (\_if routing enabled*)
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

Session data is stored in `~/.mnemoai/{profile_name}/`:

```
~/.mnemoai/
└── {profile_name}/
    ├── conversations/           # Saved conversations
    ├── profiles/                # User profiles
    ├── todos/                   # Todo list data
    ├── rag_session_id.txt       # Current RAG session
    ├── rag_store_*.faiss        # FAISS vector index (or ChromaDB directory)
    ├── chunk_cache_*.db         # SQLite chunk cache
    └── models/                  # Per-model memory (isolated by chat model)
        └── {sanitized_model}/   # e.g. global.anthropic.claude-fable-5
            ├── episodic_memory/ # Episodic memory store (FAISS or ChromaDB)
            └── playbook/        # ACE playbook strategies and metrics
```

> **Model-scoped memory:** episodic memory and the playbook live under `models/{model}/` so trying a different chat model doesn't contaminate the memory/strategies learned with another. Conversations, todos, RAG, and the user profile remain shared across models.

#### Context Compaction

To keep long conversations within the model's context window, the assistant compacts history by summarizing it:

- **Automatic** — after a turn pushes the conversation past `MAX_CONVERSATION_TOKENS`, older messages are summarized into the system prompt while the most recent `LLM.KEEP_RECENT_MESSAGES` turns are kept verbatim.
- **Manual** — run `/compact` any time (optionally `/compact <focus instructions>` to steer what the summary emphasizes). Manual compaction keeps a smaller recent window (`LLM.MANUAL_COMPACT_KEEP_RECENT`).

The kept-verbatim window is bounded by **both** a message count and a token budget (`LLM.KEEP_RECENT_TOKEN_BUDGET`, default 25% of `MAX_CONVERSATION_TOKENS`). Walking newest→oldest, a message that would exceed the budget is summarized instead of kept — so a single oversized recent message (e.g. a pasted document that alone fills the context window) cannot survive compaction verbatim.

The summary preserves topics, decisions, and **tool calls/results** (which tools ran, their inputs, and outcomes), so the agent retains actionable context after compacting.

For the full per-file reference, see [Architecture Reference](ARCHITECTURE.md).
