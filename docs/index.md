# Mnemo AI

<p align="center"><img src="assets/mnemoai-logo.png" width="120"></p>

A local agentic AI assistant with MCP (Model Context Protocol) integration, RAG capabilities, and intelligent conversation management. Built on LangGraph with LangChain for multi-provider LLM support (Ollama, Amazon Bedrock, OpenAI, Anthropic, Amazon SageMaker AI, LiteLLM).

![Demo](https://raw.githubusercontent.com/brunopistone/mnemoai/main/images/assistant-demo.gif)

## ✨ Key Features

- **🤖 Multi-Model Support**: Ollama (local), Amazon Bedrock, OpenAI, Anthropic (Claude), Amazon SageMaker AI, LiteLLM (100+ providers)
- **🔧 MCP Tool System**: Extensible tool architecture via Model Context Protocol
- **📚 RAG (Retrieval-Augmented Generation)**: Automatic document indexing and semantic search (_if enabled_)
- **💬 Advanced Chat Interface**: Multiline input, command system, conversation save/load
- **🧠 User Profile Learning**: Automatic learning from interactions for personalized responses
- **🧩 Episodic Memory**: Learns from successful task completions and retrieves similar solutions
- **📖 ACE Playbook**: Learns strategies from successes AND failures via Agentic Context Engineering
- **🔍 Web Search**: Integrated Brave Search API (_if available_)
- **🌐 Web Crawler**: Extract and index content from web pages
- **🖼️ Vision Support**: Image analysis with vision models (_if available_)
- **📁 File Operations**: Read/write/edit with support for text, CSV, JSON, PDF, DOCX
- **✏️ Precise File Editing**: Safe string replacement with validation and uniqueness checking
- **🔎 Fast Search Tools**: Glob pattern matching and ripgrep content search (10-100x faster)
- **📋 Todo Tracking**: Multi-step task management with real-time progress updates
- **⚡ Bash Execution**: Direct shell command execution with intelligent error handling
- **🛡️ Git Safety**: Protection against dangerous git operations with smart warnings
- **📝 Plan Mode**: Implementation planning workflow for complex tasks
- **🔄 Background Tasks**: Run long operations in parallel without blocking

## 📖 Project Structure

```yaml
mnemoai/                      # repo root
├── pyproject.toml                          # Packaging + `mnemoai` CLI entry point
├── requirements.txt                        # Dependencies
├── README.md                               # This file
├── pytest.ini                              # Pytest configuration
├── requirements-dev.txt                    # Dev/test dependencies
│
├── src/mnemoai/              # The single package (src layout)
│   ├── __init__.py
│   ├── __main__.py                         # `python -m mnemoai`
│   ├── main.py                             # Entry point (cli())
│   │
│   ├── client/                             # Client layer
│   │   ├── client.py                       # LangGraphClient facade (lifecycle, MCP, query)
│   │   ├── mcp_tool_wrapper.py             # MCP→LangChain adapter + MultiMCPClient (built-in + external servers)
│   │   ├── mcp_config.py                   # Loads external MCP servers from mcp.json
│   │   ├── agent/                          # Agent loop
│   │   │   ├── agent.py                    # LangGraph StateGraph agent with streaming
│   │   │   ├── router.py                   # Query classifier and routing
│   │   │   ├── orchestrator.py             # Task decomposition and worker orchestration
│   │   │   └── reasoning_utils.py          # Reasoning/thinking helpers for aux LLM calls
│   │   ├── ui/                             # User interface
│   │   │   ├── chat_interface.py           # Chat loop
│   │   │   └── spinner.py                  # Loading animations
│   │   ├── managers/                       # Business logic
│   │   │   ├── agent_conversation_manager.py  # Conversation state and token tracking
│   │   │   └── user_profile_manager.py     # User profiling and learning
│   │   └── memory/                         # Memory systems
│   │       ├── episodic_memory.py          # Episodic memory manager
│   │       ├── memory_store.py             # Curated persistent memory (MEMORY.md) store
│   │       ├── reflector.py                # ACE Reflector - extracts strategies
│   │       ├── playbook_store.py           # ACE Playbook - stores learned strategies
│   │       ├── faiss_store.py              # FAISS episodic store
│   │       └── chroma_store.py             # ChromaDB episodic store
│   │
│   ├── server/                             # MCP server layer
│   │   ├── server.py                       # FastMCP server (run as a subprocess)
│   │   ├── error_handler.py                # @tool_error_handler decorator (shared)
│   │   └── tools/                          # Tool implementations
│   │       ├── tools_manager.py            # Tool registration
│   │       ├── fs_read.py / fs_write.py / file_edit.py / file_search.py
│   │       ├── execute_bash.py / git_safety.py / todo_manager.py / plan_mode.py
│   │       ├── background_tasks.py / web_crawler.py / web_search.py
│   │       ├── describe_image.py / rag_tool.py / memory_tool.py
│   │       ├── rag/                        # RAG system (session, vector_store_controller, stores)
│   │       └── readers/                    # File readers (csv/json/pdf/docx/line/dir/search + chunking)
│   │
│   ├── models/                             # Model layer
│   │   ├── provider_params.py              # Single source of truth: per-provider config keys
│   │   ├── mantle_factory.py               # Bedrock Mantle model factory (multi-protocol)
│   │   ├── controllers/                    # Provider-dispatching controllers
│   │   │   ├── base_model_controller.py    # Minimal shared base
│   │   │   ├── llm_controller.py           # LLM initialization
│   │   │   ├── vision_model_controller.py  # Vision model initialization
│   │   │   └── embeddings_controller.py    # Embeddings initialization
│   │   └── chat_models/                    # Concrete LangChain ChatModel subclasses
│   │       ├── chat_ollama_wrapper.py      # Ollama model with penalty support
│   │       └── sagemaker_chat.py           # SageMaker ChatModel for LangChain
│   │
│   └── utils/                              # Utilities
│       ├── config.py                       # Config loader
│       ├── configurator.py                 # First-run setup + /config & /model flows
│       ├── paths.py                        # Central path helper (~/.mnemoai)
│       ├── logger.py                       # Logging utilities
│       ├── bm25.py                         # Lightweight BM25 (hybrid search)
│       ├── config.yaml.example             # Config templates (also .bedrock / .bedrock.mantle)
│       ├── mcp.json.example                # External MCP servers template
│       └── formatting/                     # Text formatting (code/url/response)
│
├── tests/                                  # Test suite (pytest)
│   ├── conftest.py                         # Puts src/ on sys.path
│   ├── unit/                               # Fast, deterministic, no deps
│   └── integration/                        # Live agent + Ollama + MCP
│
├── docs/                                   # ARCHITECTURE.md (detailed file map)
└── bash/                                   # Helper scripts
    ├── system-command-app/                 # `mnemoai` wrapper script
    ├── ollama-freeup-vram/                 # VRAM management
    └── ollama-env-mac/                     # Ollama config
```

## Next steps

- [Getting Started](getting-started.md) — install and configure the assistant
- [Usage](usage.md) — commands, feature toggles, and day-to-day use
- [Configuration](configuration.md) — full configuration reference
