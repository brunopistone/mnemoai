# Personal AI Assistant

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A local agentic AI assistant with MCP (Model Context Protocol) integration, RAG capabilities, and intelligent conversation management. Built on LangGraph with LangChain for multi-provider LLM support (Ollama, Amazon Bedrock, OpenAI, Amazon SageMaker AI, LiteLLM).

![Demo](images/assistant-demo.gif)

## ✨ Key Features

- **🤖 Multi-Model Support**: Ollama (local), Amazon Bedrock, Amazon SageMaker AI, LiteLLM (100+ providers)
- **🔧 MCP Tool System**: Extensible tool architecture via Model Context Protocol
- **📚 RAG (Retrieval-Augmented Generation)**: Automatic document indexing and semantic search (_if enabled_)
- **💬 Advanced Chat Interface**: Multiline input, command system, conversation save/load
- **🧠 User Profile Learning**: Automatic learning from interactions for personalized responses
- **🧩 Episodic Memory**: Learns from successful task completions and retrieves similar solutions
- **📖 ACE Playbook**: Learns strategies from successes AND failures via Agentic Context Engineering
- **📊 Training Data Collection**: SFT markers for quality training data
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
ai-assistant/
├── main.py                                 # Entry point
├── requirements.txt                        # Dependencies
├── README.md                               # This file
│
├── client/                                 # Client layer
│   ├── client.py                           # LangGraph client
│   ├── agent.py                            # LangGraph agent with streaming
│   ├── mcp_tool_wrapper.py                 # MCP to LangChain tool adapter
│   ├── ui/                                 # User interface
│   │   ├── chat_interface.py               # Chat loop
│   │   └── spinner.py                      # Loading animations
│   ├── managers/                           # Business logic
│   │   ├── agent_conversation_manager.py   # Conversation state and token tracking
│   │   └── user_profile_manager.py         # User profiling and learning
│   └── memory/                             # Memory systems
│       ├── episodic_memory.py              # Episodic memory manager
│       ├── reflector.py                    # ACE Reflector - extracts strategies
│       ├── playbook_store.py               # ACE Playbook - stores learned strategies
│       ├── faiss_store.py                  # FAISS episodic store
│       └── chroma_store.py                 # ChromaDB episodic store
│
├── server/                                 # MCP server layer
│   ├── server.py                           # FastMCP server
│   └── tools/                              # Tool implementations
│       ├── tools_manager.py                # Tool registration
│       ├── fs_read.py                      # File reading
│       ├── fs_write.py                     # File writing (with confirmation)
│       ├── file_edit.py                    # Precise file editing
│       ├── execute_bash.py                 # Bash execution
│       ├── file_search.py                  # Fast glob and grep search
│       ├── todo_manager.py                 # Todo list management
│       ├── error_handler.py                # Standardized error handling
│       ├── git_safety.py                   # Git operations with safety checks
│       ├── plan_mode.py                    # Implementation planning workflow
│       ├── background_tasks.py             # Background task execution
│       ├── web_crawler.py                  # Web crawler
│       ├── web_search.py                   # Web search
│       ├── describe_image.py               # Image analysis
│       ├── rag_tool.py                     # RAG tools registration
│       ├── rag/                            # RAG system
│       │   ├── session.py                  # Session management with hybrid search
│       │   ├── vector_store_controller.py  # Store abstraction layer
│       │   ├── chroma_store.py             # ChromaDB implementation
│       │   └── faiss_store.py              # FAISS implementation
│       └── readers/                        # File readers
│           ├── line_reader.py              # Text file reader
│           ├── directory_reader.py         # Directory listing
│           ├── search_reader.py            # File search
│           ├── csv_reader.py               # CSV file reader
│           ├── json_reader.py              # JSON file reader
│           ├── pdf_reader.py               # PDF reader with RAG integration
│           ├── docx_reader.py              # DOCX reader with RAG integration
│           └── chunking_helper.py          # Document chunking utilities
│
├── models/                                 # Model layer
│   ├── base_model_controller.py            # Base controller class
│   ├── llm_controller.py                   # LangChain LLM initialization
│   ├── vision_model_controller.py          # Vision model initialization
│   ├── embeddings_controller.py            # Embeddings initialization
│   └── classes/
│       └── sagemaker_chat.py               # SageMaker Handler class for Langchain
│
├── utils/                                  # Utilities
│   ├── config.py                           # Config loader
│   ├── config.yaml                         # Main config
│   ├── logger.py                           # Logging utilities
│   └── formatting/                         # Text formatting
│       ├── code_formatter.py               # Code syntax highlighting
│       ├── url_formatter.py                # URL highlighting
│       └── response_parser.py              # Response processing
│
└── bash/                                   # Helper scripts
    ├── personal-ai-assistant-wrapper.sh    # System command wrapper
    ├── ollama-freeup-vram/                 # VRAM management
    └── ollama-env-mac/                     # Ollama config
```

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

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- (Optional) Ollama installed for local models
- (Optional) AWS credentials for Bedrock/SageMaker
- (Optional) ripgrep for fast content search

### Installation

1. **Clone the repository**:

```bash
git clone https://github.com/brunopistone/personal-ai-assistant.git
cd personal-ai-assistant
```

2. **Set up Python environment** (choose one):

**Option A: venv**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Option B: uv**

```bash
uv venv
uv pip install -r requirements.txt
```

**Option C: conda**

```bash
conda create -n personal-ai-assistant python=3.11
conda activate personal-ai-assistant
pip install -r requirements.txt
```

3. **Install ripgrep (optional but recommended for fast search)**:

Ripgrep provides 10-100x faster content search than traditional grep. Required for `grep_search` tool.

**macOS:**

```bash
brew install ripgrep
```

**Ubuntu/Debian:**

```bash
sudo apt install ripgrep
```

**Fedora/RHEL:**

```bash
sudo dnf install ripgrep
```

**Windows (via Chocolatey):**

```bash
choco install ripgrep
```

**From source:**

```bash
cargo install ripgrep
```

**Verify installation:**

```bash
rg --version  # Should show ripgrep version
```

If ripgrep is not installed, the assistant will automatically fall back to using `execute_bash` with standard `grep`, but performance will be significantly slower.

4. **Configure the application**:

Edit `utils/config.yaml`:

```yaml
MODEL_ID:
  NAME: qwen3-4b-thinking-2507-q6-k:latest
  TYPE: ollama # or bedrock, sagemaker
  HOST: localhost
  PORT: 11434
```

5. **Run the assistant**:

```bash
python main.py
```

### Optional: Create System Command

Make the assistant accessible from anywhere in your terminal. The wrapper automatically detects your Python environment (venv, uv, or conda).

```bash
chmod +x bash/system-command-app/personal-ai-assistant-wrapper.sh
ln -sf $(pwd)/bash/system-command-app/personal-ai-assistant-wrapper.sh /usr/local/bin/personal-ai-assistant
```

Then run from anywhere:

```bash
personal-ai-assistant
```

See `bash/system-command-app/README.md` for details.

## 💡 Usage

### Basic Chat

Simply type your questions and press Enter. The assistant will respond using available tools when needed.

```
You: What files are in the current directory?
Assistant: [Uses fs_read tool to list directory contents]

You: Read the README.md file
Assistant: [Uses fs_read tool and displays content]
```

### Commands

| Command            | Description                                   |
| ------------------ | --------------------------------------------- |
| `/exit` or `/quit` | Exit the application                          |
| `/clear`           | Clear conversation history and RAG index      |
| `/save`            | Save current conversation                     |
| `/load <path>`     | Load a saved conversation                     |
| `/good`            | Mark last response as good (for SFT training) |

### Keyboard Shortcuts

- `Ctrl+J`: Insert new line in input
- `Enter`: Submit message
- `Ctrl+C`: Interrupt operation (press twice to exit)

### Verbose Mode

Control thinking process visibility:

```bash
python main.py              # Verbose mode (shows thinking)
python main.py --no-verbose # Hide thinking process
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
- **`mcp_tool_wrapper.py`**: MCP to LangChain adapter
  - Wraps MCP tools as LangChain BaseTool
  - Handles async/sync conversion
- **`ui/`**: User interface components
  - `chat_interface.py`: Interactive chat loop with command handling
  - `spinner.py`: Loading animations
- **`managers/`**: Business logic
  - `agent_conversation_manager.py`: Conversation state and token tracking
  - `user_profile_manager.py`: Automatic user profiling and learning
  - `dpo_collector.py`: DPO preference pair collection

#### 2. **Server Layer** (`server/`)

MCP server that provides tools to the LLM.

- **`server.py`**: FastMCP server initialization
- **`tools/`**: Tool implementations
  - `tools_manager.py`: Centralized tool registration and utilities
  - `fs_read.py`: File reading (text, CSV, JSON, PDF, DOCX)
  - `fs_write.py`: File writing with mandatory user confirmation (dry-run preview)
  - `edit.py`: Precise string replacement with validation and uniqueness checking
  - `execute_bash.py`: Shell command execution with intelligent error handling
  - `search.py`: Fast file/content search (glob patterns + ripgrep)
  - `todo.py`: Todo list management for multi-step tasks
  - `error_handler.py`: Standardized error handling decorator for all tools
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

- **Controllers** (centralized model initialization):
  - `base_model_controller.py`: Base class with shared parameter setup
  - `llm_controller.py`: LLM model initialization (Bedrock, Ollama, OpenAI, SageMaker AI)
  - `vision_model_controller.py`: Vision model initialization
  - `embeddings_controller.py`: Embedding model initialization for RAG
- **Custom implementations** (`classes/`):
  - `sagemaker_chat.py`: SageMaker Handler class for Langchain

#### 4. **Utils Layer** (`utils/`)

Shared utilities and configuration.

- `config.py`: Configuration loader
- `config.yaml`: Main configuration file
- `logger.py`: Logging utilities (stderr output)
- **`formatting/`**: Text formatting
  - `code_formatter.py`: Code syntax highlighting
  - `url_formatter.py`: URL highlighting
  - `response_parser.py`: Response processing

### Data Flow

1. **User Input** → `ChatInterface` → `LangGraphClient`
2. **Client** → Invokes LangGraph agent with MCP tools
3. **LangGraph** → Executes agent node, decides to use tools
4. **MCP Server** → Executes tool (e.g., fs_read, web_search, RAG)
5. **Tool Result** → Returned to agent via tools node
6. **LangGraph** → Continues agent loop until response complete
7. **Response** → Displayed to user via `ChatInterface`

### Session Management

Each chat session has a unique ID used for:

- RAG document indexing (session-scoped)
- Chunk caching for file summarization
- DPO pair collection

Session data is stored in `~/agent-conversations/{profile_name}/`:

```
~/agent-conversations/
└── {profile_name}/
    ├── conversations/           # Saved conversations
    ├── dpo_pairs/              # DPO training data
    ├── profiles/               # User profiles
    ├── rag_session_id.txt      # Current RAG session
    ├── rag_store_*.faiss       # FAISS vector index
    └── chunk_cache_*.db        # SQLite chunk cache
```

## 🚀 Productivity Tools

The assistant includes specialized tools for efficient code and file manipulation:

### 📋 Todo List Management

Track multi-step tasks with automatic status management:

**Tools:**

- `todo_write(todos)`: Update the todo list
- `todo_read()`: View current todos
- `todo_clear()`: Clear all todos

**Features:**

- Three states: `pending`, `in_progress`, `completed`
- Enforces exactly ONE task in progress at a time
- Real-time progress tracking
- Stored in `~/.claude/{profile}/current_todos.json`

**Usage Example:**

```
You: Implement user authentication
Assistant: [Creates todos for: database setup, API endpoints, frontend integration, testing]
Assistant: [Marks first todo as in_progress]
Assistant: [Completes each step, updating todos in real-time]
```

### 🔎 Fast Search Tools

High-performance file and content searching:

#### Glob Search (File Names)

Find files by name patterns:

```python
glob_search(pattern="**/*.py")  # All Python files recursively
glob_search(pattern="src/**/*.ts", max_results=100)  # TypeScript in src/
glob_search(pattern="test_*.py", sort_by_mtime=False)  # Unsorted for speed
```

**Parameters:**

- `pattern`: Glob pattern (e.g., `**/*.py`, `*.{yaml,json}`)
- `path`: Directory to search (default: current directory)
- `max_results`: Limit results (default: 1000, use 0 for unlimited)
- `sort_by_mtime`: Sort by modification time (default: True)

**Performance:** Best for project/codebase searches. For system-wide searches (entire home directory), the assistant automatically uses `find` command instead.

#### Grep Search (File Content)

Search within file contents using ripgrep:

```python
grep_search(pattern="class Foo")  # Find class definitions
grep_search(pattern="TODO|FIXME", file_pattern="*.py", case_insensitive=True)
grep_search(pattern="import React", output_mode="content")  # Show matched lines
```

**Parameters:**

- `pattern`: Regex pattern to search for
- `path`: Directory to search (default: current directory)
- `file_pattern`: Filter by file type (e.g., `*.py`, `*.{ts,tsx}`)
- `case_insensitive`: Case-insensitive search (default: False)
- `output_mode`: `files_with_matches` (default), `content`, or `count`
- `context_lines`: Lines of context around matches
- `max_results`: Maximum matches per file (default: 100)

**Requirements:** Requires `ripgrep` installed (see Installation section)

**Performance:** 10-100x faster than traditional grep for large codebases.

### ✏️ Precise File Editing

Safe string replacement with validation:

```python
file_edit(
    file_path="/path/to/file.py",
    old_string="def old_function():\n    pass",
    new_string="def new_function():\n    return True",
    replace_all=False  # Requires uniqueness (default)
)
```

**Safety Features:**

- Validates file exists before editing
- Checks that `old_string` exists in file
- Enforces uniqueness (prevents accidental multiple replacements)
- Provides detailed error messages with troubleshooting steps
- Returns line count changes

**Best Practice Workflow:**

1. Read the file first with `fs_read`
2. Copy the EXACT text you want to replace (including whitespace)
3. Create the new version with your changes
4. Call `file_edit` with exact strings

**Error Handling:** If the string isn't unique, the tool provides the line numbers where it appears so you can add more context.

### 🛡️ Enhanced Error Handling

All tools now provide intelligent error messages with troubleshooting guidance:

**Example Error Response:**

```json
{
  "error": true,
  "error_type": "FileNotFoundError",
  "message": "File or directory not found: /path/to/file.txt",
  "next_steps": [
    "Verify the file path is correct",
    "Use glob_search to find files by pattern",
    "Check with execute_bash('ls -la /parent/dir')",
    "Ensure you have read permissions"
  ],
  "original_error": "..."
}
```

**Handled Error Types:**

- FileNotFoundError
- PermissionError
- IsADirectoryError
- JSONDecodeError
- Encoding errors
- Command execution errors
- Timeout errors

### 📁 File Write Confirmation

`fs_write` now requires mandatory user confirmation:

**Two-Step Process:**

1. **Preview (dry_run=True)**: Shows what will happen
2. **Confirm**: User explicitly approves
3. **Execute (confirmed=True)**: Actually performs the operation

This prevents accidental file overwrites and gives users control over file system modifications.

### 🛡️ Git Safety

Safe git operations with protection against common mistakes:

**Tools:**

- `git_safe(command="...")` - Execute git commands with safety checks
- `git_status_safe()` - Comprehensive status with warnings
- `git_commit_safe(message="...", add_all=True)` - Safe commits with staging

**Protected Operations:**

| Operation                  | Protection                         |
| -------------------------- | ---------------------------------- |
| Force push to main/master  | Blocked                            |
| `git reset --hard`         | Warning + confirmation required    |
| `git push --force`         | Warning (use `--force-with-lease`) |
| `git commit --amend`       | Checks if already pushed           |
| Skip hooks (`--no-verify`) | Warning                            |
| Force delete branch (`-D`) | Warning                            |

**Example:**

```python
# Safe - uses git_safe with protections
git_safe(command="push origin feature-branch")

# Dangerous - requires confirmation
git_safe(command="reset --hard HEAD~1", allow_dangerous=True, reason="Discarding failed experiment")
```

### 📝 Plan Mode

Implementation planning workflow for complex tasks:

**Workflow:**

1. `enter_plan_mode(task_description="Add user authentication")`
2. Explore codebase with search tools
3. `add_plan_step(step_number=1, title="Create user model", description="...")`
4. `add_plan_file(file_path="models/user.py", action="create")`
5. `add_plan_risk(risk="Migration needed", mitigation="Add migration script")`
6. `present_plan()` - Show user for approval
7. `approve_plan()` + `exit_plan_mode()` - Start implementing

**When to Use:**

- New feature with multiple files
- Architectural decisions needed
- Multi-step refactoring
- Unclear requirements

**Plan Storage:** `~/.claude/current_plan.json`

### 🔄 Background Tasks

Run long operations in parallel without blocking:

**Tools:**

- `start_background_task(command="...", description="...")` - Start task
- `get_task_status(task_id="...")` - Check progress
- `get_task_output(task_id="...")` - Get output
- `list_background_tasks()` - See all tasks
- `cancel_background_task(task_id="...")` - Stop task
- `wait_for_task(task_id="...", timeout_seconds=300)` - Wait for completion

**When to Use:**

- Running full test suites
- Building large projects
- Installing dependencies
- Running linters on entire codebase
- Any command > 30 seconds

**Example:**

```python
# Start tests in background
result = start_background_task(command="pytest", description="Running tests")
# Returns: {"task_id": "abc123", ...}

# Check status later
get_task_status(task_id="abc123")

# Get output when done
get_task_output(task_id="abc123", tail_lines=50)
```

**Task Storage:** Output logs saved to `~/.claude/tasks/`

## 🔧 Configuration

### Model Configuration

The assistant supports multiple model types:

#### Amazon Bedrock

```yaml
MODEL_ID:
  NAME: us.amazon.nova-pro-v1:0
  TYPE: bedrock
  REGION: us-east-1
  TEMPERATURE: 0.1
```

#### Ollama (Local)

```yaml
MODEL_ID:
  NAME: qwen3-4b-thinking-2507-q6-k:latest
  TYPE: ollama
  HOST: localhost
  PORT: 11434
  REPETITION_PENALTY: 1.1
  PRESENCE_PENALTY: 1.5
  TEMPERATURE: 0.1
  TOP_P: 0.95
```

#### OpenAI

```yaml
MODEL_ID:
  NAME: gpt-5-mini-2025-08-07
  TYPE: openai
  STREAM: true
  REASONING_EFFORT: medium
ENV:
  OPENAI_API_KEY: your-openai-api-key
```

#### Amazon SageMaker AI

```yaml
MODEL_ID:
  NAME: your-endpoint-name
  TYPE: sagemaker
  REGION: us-east-1
  REPETITION_PENALTY: 1.1
  PRESENCE_PENALTY: 1.5
  TEMPERATURE: 0.1
  MAX_TOKENS: 4096
```

#### LiteLLM (100+ Providers)

```yaml
MODEL_ID:
  NAME: openai/your-model-name
  TYPE: litellm
  API_BASE: http://localhost:8000/v1
  API_KEY: your-api-key
  TEMPERATURE: 0.1
  MAX_TOKENS: 4096
```

### Vision Model Configuration

For Bedrock:

```yaml
VISION_MODEL_ID:
  NAME: global.anthropic.claude-haiku-4-5-20251001-v1:0
  TYPE: bedrock
  REGION: us-east-1
  TEMPERATURE: 0.3
```

For Ollama:

```yaml
VISION_MODEL_ID:
  NAME: qwen3-vl:2b
  TYPE: ollama
  HOST: localhost
  PORT: 11434
  TEMPERATURE: 0.3
```

For OpenAI:

```yaml
VISION_MODEL_ID:
  NAME: gpt-5-mini-2025-08-07
  TYPE: openai
  STREAM: true
  REASONING_EFFORT: medium
```

For SageMaker AI:

```yaml
VISION_MODEL_ID:
  NAME: your-endpoint-name
  TYPE: sagemaker
  REGION: us-east-1
  TEMPERATURE: 0.3
```

### System Prompt

The system prompt in `config.yaml` defines the assistant's behavior. Key sections:

- `<general_assistant_info>`: Basic identity and capabilities
- `<critical_response_requirement>`: Response quality standards
- `<reasoning_and_response>`: Thinking tag requirements
- `<core_principles>`: RAG-first mandate, user intent focus
- `<tool_usage>`: Tool selection rules and safety checks
- `<mandatory_rag_check>`: RAG workflow enforcement
- `<rag_query_guidelines>`: Query optimization rules

### RAG Configuration

```yaml
ENABLE_RAG: true
RAG_MAX_TOKENS: 8192
DOC_CHUNK_TOKENS: 1024
VECTOR_STORE:
  TYPE: faiss # or chromadb
```

### Episodic Memory Configuration

```yaml
ENABLE_EPISODIC_MEMORY: true
EPISODIC_MEMORY_STORE: chromadb # or faiss
EPISODIC_MEMORY:
  RETRIEVAL_THRESHOLD: 0.7 # Minimum similarity to retrieve episodes
  FOLLOW_UP_THRESHOLD: 0.4 # Similarity to detect follow-up questions
  REDUNDANCY_THRESHOLD: 0.5 # Filter episodes redundant with conversation
```

**How it works:**

- Automatically stores successful task completions with full conversation context
- Uses hybrid search (70% semantic + 30% keyword) to find similar past tasks
- **Conversation-aware injection**: Only injects episodic memory when relevant
  - Detects follow-up questions and skips injection (uses conversation context instead)
  - Filters out episodes redundant with current conversation
  - Uses semantic similarity (with embeddings) or Jaccard similarity (fallback)
- Injects compact context showing: task → tools used → outcome
- Automatic cleanup: keeps max 1000 episodes, removes entries older than 90 days

**Success detection:**

- User feedback: "thanks", "perfect", "great"
- No error markers in response
- All tools executed successfully
- Filters out simple greetings and short responses

#### Embeddings model

For Bedrock:

```yaml
EMBED_MODEL_ID:
  NAME: amazon.titan-embed-text-v2:0
  TYPE: bedrock
  REGION: us-east-1 # For bedrock/sagemaker
```

For Ollama:

```yaml
EMBED_MODEL_ID:
  NAME: mxbai-embed-large
  TYPE: ollama # or bedrock, sagemaker
  HOST: localhost
  PORT: 11434
```

For OpenAI:

```yaml
EMBED_MODEL_ID:
  NAME: text-embedding-ada-002
  TYPE: openai
```

For SageMaker:

```yaml
EMBED_MODEL_ID:
  NAME: your-endpoint-name
  TYPE: sagemaker
  REGION: us-east-1 # For bedrock/sagemaker
```

**Vector Store Options:**

- **ChromaDB** (default): Persistent vector database with built-in metadata support
- **FAISS**: Fast, in-memory vector search with disk persistence

Switch between stores by changing `VECTOR_STORE.TYPE` in config. The system uses a controller pattern, so all RAG functionality works identically regardless of the store.

## 📚 Advanced Features

### Web Search Configuration

This tool uses the Brave Search API. Obtain an API key from [Brave Search Developer Portal](https://brave.com/search/api/).

```yaml
BRAVE_API_KEY: your-api-key-here # For web search
```

### Web Crawler Configuration

Enable web page content extraction with automatic RAG integration:

```yaml
ENABLE_WEB_CRAWL: true
```

When enabled, the `web_crawler` tool:

- Extracts content from web pages as markdown
- Automatically ingests large pages (>8K tokens) into RAG (if enabled)
- Uses the same chunking configuration as PDF/DOCX readers

### RAG (Retrieval-Augmented Generation)

The RAG system automatically indexes documents for semantic search with **hybrid search** (semantic + keyword matching).

**How it works:**

1. Read a PDF/DOCX file → Automatically chunked and indexed
2. Ask questions → Assistant searches indexed documents first using hybrid search
3. Session-scoped → Cleared on `/clear` or exit

**RAG Tools:**

- `list_documents()`: Show indexed documents
- `search_in_documents(query, top_k)`: Hybrid semantic + keyword search
- `clear_documents()`: Clear RAG index

**Configuration:**

- `DOC_CHUNK_TOKENS`: Chunk size (recommended: 512-1024)
- `VECTOR_STORE.TYPE`: Choose between `faiss` or `chromadb`
- Recursive chunking with 10% overlap
- Hybrid search: 50% keyword matching + 50% semantic similarity

**Vector Store Options:**

- **ChromaDB**: Persistent vector database with metadata support (default)
- **FAISS**: Fast in-memory search with disk persistence

The system uses a **VectorStoreController** for easy switching between stores. All functionality (indexing, searching, clearing) works identically regardless of the chosen store.

### User Profile Learning

After 5+ interactions, the assistant builds a profile:

- **Cognitive style**: Analytical, creative, pragmatic, systematic
- **Domain expertise**: Python, AWS, DevOps, ML, etc.
- **Learning style**: Visual, hands-on, theoretical
- **Communication patterns**: Tone, complexity, question styles
- **Code preferences**: Testing, documentation, type hints

Profile is automatically injected into system prompt for personalization.

### Episodic Memory

The episodic memory system learns from successful task completions and retrieves similar solutions for future queries.

**How it works:**

1. **Automatic Storage**: After each successful interaction, stores:
   - Initial user query
   - Full conversation context
   - Tools used with arguments
   - Final solution
   - Timestamp

2. **Hybrid Search**: Retrieves similar episodes using:
   - 70% semantic similarity (task intent)
   - 30% keyword matching (tool names, action verbs)

3. **Context Injection**: Before processing queries, injects compact context:

   ```
   [Episodic Memory - Similar Past Tasks]
   1. "read DOCX about ML" → fs_read → success (similarity: 0.85)
   2. "analyze PDF report" → fs_read, web_search → success (similarity: 0.78)
   ```

4. **Automatic Cleanup**: Maintains bounded memory:
   - Max 1000 episodes
   - Removes entries older than 90 days
   - Runs on startup

**Success Detection:**

- User feedback: "thanks", "perfect", "great", "worked"
- No error markers in response
- All tools executed successfully
- Filters out greetings and simple acknowledgments (<300 chars, no tools)

**Storage Location:**

- FAISS: `~/agent-conversations/{profile}/episodic_memory/episodic.index`
- ChromaDB: `~/agent-conversations/{profile}/episodic_memory/`

**Configuration:**

```yaml
ENABLE_EPISODIC_MEMORY: true
EPISODIC_MEMORY_STORE: chromadb # or faiss
EMBED_MODEL_ID: # Required for both stores
  NAME: mxbai-embed-large
  TYPE: ollama
```

### ACE Playbook (Agentic Context Engineering)

The ACE Playbook learns strategies from both successes AND failures, implementing the Agentic Context Engineering framework for continuous improvement.

**How it works:**

1. **Reflector**: After each interaction, analyzes tool executions:
   - Detects failure patterns (file not found, string not found, permission denied, etc.)
   - Identifies successful strategies for specific tools (file_edit, execute_bash)
   - Extracts specific, actionable insights (not generic summaries)
   - Tracks metrics (success/failure rates, failure types) in `metrics.json`

2. **Playbook Store**: Maintains structured strategy entries:

   ```json
   {
     "context": "editing python files",
     "strategy": "Read the file first to get exact string including whitespace before using str_replace",
     "source": "Failed file_edit on 2026-02-01: string_not_found",
     "outcome": "failure",
     "tools": ["file_edit"],
     "confidence": 0.9
   }
   ```

3. **Context Injection**: Injects relevant strategies into the system prompt at startup:

   ```
   [Playbook - Learned Strategies]
   Avoid these patterns:
     ✗ [editing files]: Read the file first to get exact string before str_replace
   Effective strategies:
     ✓ [searching files]: Use glob_search instead of find for better performance
   ```

4. **Lazy Refinement**: Only deduplicates when hitting token limits, using semantic similarity if embeddings are configured.

**What gets stored:**

- **Failures**: Specific patterns like `string_not_found`, `file_not_found`, `permission_denied`, `command_failed`, etc.
- **Successes**: Only for tools with reusable patterns (file_edit, execute_bash with specific commands)
- **Not stored**: Generic successes without actionable strategies

**Key Differences from Episodic Memory:**

| Feature     | Episodic Memory       | ACE Playbook            |
| ----------- | --------------------- | ----------------------- |
| Stores      | Full task completions | Granular strategies     |
| Learns from | Successes only        | Successes AND failures  |
| Format      | Conversation context  | Structured rules        |
| Retrieval   | Semantic similarity   | Context + tool matching |

**Configuration:**

```yaml
ENABLE_PLAYBOOK: true
PLAYBOOK:
  MAX_ENTRIES: 500 # Maximum entries before refinement
  SIMILARITY_THRESHOLD: 0.85 # Threshold for merging similar strategies
  MAX_INJECT: 10 # Maximum entries to inject per query
```

**Storage Location:**

- Strategies: `~/agent-conversations/{profile}/playbook/playbook.json`
- Metrics: `~/agent-conversations/{profile}/playbook/metrics.json`

### Training Data Collection

#### Supervised Fine-Tuning (SFT)

- Use `/good` to mark high-quality responses
- Saved conversations include quality markers
- Extract labeled interactions for training

## 📦 Dependencies

All Python dependencies are listed in `requirements.txt`. The new productivity tools use only standard library features:

| Tool             | Python Packages                 | External Tools     |
| ---------------- | ------------------------------- | ------------------ |
| TodoWrite        | Standard library only           | None               |
| Edit Tool        | Standard library only           | None               |
| Glob Search      | Standard library (`glob`)       | None               |
| Grep Search      | Standard library (`subprocess`) | ripgrep (optional) |
| Error Handler    | Standard library (`functools`)  | None               |
| Git Safety       | Standard library (`subprocess`) | git                |
| Plan Mode        | Standard library (`json`, `os`) | None               |
| Background Tasks | Standard library (`threading`)  | None               |

**External Tools:**

- **ripgrep**: Required for `grep_search` tool. Install via system package manager (see Installation section). If not installed, the assistant automatically falls back to slower alternatives.

**Core Python Packages:**

- `langgraph`: Agent orchestration framework
- `langchain`, `langchain-core`: LLM abstraction layer
- `langchain-ollama`: Ollama integration
- `langchain-aws`: AWS Bedrock integration
- `langchain-openai`: OpenAI integration
- `mcp`, `mcp[cli]`: Model Context Protocol
- `ollama`: Local LLM support
- `boto3`: AWS Bedrock/SageMaker
- `tiktoken`: Token counting
- `chromadb`, `faiss-cpu`: Vector stores for RAG
- `PyPDF2`, `python-docx`: Document readers
- `Pygments`: Code syntax highlighting
- `prompt_toolkit`: Interactive CLI
- `brave-search-python-client`: Web search
- `crawl4ai`: Web crawling

## 🛠️ Development

### Adding New Tools

1. Create tool file in `server/tools/`:

```python
from mcp.server.fastmcp import FastMCP

def register_your_tool(mcp: FastMCP):
    @mcp.tool()
    async def your_tool(param: str) -> str:
        """Tool description for the LLM."""
        # Implementation
        return result
```

2. Register in `tools_manager.py`:

```python
from .your_tool import register_your_tool
register_your_tool(mcp)
```

### Adding New File Readers

1. Create reader in `server/tools/readers/`:

```python
async def read_your_format(path: str) -> str:
    """Read your custom format."""
    # Implementation
    return content
```

2. Register in `fs_read.py`:

```python
from .readers.your_reader import read_your_format
# Add to file type detection logic
```

### Switching Model Providers

The application uses **controller classes** for centralized model management. To switch providers, just update `config.yaml`:

**For LLM:**

```yaml
MODEL_ID:
  NAME: your-model-name
  TYPE: ollama # or bedrock, sagemaker
```

**For Vision:**

```yaml
VISION_MODEL_ID:
  NAME: your-vision-model
  TYPE: ollama # or sagemaker
```

**For Embeddings:**

```yaml
EMBED_MODEL_ID:
  NAME: mxbai-embed-large
  TYPE: ollama
```

The controllers (`llm_controller.py`, `vision_model_controller.py`, `embeddings_controller.py`) handle all provider-specific initialization automatically.

### Adding New Model Providers

1. Update the appropriate controller in `models/`:

```python
def initialize_model(self):
    if self.model_type == "your_provider":
        # Your provider initialization
        self.model = YourProviderModel(...)
```

2. Add configuration in `config.yaml`

## 🐛 Troubleshooting

### Common Issues

**MCP Connection Errors**

- Verify Python path in `client.py` matches your environment
- Check server path is correct

**Model Loading Issues**

- Verify model name and type in `config.yaml`
- For Ollama: Ensure model is pulled (`ollama pull model-name`)
- For AWS: Check credentials and region

**RAG Not Working**

- Ensure `ENABLE_RAG: true` in config
- Check embedding model is available
- Verify documents are being indexed with `list_documents()`

**Permission Errors**

- Ensure write permissions for `~/agent-conversations/`
- Check file paths in configuration

### Logging

Logs are output to stderr with configurable level:

```bash
LOG_LEVEL=DEBUG python main.py  # Detailed logs
LOG_LEVEL=INFO python main.py   # Normal logs (default)
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

This is a personal development project. If you'd like to use or extend it, feel free to fork the repository and adapt it to your needs!

If you use this code in your own projects, attribution to the original repository is appreciated but not required.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain)
- Uses [FastMCP](https://github.com/jlowin/fastmcp) for Model Context Protocol
- Powered by [Ollama](https://ollama.ai), [Amazon Bedrock](https://aws.amazon.com/bedrock/), and [Amazon SageMaker AI](https://aws.amazon.com/sagemaker/ai/)
