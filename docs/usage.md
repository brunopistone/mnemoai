# Usage

## 🔀 Feature Toggles

All advanced features can be independently enabled or disabled in your local `utils/config.yaml` (copied from `config.yaml.example`). Here is a quick reference:

| Feature                                                                 | Config Key                           | Default             | Dependencies                              |
| ----------------------------------------------------------------------- | ------------------------------------ | ------------------- | ----------------------------------------- |
| **RAG** (document indexing & search)                                    | `ENABLE_RAG: true`                   | `true`              | Embedding model (`RAG.EMBED_MODEL_ID`)    |
| **Episodic Memory** (learn from past tasks)                             | `ENABLE_EPISODIC_MEMORY: true`       | `true`              | Embedding model (`RAG.EMBED_MODEL_ID`)    |
| **ACE Playbook** (learn strategies from success/failure)                | `ENABLE_PLAYBOOK: true`              | `true`              | None (embeddings optional for refinement) |
| **User Profiling** (personalized responses)                             | `PROFILE.USE_PROFILING: true`        | `true`              | Activates after 5+ interactions           |
| **Web Search**                                                          | `ENABLE_WEB_SEARCH: true`            | `true`              | `BRAVE_API_KEY` configured                |
| **Web Crawler**                                                         | `ENABLE_WEB_CRAWL: true`             | `true`              | None                                      |
| **Vision** (image analysis)                                             | Configure `VISION_MODEL_ID`          | Disabled if not set | Vision-capable model                      |
| **Bash Confirmation** (prompt before each shell command)                | `REQUIRE_BASH_CONFIRMATION: true`    | `true`              | None (auto-skips when non-interactive)    |
| **Write Confirmation** (prompt before each file write)                  | `REQUIRE_WRITE_CONFIRMATION: true`   | `true`              | None (auto-skips when non-interactive)    |
| **Persistent Memory** (curated memory the agent maintains, `MEMORY.md`) | `ENABLE_MEMORY: true`                | `true`              | None                                      |
| **Memory Confirmation** (prompt before each memory write)               | `REQUIRE_MEMORY_CONFIRMATION: false` | `false`             | None (auto-skips when non-interactive)    |
| **Verbose Mode** (show thinking process)                                | CLI flag `--no-verbose`              | Enabled             | Supported by model                        |

**Dependency note:** RAG, Episodic Memory, and ACE Playbook refinement all require a working embedding model. If the embedding model is unavailable, the system falls back to SHA256-based deterministic embeddings with degraded semantic search quality. Configure `RAG.EMBED_MODEL_ID` in `config.yaml` to use a real embedding model (see [Embeddings Model](configuration.md#embeddings-model)).

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

| Command            | Description                                                                                                                                                                                                                                     |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/exit` or `/quit` | Exit the application                                                                                                                                                                                                                            |
| `/clear`           | Clear conversation history and RAG index                                                                                                                                                                                                        |
| `/save`            | Save current conversation                                                                                                                                                                                                                       |
| `/load <path>`     | Load a saved conversation                                                                                                                                                                                                                       |
| `/compact [focus]` | Summarize older turns to shrink context (optional focus instructions)                                                                                                                                                                           |
| `/config`          | Re-run the interactive configurator (overwrites `config.yaml`, then restarts the app in place to apply)                                                                                                                                         |
| `/model`           | Override just one model — chat (LLM), vision, or embeddings. Inference params (temperature, top_p, penalties, …) are reset to the model defaults on a change (they're model-specific); re-tune with `/params`. Restarts in place                |
| `/params`          | Tune a model's inference parameters (temperature, top_p, top_k, penalties, reasoning, stop, stream, …) — only the params the chosen provider supports are offered, then restart in place                                                        |
| `/mcp`             | List the configured MCP servers (built-in + any from `mcp.json`), their connection status, and tool counts                                                                                                                                      |
| `/memory`          | View the curated persistent memory (`MEMORY.md`); `/memory clear` wipes it (with a y/N confirm)                                                                                                                                                 |
| `/plan`            | Toggle **plan mode** — an enforced read-only mode. While ON, the agent investigates with read-only tools and presents a plan; file edits, shell commands, git writes, and background tasks are **hard-blocked** until you `/plan` again to exit |

### Keyboard Shortcuts

- `Ctrl+J`: Insert new line in input
- `Enter`: Submit message
- `Ctrl+C`: Interrupt operation (press twice to exit)

### Verbose Mode

Control thinking process visibility:

```bash
mnemoai              # Verbose mode (shows thinking)
mnemoai --no-verbose # Hide thinking process
# from a checkout: PYTHONPATH=src python -m mnemoai [--no-verbose]
```
