# Mnemo AI

<p align="center"><img src="assets/mnemoai-logo.png" width="120"></p>

A local agentic AI assistant for developers and power users. Mnemo AI runs in your terminal, connects to tools through MCP (Model Context Protocol), can read and edit files, use shell/git safely, search documents with RAG, and remember useful context across sessions.

It supports local and hosted model providers: **Ollama**, **Amazon Bedrock**, **Bedrock Mantle**, **OpenAI**, **Anthropic**, **Amazon SageMaker AI**, and **LiteLLM**.

![Demo](https://raw.githubusercontent.com/brunopistone/mnemoai/main/images/assistant-demo.gif)

## Quick start

```bash
uv tool install mnemoai-assistant     # or: pip install mnemoai-assistant
mnemoai
```

On first run, Mnemo AI launches an interactive setup wizard and writes your user config to:

```text
~/.mnemoai/config/config.yaml
```

You only need Python 3.11+ and access to at least one model provider. For the easiest local setup, install [Ollama](https://ollama.ai) and pull a chat model before running Mnemo AI.

## What Mnemo AI can do

- **Use multiple LLM providers** — local Ollama or hosted providers such as Bedrock, OpenAI, Anthropic, SageMaker AI, and LiteLLM.
- **Work with your files** — read, search, write, and precisely edit text, code, CSV, JSON, PDF, and DOCX files.
- **Use developer tools safely** — shell execution, git safety checks, todo tracking, plan mode, and background tasks.
- **Search and ingest knowledge** — RAG over documents, web crawling, and optional Brave web search.
- **Learn over time** — persistent memory, user profiling, episodic memory, ACE playbook strategies, and authored agent skills.
- **Handle multimodal tasks** — optional vision model support for image analysis.

## Choose your path

| Goal                                                                         | Start here                                                                                  |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Install and use Mnemo AI from scratch                                        | [Getting Started](getting-started/installation.md)                                          |
| Fix a setup or runtime problem                                               | [Troubleshooting](development/troubleshooting.md)                                           |
| Learn the chat commands and feature toggles                                  | [Usage](guides/usage.md)                                                                    |
| Configure providers, models, prompts, RAG, and memory                        | [Configuration](configuration.md)                                                           |
| Orchestrate complex tasks: query routing, workers, and sub-agents            | [Orchestration & sub-agents](guides/orchestration.md)                                       |
| Give the assistant memory: profiles, MEMORY.md, steering, episodic, playbook | [Memory & learning](guides/memory.md)                                                       |
| Add knowledge: RAG, web search, and web crawling                             | [Knowledge & web tools](guides/knowledge-and-web.md)                                        |
| Connect external MCP servers                                                 | [External MCP servers](guides/mcp.md)                                                       |
| Understand file editing, search, git safety, plan mode, and background tasks | [Productivity Tools](guides/productivity-tools.md)                                          |
| Contribute or run tests                                                      | [Development](development/index.md)                                                         |
| Understand the internal design                                               | [Architecture](getting-started/architecture.md)                                             |
| Browse the detailed per-file map (repo-only, agent/contributor reference)    | [Architecture Reference](https://github.com/brunopistone/mnemoai/blob/main/ARCHITECTURE.md) |

## Typical workflow

1. Install Mnemo AI.
2. Run `mnemoai` and complete the first-run wizard.
3. Ask a simple question or request a file operation, for example:

   ```text
   What files are in this directory?
   ```

4. Enable optional features as needed:
   - RAG/document search needs an embedding model.
   - Web search needs a Brave Search API key.
   - Vision needs a vision-capable model.
   - External MCP tools are configured in `~/.mnemoai/mcp/mcp.json`.

## Where configuration lives

For normal installs, edit:

```text
~/.mnemoai/config/config.yaml
```

You can override this with:

```bash
MNEMOAI_CONFIG=/path/to/config.yaml mnemoai
```

The first run also seeds readable examples under `~/.mnemoai/config/` and `~/.mnemoai/mcp/`.
