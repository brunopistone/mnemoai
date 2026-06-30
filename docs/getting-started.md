# Getting Started

This guide takes you from a fresh machine to a working `mnemoai` command.

## 1. Requirements

Required:

- Python 3.11+
- Access to at least one chat model provider

Choose one provider to start:

| Provider                    | What you need                                                                                |
| --------------------------- | -------------------------------------------------------------------------------------------- |
| **Ollama** (local, easiest) | Install [Ollama](https://ollama.ai), then pull a chat model such as `ollama pull qwen3.5:4b` |
| **Amazon Bedrock**          | AWS credentials with Bedrock model access in your target region                              |
| **Bedrock Mantle**          | AWS credentials or a Bedrock API key, plus a Mantle model available in your account/region   |
| **Amazon SageMaker AI**     | AWS credentials and a deployed SageMaker endpoint                                            |
| **OpenAI**                  | `OPENAI_API_KEY` environment variable                                                        |
| **Anthropic**               | `ANTHROPIC_API_KEY` environment variable                                                     |
| **LiteLLM**                 | A LiteLLM-compatible provider, API base, and credentials as needed                           |

Optional, depending on features you enable:

- **Embedding model** — needed for high-quality RAG, episodic memory, and ACE playbook refinement.
- **Vision model** — needed for image analysis.
- **Brave Search API key** — needed for web search.
- **ripgrep** — recommended for fast content search.

## 2. Install Mnemo AI

Recommended isolated install:

```bash
uv tool install mnemoai-assistant
```

Alternatives:

```bash
pipx install mnemoai-assistant
# or
pip install mnemoai-assistant
```

The published package name is `mnemoai-assistant`; the terminal command and Python import package are both `mnemoai`.

Upgrade later with:

```bash
uv tool upgrade mnemoai-assistant
# or: pip install -U mnemoai-assistant
```

## 3. First run setup

Start the assistant:

```bash
mnemoai
```

If no config exists, Mnemo AI opens an interactive setup wizard. It asks for:

- chat model provider and model name;
- provider connection details, such as Ollama host/port, AWS region, SageMaker input format, LiteLLM API base/key, or Mantle protocol;
- optional vision and embedding model settings;
- profile name;
- optional Brave Search API key;
- feature toggles such as RAG, memory, web crawling, routing, and orchestration.

The wizard writes your user config to:

```text
~/.mnemoai/config/config.yaml
```

You can edit that file later or run `/config` inside Mnemo AI to re-run the configurator.

## 4. Verify it works

After setup, try a simple prompt:

```text
What files are in the current directory?
```

If the assistant lists files or uses the file-reading tools, the core loop is working.

Useful startup flags:

```bash
mnemoai              # verbose mode: shows thinking/reasoning when available
mnemoai --no-verbose # hides thinking/reasoning output
```

## 5. Ollama quick setup

For a fully local setup:

```bash
ollama pull qwen3.5:4b
mnemoai
```

If you enable RAG, episodic memory, or ACE playbook refinement, also pull an embedding model and configure it under `RAG.EMBED_MODEL_ID`:

```bash
ollama pull qwen3-embedding:0.6b
```

A minimal Ollama config looks like this:

```yaml
MODEL_ID:
  NAME: qwen3.5:4b
  TYPE: ollama
  HOST: localhost
  PORT: 11434
  TEMPERATURE: 0.6

PROFILE:
  NAME: default

ENABLE_RAG: false
ENABLE_EPISODIC_MEMORY: false
ENABLE_PLAYBOOK: false
ENABLE_WEB_SEARCH: false
ENABLE_WEB_CRAWL: false
```

For normal installs, save manual configs at `~/.mnemoai/config/config.yaml`.

## 6. Where config files live

Config resolution order, first match wins:

1. `$MNEMOAI_CONFIG` — explicit config path.
2. `~/.mnemoai/config/config.yaml` — normal user config for installed `mnemoai`.
3. `~/.mnemoai/config.yaml` — legacy flat location.
4. `<package>/utils/config.yaml` — package-relative fallback, mainly useful for source checkouts.

On first run, Mnemo AI also seeds examples you can copy or inspect:

```text
~/.mnemoai/config/config.yaml.example
~/.mnemoai/config/config.yaml.bedrock.example
~/.mnemoai/config/config.yaml.bedrock.mantle.example
~/.mnemoai/mcp/mcp.json.example
```

Prompts live separately in:

```text
~/.mnemoai/config/prompts.yaml
```

## 7. Recommended optional tools

Install ripgrep for faster content search:

=== "macOS"

    ```bash
    brew install ripgrep
    ```

=== "Ubuntu/Debian"

    ```bash
    sudo apt install ripgrep
    ```

=== "Fedora/RHEL"

    ```bash
    sudo dnf install ripgrep
    ```

Verify:

```bash
rg --version
```

Without ripgrep, Mnemo AI falls back to slower grep-based searches.

## 8. Developer install from a checkout

Use this path if you want to edit the source.

```bash
git clone https://github.com/brunopistone/mnemoai.git
cd mnemoai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m mnemoai
```

Or install the checkout as a command:

```bash
uv tool install .        # or: pipx install .
mnemoai
```

For live source edits without reinstalling, keep using:

```bash
PYTHONPATH=src python -m mnemoai
```

You can also use the wrapper under `bash/system-command-app/` if you want a `mnemoai` command that runs your working tree directly.

## 9. Next steps

- Learn commands and feature toggles in [Usage](usage.md).
- Configure providers and advanced model parameters in [Configuration](configuration.md).
- Add RAG, external MCP servers, web tools, memory, and skills in [Advanced Features](advanced-features.md).
- See [Development](development.md) if you want to run tests or contribute.
