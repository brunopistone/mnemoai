# Getting Started

## 🚀 Quick Start

### Prerequisites

**Required:**

- Python 3.11+
- At least **one LLM provider** configured and accessible (see below)

**LLM Providers (choose at least one):**

| Provider                                            | Requirements                                                                      |
| --------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Ollama** (local, recommended for getting started) | [Install Ollama](https://ollama.ai), then pull a model: `ollama pull qwen3:4b`    |
| **Amazon Bedrock**                                  | AWS CLI configured (`aws configure`) with Bedrock access in your region           |
| **Amazon SageMaker AI**                             | AWS CLI configured with a deployed SageMaker endpoint                             |
| **OpenAI**                                          | Set `OPENAI_API_KEY` environment variable                                         |
| **Anthropic** (Claude API)                          | Set `ANTHROPIC_API_KEY` environment variable                                      |
| **LiteLLM**                                         | Depends on the underlying provider (see [LiteLLM docs](https://docs.litellm.ai/)) |

**Optional:**

- **ripgrep** — 10-100x faster content search (see installation below)
- **Embedding model** — Required if you enable RAG, Episodic Memory, or ACE Playbook (see [Feature Toggles](usage.md#feature-toggles))
- **Vision model** — Required for image analysis (`describe_image` tool)
- **Brave Search API key** — Required for web search ([get one here](https://brave.com/search/api/))

### Installation

#### Recommended: install from PyPI

The published package is **[`mnemoai-assistant`](https://pypi.org/project/mnemoai-assistant/)** (the import name and the CLI command are both `mnemoai`). No clone needed — install it into an isolated environment and get the `mnemoai` command on your PATH:

```bash
uv tool install mnemoai-assistant     # or: pipx install mnemoai-assistant
```

Or into the current environment with pip:

```bash
pip install mnemoai-assistant
```

Then configure a user config (see step 4 below) and run:

```bash
mnemoai            # verbose (shows thinking)
mnemoai --no-verbose
```

To upgrade: `uv tool upgrade mnemoai-assistant` (or `pip install -U mnemoai-assistant`). To remove: `uv tool uninstall mnemoai-assistant`.

> This is the best choice if you just want to use the assistant. Install from a checkout (below) instead if you plan to edit the source.

#### Install from a checkout

1. **Clone the repository**:

```bash
git clone https://github.com/brunopistone/mnemoai.git
cd mnemoai
```

2. **Install the assistant** (choose one):

#### Option 1: install as a CLI command (`uv tool install`)

This installs the project into its own isolated environment and puts `mnemoai` on your PATH, so you can run it from any directory (macOS and Linux) without activating anything:

```bash
uv tool install .        # or: pipx install .
```

Then configure a user config (see step 4) and run:

```bash
mnemoai            # verbose (shows thinking)
mnemoai --no-verbose
```

To upgrade after pulling changes: `uv tool install --force .`. To remove: `uv tool uninstall mnemoai`.

> Pick "run from a checkout" below instead if you plan to actively edit the code, since that runs your working tree directly with no reinstall step.

#### Option 2: run from a checkout

Set up an environment (choose one), which lets you run the assistant directly from the repo while editing the source live. Because the code uses a `src/` layout, run it as a module with `src/` on the path:

```bash
PYTHONPATH=src python -m mnemoai            # verbose
PYTHONPATH=src python -m mnemoai --no-verbose
```

(Or `pip install -e .` once, then just `mnemoai`.)

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
conda create -n mnemoai python=3.11
conda activate mnemoai
pip install -r requirements.txt
```

**Get the `mnemoai` command for a checkout install**

So you don't have to `cd` into the repo every time, symlink the bundled wrapper script onto your PATH. It activates the project environment, then runs the app (`PYTHONPATH=src python -m mnemoai`):

```bash
chmod +x bash/system-command-app/mnemoai-wrapper.sh
ln -sf "$(pwd)/bash/system-command-app/mnemoai-wrapper.sh" /usr/local/bin/mnemoai
```

Now `mnemoai` works from any directory and always reflects your latest edits. The wrapper auto-activates a project-local `.venv` (Options A and B) if present, otherwise it falls back to a conda env named `mnemoai` (Option C) — edit the script if your environment differs.

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

**First-run setup (easiest).** If you start the assistant and no config is found, an interactive configurator runs automatically. It walks you through: the LLM provider (Ollama / Bedrock / Mantle / OpenAI / Anthropic / Amazon SageMaker AI / LiteLLM) plus chat model, connection details (Ollama host/port; AWS region; for Mantle the API protocol — chat_completions / responses / anthropic; SageMaker region + input format; LiteLLM API base/key; OpenAI uses `OPENAI_API_KEY`; Anthropic uses `ANTHROPIC_API_KEY` with an optional base URL), optional max output tokens (blank or `none` uses the provider default), and a mandatory max context window (defaults to 65536); the vision model (reusing the chat model's host/region, with its own Mantle protocol and optional max output tokens); your profile name; an optional Brave Search key; and each feature toggle (RAG, episodic memory, ACE playbook, web crawler, query routing, orchestration, user profiling). Every prompt is pre-filled with the template's default, so you can press Enter through the ones you don't care about. It then writes a ready-to-use `~/.mnemoai/config/config.yaml` from the matching template. Just run:

```bash
mnemoai      # or, from a checkout: PYTHONPATH=src python -m mnemoai
```

and follow the prompts. You can re-edit the generated file any time to fine-tune models, prompts, and feature toggles.

**Manual setup.** Prefer to write it yourself? Copy a template (they live inside the package, under `src/mnemoai/utils/`):

```bash
cp src/mnemoai/utils/config.yaml.example src/mnemoai/utils/config.yaml
```

Edit that `config.yaml` with your settings. This file is git-ignored to protect your API keys. At minimum, configure your LLM provider.

The config file is resolved in this order (first match wins):

1. `$MNEMOAI_CONFIG` — explicit path (handy for switching between provider configs)
2. `~/.mnemoai/config/config.yaml` — **user config used by the installed `mnemoai` command**
3. `~/.mnemoai/config.yaml` — legacy pre-subfolder location (still read if present)
4. `<package>/utils/config.yaml` — package-relative fallback (used when running from a checkout)

On first run mnemoai seeds `~/.mnemoai/config/` and `~/.mnemoai/mcp/` with copies of the bundled examples (`config.yaml*.example`, `mcp.json.example`) so you have them to read right next to your live files. If you installed the CLI with `uv tool install` (the recommended option), put your config in the user location:

```bash
# Examples are auto-copied on first run; just copy one to config.yaml and edit:
cp ~/.mnemoai/config/config.yaml.example        ~/.mnemoai/config/config.yaml
# or, for Bedrock / Mantle:
# cp ~/.mnemoai/config/config.yaml.bedrock.example        ~/.mnemoai/config/config.yaml
# cp ~/.mnemoai/config/config.yaml.bedrock.mantle.example ~/.mnemoai/config/config.yaml
```

At minimum, configure your LLM provider:

**For Ollama (quickest setup):**

```bash
# Pull a model first
ollama pull qwen3:4b
```

```yaml
# utils/config.yaml (minimal)
MODEL_ID:
  NAME: qwen3:4b
  TYPE: ollama
  HOST: localhost
  PORT: 11434
  TEMPERATURE: 0.6

# Profile name (used for session data isolation)
PROFILE:
  NAME: default

# Everything else can be left at defaults or disabled
ENABLE_RAG: false
ENABLE_EPISODIC_MEMORY: false
ENABLE_PLAYBOOK: false
ENABLE_WEB_SEARCH: false
ENABLE_WEB_CRAWL: false
```

See [Configuration](configuration.md) for all options and [Feature Toggles](usage.md#feature-toggles) for enabling advanced features.

5. **Run the assistant**:

If you installed with `uv tool install` (recommended), run the command from anywhere:

```bash
mnemoai
```

If you set up a checkout and symlinked the wrapper, the same command works. Otherwise, run it from the repo directory:

```bash
PYTHONPATH=src python -m mnemoai
```

See `bash/system-command-app/README.md` for details on the wrapper script.
