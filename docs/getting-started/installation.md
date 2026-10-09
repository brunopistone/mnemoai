# Getting Started

This guide takes you from a fresh machine to a working `mnemoai` command.

## 1. Requirements

Required:

- macOS or glibc-based Linux (arm64/x86_64), with `curl` and `tar`
- Python 3.11+ only for manual installs; the one-command installer manages Python 3.12
- Access to at least one chat model provider

Choose one provider to start:

| Provider                       | What you need                                                                                                           |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| **Ollama** (local, easiest)    | Install [Ollama](https://ollama.ai), then pull a chat model such as `ollama pull qwen3.5:4b`                            |
| **MLX** (local, Apple Silicon) | An MLX server such as [`mlx-openai-server`](https://github.com/cubist38/mlx-openai-server) running, with a model loaded |
| **Amazon Bedrock**             | AWS credentials with Bedrock model access in your target region                                                         |
| **Bedrock Mantle**             | AWS credentials or a Bedrock API key, plus a Mantle model available in your account/region                              |
| **Amazon SageMaker AI**        | AWS credentials and a deployed SageMaker endpoint                                                                       |
| **OpenAI**                     | `OPENAI_API_KEY` environment variable                                                                                   |
| **Anthropic**                  | `ANTHROPIC_API_KEY` environment variable                                                                                |
| **LiteLLM**                    | A LiteLLM-compatible provider, API base, and credentials as needed                                                      |

Optional, depending on features you enable:

- **Embedding model** — needed for high-quality RAG, episodic memory, and ACE playbook refinement.
- **Vision model** — needed for image analysis.
- **Brave Search API key** — needed for web search.
- **ripgrep** — recommended for fast content search.

## 2. Install Mnemo AI

Recommended one-command install (macOS/Linux):

```bash
curl -fsSL https://raw.githubusercontent.com/brunopistone/mnemoai/main/install.sh | sh
```

Re-run the same command to update to the latest stable **GitHub release**. The
installer resolves the release when it runs, pins its wheel by SHA-256, and checks
the installed version before reporting success. GitHub publication no longer has
to wait for PyPI indexing. If release discovery or checksum verification fails,
installation stops rather than silently selecting an older release.

The installer uses `uv` with an isolated, managed Python 3.12 runtime. If `uv` is missing, it
downloads a pinned binary and verifies its SHA-256 checksum before executing it.
Python bytecode is compiled during installation rather than on the first launch.
This is still the Python application, not a native rewrite.

It does **not** use `sudo`, modify shell profiles, remove an independently installed
copy, or change `~/.mnemoai` configuration, credentials, memory, or conversations. The default
command is `~/.local/bin/mnemoai`; the runtime lives in
`~/.local/share/mnemoai-runtime`. If PATH selects an older installation, the installer
warns you: put `~/.local/bin` **first** on PATH (and restart your shell), or run
`~/.local/bin/mnemoai` directly. For example, in bash/zsh:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

If the destination already contains a command not owned by this installer, it
refuses **before** updating the runtime or removing any old launcher. Choose a
different `MNEMOAI_BIN_DIR`, or migrate that command yourself.

Inspect the script before running it if you prefer:

```bash
curl -fsSLo install.sh https://raw.githubusercontent.com/brunopistone/mnemoai/main/install.sh
less install.sh
sh install.sh
sh install.sh --version 1.29.0  # optional explicit PyPI version; bypasses latest-release lookup
```

`MNEMOAI_INSTALL_DIR` and `MNEMOAI_BIN_DIR` override the runtime and command
directories (absolute paths). When piping, set overrides on the `sh` process:

```bash
curl -fsSL https://raw.githubusercontent.com/brunopistone/mnemoai/main/install.sh |
  MNEMOAI_BIN_DIR="$HOME/bin" sh
```

From a checkout, `sh install.sh` runs the same installer. Release testing can use
`sh install.sh --wheel /absolute/path/mnemoai_assistant-VERSION-py3-none-any.whl`.
Windows and musl-based Linux are not supported by this installer.

The completion message shows the verified installed version and executable path.
If the plain `mnemoai` command still starts an older version, use that printed path
directly and check `type -a mnemoai`. After correcting PATH, restart your shell,
or run `hash -r` (bash) / `rehash` (zsh), to clear a cached executable location.

Alternatives with your own Python:

```bash
uv tool install mnemoai-assistant
pipx install mnemoai-assistant
# or
pip install mnemoai-assistant
```

The published package name is `mnemoai-assistant`; the terminal command and Python import package are both `mnemoai`.

For a **manual** installation, upgrade with its original package manager:

```bash
uv tool upgrade mnemoai-assistant
# or: pipx upgrade mnemoai-assistant
# or: pip install -U mnemoai-assistant
```

### Uninstall

Close the assistant before uninstalling. For an installation made with `install.sh`:

```bash
mnemo_runtime="${MNEMOAI_INSTALL_DIR:-$HOME/.local/share/mnemoai-runtime}"
mnemo_uv="$(command -v uv || printf '%s' "$mnemo_runtime/bootstrap/0.12.23/uv")"
UV_TOOL_DIR="$mnemo_runtime/tools" \
UV_TOOL_BIN_DIR="${MNEMOAI_BIN_DIR:-$HOME/.local/bin}" \
"$mnemo_uv" --no-config tool uninstall mnemoai-assistant
```

For custom installation directories, set the same `MNEMOAI_INSTALL_DIR` and
`MNEMOAI_BIN_DIR` values you used when installing. The command removes the
application environment and its launcher, not independently installed copies.
You may then move the now-unused runtime directory to Trash to reclaim its
private Python, bootstrap files, and cached release wheels.

Keep `~/.mnemoai` to preserve configuration, memory, and conversations.
For a manual pip/pipx/uv installation, use that package manager's uninstall
command in the original environment instead.

## 3. First run setup

Start the assistant:

```bash
mnemoai
```

If no config exists, Mnemo AI opens an interactive setup wizard. It asks for:

- chat model provider and model name;
- provider connection details, such as Ollama or MLX server host/port, AWS region, SageMaker input format, LiteLLM API base/key, or Mantle protocol;
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

If the assistant lists files or uses the file-reading tools, the core loop is working. If something doesn't work, see [Troubleshooting](../development/troubleshooting.md).

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

Here is a **deliberately minimal, everything-off** Ollama config — the smallest thing that runs. This is _not_ what the first-run wizard writes: the bundled template (and the shipped [`config.yaml.example`](../configuration.md#complete-example-config)) enable RAG, episodic memory, the playbook, web search, and web crawling by default. Start minimal and switch features on as you need them:

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

To see the full, annotated defaults instead, see the [complete example config](../configuration.md#complete-example-config). For normal installs, save manual configs at `~/.mnemoai/config/config.yaml`.

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
~/.mnemoai/config/config.yaml.mlx.example
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

Without ripgrep, the `grep_search` tool is unavailable: it returns
`ripgrep (rg) not installed` instead of searching. There is no fallback.
`glob_search` (filename patterns) works regardless — it uses the Python standard
library.

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

- Learn commands and feature toggles in [Usage](../guides/usage.md).
- Configure providers and advanced model parameters in [Configuration](../configuration.md).
- Add RAG, external MCP servers, web tools, memory, and skills — browse the [Guides](../guides/index.md).
- See [Development](../development/index.md) if you want to run tests or contribute.
