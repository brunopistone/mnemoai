# Development

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
- `langchain-openai`: OpenAI integration (also used for Bedrock Mantle OpenAI/Responses protocols)
- `langchain-anthropic`: Anthropic integration (Bedrock Mantle `anthropic` protocol)
- `aws-bedrock-token-generator`: Bearer-token auth for Bedrock Mantle
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

### Testing

The test suite uses `pytest` and is split into two tiers under `tests/`:

- **`tests/unit/`** — fast, deterministic tests for pure logic (BM25, reasoning helpers, response parsing, subtask parsing, the tool error handler, git-safety command classification, file editing/search, bash timeout handling, and episodic-memory heuristics). No LLM, Ollama, or network required, so they run in seconds and don't need a `config.yaml`.
- **`tests/integration/`** — end-to-end tests that drive the real agent against a live Ollama server and the MCP subprocess (routing, tool calls, bash timeout, no silent empty turns). Marked with `@pytest.mark.integration` and **auto-skipped** unless a runtime `utils/config.yaml` exists and the configured Ollama host is reachable.

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run everything (integration auto-skips if Ollama/config aren't available)
python -m pytest

# Unit tier only (fast — good for CI and pre-commit)
python -m pytest tests/unit

# Integration tier only (requires Ollama running + a real config.yaml)
python -m pytest -m integration

# Run a single file
python -m pytest tests/unit/test_bm25.py
```

When adding new code, keep import-time side effects independent of `config.yaml` so the module stays unit-testable.

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
RAG:
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

## 🔧 Ollama Utilities (Optional)

The `bash/` directory contains helper scripts for Ollama users on macOS and Linux.

### Ollama Environment Setup (macOS)

Sets Ollama performance environment variables at boot and launches the Ollama app:

```bash
# Variables set: OLLAMA_FLASH_ATTENTION=1, OLLAMA_KV_CACHE_TYPE=q8_0, OLLAMA_NUM_GPU=999
```

**Setup:**

1. Edit `bash/ollama-env-mac/ollama.environment.plist` (no changes needed for defaults)
2. Copy to LaunchAgents:

```bash
cp bash/ollama-env-mac/ollama.environment.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/ollama.environment.plist
```

### VRAM Cleaner

Automatically unloads idle Ollama models from VRAM to free GPU memory. Useful when running multiple models or when GPU memory is limited.

**macOS (LaunchAgent, runs every 60 seconds):**

1. Edit `bash/ollama-freeup-vram/com.ollama.vramcleaner.plist`:
   - Replace `<PATH_TO_FOLDER>` with the actual path to this repository
   - Replace `<PATH_TO_USER_HOME>` with your home directory
2. Install:

```bash
cp bash/ollama-freeup-vram/com.ollama.vramcleaner.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.ollama.vramcleaner.plist
```

**Linux (systemd):**

1. Edit `bash/ollama-freeup-vram/ollama-vram-cleaner.service`:
   - Replace `<PATH_TO_FOLDER>` with the actual path
2. Install:

```bash
sudo cp bash/ollama-freeup-vram/ollama-vram-cleaner.service /etc/systemd/system/
sudo systemctl enable ollama-vram-cleaner
sudo systemctl start ollama-vram-cleaner
```

See `bash/ollama-freeup-vram/README.md` and `bash/ollama-env-mac/README.md` for more details.

## 🐛 Troubleshooting

### Common Issues

**MCP Connection Errors**

- Verify Python path in `client.py` matches your environment
- Check server path is correct
- Ensure all dependencies are installed (`pip install -r requirements.txt`)

**Model Loading Issues**

- Verify model name and type in `config.yaml`
- For Ollama: Ensure Ollama is running (`ollama serve`) and model is pulled (`ollama pull model-name`)
- For AWS Bedrock: Check credentials (`aws sts get-caller-identity`), region, and model access
- For OpenAI: Ensure `OPENAI_API_KEY` environment variable is set

**RAG / Episodic Memory Not Working**

- Ensure `ENABLE_RAG: true` (or `ENABLE_EPISODIC_MEMORY: true`) in config
- Verify embedding model is configured and available (`RAG.EMBED_MODEL_ID` in config)
- For Ollama embeddings: ensure the embedding model is pulled (`ollama pull mxbai-embed-large`)
- Check logs for "fallback embeddings" warnings — this means the real model is unreachable
- Verify documents are being indexed with `list_documents()`

**Permission Errors**

- Ensure write permissions for `~/.mnemoai/`
- Ensure write permissions for `~/.mnemoai/` (the app home: config, plans, tasks, per-profile state)
- Check file paths in configuration

**Import Errors on Startup**

- Some dependencies (chromadb, faiss-cpu, crawl4ai) can be tricky to install. Check platform-specific instructions.
- On Apple Silicon: `faiss-cpu` may require `pip install faiss-cpu --no-cache-dir`

### Logging

Logs are output to stderr with configurable level:

```bash
LOG_LEVEL=DEBUG mnemoai  # Detailed logs
LOG_LEVEL=INFO mnemoai   # Normal logs (default)
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
