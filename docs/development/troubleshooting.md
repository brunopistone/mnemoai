# Troubleshooting

Diagnose the failures you're most likely to hit, by symptom.

| Symptom                                            | Go to                                                                             |
| -------------------------------------------------- | --------------------------------------------------------------------------------- |
| Startup hangs, or tools are missing entirely       | [The assistant starts but has no tools](#the-assistant-starts-but-has-no-tools)   |
| `grep_search` returns `ripgrep (rg) not installed` | [Content search doesn't work](#content-search-doesnt-work)                        |
| A model error on the first prompt                  | [The model fails to load](#the-model-fails-to-load)                               |
| Searches find nothing in indexed documents         | [RAG or episodic memory returns nothing](#rag-or-episodic-memory-returns-nothing) |
| `fallback embeddings` in the logs                  | [Semantic search quality is degraded](#semantic-search-quality-is-degraded)       |
| A write or edit is refused                         | [A file write is refused](#a-file-write-is-refused)                               |
| `ImportError` before the prompt appears            | [Import errors on startup](#import-errors-on-startup)                             |

## The assistant starts but has no tools

The MCP server runs as a subprocess (`python -m mnemoai.server.server`). If it
fails to start, the client still comes up — but the model has nothing to call.

1. Run with `LOG_LEVEL=DEBUG mnemoai` and look for the server's stderr.
2. Check the MCP transport log at `~/.mnemoai/logs/mcp.log`, which captures the
   subprocess's own output across runs.
3. Confirm the interpreter running `mnemoai` can import the package:
   `python -c "import mnemoai; print(mnemoai.__file__)"`. A mismatch here is the
   usual cause when the app was installed into one environment and launched from
   another.
4. Run `/mcp` in the app to list the servers that did connect.

If an **external** MCP server from `~/.mnemoai/mcp.json` is the one failing, the
built-in tools still load — external server failures are skipped rather than
fatal. See [External MCP servers](../guides/mcp.md).

## Content search doesn't work

```
ripgrep (rg) not installed. Install with: brew install ripgrep (macOS) or apt install ripgrep (Linux)
```

`grep_search` requires ripgrep and has **no fallback**. Install it, then verify
with `rg --version`. Filename search (`glob_search`) is unaffected — it uses the
Python standard library.

## The model fails to load

Check `MODEL_ID.TYPE` and `MODEL_ID.NAME` in `config.yaml` first (`/model` edits
both), then the provider's own prerequisite:

| Provider             | Verify with                                                           |
| -------------------- | --------------------------------------------------------------------- |
| `ollama`             | `ollama serve` is running, and `ollama pull <model>` has been run     |
| `bedrock` / `mantle` | `aws sts get-caller-identity`, plus region and model access           |
| `openai`             | `OPENAI_API_KEY` is set (via the config `ENV:` block or the shell)    |
| `anthropic`          | `ANTHROPIC_API_KEY` is set                                            |
| `sagemaker`          | The endpoint name is correct and `InService` in the configured region |

## RAG or episodic memory returns nothing

Both are off unless enabled, and both need a working embedding model.

1. Confirm the feature is on: `ENABLE_RAG: true` or
   `ENABLE_EPISODIC_MEMORY: true`. `/features` shows the current state.
2. Confirm `RAG.EMBED_MODEL_ID` points at a model that is actually reachable.
3. Ask the assistant to run `list_documents` — if it reports none, nothing was
   ingested. Only large documents are offloaded to RAG; a small file is returned
   inline and never indexed.
4. Episodic recall is additionally filtered: results below the retrieval
   threshold are dropped, and injection is skipped for very short follow-up
   prompts. See [Memory & learning](../guides/memory.md).

## Semantic search quality is degraded

```
Using fallback embeddings
```

The configured embedding model was unreachable, so deterministic SHA256 vectors
were used instead. They keep the app running but carry no semantic meaning —
retrieval will look random. Fix the embedding model rather than tuning
thresholds: for Ollama, `ollama pull qwen3-embedding:0.6b`.

## A file write is refused

Three different guards produce three different messages:

| Message contains                         | Cause                                            | What to do                                               |
| ---------------------------------------- | ------------------------------------------------ | -------------------------------------------------------- |
| `read it first`                          | The file was never read this session             | Read the file, then retry the write                      |
| `changed on disk since you last read it` | The file changed after it was read               | Re-read it so the edit is based on current content       |
| `protected system directory`             | The target is under `/etc`, `/usr`, `/System`, … | Write somewhere in your home, project, or temp directory |

This is the read-before-write gate and the path policy. See
[Safety & permissions](../guides/safety.md).

## Import errors on startup

Some dependencies are heavy and platform-sensitive.

- `faiss-cpu` on Apple Silicon: `pip install faiss-cpu --no-cache-dir`
- `chromadb` / `crawl4ai`: check their platform-specific instructions; both are
  only needed if RAG or the web crawler is enabled
- If you installed from a checkout, re-run `pip install -e ".[dev]"` — the
  runtime extras alone don't cover the test and docs tooling

## Permission errors

Ensure the app home is writable: `~/.mnemoai/` holds config, prompts, plans,
tasks, logs, skills, agents, and all per-profile state. Override the location
with `MNEMOAI_HOME` if the default isn't writable.

## Read the logs

Logs go to **stderr**, at `WARNING` by default:

```bash
LOG_LEVEL=DEBUG mnemoai  # everything, including MCP traffic
LOG_LEVEL=INFO mnemoai   # lifecycle events
mnemoai                  # warnings and errors only
```

The MCP subprocess also writes to `~/.mnemoai/logs/mcp.log`, which persists
across runs and is the right place to look when startup fails before the prompt
appears.

## See also

- [Configuration](../configuration.md): every config key, with defaults
- [Safety & permissions](../guides/safety.md): what is blocked, and what asks first
- [The `~/.mnemoai` directory](../getting-started/mnemoai-directory.md): where state lives on disk
