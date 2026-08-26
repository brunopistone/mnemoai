# Troubleshooting

Diagnose the failures you're most likely to hit, by symptom.

**Run `/doctor` first.** It checks this install for the usual causes — which
`config.yaml` is actually loaded, whether the provider's credentials resolve and
its endpoint answers, whether `rg`/`git`/`bash` are on `PATH`, how many MCP tools
came up and whether every declared server connected, and whether an enabled
feature is missing its dependency. Most of the sections below start with a check
it already performed for you ([details](../guides/usage.md#checking-your-install)).

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

## A tool times out after 300 seconds

```
Tool execution error: 'glob_search' did not respond within 300s. The call was not retried …
```

The number is `LLM.MCP_CALL_TIMEOUT`, and the tool named in the message is
usually the **victim**, not the cause — especially when its real work takes
milliseconds. The MCP server is one subprocess with one event loop, so a tool
whose body blocks that loop holds up every other agent's call behind it until
their timeouts fire. Expect it in bursts, while several agents are working in
parallel.

1. If the failing call is a genuinely long one — a slow build under
   `execute_bash`, `wait_for_task` on a long background job — raise
   `LLM.MCP_CALL_TIMEOUT`. That is the case the message's own hint is about.
2. If it is a fast tool that timed out anyway, something else was blocking the
   server. Check the log for a long-running tool that started before it (see
   [Read the logs](#read-the-logs)); a tool added locally is the first suspect —
   write it as a plain `def` unless its body really awaits, so it is offloaded to
   a worker thread (`python -m pytest tests/unit/test_thread_offload.py` fails on
   an `async def` that never awaits, helpers included).
3. `glob_search` bounds itself at 30 seconds and returns what it found with
   `timed_out` set, so a slow filename search reports a partial result rather
   than hanging. Seeing that instead means the pattern is aimed at too large a
   tree: narrow it, point `path` at a subdirectory, or use `find` via
   `execute_bash`.

## `529 overloaded_error` / `Router classification failed`

```
WARNING — Stream connection failed (Error code: 529 … 'overloaded_error');
          retrying turn on a fresh connection in 1.0s (attempt 1/6)
WARNING — Router classification failed (Error code: 529 …); binding the full toolset
WARNING — Task decomposition failed: Error code: 529 …; using single subtask
```

A `529` (or a `503`/`overloaded`) is the provider saying it is busy, not a
misconfiguration. It is transient and retried automatically — the turn on a fresh
connection, and the smaller internal calls (routing, decomposition, each
compaction summary batch) up to three attempts each, with jittered backoff and
the provider's own `retry-after` honored when it sends one.

1. Nothing to do if the retry succeeds; the lines are informational.
2. The two `WARNING` lines above mean the retries were exhausted and the app
   proceeded on its fallback: every tool was bound instead of a routed subset, or
   the request ran as one task instead of a decomposed one. The answer is still
   produced — resend the message if the result looks coarser than usual.
3. A burst of them means genuine provider load. `LLM.MAX_RETRIES`,
   `LLM.RETRY_DELAY` and `LLM.RETRY_BACKOFF` govern how long the app waits;
   raising `RETRY_DELAY` helps more than raising `MAX_RETRIES` when the overload
   lasts more than a few seconds.
4. Concurrency multiplies it: orchestrator waves, `spawn_agent` sub-agents and the
   compaction summary all call the provider at once. Lower
   `LLM.SUBAGENT_MAX_CONCURRENCY` if overloads cluster around parallel work.

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
