# Troubleshooting

Diagnose the failures you're most likely to hit, by symptom.

**Run `/doctor` first.** It checks this install for the usual causes — which
`config.yaml` is actually loaded, whether the provider's credentials resolve and
its endpoint answers, whether `rg`/`git`/`bash` are on `PATH`, how many MCP tools
came up and whether every declared server connected, and whether an enabled
feature is missing its dependency. Most of the sections below start with a check
it already performed for you ([details](../guides/usage.md#checking-your-install)).

| Symptom                                                     | Go to                                                                             |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Startup hangs, or tools are missing entirely                | [The assistant starts but has no tools](#the-assistant-starts-but-has-no-tools)   |
| `grep_search` returns `ripgrep (rg) not installed`          | [Content search doesn't work](#content-search-doesnt-work)                        |
| A model error on the first prompt                           | [The model fails to load](#the-model-fails-to-load)                               |
| Searches find nothing in indexed documents                  | [RAG or episodic memory returns nothing](#rag-or-episodic-memory-returns-nothing) |
| `fallback embeddings` in the logs                           | [Semantic search quality is degraded](#semantic-search-quality-is-degraded)       |
| `Connection was closed before we received a valid response` | [A turn dies on a dropped connection](#a-turn-dies-on-a-dropped-connection)       |
| `No first token for Ns` on a long conversation              | [A turn dies on a dropped connection](#a-turn-dies-on-a-dropped-connection)       |
| A write or edit is refused                                  | [A file write is refused](#a-file-write-is-refused)                               |
| `ImportError` before the prompt appears                     | [Import errors on startup](#import-errors-on-startup)                             |

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

## A turn dies on a dropped connection

```
WARNING — Stream connection failed (No stream data for 120s (connection likely dropped));
          retrying turn on a fresh connection in 1.2s (attempt 1/6)
ERROR   — Model request failed: Connection was closed before we received a valid
          response from endpoint URL: ".../converse-stream".
● The model request failed with an error I can't recover from automatically.
```

Both lines are retried automatically as of 1.12.3. If you see this on an older
version, two separate causes were at work — and the giveaway for each is how large
the context was (the footer's meter, or the `[Context: N tokens]` line printed with
the turn on those versions):

1. **A genuinely dropped socket.** botocore words this "Connection **was** closed",
   which the transient-error classifier didn't recognize, so the most retryable
   failure there is got no retry at all and ended the turn. Fixed by matching
   botocore's own phrasings.
2. **A long prompt mistaken for a dead stream.** `LLM.STREAM_IDLE_TIMEOUT`
   (default 120s) used to police the wait for the _first_ token as well as the gaps
   between chunks. That first wait is the model reading your whole prompt before it
   answers — minutes on a large conversation, not seconds — so the watchdog killed
   healthy turns, and each retry re-sent the same prompt and re-paid the same wait.
   The first-token window is now derived from `LLM.REQUEST_TIMEOUT` (default 600s)
   instead, while `STREAM_IDLE_TIMEOUT` still guards the running stream.

If it persists after upgrading, the context itself is the problem — a turn large
enough to exceed `REQUEST_TIMEOUT` before the first token can't be rescued by
retrying, only by sending less:

- Run `/context` to see what the next turn pays for, and `/compact` to summarize
  the conversation now.
- Lower `MAX_CONVERSATION_TOKENS`. It sets the compaction trigger too (80% of it by
  default), so a value matching the model's full window — e.g. `1000000` — means
  compaction effectively never runs and every turn re-sends the whole history.
  A ceiling well below the model's limit keeps turns fast; the model's context
  window is a hard maximum, not a target.
- `/branch` to continue from an earlier point without the accumulated history.

## A turn ends with "I couldn't compact it further"

```
Compacted: summarized 1209 older messages, kept 2 recent.
WARNING — Context overflow: prompt is too long: 3155357 tokens > 1000000 maximum;
          compacted and retrying
WARNING — Context overflow: prompt is too long: 3157792 tokens > 1000000 maximum;
          could not compact
● The conversation grew past the model's context window and I couldn't compact it
  further. Use /clear to start fresh, or /compact <focus>.
```

Fixed in 1.12.4. The tell is the arithmetic: the second overflow is a few thousand
tokens above the first, and it lands _after_ a compaction that reduced history to
two messages — so the message is wrong, there was plenty left to compact. Three
separate defects lined up, and on older versions all three are worth knowing about:

1. **The compaction didn't reach the rest of the turn.** A running turn re-reads the
   history it started with, so only the immediate retry saw the smaller prompt; the
   next model call re-sent the original one, and by then compaction genuinely had
   nothing left to give.
2. **The turn then undid its own compaction.** Everything summarized away was
   appended back when the turn ended — and written to the session transcript as
   this turn's work — so the following turn compacted the same history again
   (`1209 older messages`, then `1165`).
3. **The pre-flight check was skipped entirely.** It uses the context size the
   provider reported for the previous turn, and `/branch`, `/load` and `--resume`
   replace history wholesale. Since the transcript keeps every message compaction
   had summarized away, restoring it makes history much _larger_ while the
   leftover count still describes the small version — so the check passed and the
   first turn went straight to a provider-side overflow.

Defect 3 is why this shows up right after a `/branch`, `/load` or `--resume` of a
long conversation. On an older version, `/compact` immediately after restoring one
avoids it; on 1.12.4 the next turn measures the history that is actually there, and
since 1.12.6 restoring rebuilds the compacted state rather than re-inflating it
(see [below](#a-resumed-session-reports-a-much-larger-context-than-when-i-closed-it)).

## A resumed session reports an impossible context size

```
[Context: 12650351 tokens]
[Resumed session 20260827_095050_7757_521593]
```

Twelve million tokens into a one-million-token window is not a display glitch —
the history really was that large, and the next message would have overflowed.
Fixed in 1.12.5: each resume used to re-save every tool result wrapped inside a
copy of itself with all the quotes escaped again, so the same results roughly
**doubled on every resume** while adding no content. File reads in the reported
session had reached 1.07M characters each, and ~90% of the 21M-character
transcript was backslashes.

Upgrading is the whole fix — there is nothing to delete or migrate. Opening an
affected conversation (`--resume`, `--continue` or `/load`) unwraps the nested
copies as it reads and re-saves it clean; the reported session dropped from 21M to
2.1M characters with no content lost. To see the effect, run `/context` after
resuming.

One related note if the number is large but plausible: `MAX_TOOL_RESULT_CHARS`
caps a tool result when it is first produced, not when it is restored, and it
derives from `MAX_CONVERSATION_TOKENS` — at `1000000` a single result may be
400,000 characters. If the number is instead much larger than it was when you
closed the session, see the next section.

## A resumed session reports a much larger context than when I closed it

```
[Context: 235793 tokens]     ← before closing
[Context: 1166221 tokens]    ← after --resume
```

Fixed in 1.12.6. The transcript holds every turn's full text — deliberately,
because nothing you said should ever be lost from disk — but restoring it used to
replay all of that text, which brought back everything a previous `/compact` had
summarized away. So the context came back at its pre-compaction size (here 5×
larger, past the model's window), the first message after resuming had to summarize
the whole conversation again, and the summary you had already paid for was
discarded. A compaction now records what replaced that history, and a restore
rebuilds the state the session ended in, for `--resume`, `/load` and `/branch`
alike. The conversation still replays on screen in full — what came back is visible
in the footer's context meter, which reads about what it did when you closed the
session.

Nothing needs migrating, but note what a checkpoint is and isn't:

- It is written **when a compaction happens**, so a session that never compacted is
  unaffected, and a session compacted on an older version has no checkpoint to
  restore — the first `/compact` after upgrading creates one.
- `/save` records the active summary too, so a `/load` of a compacted conversation
  no longer resumes mid-thread with the earlier history missing. Files saved before
  1.12.6 load exactly as they did.
- A `/branch` **before** the compaction point deliberately forks the raw history —
  that is the point of rewinding to there.
- A compaction that only trimmed old tool-result bodies (the cheap pass that runs
  before any summarizing) is checkpointed as well, but only since **1.12.7** — on
  1.12.6 that case still came back at full size, and with no summary involved
  nothing on screen hinted that anything had been reclaimed.

On an older version, `/compact` immediately after resuming is the workaround; it
costs a summarization but brings the context back down before the first real turn.

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

Two destinations, with different jobs.

**On screen**, a problem is one line in the app's own shape — no timestamp,
logger name or level word, and never a stack trace (the terminal is a
conversation, and a trace there buries the answer above it). The mark carries
the severity: `✗` an error, `!` a warning, `·` anything lower.

```
✗ Query failed: division by zero (traceback → ~/.mnemoai/logs/mnemoai.log)
```

The pointer at the end appears only when something *was* left out — a
traceback, or a message too long for one line.

**On disk**, `~/.mnemoai/logs/mnemoai.log` has the whole record — traceback,
thread name, and the surrounding `INFO` lifecycle lines — for both the app's own
errors and anything a library or the standard library logged. Start here for any
"it went wrong and I couldn't see why". The file rotates at 2 MB (two
generations kept) and every file under `logs/` is deleted after
`LOG_MAX_AGE_DAYS` days (default 7; `0` keeps them forever). `/doctor` prints
the path and the retention it's using.

The screen threshold is `WARNING` by default:

```bash
LOG_LEVEL=DEBUG mnemoai  # everything, tracebacks included, on screen too
LOG_LEVEL=INFO mnemoai   # lifecycle events
mnemoai                  # warnings and errors only, one line each
```

The MCP subprocess writes separately to `~/.mnemoai/logs/mcp.log`, which is the
right place to look when startup fails before the prompt appears.

## See also

- [Configuration](../configuration.md): every config key, with defaults
- [Safety & permissions](../guides/safety.md): what is blocked, and what asks first
- [The `~/.mnemoai` directory](../getting-started/mnemoai-directory.md): where state lives on disk
