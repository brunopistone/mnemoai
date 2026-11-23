# Runtime settings

## General Parameters { #general-parameters }

```yaml
# Context window size (passed to model as num_ctx for Ollama)
MAX_CONVERSATION_TOKENS: 65536

# Maximum tokens when reading documents (CSV, JSON, text files)
DOC_MAX_TOKENS: 16384

# Days a recorded session is kept for `--resume` (0 disables recording entirely).
# Saved conversations (/save) are separate and never expire.
SESSION_MAX_AGE_DAYS: 30

# Days a log file under ~/.mnemoai/logs/ is kept (0 disables the sweep).
LOG_MAX_AGE_DAYS: 7

# Profile configuration
PROFILE:
  NAME: default # Used for session data isolation (~/.mnemoai/{NAME}/)
  USE_PROFILING: true # Enable automatic user profiling
```

`SESSION_MAX_AGE_DAYS` controls the automatic session transcripts that
`mnemoai --resume` restores (see [Resuming a session](../features/sessions.md#resuming-a-session)).
Sessions are grouped by the directory you launched from, so resuming in a project
only offers that project's sessions. Set it to `0` to stop recording sessions
altogether; this never affects `/save` / `/load`.

`LOG_MAX_AGE_DAYS` expires everything under `~/.mnemoai/logs/` — the app log
(`mnemoai.log`, where the tracebacks the terminal deliberately doesn't print end
up) and the external MCP subprocess log. The app log also rotates at 2 MB, keeping two
generations, so one very noisy `LOG_LEVEL=DEBUG` run can't fill the disk before
the next sweep. Set it to `0` to keep logs indefinitely. See
[Read the logs](../development/troubleshooting.md#read-the-logs).

## LLM Interaction Configuration { #llm-interaction-configuration }

```yaml
LLM:
  ENABLE_THINKING: true # Enable thinking tags (verbose mode)
  RETRY_ENABLED: true # Retry failed LLM calls
  MAX_RETRIES: 5 # Maximum retry attempts; also caps retries of a
  # transient *empty* model response
  RETRY_DELAY: 1.0 # Seconds between retries
  RETRY_BACKOFF: 2.0 # Exponential backoff multiplier
  MAX_OUTPUT_CONTINUE_RETRIES: 3 # Auto-continue a turn cut off by MAX_TOKENS
  # (reasoning + answer exceeded the output
  # budget); 0 disables. See below.
  STREAM_IDLE_TIMEOUT: 120 # Abandon a streaming read that goes silent this
  # long (dead socket, e.g. laptop sleep) and
  # re-run the turn on a fresh connection;
  # 0 disables. See below.
  SUMMARIZATION_THINK: false # Include thinking in summarization
  TOKEN_COUNTING:
    OLLAMA_CHARS_PER_TOKEN: 3.0 # Ollama: chars per token (no tokenizer available)
    ANTHROPIC_MULTIPLIER: 1.5 # Per-provider safety multiplier; see below
  # --- Context management (compaction + overflow protection) ---
  KEEP_RECENT_MESSAGES: 6 # Turns kept verbatim on auto-compaction
  MANUAL_COMPACT_KEEP_RECENT: 2 # Smaller window for the manual /compact command
  KEEP_RECENT_TOKEN_BUDGET: 16384 # Also bound the kept window by tokens
  # COMPACT_HIGH_WATER_TOKENS  # Proactively compact before a turn when history
  # exceeds this. Auto-derives to 80% of
  # MAX_CONVERSATION_TOKENS when unset; 0 disables.
  # MAX_TOOL_RESULT_CHARS      # Cap one tool result (~4 chars/token) so a
  # runaway result can't overflow the window
  # (head+tail kept with a note). Auto-derives to
  # 10% of the window (in chars) when unset;
  # 0 disables.
  # TOOL_EVICTION_KEEP_RECENT: 8  # Messages kept verbatim by the tool-result
  # eviction layer (below).
  # EVICTED_TOOL_RESULT_CHARS: 500 # Char cap an OLD tool result is shrunk to
  # before any LLM summary. 0 disables the layer.
  RECURSION_LIMIT: 200 # Max model<->tool steps per execution segment
  MCP_CALL_TIMEOUT: 300 # Transport-layer timeout for one MCP tool call (s)
```

**Token counting.** `TOKEN_COUNTING` tunes the local estimate used when no current
provider measurement is available, including after resume/load or compaction.
The footer, `/context`, `/usage`'s context line and restore preflight share this
full-input estimate: live system prompt, steering/plan instructions, tool
definitions, message text, reasoning, tool calls and approximate framing.
Estimates are marked `~` in the footer and context notices.

Repeated tool failures and unchanged reads are also checked before another
execution: after three matching observations, the model is told to reuse the
result or change approach. New information, file changes or user guidance allow
recovery. Three further attempts without new information stop that actor
as incomplete, preserving completed work. Successful mutations and active job
polling remain available; a worker's stop does not cancel its parent.

When available, `usage_metadata.input_tokens` is the provider-reported size of
the **last request**, not a measurement of an unsent next request. `/usage`'s
input/output totals are separate cumulative spending, not this context estimate.

The estimate adds headroom but is not a guaranteed upper bound: tokenizers and
request framing vary by model and provider. Text is tokenized with tiktoken's
`o200k_base`, then scaled per provider family:

| Provider `TYPE`                   | Multiplier            | Override key                                |
| --------------------------------- | --------------------- | ------------------------------------------- |
| `openai`                          | 1.0 (tokenizer basis) | `OPENAI_MULTIPLIER`                         |
| `anthropic`, `mantle`             | 1.5                   | `ANTHROPIC_MULTIPLIER`, `MANTLE_MULTIPLIER` |
| `bedrock`, `sagemaker`, `litellm` | 1.35                  | `BEDROCK_MULTIPLIER`, …                     |
| anything else                     | 1.35                  | `<TYPE>_MULTIPLIER`                         |

`ollama` is the exception: no tokenizer is available, so the count is
`len(text) / OLLAMA_CHARS_PER_TOKEN` (default `3.0`).

`mlx` has no entry of its own and so takes the 1.35 fallback: one MLX server can
serve any tokenizer family, so a fixed per-provider figure would be a guess, and
the fallback adds headroom without promising exact counts.
Set `MLX_MULTIPLIER` if you want it tighter for the model you actually run.

!!! note "Two vestigial keys under `TOKEN_COUNTING`"

    Older config templates list `FALLBACK_MODEL` and `OLLAMA_APPROXIMATION`.
    `FALLBACK_MODEL` is no longer read by anything and can be deleted.
    `OLLAMA_APPROXIMATION` (default `1.3`) still applies, but only to the
    episodic-memory size budget — not to conversation token counting.

**Context management.** The conversation is kept under `MAX_CONVERSATION_TOKENS`
by summarizing older turns into the system prompt while keeping recent ones
verbatim — automatically when over budget, or manually via `/compact`. Several
layers prevent a single oversized turn from breaking the loop:

1. **Tool-result cap** (`MAX_TOOL_RESULT_CHARS`) — one runaway result (e.g. a
   `grep_search` with a huge `max_results`) is truncated head+tail with a note,
   so it can never alone exceed the context window. Auto-derives to 10% of the
   window (in chars) when unset, scaling with the model.
2. **Pre-flight compaction, layered** (`COMPACT_HIGH_WATER_TOKENS`) — before a
   turn, if the accumulated history is over the high-water mark, it is compacted.
   The mark auto-derives to 80% of `MAX_CONVERSATION_TOKENS` when unset. The
   cheapest layer runs first: **tool-result eviction** shrinks the bodies of
   _old_ tool results (grep/read/web dumps outside the recent window, which carry
   most of the context but are rarely needed verbatim once acted on) to a short
   head plus a marker, with **no model call** — recent turns stay verbatim and no
   message is dropped. If that alone gets back under budget, the expensive
   summary is skipped; otherwise it falls through to the full LLM summary. Either
   way the reduced state is checkpointed in the session transcript, so a resume
   comes back to it rather than to the full-size history. Tune
   with `TOOL_EVICTION_KEEP_RECENT` (messages kept verbatim, default 8) and
   `EVICTED_TOOL_RESULT_CHARS` (shrink target, default 500; 0 disables the layer).
3. **Overflow backstop** — if a request still exceeds the window, the turn ends
   with a clear message and compacts for the next turn instead of retrying the
   same oversized prompt in a loop.

Those three guard the _input_ side (prompt too large). The _output_ side has its
own recovery:

4. **Output-token auto-continue** (`MAX_OUTPUT_CONTINUE_RETRIES`) — when the
   model's response is cut off by `MODEL_ID.MAX_TOKENS` mid-turn (common with
   `REASONING_EFFORT: high`/`max` on a large context — reasoning plus a partial
   answer or tool call exhaust the output budget), the agent feeds the partial
   turn back and resumes ("continue where you left off"), accumulating the answer,
   up to this many attempts (default 3; 0 disables). It stops early when a
   continuation finishes cleanly or emits a tool call. You never have to type
   "continue"; if the retries are exhausted it surfaces a message to raise
   `MAX_TOKENS`.
5. **Stalled-stream recovery** (`STREAM_IDLE_TIMEOUT`) — a streaming read is a
   blocking socket read; if the connection dies silently (e.g. the laptop sleeps),
   it would otherwise park the worker thread forever and freeze the whole UI. A
   per-chunk idle-timeout watchdog abandons a stream that goes quiet for this long
   (default 120s; 0 disables) and the turn is **re-run on a fresh connection with
   exponential backoff** — the same recovery path used for a transient network
   drop (reset/timeout/5xx). The partial response is discarded (a dropped stream
   can't resume mid-generation), but the conversation continues; if reconnection
   keeps failing it ends with a "lost the connection — send your message again"
   message rather than hanging.
