# Context and token usage

## Seeing where the context goes { #seeing-where-the-context-goes }

`/usage` answers "what has this session spent". `/context` answers the other
question — "what is my next turn paying for, and which part of it can I shrink":

```
Context window — 48,120 tokens of 1,000,000 (5%)
  [█░░░░░░░░░░░░░░░░░░░░░░░]

  System prompt                                 9,930   21%
  Persistent memory (MEMORY.md)                   430    1%
  Skills listing                                  210    0%

  Steering: 2 files                            14,880   31%
    ~/.mnemoai/STEERING.md                        640
    ~/dev/project/CLAUDE.md                    14,240

  Tool schemas: 41 tools                       11,300   23%

  Conversation: 38 messages                    11,370   24%
    tool results                                8,960
    assistant replies                           1,880
    your prompts                                  530

  Free: 951,880 tokens  ·  compaction starts at 800,000
```

The two rows worth looking at are the ones nothing else surfaces: **steering files**
and **tool schemas** are re-sent verbatim on every single call and
[compaction](../reference/commands.md#commands) can never reclaim them, so a 14k-token repo `CLAUDE.md` is a
permanent tax on every turn — trimming the file is the only thing that helps. The
conversation row is the part `/compact` shrinks.

The **total** is exact (the provider's own count for the last turn, the same number
the [status footer](terminal.md#the-status-footer) shows); the **split** is estimated and scaled onto it,
so the percentages are meaningful even though the per-part counts are approximate.
After a restore, compaction, or a request with missing usage, the measurement is
unavailable and the report explicitly switches to the full-input estimate.

## Checking token usage { #checking-token-usage }

`/usage` shows what this session has actually cost in tokens:

```
Token usage this session (as reported by the provider)

  anthropic.claude-opus-5
    12 calls  ·  in 103,204  ·  out 307  ·  total 103,511
    cache: 41,208 read  ·  0 written

  Current context: 10,741 tokens (provider-reported last input)
```

**It counts work you never see.** Sub-agents, orchestrator workers, decomposition,
aggregation, routing, compaction summaries, review and recovery calls are included.
Retries with missing usage are counted as unmeasured attempts, not silently
treated as free requests.

**The cache line is prompt caching at work.** On the providers that support it the
stable start of every request — system prompt, tool definitions, prior turns — is
cached and re-read instead of re-charged, so `cache: … read` grows as a turn makes
tool calls. See
[`PROMPT_CACHE`](../configuration/parameters.md#prompt_cache-reuse-the-prompt-prefix-instead-of-re-paying-for-it)
for what it applies to and how to turn it off.

**`/usage` and the footer's context meter measure different things.** The footer
shows the last measured request size, or an estimate of current input when that
measurement is unavailable. `/usage` is cumulative spend since the session
started (or since your last `/clear`, which resets it). A long conversation has a
large context; a conversation with lots of delegated work has large usage. They move
independently.

Resume/load restores conversation context and its compaction summary, not past
spending. A freshly resumed process therefore starts with zero spending; loading
history into an existing process does not add its historical tokens to `/usage`.

!!! note "No dollar figure, on purpose"

    Token pricing doesn't apply uniformly across the providers mnemoai supports:
    Ollama and a local OpenAI-compatible server cost nothing per token, SageMaker
    bills by endpoint-hour rather than by token, and LiteLLM can proxy any model at
    prices this process has no way to know. A hardcoded price table would produce
    confidently wrong numbers, so `/usage` reports tokens and tells you whose
    numbers they are.

    For the same reason the totals are **reported, not measured**: they come from each
    provider's own `usage_metadata`. Not every provider populates it — if some calls
    report nothing, the report says so and treats the total as a lower bound rather
    than quietly counting them as zero.
