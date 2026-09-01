# Orchestration & sub-agents

## Query Routing

When enabled, the assistant classifies each query before processing it and routes it to a specialized tool subset. This reduces noise for the model and improves response quality.

**Categories:**

| Route       | Description                                 | Tools Available                                              |
| ----------- | ------------------------------------------- | ------------------------------------------------------------ |
| `simple_qa` | Greetings, explanations, general knowledge  | None (direct LLM answer)                                     |
| `code`      | File ops, code editing, git, shell commands | fs_read, fs_write, file_edit, execute_bash, git, search, etc |
| `research`  | Web search, URL fetching                    | web_search, web_crawler                                      |
| `knowledge` | Document reading, indexing, RAG queries     | pdf/csv/docx/json readers, RAG tools, fs_read                |
| `full`      | Multi-category or ambiguous tasks           | All tools (fallback)                                         |

**How it works:**

1. A lightweight LLM call classifies the query into one of the categories above
2. The agent node binds only the tools for that category
3. If a query spans multiple categories, it routes to `full` (all tools)
4. The classifier prompt is customizable via `ROUTING_PROMPT` in `prompts.yaml`
   (all prompts live there, separate from `config.yaml`)

**Configuration:**

```yaml
# config.yaml — toggle
ENABLE_ROUTING: true

# prompts.yaml — the classifier prompt
ROUTING_PROMPT: |
  # Custom classifier prompt (optional, has a sensible default)
  ...
```

**Classification can run on its own model.** It produces a one-word label, so a
large reasoning model spends real latency on it — on every turn. Point the
`ROUTER` area at something small:

```yaml
AREA_MODELS:
  ROUTER: qwen3.5:1.7b
```

`/model` sets it without editing YAML (it lists a **Router model** row, which
starts by asking whether to keep using the chat model). See
[per-area models](../configuration.md#per-area-models-area_models) for the full
shape (any partial `MODEL_ID`, including a different provider).

## Orchestrator-Workers

When enabled alongside routing, tasks classified as `full` (spanning multiple categories) are automatically decomposed into focused subtasks executed by specialized workers.

**How it works:**

1. **Orchestrator**: An LLM call decomposes the complex query into subtasks, each assigned a category (code, research, knowledge, etc.) and an optional `depends_on` (which earlier subtasks it needs).
2. **Workers**: Each subtask runs with only the tools for its category. Subtasks are **scheduled by their dependencies** — independent subtasks run **in parallel** (bounded by `SUBAGENT_MAX_CONCURRENCY`), while a subtask that declared a `depends_on` waits for exactly those results (threaded into its prompt). A purely linear plan still runs in order.
3. **Aggregator**: If there were multiple subtasks, a final LLM call synthesizes all worker results into a single coherent response.

**Example flow for "Read this PDF and write a summary to a file" (a dependency chain):**

```
Orchestrator decomposes into:
  [Step 1/2: Read and summarize the PDF document]        → knowledge worker
  [Step 2/2: Write the summary to summary.md]            → code worker (depends_on step 1)
  [Synthesizing results...]                               → aggregator
```

**Example flow for "Investigate X and Y independently" (parallel):**

```
Orchestrator decomposes into (no dependencies between them):
  [Running 2 steps in parallel]
  [Step 1/2: Investigate X]   → worker  ┐ run
  [Step 2/2: Investigate Y]   → worker  ┘ together
  [Synthesizing results...]                               → aggregator
```

**Configuration:**

```yaml
# config.yaml — toggles
ENABLE_ROUTING: true # Required
ENABLE_ORCHESTRATION: true # Activates orchestrator for 'full' route

# prompts.yaml — customize the prompts (optional; sensible defaults bundled)
# ORCHESTRATOR_PROMPT: |      # decomposition prompt (teaches the depends_on field)
# AGGREGATOR_PROMPT: |        # synthesis prompt

# LLM — bounds how many independent subtasks run at once (shared with sub-agents)
LLM:
  SUBAGENT_MAX_CONCURRENCY: 4 # 1 = force sequential
```

**Decomposition can run on its own model too** — with the opposite trade-off to
the router. A bad split wastes the whole task, so this is the one internal call
that may deserve a _bigger_ model than the one answering:

```yaml
AREA_MODELS:
  ROUTER: qwen3.5:1.7b # small: it only picks a label
  ORCHESTRATOR: qwen3.5:32b # large: the split decides everything downstream
```

The workers and the aggregator keep using `MODEL_ID` — the aggregator writes the
answer you read, so it stays on the main model by design. Both areas are offered
by the first-run configurator (once the toggle above is on) and by `/model`
afterwards. Details in
[per-area models](../configuration.md#per-area-models-area_models).

**When orchestration is disabled**, `full` routes use all tools in a single agent loop (the previous behavior). No regression. Distinct from model-initiated **sub-agents** (`spawn_agent`): the orchestrator is framework-driven (it decomposes complex `full` queries for you), while sub-agents are the model's own on-demand delegation — both now share the same bounded concurrency engine.

## Sub-agents (`spawn_agent`)

Distinct from the orchestrator (which the framework drives), sub-agents are
**model-initiated**: the assistant can call the `spawn_agent` tool to hand a
self-contained task to a fresh sub-agent that runs in its **own isolated
context** and returns **only its final report**. The sub-agent's intermediate
tool calls never enter the main conversation, so the assistant's context stays
clean during search-heavy or multi-step work.

**How it works:**

1. The assistant calls `spawn_agent(agent_type, prompt, description)` with a
   complete, self-contained brief (the sub-agent doesn't see the conversation).
2. The sub-agent runs its own model↔tool loop with its type's tool allowlist and
   system prompt, then returns a concise report.
3. The assistant summarizes that report for you (the raw result isn't shown
   directly).

The sub-agent runs **quietly**: its internal steps (searches, file reads,
reasoning) don't stream to your terminal — you see a compact status line
(`… N sub-agents running`) while it works, and only its final report comes back.
For **independent** investigations the assistant can spawn several sub-agents in
one turn and they run **in parallel** (bounded by `LLM.SUBAGENT_MAX_CONCURRENCY`,
default 4; set it to 1 to force sequential). A destructive tool used inside a
sub-agent still asks for confirmation the same way the main assistant does.

**Background sub-agents.** For a long task you don't want to wait on, the
assistant can run a sub-agent **in the background**: the call returns right away
and you keep working. When it finishes, its report **surfaces automatically** —
if you're idle, the assistant speaks up on its own to deliver it; if you're
mid-conversation, it's folded into your next turn. Because a background sub-agent
has no terminal to prompt on, it **automatically skips any destructive tool that
isn't already approved** — so background work is safe by default; use it for
read-only investigation or when the needed actions were pre-approved. The
assistant can also **resume** a finished sub-agent with a follow-up ("now also
check the tests"), which continues it with its prior work as context — this
works even after restarting the app or loading a saved conversation, because
each run's brief and report are saved to disk.

**Built-in agent types:**

| Type              | Tools                   | Use for                                                   |
| ----------------- | ----------------------- | --------------------------------------------------------- |
| `general-purpose` | All tools               | Multi-step tasks that may research **and** make changes   |
| `explore`         | Read-only (search/read) | Locating code, tracing how things work, gathering context |
| `plan`            | Read-only (search/read) | Investigating, then returning a step-by-step plan         |

`explore` and `plan` cannot edit files or run commands. Nested spawning is
blocked (a sub-agent can't spawn its own sub-agents). Each type's system prompt
lives in `prompts.yaml` (`SUBAGENT_GENERAL_PROMPT`, `SUBAGENT_EXPLORE_PROMPT`,
`SUBAGENT_PLAN_PROMPT`) and can be customized.

**Custom agent types.** Define your own by dropping a markdown file in
`~/.mnemoai/agents/` (one file per agent, e.g. `reviewer.md`):

```markdown
---
name: reviewer # optional; defaults to the filename (reviewer)
description: Reviews a diff for bugs and style issues. Read-only.
tools: fs_read, grep_search, glob_search # optional allowlist; omit or "*" = all tools
---

You are a meticulous code reviewer. Given a diff or set of files, identify
bugs, edge cases, and style issues, then return a concise prioritized report.
```

- `description` is required (it's what the assistant reads to decide when to
  spawn the type); `name` and `tools` are optional.
- `tools` may be a YAML list or a comma-separated string; omit it (or use `*`)
  for the full toolset. `fs_read`/`describe_image` are always available;
  `spawn_agent` (no nested spawning) and `ask_user_question` (a sub-agent has no
  direct user — see [Questions the Assistant Asks
  You](productivity-tools.md#questions-the-assistant-asks-you)) are always
  removed, regardless of the list.
- The markdown body is the sub-agent's system prompt.
- A custom type **overrides** a built-in of the same name.
- Bad or incomplete files are skipped (reported, never fatal).

There's no `/agents` command: agents are authored as files and discovered
automatically. All available types (built-in + custom) are listed to the
assistant, which decides when to spawn one — you don't invoke them directly.
