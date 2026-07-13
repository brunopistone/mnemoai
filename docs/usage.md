# Usage

## 🔀 Feature Toggles

All advanced features can be independently enabled or disabled in your config file. For normal installs, edit `~/.mnemoai/config/config.yaml`; source checkouts can also use `MNEMOAI_CONFIG` or the package-relative fallback. Here is a quick reference:

!!! note "About the Default column"
It reflects the bundled config template the setup wizard writes — what almost every install starts from. If you hand-edit `config.yaml` and omit a key entirely, the code falls back instead: RAG, Episodic Memory, ACE Playbook, Query Routing, Orchestrator-Workers, Web Search, and Web Crawler all fall back to **off**; Persistent Memory and Skills fall back to **on**.

| Feature                                                                 | Config Key                           | Default             | Dependencies                                                                           |
| ----------------------------------------------------------------------- | ------------------------------------ | ------------------- | -------------------------------------------------------------------------------------- |
| **RAG** (document indexing & search)                                    | `ENABLE_RAG: true`                   | `true`              | Embedding model (`RAG.EMBED_MODEL_ID`)                                                 |
| **Episodic Memory** (learn from past tasks)                             | `ENABLE_EPISODIC_MEMORY: true`       | `true`              | Embedding model (`RAG.EMBED_MODEL_ID`)                                                 |
| **ACE Playbook** (learn strategies from success/failure)                | `ENABLE_PLAYBOOK: true`              | `true`              | None (embeddings optional for refinement)                                              |
| **Query Routing** (classify each query, bind a tool subset)             | `ENABLE_ROUTING: true`               | `true`              | None ([details](advanced-features.md#query-routing))                                   |
| **Orchestrator-Workers** (decompose complex tasks into subtasks)        | `ENABLE_ORCHESTRATION: true`         | `true`              | Requires `ENABLE_ROUTING: true` ([details](advanced-features.md#orchestrator-workers)) |
| **User Profiling** (personalized responses)                             | `PROFILE.USE_PROFILING: true`        | `true`              | Activates after 5+ interactions                                                        |
| **Web Search**                                                          | `ENABLE_WEB_SEARCH: true`            | `true`              | `BRAVE_API_KEY` configured                                                             |
| **Web Crawler**                                                         | `ENABLE_WEB_CRAWL: true`             | `true`              | None                                                                                   |
| **Vision** (image analysis)                                             | Configure `VISION_MODEL_ID`          | Disabled if not set | Vision-capable model                                                                   |
| **Bash Confirmation** (prompt before each shell command)                | `REQUIRE_BASH_CONFIRMATION: true`    | `true`              | None (auto-skips when non-interactive)                                                 |
| **Write Confirmation** (prompt before each file write)                  | `REQUIRE_WRITE_CONFIRMATION: true`   | `true`              | None (auto-skips when non-interactive)                                                 |
| **Persistent Memory** (curated memory the agent maintains, `MEMORY.md`) | `ENABLE_MEMORY: true`                | `true`              | None                                                                                   |
| **Memory Confirmation** (prompt before each memory write)               | `REQUIRE_MEMORY_CONFIRMATION: false` | `false`             | None (auto-skips when non-interactive)                                                 |
| **Verbose Mode** (show thinking process)                                | CLI flag `--no-verbose`              | Enabled             | Supported by model                                                                     |

**Dependency note:** RAG, Episodic Memory, and ACE Playbook refinement all require a working embedding model. If the embedding model is unavailable, the system falls back to SHA256-based deterministic embeddings with degraded semantic search quality. Configure `RAG.EMBED_MODEL_ID` in `config.yaml` to use a real embedding model (see [Embeddings Model](configuration.md#embeddings-model)).

## 💡 Usage

### Basic Chat

Simply type your questions and press Enter. The assistant will respond using available tools when needed.

```
You: What files are in the current directory?
Assistant: [Uses fs_read tool to list directory contents]

You: Read the README.md file
Assistant: [Uses fs_read tool and displays content]
```

### Commands

| Command            | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/exit` or `/quit` | Exit the application                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `/clear`           | Clear conversation history and RAG index                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `/save`            | Save the current conversation. After a `/load` (or an earlier `/save`), a bare `/save` **overwrites that same file**; `/save <path>` saves to a specific file/dir; a fresh conversation (after `/clear`) saves to a new timestamped file                                                                                                                                                                                                                                                                                                           |
| `/load <path>`     | Load a saved conversation (no path → pick from a list). The loaded file becomes the one a later `/save` writes back to                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `/compact [focus]` | Summarize older turns to shrink context (optional focus instructions)                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `/config`          | Re-run the interactive configurator (overwrites `config.yaml`, then restarts the app in place to apply)                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `/model`           | Override just one model — chat (LLM), vision, or embeddings. Inference params (temperature, top_p, penalties, …) are reset to the model defaults on a change (they're model-specific); re-tune with `/params`. Restarts in place                                                                                                                                                                                                                                                                                                                   |
| `/params`          | Tune a model's inference parameters (temperature, top_p, top_k, penalties, reasoning, stop, stream, …) — only the params the chosen provider supports are offered, then restart in place                                                                                                                                                                                                                                                                                                                                                           |
| `/mcp`             | List the configured MCP servers (built-in + any from `mcp.json`), their connection status, and tool counts                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `/skills`          | List installed skills (name + description); `/skills <name>` previews a skill's full instructions. See [Agent Skills](#agent-skills) below                                                                                                                                                                                                                                                                                                                                                                                                         |
| `/memory`          | View the curated persistent memory (`MEMORY.md`); `/memory clear` wipes it (with a y/N confirm)                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `/plan`            | Toggle **plan mode** — an enforced read-only mode. While ON, the agent investigates and, when ready, presents a plan you **approve (y), edit in `$EDITOR` (e), or keep refining (n)**; approving turns plan mode off and the agent executes the plan in the same turn (saved to `plans/plan_<ts>.md`). Read-only shell commands (`ls`, `cat`, `grep`, `git status/log/diff`, …) still run; file edits, mutating shell, git writes, and background tasks are **hard-blocked** while ON. See [Plan Mode](productivity.md#plan-mode) for full details |

### The prompt

On an interactive terminal the input `>` stays **pinned at the bottom** of the
screen while the assistant works; the answer, its reasoning, and tool activity
stream **above** it (native scrollback and copy/paste are preserved). You can
keep typing during a turn — see the shortcuts below.

### Keyboard Shortcuts

- `Ctrl+J`: Insert a new line in the input (`Enter` submits)
- `Enter`: Submit the message. **While the assistant is working**, a submitted
  message is **queued** (shown as a dim `> … (queued)` line) and runs after the
  current turn finishes — it never starts a second query in parallel.
- `Esc`: Cancel the turn that's currently running
- `Ctrl+C`: Cancel the running turn; when nothing is running, press twice (or
  `Ctrl+D`) to exit
- `/` then a letter: shows a slash-command completion menu (↑/↓ to move,
  `Enter`/`Tab` to accept)

In dialogs (the `/load` picker and the `/config`, `/model`, `/params` prompts):
arrow keys move, `Enter` confirms, `Esc` cancels. These briefly take over the
full screen, then return you to the pinned prompt.

### Verbose Mode

Control thinking process visibility:

```bash
mnemoai              # Verbose mode (shows thinking)
mnemoai --no-verbose # Hide thinking process
# from a checkout: PYTHONPATH=src python -m mnemoai [--no-verbose]
```

## 🧩 Agent Skills

**Skills** are authored, reusable procedures the assistant loads **on demand** —
ideal for multi-step tasks you do repeatedly and want done a specific way (a
release checklist, "add a new API endpoint", a report format). They're distinct
from persistent memory (always-on facts) and the learned playbook: a skill is an
_authored procedure_ the model follows when the task matches.

Skills use **three-tier progressive disclosure**, so installing many is cheap:

1. **Always-on (tiny):** only each skill's `name` + `description` is added to the
   system prompt, so the model knows what's available.
2. **On trigger:** when your request matches a skill, the model loads that skill's
   **full instructions** (its `SKILL.md` body) into the conversation and follows
   them — no extra cost until then.
3. **On demand:** any reference files or scripts the skill bundles are read or run
   only if the procedure needs them.

### Creating a skill

Add a directory under `~/.mnemoai/skills/`, with a `SKILL.md` inside:

```
~/.mnemoai/skills/
└── commit-message/
    ├── SKILL.md          # required
    ├── reference.md      # optional — read on demand
    └── scripts/          # optional — run on demand
```

`SKILL.md` is YAML frontmatter + a markdown body of instructions:

```markdown
---
name: Conventional Commit Message
description: Use when the user asks to write or improve a git commit message...
---

# Conventional Commit Message

Step-by-step instructions the model follows once this skill is loaded...
```

- **`name`** and **`description`** are required. The **directory name** is the id
  the model uses to load the skill (`commit-message` above).
- Write the **description "pushy"** — start with _"Use when the user…"_ and include
  the phrases a user would actually say. The model decides whether to trigger a
  skill from this description, and tends to under-trigger if it's vague.
- Skills are seeded on first run: a `commit-message` example to copy, a
  **`skill-creator`** skill (just ask the assistant to "create a skill for X" and
  it writes a well-formed `SKILL.md` for you), and a **`steering-creator`** skill
  (ask it to "create a STEERING.md" or "document how to work in this project" and
  it investigates the repo and writes a well-formed `STEERING.md` following best
  practices).

### Using and managing skills

- The assistant triggers a matching skill automatically — no special syntax needed.
- `/skills` lists installed skills; if a skill is malformed it's shown under
  **Skipped** with the reason (e.g. missing `description`) so you can fix it.
- `/skills <name>` previews a skill's full body.
- Toggle the whole feature with `ENABLE_SKILLS` (default `true`) in `config.yaml`.
