# Usage

## 🔀 Feature Toggles

All advanced features can be independently enabled or disabled in your local `utils/config.yaml` (copied from `config.yaml.example`). Here is a quick reference:

| Feature                                                                 | Config Key                           | Default             | Dependencies                              |
| ----------------------------------------------------------------------- | ------------------------------------ | ------------------- | ----------------------------------------- |
| **RAG** (document indexing & search)                                    | `ENABLE_RAG: true`                   | `true`              | Embedding model (`RAG.EMBED_MODEL_ID`)    |
| **Episodic Memory** (learn from past tasks)                             | `ENABLE_EPISODIC_MEMORY: true`       | `true`              | Embedding model (`RAG.EMBED_MODEL_ID`)    |
| **ACE Playbook** (learn strategies from success/failure)                | `ENABLE_PLAYBOOK: true`              | `true`              | None (embeddings optional for refinement) |
| **User Profiling** (personalized responses)                             | `PROFILE.USE_PROFILING: true`        | `true`              | Activates after 5+ interactions           |
| **Web Search**                                                          | `ENABLE_WEB_SEARCH: true`            | `true`              | `BRAVE_API_KEY` configured                |
| **Web Crawler**                                                         | `ENABLE_WEB_CRAWL: true`             | `true`              | None                                      |
| **Vision** (image analysis)                                             | Configure `VISION_MODEL_ID`          | Disabled if not set | Vision-capable model                      |
| **Bash Confirmation** (prompt before each shell command)                | `REQUIRE_BASH_CONFIRMATION: true`    | `true`              | None (auto-skips when non-interactive)    |
| **Write Confirmation** (prompt before each file write)                  | `REQUIRE_WRITE_CONFIRMATION: true`   | `true`              | None (auto-skips when non-interactive)    |
| **Persistent Memory** (curated memory the agent maintains, `MEMORY.md`) | `ENABLE_MEMORY: true`                | `true`              | None                                      |
| **Memory Confirmation** (prompt before each memory write)               | `REQUIRE_MEMORY_CONFIRMATION: false` | `false`             | None (auto-skips when non-interactive)    |
| **Verbose Mode** (show thinking process)                                | CLI flag `--no-verbose`              | Enabled             | Supported by model                        |

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

| Command            | Description                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/exit` or `/quit` | Exit the application                                                                                                                                                                                                                                                                                                                                                                          |
| `/clear`           | Clear conversation history and RAG index                                                                                                                                                                                                                                                                                                                                                      |
| `/save`            | Save current conversation                                                                                                                                                                                                                                                                                                                                                                     |
| `/load <path>`     | Load a saved conversation                                                                                                                                                                                                                                                                                                                                                                     |
| `/compact [focus]` | Summarize older turns to shrink context (optional focus instructions)                                                                                                                                                                                                                                                                                                                         |
| `/config`          | Re-run the interactive configurator (overwrites `config.yaml`, then restarts the app in place to apply)                                                                                                                                                                                                                                                                                       |
| `/model`           | Override just one model — chat (LLM), vision, or embeddings. Inference params (temperature, top_p, penalties, …) are reset to the model defaults on a change (they're model-specific); re-tune with `/params`. Restarts in place                                                                                                                                                              |
| `/params`          | Tune a model's inference parameters (temperature, top_p, top_k, penalties, reasoning, stop, stream, …) — only the params the chosen provider supports are offered, then restart in place                                                                                                                                                                                                      |
| `/mcp`             | List the configured MCP servers (built-in + any from `mcp.json`), their connection status, and tool counts                                                                                                                                                                                                                                                                                    |
| `/skills`          | List installed skills (name + description); `/skills <name>` previews a skill's full instructions. See [Agent Skills](#-agent-skills) below                                                                                                                                                                                                                                                   |
| `/memory`          | View the curated persistent memory (`MEMORY.md`); `/memory clear` wipes it (with a y/N confirm)                                                                                                                                                                                                                                                                                               |
| `/plan`            | Toggle **plan mode** — an enforced read-only mode. While ON, the agent investigates and presents a plan; **read-only** shell commands (`ls`, `cat`, `grep`, `git status/log/diff`, …) still run and it may draft the plan to a `.md` file under the plans dir, but file edits, mutating shell commands, git writes, and background tasks are **hard-blocked** until you `/plan` again to exit |

### Keyboard Shortcuts

- `Ctrl+J`: Insert new line in input
- `Enter`: Submit message
- `Ctrl+C`: Interrupt operation (press twice to exit)

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
- Two skills are seeded on first run: a `commit-message` example to copy, and a
  **`skill-creator`** skill — just ask the assistant to "create a skill for X" and
  it loads that guidance and writes a well-formed `SKILL.md` for you.

### Using and managing skills

- The assistant triggers a matching skill automatically — no special syntax needed.
- `/skills` lists installed skills; if a skill is malformed it's shown under
  **Skipped** with the reason (e.g. missing `description`) so you can fix it.
- `/skills <name>` previews a skill's full body.
- Toggle the whole feature with `ENABLE_SKILLS` (default `true`) in `config.yaml`.
