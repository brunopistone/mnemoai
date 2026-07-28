# Usage

## 🔀 Feature Toggles

All advanced features can be independently enabled or disabled in your config file. For normal installs, edit `~/.mnemoai/config/config.yaml`; source checkouts can also use `MNEMOAI_CONFIG` or the package-relative fallback. Here is a quick reference:

!!! note "About the Default column"

    **The Default column shows what the setup wizard writes — the state almost every install starts from.** If you instead hand-edit `config.yaml` and omit a key entirely, the code falls back differently: RAG, Episodic Memory, ACE Playbook, Query Routing, Orchestrator-Workers, Web Search, and Web Crawler fall back to **off**; Persistent Memory and Skills fall back to **on**.

| Feature                                                                   | Config Key                             | Default             | Dependencies                                                                       |
| ------------------------------------------------------------------------- | -------------------------------------- | ------------------- | ---------------------------------------------------------------------------------- |
| **RAG** (document indexing & search)                                      | `ENABLE_RAG: true`                     | `true`              | Embedding model (`RAG.EMBED_MODEL_ID`)                                             |
| **Episodic Memory** (learn from past tasks)                               | `ENABLE_EPISODIC_MEMORY: true`         | `true`              | Embedding model (`RAG.EMBED_MODEL_ID`)                                             |
| **ACE Playbook** (learn strategies from success/failure)                  | `ENABLE_PLAYBOOK: true`                | `true`              | None (embeddings optional for refinement)                                          |
| **Query Routing** (classify each query, bind a tool subset)               | `ENABLE_ROUTING: true`                 | `true`              | None ([details](orchestration.md#query-routing))                                   |
| **Orchestrator-Workers** (decompose complex tasks into subtasks)          | `ENABLE_ORCHESTRATION: true`           | `true`              | Requires `ENABLE_ROUTING: true` ([details](orchestration.md#orchestrator-workers)) |
| **User Profiling** (personalized responses)                               | `PROFILE.USE_PROFILING: true`          | `true`              | Activates after 5+ interactions                                                    |
| **Web Search**                                                            | `ENABLE_WEB_SEARCH: true`              | `true`              | `BRAVE_API_KEY` configured                                                         |
| **Web Crawler**                                                           | `ENABLE_WEB_CRAWL: true`               | `true`              | None                                                                               |
| **Vision** (image analysis)                                               | Configure `VISION_MODEL_ID`            | Disabled if not set | Vision-capable model                                                               |
| **Bash Confirmation** (prompt before each shell command)                  | `REQUIRE_BASH_CONFIRMATION: true`      | `true`              | None (auto-skips when non-interactive)                                             |
| **Write Confirmation** (prompt before each file write)                    | `REQUIRE_WRITE_CONFIRMATION: true`     | `true`              | None (auto-skips when non-interactive)                                             |
| **Persistent Memory** (curated memory the agent maintains, `MEMORY.md`)   | `ENABLE_MEMORY: true`                  | `true`              | None                                                                               |
| **Memory Confirmation** (prompt before each memory write)                 | `REQUIRE_MEMORY_CONFIRMATION: false`   | `false`             | None (auto-skips when non-interactive)                                             |
| **Memory Auto-Extraction** (background turn-end auto-save to `MEMORY.md`) | `ENABLE_MEMORY_AUTO_EXTRACTION: false` | `false`             | Writes without a prompt; one extra background model call per turn                  |
| **Verbose Mode** (show thinking process)                                  | CLI flag `--no-verbose`                | Enabled             | Supported by model                                                                 |
| **Session Recording** (resume a past session with `--resume`)             | `SESSION_MAX_AGE_DAYS: 30`             | `30` days           | `0` disables recording ([details](#resuming-a-session))                            |

**Dependency note:** RAG, Episodic Memory, and ACE Playbook refinement all require a working embedding model. If the embedding model is unavailable, the system falls back to SHA256-based deterministic embeddings with degraded semantic search quality. Configure `RAG.EMBED_MODEL_ID` in `config.yaml` to use a real embedding model (see [Embeddings Model](../configuration.md#embeddings-model)).

**Toggle them from the app:** run **`/features`** for a checklist of these `ENABLE_*` features — flip any on/off without editing `config.yaml`. Turning one on prompts for anything it needs (a Brave API key for web search, an embeddings model for RAG / episodic memory).

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

| Command            | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/exit` or `/quit` | Exit the application                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `/clear`           | Clear conversation history and RAG index                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `/save`            | Save the current conversation. After a `/load` (or an earlier `/save`), a bare `/save` **overwrites that same file**; `/save <path>` saves to a specific file/dir; a fresh conversation (after `/clear`) saves to a new timestamped file                                                                                                                                                                                                                                                                                                                 |
| `/load <path>`     | Load a saved conversation (no path → pick from a list; the picker has a **Delete** button to remove a saved conversation after a Yes/No confirm, then reopens). The loaded file becomes the one a later `/save` writes back to                                                                                                                                                                                                                                                                                                                           |
| `/compact [focus]` | Summarize older turns to shrink context (optional focus instructions)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `/config`          | Re-run the interactive configurator — chat, vision (with a "same as chat?" shortcut + provider choice), and embeddings models, then feature toggles. `Ctrl+B` steps back within a model section. Overwrites `config.yaml` and restarts in place                                                                                                                                                                                                                                                                                                          |
| `/model`           | Set up one model — chat (LLM), vision, or embeddings. Vision offers a "use the same model as Chat?" shortcut. Inference params reset to model defaults on a change (re-tune with `/params`); restarts in place                                                                                                                                                                                                                                                                                                                                           |
| `/params`          | Tune a model's inference parameters (temperature, top_p, top_k, penalties, reasoning, stop, stream, …) — only the params the chosen provider supports are offered, then restart in place                                                                                                                                                                                                                                                                                                                                                                 |
| `/features`        | Enable/disable app features — a checklist of RAG, episodic memory, playbook, web search/crawl, routing, orchestration, memory, skills. Turning one on prompts for anything it needs (Brave API key, embeddings model); restarts in place                                                                                                                                                                                                                                                                                                                 |
| `/mcp`             | List the configured MCP servers (built-in + any from `mcp.json`), their connection status, and tool counts                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `/skills`          | List installed skills (name + description); `/skills <name>` previews a skill's full instructions. See [Agent Skills](agent-skills.md) below                                                                                                                                                                                                                                                                                                                                                                                                             |
| `/memory`          | View the curated persistent memory (`MEMORY.md`); `/memory clear` wipes it (with a y/N confirm)                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `/plan`            | Toggle **plan mode** — an enforced read-only mode. While ON, the agent investigates and, when ready, presents a plan you **approve (y), edit in `$EDITOR` (e), or keep refining (n)**; approving turns plan mode off and the agent executes the plan in the same turn (saved to `plans/plan_<ts>.md`). Read-only shell commands (`ls`, `cat`, `grep`, `git status/log/diff`, …) still run; file edits, mutating shell, git writes, and background tasks are **hard-blocked** while ON. See [Plan Mode](productivity-tools.md#plan-mode) for full details |

### The prompt

On an interactive terminal the input `>` stays **pinned at the bottom** of the
screen while the assistant works; the answer, its reasoning, and tool activity
stream **above** it (native scrollback and copy/paste are preserved). You can
keep typing during a turn — see the shortcuts below.

### Keyboard Shortcuts

- `Ctrl+J`: Insert a new line in the input (`Enter` submits)
- `Enter`: Submit the message. **While the assistant is working**, a submitted
  message is **queued** (shown as a dim `> … (queued)` line) and runs as its own
  turn after the current one finishes — it's never folded into the running turn
  and never starts a second query in parallel.
- `Esc`: Cancel the turn that's currently running
- `Ctrl+C`: Cancel the running turn; when nothing is running, press twice (or
  `Ctrl+D`) to exit
- `/` then a letter: shows a slash-command completion menu (↑/↓ to move,
  `Enter`/`Tab` to accept)

### Pasting long text

Pasting a long block (a transcript, a file, a stack trace) collapses it into a
compact **`[Pasted text #N +M lines]`** placeholder in the input instead of
flooding the prompt — `M` is the number of line breaks. The full text is kept
aside; when you submit, the **assistant receives the complete paste**, and the
scrollback shows it **dimmed (gray)** so it's easy to tell apart from what you
typed. A large paste is shown **capped** in the scrollback — the first and last
few lines with a `… +N lines …` marker in between — so it never floods the
screen (the assistant still gets all of it). The collapsed view is only while
you're composing. Backspace over the placeholder deletes the **whole block at
once** (not character by character). Short pastes insert normally.

In dialogs (the `/load` picker and the `/config`, `/model`, `/params`,
`/features` prompts): arrow keys move, `Enter` confirms, `Esc` cancels, and
`Ctrl+B` steps back to the previous prompt within a model-section flow. These
briefly take over the full screen, then return you to the pinned prompt.

### Verbose Mode

Control thinking process visibility:

```bash
mnemoai              # Verbose mode (shows thinking)
mnemoai --no-verbose # Hide thinking process
# from a checkout: PYTHONPATH=src python -m mnemoai [--no-verbose]
```

### Resuming a Session

Every session is recorded automatically, **scoped to the directory you launched
from** — so resuming inside a project only ever offers that project's sessions:

```bash
mnemoai --resume              # pick from this directory's recent sessions
mnemoai --resume <session-id> # resume that session directly
mnemoai --continue            # resume the most recent one, no prompt
```

`--resume` with no value lists this directory's sessions newest-first, showing how
long ago each one ran, how many turns it had, and the opening prompt **as you
typed it** — retrieved memory, steering instructions and other context the
assistant adds behind the scenes are left out of the label:

```
   4m ago    6 turns  refactor the FSDP config parser
  2h ago    2 turns  why does the loader crash on startup?
   3d ago   14 turns  add a --resume flag
```

Pick one and the conversation replays into the terminal and continues where you
left off, with the restored conversation shown just above the prompt. `Esc`
cancels and **exits** — since you asked to resume, it won't quietly start a new
conversation instead; run `mnemoai` without a flag for that.

!!! note "`--resume` and `/save` are separate things"

    `--resume` is **automatic** — every session is recorded without you asking, and
    old ones expire (30 days by default; set `SESSION_MAX_AGE_DAYS`, or `0` to
    turn session recording off entirely).

    [`/save` and `/load`](#commands) are **yours to curate** — a conversation you
    deliberately keep, under a name or path you choose, in
    `~/.mnemoai/{profile}/conversations/`. Saved conversations are **never expired
    or deleted** by session cleanup, and resuming a session never overwrites one
    (after a `--resume`, a bare `/save` writes a new file rather than adopting the
    resumed session).

Sessions live in `~/.mnemoai/{profile}/sessions/{directory}/` as append-only
files, one per session. A turn that was **cancelled or that failed** is recorded
too — so a resumed session shows exactly what the live one did, including a
question you interrupted and one a dropped connection cut short.

**Resuming carries the whole conversation forward.** When you resume (or `/load`),
the restored history is copied into the new session's file, so that file is a
complete record on its own and resuming *it* later replays everything — not just
what you added after the restore. The session you resumed from is never modified,
so you can always go back and resume the same point again.

A launch you never typed into records nothing resumable, so its file is removed
when you exit — only sessions with at least one exchange are offered, which also
means a resume you didn't ask anything in won't appear as a duplicate of the
session it restored. If this directory has no sessions yet, `--resume` says so
and starts a normal session.
