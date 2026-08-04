# Usage

## 🔀 Feature Toggles

All advanced features can be independently enabled or disabled in your config file. For normal installs, edit `~/.mnemoai/config/config.yaml`; source checkouts can also use `MNEMOAI_CONFIG` or the package-relative fallback. Here is a quick reference:

!!! note "About the Default column"

    **The Default column shows what the setup wizard writes — the state almost every install starts from.** If you instead hand-edit `config.yaml` and omit a key entirely, the code falls back differently: RAG, Episodic Memory, ACE Playbook, Query Routing, Orchestrator-Workers, Web Search, and Web Crawler fall back to **off**; Persistent Memory and Skills fall back to **on**.

| Feature                                                                       | Config Key                             | Default             | Dependencies                                                                       |
| ----------------------------------------------------------------------------- | -------------------------------------- | ------------------- | ---------------------------------------------------------------------------------- |
| **RAG** (document indexing & search)                                          | `ENABLE_RAG: true`                     | `true`              | Embedding model (`RAG.EMBED_MODEL_ID`)                                             |
| **Episodic Memory** (learn from past tasks)                                   | `ENABLE_EPISODIC_MEMORY: true`         | `true`              | Embedding model (`RAG.EMBED_MODEL_ID`)                                             |
| **ACE Playbook** (learn strategies from success/failure)                      | `ENABLE_PLAYBOOK: true`                | `true`              | None (embeddings optional for refinement)                                          |
| **Query Routing** (classify each query, bind a tool subset)                   | `ENABLE_ROUTING: true`                 | `true`              | None ([details](orchestration.md#query-routing))                                   |
| **Orchestrator-Workers** (decompose complex tasks into subtasks)              | `ENABLE_ORCHESTRATION: true`           | `true`              | Requires `ENABLE_ROUTING: true` ([details](orchestration.md#orchestrator-workers)) |
| **User Profiling** (personalized responses)                                   | `PROFILE.USE_PROFILING: true`          | `true`              | Activates after 5+ interactions                                                    |
| **Web Search**                                                                | `ENABLE_WEB_SEARCH: true`              | `true`              | `BRAVE_API_KEY` configured                                                         |
| **Web Crawler**                                                               | `ENABLE_WEB_CRAWL: true`               | `true`              | None                                                                               |
| **Vision** (image analysis)                                                   | Configure `VISION_MODEL_ID`            | Disabled if not set | Vision-capable model                                                               |
| **Bash Confirmation** (prompt before each shell command)                      | `REQUIRE_BASH_CONFIRMATION: true`      | `true`              | None (auto-skips when non-interactive)                                             |
| **Write Confirmation** (prompt before each file write)                        | `REQUIRE_WRITE_CONFIRMATION: true`     | `true`              | None (auto-skips when non-interactive)                                             |
| **Persistent Memory** (curated memory the agent maintains, `MEMORY.md`)       | `ENABLE_MEMORY: true`                  | `true`              | None                                                                               |
| **Memory Confirmation** (prompt before each memory write)                     | `REQUIRE_MEMORY_CONFIRMATION: false`   | `false`             | None (auto-skips when non-interactive)                                             |
| **Git Override Confirmation** (prompt before overriding a git safety refusal) | `REQUIRE_GIT_CONFIRMATION: true`       | `true`              | None (auto-skips when non-interactive)                                             |
| **Memory Auto-Extraction** (background turn-end auto-save to `MEMORY.md`)     | `ENABLE_MEMORY_AUTO_EXTRACTION: false` | `false`             | Writes without a prompt; one extra background model call per turn                  |
| **Verbose Mode** (show thinking process)                                      | CLI flag `--no-verbose`                | Enabled             | Supported by model                                                                 |
| **Session Recording** (resume a past session with `--resume`)                 | `SESSION_MAX_AGE_DAYS: 30`             | `30` days           | `0` disables recording ([details](#resuming-a-session))                            |

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

| Command                    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/exit` or `/quit`         | Exit the application                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `/clear`                   | Clear conversation history and RAG index                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `/save`                    | Save the current conversation. After a `/load` (or an earlier `/save`), a bare `/save` **overwrites that same file**; `/save <path>` saves to a specific file/dir; a fresh conversation (after `/clear`) saves to a new timestamped file                                                                                                                                                                                                                                                                                                                                |
| `/load <path>`             | Load a saved conversation (no path → pick from a list; the picker has a **Delete** button to remove a saved conversation after a Yes/No confirm, then reopens). The loaded file becomes the one a later `/save` writes back to                                                                                                                                                                                                                                                                                                                                          |
| `/usage`                   | Token totals for this session, per model — input, output and cache tokens, counting **every** model call including sub-agents, orchestrator workers and the query router. Cumulative spend, not the size of your conversation ([details](#checking-token-usage))                                                                                                                                                                                                                                                                                                        |
| `/export [md\|txt] [path]` | Write the conversation as a **shareable transcript** — readable Markdown (default) or plain text — into the **current directory** unless you give a path. Not the same as `/save`: an export is a one-way artifact for pasting into a bug report or PR, not something `/load` can read back. Tool _calls_ appear as one-line summaries; tool _results_ and injected context are left out. Add `reasoning` to include thinking blocks ([details](#exporting-a-transcript))                                                                                               |
| `/branch [turn]`           | **Fork this session** and carry on in the copy. No argument → pick the turn to branch after; `/branch 3` branches directly. The original session is **never modified** — it stays resumable with `--resume` ([details](#branching-a-session))                                                                                                                                                                                                                                                                                                                           |
| `/compact [focus]`         | Summarize older turns to shrink context (optional focus instructions)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `/config`                  | Re-run the interactive configurator — chat, vision (with a "same as chat?" shortcut + provider choice), and embeddings models, then feature toggles. `Ctrl+B` steps back within a model section. Overwrites `config.yaml` and restarts in place                                                                                                                                                                                                                                                                                                                         |
| `/model`                   | Set up one model — chat (LLM), vision, or embeddings. Vision offers a "use the same model as Chat?" shortcut. Inference params reset to model defaults on a change (re-tune with `/params`); restarts in place                                                                                                                                                                                                                                                                                                                                                          |
| `/params`                  | Tune a model's inference parameters (temperature, top_p, top_k, penalties, reasoning, stop, stream, …) — only the params the chosen provider supports are offered. Applied **without restarting**, so the current conversation continues                                                                                                                                                                                                                                                                                                                                |
| `/features`                | Enable/disable app features — a checklist of RAG, episodic memory, playbook, web search/crawl, routing, orchestration, memory, skills. Turning one on prompts for anything it needs (Brave API key, embeddings model); restarts in place                                                                                                                                                                                                                                                                                                                                |
| `/mcp`                     | List the configured MCP servers (built-in + any from `mcp.json`), their connection status, and tool counts                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `/skills`                  | List installed skills (name + description); `/skills <name>` previews a skill's full instructions. See [Agent Skills](agent-skills.md) below                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `/memory`                  | View the curated persistent memory (`MEMORY.md`); `/memory clear` wipes it (with a y/N confirm)                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `/plan`                    | Toggle **plan mode** — an enforced read-only mode. While ON, the agent investigates and, when ready, presents a plan you **approve (y), edit in `$EDITOR` (e), or keep refining (n)**; approving turns plan mode off and the agent executes the plan in the same turn (saved to `plans/plan_<ts>.md`). Read-only shell commands (`ls`, `cat`, `grep`, `git status/log/diff`, …) still run; file edits, mutating shell, git writes, and background tasks are **hard-blocked** while ON. See [plan mode](safety.md#investigate-before-changing-anything) for full details |

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
   4m ago    6 turns  refactor the FSDP config parser (continued)
  2h ago    2 turns  why does the loader crash on startup?
   3d ago   14 turns  add a --resume flag
```

The turn count describes the **whole conversation you'd restore**, inherited
history included — so a chat you've resumed several times shows its full length,
not just what you typed since the last resume. `(continued)` marks a session that
carried on from an earlier one.

**One row per conversation, not per file.** Resuming records a new session seeded
with everything that came before, so a chat you resumed three times exists as four
files on disk. Only the most recent is offered: the earlier ones are contained in
it, and that's the one you want to continue. Nothing is deleted — and if you want
an earlier point specifically, `mnemoai --resume <session-id>` still restores that
exact one. (A [`/branch`](#branching-a-session) fork is _not_ collapsed, since it
diverges rather than continues.)

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
complete record on its own and resuming _it_ later replays everything — not just
what you added after the restore. The session you resumed from is never modified,
so you can always go back and resume the same point again.

A launch you never typed into records nothing resumable, so its file is removed
when you exit — only sessions with at least one exchange are offered, which also
means a resume you didn't ask anything in won't appear as a duplicate of the
session it restored. If this directory has no sessions yet, `--resume` says so
and starts a normal session.

### Branching a session

`/branch` **forks the current conversation** so you can try a different direction
without losing the one you have. Use it when a conversation has gone somewhere you
don't want — a wrong assumption three turns back, a tangent that ate the context —
and you'd rather rewind than start over and re-explain everything.

```
/branch          # pick the turn to branch after
/branch 3        # branch straight after turn 3
```

With no argument you get a list of this session's turns, labelled by what you
asked:

```
1. refactor the FSDP config parser
2. also handle the sharding edge case
3. now add tests for it  (latest)
```

Branching after turn 2 gives you a session containing turns 1–2, and you continue
from there — turn 3 stays behind in the original. **The original session is copied,
never changed**, so it remains resumable exactly as it was; if the branch turns out
to be a dead end, `--resume` the original and nothing is lost. That also means
branching is cheap: there's no "are you sure", because nothing is destroyed.

Forks are marked in the `--resume` picker, since a branch inherits its parent's
opening prompt and the two rows would otherwise look identical:

```
   2m ago    1 turn   refactor the FSDP config parser (branch @ turn 2)
   9m ago    3 turns  refactor the FSDP config parser
```

!!! note "Branching moves the conversation, not your files"

    A branch rewinds the **conversation only**. Files the abandoned turns wrote are
    still on disk — mnemoai doesn't snapshot and restore your working tree. So after
    branching past a turn that edited code, the assistant's picture of those files
    comes from the branch's history: say what you want undone, or check `git diff`
    before continuing.

### Checking token usage

`/usage` shows what this session has actually cost in tokens:

```
Token usage this session (as reported by the provider)

  anthropic.claude-opus-5
    12 calls  ·  in 103,204  ·  out 307  ·  total 103,511
    cache: 41,208 read  ·  0 written

  Current context: 10,741 tokens (what the next turn re-sends)
```

**It counts work you never see.** Sub-agents, orchestrator workers and the query
router are all real model calls, and they're usually where the tokens go — a single
delegated research task can spend an order of magnitude more than the turn that
triggered it. Those are exactly the numbers worth surfacing, so they're included.

**`/usage` and `[Context: N]` measure different things.** The context line after each
turn is how big the prompt is _right now_ — what the next turn re-sends, and what
[compaction](#commands) shrinks. `/usage` is cumulative spend since the session
started (or since your last `/clear`, which resets it). A long conversation has a
large context; a conversation with lots of delegated work has large usage. They move
independently.

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

### Exporting a transcript

`/export` writes the conversation as a **readable transcript** — the thing you paste
into a bug report, a PR description, or a message to someone else:

```
/export                  # Markdown, into the current directory
/export txt              # plain text instead
/export ~/notes/         # into a directory (name generated from your first prompt)
/export bug-report.md    # to an exact file
/export reasoning        # include the assistant's thinking blocks
```

This is **not** `/save`. `/save` writes JSON that `/load` can read back; an export is
one-way and optimized for a human reader:

- **Tool calls** appear as one-line summaries (`fs_read(path=…, mode=Search)`); a
  file body passed as an argument is replaced by its size rather than pasted in.
- **Tool results are left out.** A few thousand lines of file content is the single
  biggest source of noise in a transcript, and the answer already says what mattered.
- **Injected context is stripped** — retrieved memories and steering instructions
  were never something you typed, and unstripped they dominate the export.
- **Reasoning is off by default** (thinking blocks usually dwarf the conversation);
  `/export reasoning` includes them, collapsed in Markdown.

The filename is derived from your opening prompt
(`conversation_20260730_161738_reply-with-exactly-red.md`), so exports are
identifiable in a directory listing instead of being one of many timestamps.
