# Usage

## 🔀 Feature Toggles

All advanced features can be independently enabled or disabled in your config file. For normal installs, edit `~/.mnemoai/config/config.yaml`; source checkouts can also use `MNEMOAI_CONFIG` or the package-relative fallback. Here is a quick reference:

!!! note "About the Default column"

    **The Default column shows what the setup wizard writes — the state almost every install starts from.** If you instead hand-edit `config.yaml` and omit a key entirely, the code falls back differently: RAG, Episodic Memory, ACE Playbook, Query Routing, Orchestrator-Workers, Web Search, and Web Crawler fall back to **off**; Persistent Memory and Skills fall back to **on**.

| Feature                                                                       | Config Key                             | Default             | Dependencies                                                                                  |
| ----------------------------------------------------------------------------- | -------------------------------------- | ------------------- | --------------------------------------------------------------------------------------------- |
| **RAG** (document indexing & search)                                          | `ENABLE_RAG: true`                     | `true`              | Embedding model (`RAG.EMBED_MODEL_ID`)                                                        |
| **Episodic Memory** (learn from past tasks)                                   | `ENABLE_EPISODIC_MEMORY: true`         | `true`              | Embedding model (`RAG.EMBED_MODEL_ID`)                                                        |
| **ACE Playbook** (learn strategies from success/failure)                      | `ENABLE_PLAYBOOK: true`                | `true`              | None (embeddings optional for refinement)                                                     |
| **Query Routing** (classify each query, bind a tool subset)                   | `ENABLE_ROUTING: true`                 | `true`              | None ([details](orchestration.md#query-routing))                                              |
| **Orchestrator-Workers** (decompose complex tasks into subtasks)              | `ENABLE_ORCHESTRATION: true`           | `true`              | Requires `ENABLE_ROUTING: true` ([details](orchestration.md#orchestrator-workers))            |
| **User Profiling** (personalized responses)                                   | `PROFILE.USE_PROFILING: true`          | `true`              | Activates after 5+ interactions                                                               |
| **Web Search**                                                                | `ENABLE_WEB_SEARCH: true`              | `true`              | `BRAVE_API_KEY` configured                                                                    |
| **Web Crawler**                                                               | `ENABLE_WEB_CRAWL: true`               | `true`              | None                                                                                          |
| **Vision** (image analysis)                                                   | Configure `VISION_MODEL_ID`            | Disabled if not set | Vision-capable model                                                                          |
| **Bash Confirmation** (prompt before each shell command)                      | `REQUIRE_BASH_CONFIRMATION: true`      | `true`              | None (auto-skips when non-interactive)                                                        |
| **Write Confirmation** (prompt before each file write)                        | `REQUIRE_WRITE_CONFIRMATION: true`     | `true`              | None (auto-skips when non-interactive)                                                        |
| **Persistent Memory** (curated memory the agent maintains, `MEMORY.md`)       | `ENABLE_MEMORY: true`                  | `true`              | None                                                                                          |
| **Memory Confirmation** (prompt before each memory write)                     | `REQUIRE_MEMORY_CONFIRMATION: false`   | `false`             | None (auto-skips when non-interactive)                                                        |
| **Git Override Confirmation** (prompt before overriding a git safety refusal) | `REQUIRE_GIT_CONFIRMATION: true`       | `true`              | None (auto-skips when non-interactive)                                                        |
| **Memory Auto-Extraction** (background turn-end auto-save to `MEMORY.md`)     | `ENABLE_MEMORY_AUTO_EXTRACTION: false` | `false`             | Writes without a prompt; one extra background model call per turn                             |
| **Verbose Mode** (show thinking process)                                      | CLI flag `--no-verbose`                | Enabled             | Supported by model                                                                            |
| **Session Recording** (resume a past session with `--resume`)                 | `SESSION_MAX_AGE_DAYS: 30`             | `30` days           | `0` disables recording ([details](#resuming-a-session))                                       |
| **`@`-mention size** (per file attached with `@` in a prompt)                 | `MENTIONS.MAX_FILE_CHARS: 20000`       | `20000` chars       | `0` lifts the cap ([details](#attaching-a-file-with))                                         |
| **Turn-end notification** (bell + desktop notification when a long turn ends) | `NOTIFY.AFTER_SECONDS: 30`             | `30` seconds        | `0` disables; `NOTIFY.BELL` / `NOTIFY.DESKTOP` ([details](#when-the-terminal-wants-you-back)) |
| **Log retention** (days a file under `~/.mnemoai/logs/` is kept)              | `LOG_MAX_AGE_DAYS: 7`                  | `7` days            | `0` keeps them forever ([details](../development/troubleshooting.md#read-the-logs))           |

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
| `/help`                    | Show the command reference and keyboard keys — the same box the launch banner prints, brought back after it has scrolled away                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `/exit` or `/quit`         | Exit the application                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `/clear`                   | Clear conversation history and RAG index                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `/rewind`                  | **Take back your last prompt** and everything the turn produced, from the conversation and the transcript. Moves the conversation only — **files on disk are untouched** ([details](#taking-back-your-last-prompt))                                                                                                                                                                                                                                                                                                                                                     |
| `/context`                 | Break down **what is filling the context window** right now — system prompt, steering files, tool schemas, conversation — with each part's share and how much room is left ([details](#seeing-where-the-context-goes))                                                                                                                                                                                                                                                                                                                                                  |
| `/save`                    | Save the current conversation. After a `/load` (or an earlier `/save`), a bare `/save` **overwrites that same file**; `/save <path>` saves to a specific file/dir; a fresh conversation (after `/clear`) saves to a new timestamped file                                                                                                                                                                                                                                                                                                                                |
| `/load <path>`             | Load a saved conversation (no path → pick from a list; the picker has a **Delete** button to remove a saved conversation after a Yes/No confirm, then reopens). The loaded file becomes the one a later `/save` writes back to                                                                                                                                                                                                                                                                                                                                          |
| `/usage`                   | Token totals for this session, per model — input, output and cache tokens, counting **every** model call including sub-agents, orchestrator workers and the query router. Cumulative spend, not the size of your conversation ([details](#checking-token-usage))                                                                                                                                                                                                                                                                                                        |
| `/files`                   | **What this session touched** — every file it read, wrote or you attached with `@`, newest first, with how many times each came up. Includes sub-agent and parallel-wave work, and survives compaction ([details](#what-this-session-touched))                                                                                                                                                                                                                                                                                                                          |
| `/diff [path]`             | **Uncommitted changes, with this session's edits marked `✎`** — staged, unstaged and untracked in one list with `+`/`−` counts. `/diff <path>` shows one file's colored diff. Read-only: it never stages, stashes or checks anything out ([details](#seeing-what-changed))                                                                                                                                                                                                                                                                                              |
| `/copy [code\|N]`          | **Copy the last answer to the clipboard** without the terminal's line wrapping. `/copy code` takes its last fenced code block, `/copy 2` the answer before last. Uses a local helper or OSC 52, so it works over SSH too ([details](#copying-an-answer))                                                                                                                                                                                                                                                                                                                |
| `/export [md\|txt] [path]` | Write the conversation as a **shareable transcript** — readable Markdown (default) or plain text — into the **current directory** unless you give a path. Not the same as `/save`: an export is a one-way artifact for pasting into a bug report or PR, not something `/load` can read back. Tool _calls_ appear as one-line summaries; tool _results_ and injected context are left out. Add `reasoning` to include thinking blocks ([details](#exporting-a-transcript))                                                                                               |
| `/branch [turn]`           | **Fork this session** and carry on in the copy. No argument → pick the turn to branch after; `/branch 3` branches directly. The original session is **never modified** — it stays resumable with `--resume` ([details](#branching-a-session))                                                                                                                                                                                                                                                                                                                           |
| `/rename [title]`          | **Name this session** so it's recognizable in the `--resume` picker instead of being labelled by its first prompt. No argument shows the current name; `/rename clear` removes it ([details](#naming-a-session))                                                                                                                                                                                                                                                                                                                                                        |
| `/doctor`                  | **Check this install for problems** — config, provider credentials, required binaries, MCP servers, enabled features and their dependencies, and the files that grow. Local and read-only ([details](#checking-your-install))                                                                                                                                                                                                                                                                                                                                           |
| `/hooks`                   | List the **tool hooks** this session runs — your own commands fired before or after each tool call, from `~/.mnemoai/hooks/hooks.json`. See [Tool hooks](hooks.md)                                                                                                                                                                                                                                                                                                                                                                                                      |
| `/compact [focus]`         | Summarize older turns to shrink context (optional focus instructions)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `/config`                  | Re-run the interactive configurator — chat, vision (with a "same as chat?" shortcut + provider choice), and embeddings models, then feature toggles. `Ctrl+B` steps back within a model section. Overwrites `config.yaml` and restarts in place                                                                                                                                                                                                                                                                                                                         |
| `/model`                   | Set up one model — chat (LLM), vision, or embeddings. Vision offers a "use the same model as Chat?" shortcut. Inference params reset to model defaults on a change (re-tune with `/params`); restarts in place                                                                                                                                                                                                                                                                                                                                                          |
| `/params`                  | Tune a model's inference parameters (temperature, top_p, top_k, penalties, reasoning, stop, stream, …) — only the params the chosen provider supports are offered. Applied **without restarting**, so the current conversation continues                                                                                                                                                                                                                                                                                                                                |
| `/features`                | Enable/disable app features — a checklist of RAG, episodic memory, playbook, web search/crawl, routing, orchestration, memory, skills. Turning one on prompts for anything it needs (Brave API key, embeddings model); restarts in place                                                                                                                                                                                                                                                                                                                                |
| `/mcp`                     | List the configured MCP servers (built-in + any from `mcp.json`), their connection status, and tool counts                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `/skills`                  | List installed skills (name + description); `/skills <name>` previews a skill's full instructions. See [Agent Skills](agent-skills.md) below                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `/memory`                  | View the curated persistent memory (`MEMORY.md`); `/memory clear` wipes it (with a y/N confirm)                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `/plan`                    | Toggle **plan mode** — an enforced read-only mode. While ON, the agent investigates and, when ready, presents a plan you **approve (y), edit in `$EDITOR` (e), or keep refining (n)**; approving turns plan mode off and the agent executes the plan in the same turn (saved to `plans/plan_<ts>.md`). Read-only shell commands (`ls`, `cat`, `grep`, `git status/log/diff`, …) still run; file edits, mutating shell, git writes, and background tasks are **hard-blocked** while ON. See [plan mode](safety.md#investigate-before-changing-anything) for full details |

Beyond these, any `*.md` file you drop in `~/.mnemoai/commands/` becomes a command
you can type — see [Your own slash commands](#your-own-slash-commands).

### Your own slash commands

A prompt you retype often can become a command. Every `*.md` file in
`~/.mnemoai/commands/` is one, and **the file name is the command**:

```text
~/.mnemoai/commands/standup.md   →   /standup
```

The body of the file is the prompt that gets sent. Whatever you type after the
name is substituted in:

| Placeholder  | Becomes                                   |
| ------------ | ----------------------------------------- |
| `$ARGUMENTS` | everything after the command name         |
| `$1` … `$9`  | the 1st … 9th word after the command name |

If the body uses no placeholder at all, your arguments are **appended** — so a
one-line instruction plus a target works with no markup at all.

Two optional frontmatter keys document the command in the `/` menu and in `/help`:

```markdown
---
description: Review a diff for correctness and tests
argument_hint: <path or branch>
---

Review $ARGUMENTS. Read the changed files first, then judge whether the tests
cover the new behavior. Don't change anything.
```

A bundled `explain.md` is installed as a working example, next to a `_README.md`
that repeats this page's rules.

Worth knowing:

- **A built-in always wins.** A file named after one (`save.md`, `plan.md`, …) is
  skipped and `/doctor` says so, since the command would never fire.
- **Files starting with `_` or `.` are ignored**, so notes and drafts can live in
  the same directory.
- **Edits apply to the next line you type** — no restart. New files show up in the
  `/` menu immediately.
- **This directory only.** Commands are never read from a project you clone: what a
  name you type expands to is yours to decide.
- **The model never learns a command was involved** — the expansion _is_ your
  prompt, so the turn behaves like any other (a dim `⌘ /name · file.md` line
  records which file ran). For instructions the _assistant_ should reach for on its
  own, write a [skill](agent-skills.md) instead.

### Attaching a file with `@`

Type `@` anywhere in your prompt and paths complete as you type:

```text
> does @src/mnemoai/client/doctor.py check for ripgrep?
```

When you submit, **every `@path` in the line is read and sent with the question**.
That's the difference from typing a path in prose: the file is already there, so
the answer doesn't start with a round of searching for it — and it doesn't depend
on the assistant deciding the file was worth opening.

Completion works two ways:

- **A bare name searches the project by file name.** `@chat_int` finds
  `src/mnemoai/client/ui/chat_interface.py` — you rarely remember the directory.
  Files ignored by git are skipped.
- **A path completes directory by directory.** Anything with a `/`, or starting at
  `~`, completes one segment at a time, so a file **outside** the project is
  reachable too. Accepting a directory keeps its trailing `/`, so the next
  keystroke continues inside it.

A mentioned **directory** contributes its listing (names only, not the contents of
everything in it) — useful for "what's in `@src/mnemoai/server/tools`".

Each mention is confirmed with a dim line:

```text
@src/mnemoai/client/doctor.py · 486 lines
@notes.mdd · no such file
```

The second line is the reason this is worth printing: attaching nothing looks
exactly like attaching the right file, and the question still goes through — so
without it you'd get a confident answer about a file that was never sent.

Worth knowing:

- **Limits.** Up to 10 files per message, 60 000 characters in total, and 20 000
  characters per file. A file that gets cut says so in the prompt, so the assistant
  knows to read the rest itself rather than assume the tail was empty. Raise or
  remove the per-file cap with `MENTIONS.MAX_FILE_CHARS` in `config.yaml` (`0`
  disables it). Keep them in mind: unlike a tool result, what you attach yourself
  stays in the conversation at full size until it's summarized.
- **It's the same `@` as steering files.** `@path` references inside
  [`STEERING.md`](memory.md#steering-steeringmd) work the same way, and a mentioned file's own
  `@`-references are followed too.
- **Anything that isn't a readable path is left alone**, so `@staticmethod`,
  `@someone` and an email address keep meaning what they say. A path-looking
  mention that doesn't exist gets the gray `no such file` line above.
- **Binary files aren't inlined** (they're reported as `not a text file`) — ask
  about an image and the vision tool handles it instead.
- **Attaching isn't reading.** Before editing a file, the assistant still opens it
  with its own read tool; a mention doesn't stand in for that.

### The prompt

On an interactive terminal the input `>` stays **pinned at the bottom** of the
screen while the assistant works; the answer, its reasoning, and tool activity
stream **above** it (native scrollback and copy/paste are preserved). You can
keep typing during a turn — see the shortcuts below.

### The status footer

Under the input sits a dim one-line footer with the three facts that apply to
whatever you type next:

```
claude-opus-5 · bedrock          ~/dev/project          ▓░░░░░░░ 90.1k · 9%
```

the **model** (and provider) the turn will go to, the **directory** the session is
running in — which is what the file and shell tools act on — and the **context
meter**: how large the next prompt is and how much of the model's window that
fills. The meter turns **amber past 70%** and **red past 90%**, which is the point
at which [compaction](#commands) is close (it starts at 80% by default), so a long
session tells you it's getting long before a summary interrupts it.

The count is the provider's own number for the last turn. Before the first turn —
and right after a `--resume`, where no turn has run yet — it's a local estimate and
shown with a `~`; estimates run high, so the first real turn usually moves it down.
A narrow terminal drops the path, then the provider, keeping the meter.

Off an interactive terminal (a pipe, CI) there's no footer, so the context size is
printed after each turn as `[Context: N tokens]` instead.

### When the terminal wants you back

A long turn is exactly the turn you stop watching. Three moments therefore ring
the terminal:

- a **turn finishes** after running longer than `NOTIFY.AFTER_SECONDS` (30s by
  default),
- a **confirmation prompt or a question is waiting** on you — the work is stopped
  until it's answered, so this one is sent however short the turn has been,
- a **background sub-agent's report arrives** while you're idle.

Each sends two things: the **terminal bell**, which every terminal has and which
tmux and screen turn into a window-activity flag, and an **OSC 9 desktop
notification** (`mnemoai · done in 4m12s · project`), which iTerm2, WezTerm,
kitty, Windows Terminal and others raise as a real system notification —
terminals that don't understand it ignore it. Inside tmux or screen the sequence
is wrapped so it reaches the outer terminal instead of being swallowed.

Nothing is sent when output isn't going to a terminal (a pipe, CI), nor for a
turn **you** cancelled, and two notifications closer together than 10 seconds
collapse into one — a task confirming eight writes is one interruption, not eight.

```yaml
NOTIFY:
  AFTER_SECONDS: 30 # a turn must run this long for its end to notify (0 = never)
  BELL: true # the terminal bell
  DESKTOP: true # the OSC 9 desktop notification
```

Set both `BELL` and `DESKTOP` to `false` to stay silent entirely; `AFTER_SECONDS: 0`
silences only the turn-end half, keeping the ones that are waiting on you.

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
- `@` then part of a path: completes files and directories anywhere in the line —
  see [attaching a file with `@`](#attaching-a-file-with)

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

**A compacted session comes back compacted.** If the conversation was
[compacted](#commands) — by you, or automatically once it outgrew the window — the
transcript records the summary alongside the turns that stayed verbatim, so
resuming restores the state the session **ended** in: the same summary, the same
recent turns, and roughly the same context size the footer last showed you. The
full text is still in the file and still replays into the terminal, so you can
scroll back and read anything the summary condensed — it just isn't re-sent to the
model, which is what a compaction was for.

The same holds when the context was reclaimed **without** a summary. Before
summarizing anything, mnemoai first trims the bodies of older tool results — a
cheap pass that costs no model call and often frees enough on its own. A resume
brings back those trimmed results too, so the context you come back to matches the
one you left in that case as well.

A launch you never typed into records nothing resumable, so its file is removed
when you exit — only sessions with at least one exchange are offered, which also
means a resume you didn't ask anything in won't appear as a duplicate of the
session it restored. If this directory has no sessions yet, `--resume` says so
and starts a normal session.

### Naming a session

The picker labels every session with its opening prompt, which stops being enough
once a project has a few of them — especially after a `--resume` or a `/branch`,
where several rows legitimately share the same first question. `/rename` gives the
current session a name of your own:

```
/rename hooks phase 1        # name it
/rename                      # show the current name
/rename clear                # back to the first-prompt label
```

```
   4m ago    6 turns  hooks phase 1
  2h ago    2 turns  why does the loader crash on startup?
```

The name is appended to the session file like everything else, so renaming twice
just means the last name wins, and it **survives a resume** — same conversation,
new file, same name. A [`/branch`](#branching-a-session) fork deliberately does
**not** inherit it: the fork and its parent already share an opening prompt, and
sharing the name too would put you back where you started. Naming a session you
never type into doesn't keep its empty file alive.

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

If the conversation was [compacted](#commands), branching after that point carries
the summary with it, exactly as [resuming](#resuming-a-session) does — while
branching to a turn _before_ it forks the raw history instead, which is a way to get
the uncompacted detail of those turns back into a conversation.

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

### Taking back your last prompt

Sometimes the prompt itself was the mistake — the wrong file, the wrong framing, a
question that sent the whole turn somewhere you didn't want it to go. `/rewind`
drops that prompt **and everything the turn produced** from the conversation, so
the context is what it was the moment before you pressed Enter:

```
> /rewind

⟲ withdrew your last prompt
  "why is the FSDP config parser slow"
  14 messages dropped from the conversation and transcript.

  Files on disk are untouched — a rewind moves the conversation only.
```

The echoed prompt is there so you can see _which_ turn went, and the count covers
the whole turn: the answer, the tool calls, their results. Run it twice and you walk
back two turns. It looks for the last thing **you** typed, not the last message with
your name on it, so a turn made of tool results or an [auto-delivered sub-agent
report](orchestration.md#sub-agents-spawn_agent) can't be mistaken for a prompt.

Its scope is deliberately narrow, and it is the one thing to be clear about:

!!! warning "A rewind moves the conversation, not your files"

    Files the withdrawn turn wrote are **still on disk** — exactly as with
    [`/branch`](#branching-a-session). Undoing an edit is `git` territory (or
    [`/diff`](#seeing-what-changed), to see what there is to undo); a command that
    rolled back half a turn would be worse than one that tells you which half it
    does.

Three more things worth knowing:

- **It refuses after a compaction.** If the conversation was
  [compacted](#commands) during or right after that turn, the summary standing in
  for it can't be un-summarized, so `/rewind` says so and changes nothing rather
  than half-applying. Older compactions don't get in the way.
- **The transcript records the withdrawal; it doesn't erase it.** Session
  transcripts are append-only, so the turn is marked withdrawn: it stops being part
  of the conversation, a [`/branch`](#branching-a-session) can no longer fork at it,
  and `--resume` won't bring it back — but the text stays in the file. A rewind is
  an undo, not a redaction.
- **What the session _learned_ from the turn stays**, and the notice names which
  stores that means. [Episodic memory](memory.md#episodic-memory) keeps entries by
  similarity and the playbook folds a repeat strategy into an existing one, so
  there's nothing precise left to delete; `/memory` is editable if a
  [curated](memory.md#persistent-memory-memorymd) fact needs to go.

### Seeing where the context goes

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
[compaction](#commands) can never reclaim them, so a 14k-token repo `CLAUDE.md` is a
permanent tax on every turn — trimming the file is the only thing that helps. The
conversation row is the part `/compact` shrinks.

The **total** is exact (the provider's own count for the last turn, the same number
the [status footer](#the-status-footer) shows); the **split** is estimated and scaled onto it,
so the percentages are meaningful even though the per-part counts are approximate.
Before the first turn of a session there is nothing to scale to and the report says
so.

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

**The cache line is prompt caching at work.** On the providers that support it the
stable start of every request — system prompt, tool definitions, prior turns — is
cached and re-read instead of re-charged, so `cache: … read` grows as a turn makes
tool calls. See
[`PROMPT_CACHE`](../configuration.md#prompt_cache-reuse-the-prompt-prefix-instead-of-re-paying-for-it)
for what it applies to and how to turn it off.

**`/usage` and the footer's context meter measure different things.** The footer is
how big the prompt is _right now_ — what the next turn re-sends, and what
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

### What this session touched

`/files` answers the question a long session makes hard: which files did it actually
open, and which did it change?

```
Files this session

  Changed (2)
    ✎ tests/unit/test_diff_report.py      1 edit
    ✎ src/mnemoai/client/diff_report.py   2 edits · 1 read

  Attached with @ (1)
    @ docs/guides/usage.md                attached

  Read (2)
    · src/mnemoai/client/ui/turn_view.py  2 reads
    · src/mnemoai/client/agent/agent.py   1 read
```

Most recently touched first, **changed files first** — that's the group with
consequences on disk. Worth knowing:

- **It counts the work you don't see.** A sub-agent's edits and a parallel wave's
  reads are in the list, because the record is taken where every tool call passes
  through rather than from what got printed.
- **Two spellings are one row.** `./src/x.py`, `src/x.py` and `~/proj/src/x.py`
  resolve to the same file, so the counts are the real counts.
- **It survives compaction.** A file read an hour ago may have had its content
  summarized out of the context since; the touch still counts, and the report says
  so rather than implying the content is still in the window.
- **A tree-wide search isn't a touch.** `grep_search` over a directory produces no
  file you can go and look at, so only the tools that name a single file are recorded.
- `/clear` resets it, since a cleared conversation is a new session.

### Seeing what changed

`/diff` is `git status` with the one column git can't give you — **which of these
files this conversation wrote**:

```
Changes in ~/development/mnemoai (dev)

  ✎ src/mnemoai/client/diff_report.py        +402    new
  ✎ src/mnemoai/client/ui/chat_interface.py  +24 -1
  ✎ CHANGELOG.md                             +31
    docs/guides/usage.md                     +6 -2
    assets/logo.png                          binary

  5 files · +463 -3 · ✎ 3 written this session
  /diff <path> shows one file's diff.
```

Staged, unstaged and untracked files in one list. A `✎` marks the ones the session
wrote (they sort first); **everything unmarked was already dirty when you started**,
which is the distinction that decides what's safe to commit. `/diff <path>` shows
that one file's unified diff, colored, with an untracked file rendered as the
all-additions diff it effectively is.

**It is read-only by construction.** The only git it runs is `rev-parse`, `diff` and
`ls-files` — there is no code path that could stage, stash or check anything out. It
is also bounded (a long list and a long diff both collapse), and when something is
cut it prints the exact `git` command that shows the rest.

### Copying an answer

Selecting a streamed answer with the mouse takes the terminal's line wrapping with
it, which is why a copied code block so often has to be reflowed by hand. `/copy`
takes the message as the model wrote it:

```
/copy          # the last answer
/copy code     # just its last fenced code block
/copy 2        # the answer before last
```

Two transports, tried in the order that's right for where you are. A **local helper**
(`pbcopy`, `wl-copy`, `xclip`, `xsel`, `clip.exe`) when one is installed, and
**OSC 52** — the escape sequence that asks the terminal itself to set the clipboard —
otherwise, so a machine with no helper still works. **Over SSH the terminal goes
first**, because a helper running on the far end would set the clipboard of a machine
nobody is sitting at; that's the case OSC 52 exists for. Inside tmux or screen the
sequence is wrapped in a passthrough so it reaches the outer terminal.

The notice says what was copied, how big it was and which transport carried it
(`Copied the last sh block (1 line, 9 chars) via pbcopy.`) — a clipboard write is
otherwise invisible until you paste. If neither transport is available it says so and
points at `/export`, which writes to a file instead.

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

### Checking your install

`/doctor` inspects this install and reports what's wrong with it — the first thing
to run when something behaves strangely, and the thing to paste into an issue:

```
Doctor — 1 problem, 1 warning

  Install
    · mnemoai  1.12.0
    · python  3.12.14 (/opt/homebrew/opt/python@3.12/bin/python3.12)
    · platform  Darwin 25.6.0
    ✓ app home  ~/.mnemoai

  Configuration
    ✓ config.yaml  ~/.mnemoai/config/config.yaml
    ✓ prompts.yaml  ~/.mnemoai/config/prompts.yaml

  Provider
    ✓ model  bedrock: us.anthropic.claude-sonnet-4-5-20250929-v1:0
    ✓ aws credentials  resolved, region us-east-1
    ✓ prompt cache  on, 1h TTL

  Tools
    ✗ rg  not on PATH — grep_search will not work
      → Install rg.
    ✓ git  /opt/homebrew/bin/git
    ✓ bash  /opt/homebrew/bin/bash
    ✓ MCP tools  41 registered

  Features
    ✓ chromadb (chromadb)  installed

  State
    ✓ sessions here  10 recorded, kept 30 days
    · logs  ~/.mnemoai/logs/mnemoai.log — 34 KB, kept 7 days
    ! MEMORY.md  2192 / 2200 chars (~/.mnemoai/bpistone/MEMORY.md)
      → Nearly full; the next entry may push an older one out.
    · steering  ~/dev/project/CLAUDE.md — 14240 chars, injected every turn
```

Each line is `✓` fine, `!` worth knowing, `✗` broken, or `·` informational, and a
problem comes with the command that fixes it. What it covers:

- **Configuration** — which `config.yaml` and `prompts.yaml` are **actually loaded**.
  ("I edited `config.yaml` and nothing changed" is nearly always a different file
  being live — an `MNEMOAI_CONFIG` override or a checkout fallback — and this is the
  only place that tells you.)
- **Provider** — the model and protocol in use, whether its credentials resolve,
  whether the endpoint answers (a 2-second local probe for Ollama and
  OpenAI-compatible servers), and whether prompt caching applies to it. Keys are
  never printed, only whether one was found.
- **Tools** — `rg` (required by `grep_search`, with no fallback), `git`, `bash`, how
  many MCP tools loaded, and whether every server declared in `mcp.json` actually
  connected. A server that **failed** is a warning rather than being reported as "not
  configured", which is the distinction that sends you looking in the right place.
- **Features** — for each feature you switched on, whether what it needs is present:
  the vector store package for RAG, a `BRAVE_API_KEY` for web search.
- **State** — the files that grow and then bite: `MEMORY.md` near its cap (it starts
  dropping older entries), steering files and the size each one injects **every**
  turn, how many sessions this directory has recorded, and where the app log is
  (the terminal shows one line per error; the traceback behind it is only there).
  Your own [slash commands](#your-own-slash-commands) are listed here too, along
  with any file that was **skipped** and why — otherwise a rejected command is
  indistinguishable from a feature that doesn't work.

It's local, cheap and read-only: no model call, nothing written, no network beyond
that one local port probe.
