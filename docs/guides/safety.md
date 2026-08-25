# Control what Mnemo AI can do

Decide what runs without asking, what asks first, and what is refused outright.

This page covers:

- [Approve actions as they happen](#approve-actions-as-they-happen) — the `Proceed?` prompt and its toggles
- [Investigate before changing anything](#investigate-before-changing-anything) — plan mode
- [Understand the safety floor](#understand-the-safety-floor) — what is refused even with confirmations off
- [Keep git operations recoverable](#keep-git-operations-recoverable) — git-specific protections
- [Why a write was refused](#why-a-write-was-refused) — the read-before-write gate

## Approve actions as they happen

Destructive tools stop and ask before they run — shell commands (`execute_bash`)
and file modifications (`fs_write`, `file_edit`):

```
▶ Run shell command?
  rm -rf build/
  Proceed? (y/N/a=allow all this session):

▶ Write to file?
  create ~/script.py
  Proceed? (y/N/a=allow all this session):
```

Only an explicit `y`/`yes` proceeds. `a` trusts that whole category for the rest
of the session, so a multi-step task doesn't re-prompt. Anything else — including
Enter — declines, and the model is told you declined rather than being left to
retry.

This is a **hard gate enforced client-side**: the prompt fires regardless of what
the model does, because the client owns the terminal. The MCP server is a piped
subprocess and cannot prompt you.

| Toggle                        | Default | Gates                           |
| ----------------------------- | ------- | ------------------------------- |
| `REQUIRE_BASH_CONFIRMATION`   | `true`  | `execute_bash`                  |
| `REQUIRE_WRITE_CONFIRMATION`  | `true`  | `fs_write`, `file_edit`         |
| `REQUIRE_MEMORY_CONFIRMATION` | `false` | `memory` writes to `MEMORY.md`  |
| `REQUIRE_GIT_CONFIRMATION`    | `true`  | Overriding a git safety refusal |

Set any to `false` for a trusted or automated setup. Non-interactive runs (no TTY
— tests, pipes, CI) auto-proceed so they can't hang waiting for a keypress.

**Tired of confirming the same harmless command?** A [tool hook](hooks.md) can
answer the prompt for you — a `PreToolUse` hook that prints
`{"decision": "allow"}` for `git status` waives that one prompt without turning
the whole `execute_bash` gate off. A hook can also **deny** a call outright, which
is enforced everywhere the prompt is. What it can't do is reach past plan mode or
the safety floor below.

!!! note "Sub-agents can't approve on your behalf"

    A background [sub-agent](orchestration.md) has no terminal, so it cannot
    prompt. An untrusted destructive tool there **auto-denies** — the safe
    direction. It proceeds only if you already trusted that category with `a`.

## Investigate before changing anything

Toggle plan mode with `/plan` before asking for a complex change. The assistant
investigates and drafts a plan without touching anything, then you approve and it
executes.

Reach for it when the change spans multiple files, needs an architectural
decision, or the requirements aren't settled yet.

While plan mode is **on**, only read-only tools work: file reads, `glob_search`
and `grep_search`, read-only shell (`ls`, `cat`, `git status`, `git log`,
`git diff`, …), web search, and document readers. Any attempt to edit files, run
mutating shell, perform git writes, or start a background task is hard-blocked
client-side — regardless of what the model tries.

When the plan is ready the assistant calls `exit_plan_mode`, which shows it and
asks how to proceed:

| Key | Effect                                                                           |
| --- | -------------------------------------------------------------------------------- |
| `y` | Approve and run — plan mode turns off and the plan executes **in the same turn** |
| `e` | Open the plan in `$EDITOR` to tweak it, then re-review                           |
| `n` | Keep planning — stays read-only and keeps refining                               |

The approved plan is saved to `~/.mnemoai/plans/plan_<timestamp>.md`, so it
survives compaction and can be re-read later.

**Pre-approved commands.** A plan can declare the shell commands it will run
during execution — tests, builds, installs. Approving the plan pre-approves them,
so they run without a per-command `Proceed?` prompt while the plan executes;
anything not on the list still prompts. These pre-approvals are scoped to that
plan and are cleared when you `/clear` or re-enter plan mode.

Plan mode is enforced at the same client-side chokepoint as the confirmation
gate, so it holds across the normal loop and the orchestrator workers alike. A
misbehaving local model cannot mutate anything while it is on.

## Understand the safety floor

The confirmation prompt is a convenience layer you can switch off. Below it sits
a floor that is **always on** and lives inside the MCP server, so it holds even
with confirmations disabled, in non-interactive runs, or if the server is driven
by a different client.

| Layer                        | What it does                                                         | Can be disabled?     |
| ---------------------------- | -------------------------------------------------------------------- | -------------------- |
| Your [tool hooks](hooks.md)  | Whatever you scripted — can deny a call, or waive its prompt         | Yes (it's your file) |
| Client confirmation prompt   | Asks `Proceed? (y/N/a)` before each shell, file, or memory operation | Yes (`REQUIRE_*`)    |
| Plan mode, while on          | Hard-blocks every mutating and executing tool                        | Yes (`/plan`)        |
| **Server-side safety floor** | Refuses system-destroying commands and system-directory writes       | **No**               |

Shell commands (`execute_bash`, `start_background_task`) are refused when they
would:

- recursively force-delete a root or home target — `rm -rf /`, `rm -rf ~`,
  `rm -rf /*`, including `rm -rf / --no-preserve-root` and variants that hide the
  target behind trailing flags or a redirection
- create a filesystem or overwrite a raw device — `mkfs…`, `dd of=/dev/…`,
  `shred`/`wipefs` on a device
- change the machine power state — `shutdown`, `reboot`, `halt`, `poweroff`,
  `init 0/6`
- fork-bomb the machine — `:(){ :|:& };:`
- **write into a protected system directory** — `echo x > /etc/hosts`,
  `sh -c '… > /etc/…'`, `tee /etc/…`, `cp … /usr/local/bin/…`, `sed -i` on a
  system file. Shell writes are classified with the same path policy as the file
  tools, so the two cannot disagree.

File writes (`fs_write`, `file_edit`) are refused when the target is inside a
critical system directory — `/`, `/etc`, `/bin`, `/usr`, `/boot`, `/dev`,
`/System`, `/Library`, … — with `..` and `~` resolved first, so a symlink can't
step around the check. Your home, project trees, temp directories, and the app
home stay fully writable. `/dev/null` and the other standard sinks stay writable
too, so `2>/dev/null` keeps working.

The floor is deliberately **narrow**. Scoped, everyday-destructive commands like
`rm -rf build/` or `git reset --hard` are _not_ blocked here — they stay gated by
the confirmation prompt. The goal is to make irreversible system damage
impossible, not to second-guess normal edits.

**Web requests** go through a URL policy: only `http`/`https`, and every resolved
address must be public. Loopback, link-local, private ranges, and cloud metadata
endpoints such as `169.254.169.254` are refused, so a page can't talk the crawler
into fetching your instance credentials.

## Keep git operations recoverable

Use `git_safe`, `git_status_safe`, and `git_commit_safe` rather than driving git
through `execute_bash` — they add checks that plain git has no reason to.

| Operation                                                  | Protection                                              |
| ---------------------------------------------------------- | ------------------------------------------------------- |
| Force push to `main`/`master`                              | **Blocked** — cannot be overridden                      |
| Option injection (`-c`, `--exec-path`, `--upload-pack`, …) | **Blocked** — these make git run arbitrary commands     |
| `git reset --hard`                                         | Asks first                                              |
| `git push --force`                                         | Asks first (prefer `--force-with-lease`)                |
| `git commit --amend`                                       | Checks whether the commit was already pushed, then asks |
| `git clean -f` / `-fd`                                     | Asks first                                              |
| Force-delete branch (`-D`)                                 | Asks first                                              |
| Skip hooks (`--no-verify`)                                 | Asks first                                              |

Checks run against the command as **git will receive it**, not the raw string, so
quoting can't smuggle a flag past them: `push origin main --for"ce"` is still
recognized as a force push. Conversely, a commit message that merely _mentions_ a
dangerous operation is not treated as one.

When an operation needs confirmation, **you** are asked — not the model:

```
▶ Proceed with flagged operation?
  git reset --hard HEAD~1 — Hard reset will discard all uncommitted changes permanently.
  Proceed? (y/N/a=allow all this session):
```

Decline and the model is told you declined and instructed not to retry with an
override. This is gated by `REQUIRE_GIT_CONFIRMATION` (default `true`).

## Why a write was refused

A write can also be refused for a reason that has nothing to do with safety
policy: the assistant must **read a file before writing it**, and its read must
still be current.

| Message contains                         | Cause                                | Fix                                             |
| ---------------------------------------- | ------------------------------------ | ----------------------------------------------- |
| `read it first`                          | The file was never read this session | Read it, then retry                             |
| `changed on disk since you last read it` | The file changed after it was read   | Re-read so the edit is based on current content |

This exists because a blind edit to a file that changed underneath silently
discards someone else's work — including your own edits in another editor.

## See also

- [Configuration](../configuration.md): the `REQUIRE_*` toggles and every other key
- [Tool hooks](hooks.md): script your own allow/deny rules around tool calls
- [Tools reference](../reference/tools.md): every tool and its parameters
- [Sub-agents & orchestration](orchestration.md): how confirmations behave in a delegated run
- [Troubleshooting](../development/troubleshooting.md): diagnosing a refused write
