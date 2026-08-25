# Tool hooks

Run your own shell command every time the assistant uses a tool — before the
call, after it, or after it fails.

Hooks exist for the things a prompt cannot reliably enforce. "Always format the
files you write" is an instruction the model follows most of the time; a
`PostToolUse` hook that runs your formatter happens every time. The same goes the
other way: "never touch `secrets/`" becomes a rule instead of a hope.

Typical uses:

- **Format or lint** every file the assistant writes.
- **Refuse** writes under a path you never want modified.
- **Auto-approve** the commands you're tired of confirming (`git status`, `ls`).
- **Hand the model a hint** when a tool fails, so it recovers instead of retrying
  the same thing.

## The file

Hooks live in **one** file — `~/.mnemoai/hooks/hooks.json`. A commented
`hooks.json.example` is seeded next to it on install; copy it and edit:

```bash
cp ~/.mnemoai/hooks/hooks.json.example ~/.mnemoai/hooks/hooks.json
```

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "fs_write",
        "hooks": [
          {
            "type": "command",
            "command": "my-formatter-wrapper.sh",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

| Field     | Meaning                                                                              |
| --------- | ------------------------------------------------------------------------------------ |
| event     | `PreToolUse`, `PostToolUse`, or `PostToolUseFailure`                                 |
| `matcher` | Glob over the **tool name** (`fs_write`, `git_*`, `*` for every tool). Default `*`   |
| `type`    | `"command"` — the only type in this release                                          |
| `command` | The shell command to run (real bash, not `/bin/sh`)                                  |
| `timeout` | Seconds before the hook is abandoned (default 30, max 600). Overrunning never blocks |

Every hook whose matcher matches runs, in the order you wrote them. Keys starting
with `//` are ignored, so you can comment the file.

Run **`/hooks`** in the app to see what's loaded:

```
Tool hooks
  config: ~/.mnemoai/hooks/hooks.json

  PreToolUse
    fs_write             python -c "import json,sys; p=json.load(sys.stdi…  (10s)

  PostToolUse
    fs_write             ruff format "$(…)"  (30s)

  2 hook(s), read at startup — restart to apply an edit.
```

A hook that doesn't parse is reported there (and at startup) instead of silently
disappearing — one malformed entry never costs you the rest of the file.

## What a hook receives

The call is written to the hook's **stdin as JSON**:

```json
{
  "session_id": "session_20260825_143210_8123_a4f1",
  "cwd": "/Users/you/dev/project",
  "hook_event_name": "PreToolUse",
  "tool_name": "fs_write",
  "tool_input": { "path": "/Users/you/dev/project/app.py", "content": "…" },
  "tool_response": "…"
}
```

`tool_response` is present only on the two post-call events. Long string values in
`tool_input` are capped (a written file body can be megabytes); anything clipped is
listed under a `_truncated` key so a hook can tell a short value from a cut one.

`MNEMOAI_HOOK_EVENT` and `MNEMOAI_TOOL_NAME` are also exported, so a one-line hook
needn't parse JSON at all.

## What a hook can say back

| The hook…                       | Effect                                                                                   |
| ------------------------------- | ---------------------------------------------------------------------------------------- |
| exits `0`                       | The call proceeds. Plain stdout is shown to **you** and logged — never sent to the model |
| exits `2`                       | **Deny**: the call is blocked and stderr is given to the model as the reason             |
| exits anything else, or crashes | Non-blocking error: a notice, a log line, and the call proceeds                          |
| overruns its `timeout`          | Non-blocking: a notice, a log line, and the call proceeds                                |
| prints JSON on stdout           | Honored — see below                                                                      |

```json
{ "decision": "deny", "reason": "Writes under secrets/ are blocked locally." }
{ "decision": "allow" }
{ "additionalContext": "This repo pins ruff 0.6 — use that, not the system one." }
```

`additionalContext` is the **only** part of a hook's output that reaches the
model: it's appended to the tool result. Everything else a hook prints is for you.
That split is deliberate — a formatter that runs on every write shouldn't be
narrating itself into the context window.

When several hooks match one call, `additionalContext` is concatenated and the
first **deny** stops the rest. An `allow` never short-circuits, so a later deny
still wins: the safe answer must not depend on the order you happened to write
your hooks in.

## Where a hook sits, and what it cannot do

A hook is one gate among several, and it is not the outermost one:

```
server-side safety floor (catastrophic commands, system-path writes)
  → plan mode hard block
    → PreToolUse hook              ← here
      → the Proceed? confirmation prompt
        → the tool runs
```

So a hook's **`deny` is honored anywhere**, but its **`allow` reaches exactly one
gate**: it satisfies the confirmation prompt for that one call. It cannot unblock a
tool [plan mode](safety.md#investigate-before-changing-anything) has blocked, and it
cannot reach the [server-side safety floor](safety.md#understand-the-safety-floor),
which runs inside the MCP subprocess and still refuses a catastrophic command. A
config file is not a way to widen what the app is allowed to do — only to narrow
it, or to waive one prompt you'd have answered `y` to anyway.

!!! warning "Hooks are code, so the file is app-home-only and read at startup"

    Unlike [`STEERING.md`](memory.md#steering-steeringmd) — read-only
    text that can live in any project and is re-read every turn — a hooks file runs
    arbitrary commands. Two consequences:

    - **Nothing outside `~/.mnemoai/hooks/` is read.** A `hooks.json` arriving with
      a `git clone` would otherwise be remote code execution the first time you
      edited a file in that repo.
    - **Hooks are snapshotted at startup.** Editing the file mid-session doesn't
      change what's running — restart to apply. What you approved when the session
      began is what fires for the rest of it.

## Behavior worth knowing

- **A hook can never wedge the agent.** Every failure path — non-zero exit, crash,
  missing binary, timeout — is reported and skipped, and the tool proceeds. The only
  code that blocks is `2`.
- **A hook must never wait for input.** They run on tool-calling threads, including
  parallel orchestrator workers and background sub-agents, which have no terminal. A
  hook that prompts hangs on a keypress that can never arrive.
- **Hooks fire for delegated work too** — a sub-agent's writes are still writes.
  Keep them thread-safe.
- **Real bash.** Commands run under `bash`, not `/bin/sh`, so `[[ … ]]`, arrays and
  `source` behave as written rather than depending on the host's `/bin/sh`.
- **When a hook does something, you see it** — a dim notice in the scrollback says
  which hook blocked a call, approved one, or added context.

## Examples

Block writes under a directory you never want touched:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "fs_write",
        "hooks": [
          {
            "type": "command",
            "command": "python -c \"import json,sys; p=json.load(sys.stdin).get('tool_input',{}).get('path',''); sys.exit(0) if '/secrets/' not in p else (sys.stderr.write('Blocked by a local hook.'), sys.exit(2))\"",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

Stop confirming read-only git commands:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "execute_bash",
        "hooks": [
          {
            "type": "command",
            "command": "python -c \"import json,sys; c=json.load(sys.stdin).get('tool_input',{}).get('command','').strip(); print(json.dumps({'decision':'allow'}) if c.startswith(('git status','git log','git diff')) else '')\"",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

Give the model something to act on when a tool fails:

```json
{
  "hooks": {
    "PostToolUseFailure": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "echo '{\"additionalContext\": \"Re-read the file before retrying — it may have changed on disk.\"}'",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

## See also

- [Safety & permissions](safety.md): the confirmation prompt, plan mode, and the floor a hook sits below
- [Usage & commands](usage.md#commands): `/hooks`, and `/doctor` for checking an install
- [Tools reference](../reference/tools.md): the tool names to match on
