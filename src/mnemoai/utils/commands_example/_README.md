# Your own slash commands

Every `*.md` file in this directory is a command you can type. **The file name is
the command**: `deploy.md` → `/deploy`. The body is the prompt that gets sent when
you type it, so a command is a shortcut for something you'd otherwise retype.

    ~/.mnemoai/commands/standup.md   →   /standup

## Arguments

Whatever you type after the command name is available in the body:

| Placeholder  | Becomes                                   |
| ------------ | ----------------------------------------- |
| `$ARGUMENTS` | everything after the command name         |
| `$1` … `$9`  | the 1st … 9th word after the command name |

If the body uses no placeholder at all, your arguments are appended to the end —
so a one-line instruction plus a target works without any markup.

## Frontmatter (optional)

```markdown
---
description: Summarize what changed since yesterday
argument_hint: <since when>
---

Summarize the git commits since $ARGUMENTS …
```

Both keys are optional. `description` is what the `/` menu and `/help` show;
without it the first line of the body is used. `argument_hint` documents what to
type after the name. A plain markdown file with no frontmatter is a valid command.

## Rules worth knowing

- A file whose name is already a built-in (`/save`, `/plan`, …) is **skipped** —
  the built-in wins, so the file would never fire.
- Files starting with `_` or `.` are ignored: notes and drafts (like this file) can
  live here without becoming commands.
- Edits apply on the next line you type — no restart.
- Commands are read from this directory only, never from a project you clone.
- Expanding a command is a normal turn: the model sees your prompt, not the
  command name. For instructions the _model_ should reach for on its own, write a
  skill (`~/.mnemoai/skills/`) instead.
