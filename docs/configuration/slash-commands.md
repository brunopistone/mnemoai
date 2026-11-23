# Custom slash commands { #your-own-slash-commands }

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
  own, write a [skill](../guides/agent-skills.md) instead.
