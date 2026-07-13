---
name: Steering Creator
description: Use when the user asks to create, write, author, set up, generate, or improve a STEERING.md file, or asks the assistant to "init" / "learn this project" / "document how to work here" / "remember these conventions". Guides writing a well-formed STEERING.md of always-on user instructions.
version: 1
---

# Steering Creator

Author (or improve) a `STEERING.md` — the user-authored file of **always-on
instructions** the assistant follows every turn: build/test commands, code
conventions, project layout, workflows, and "always do X" rules. It is the
user's counterpart to the agent-curated `MEMORY.md`; the assistant never edits
STEERING.md on its own, so writing it is always an explicit request like this.

## Where STEERING.md lives

Two optional levels, combined global-first then project (project wins):

- **Global:** `~/.mnemoai/STEERING.md` — applies in every session, everywhere.
  Use it for personal, cross-project preferences (tone, default language,
  machine-specific rules).
- **Project:** `./STEERING.md` at the repository root — checked into the repo and
  shared with the team. Use it for this project's conventions.

Pick the level from the user's intent: project conventions → project file;
personal always-on preferences → global file. When unsure, ask. Default to the
**project root** for anything project-specific.

It's applied verbatim on every turn (re-read from disk, never summarized), so it
must stay tight — aim for **under ~200 lines**. Everything in it costs context
each turn, so include only what genuinely changes the assistant's behavior.

## How to write a good STEERING.md

The goal is what a new collaborator would need to be productive here. If this is
an "init"-style request for an existing project, first **investigate** before
writing: read the README, look for build/test config (`package.json`,
`pyproject.toml`, `Makefile`, CI workflows), and skim the directory layout, so
the file reflects the real project, not guesses.

Write it as direct instructions to the assistant. Favor these principles:

- **Be specific, not vague.** "Use 4-space indentation and `ruff` for linting"
  beats "format code properly." Concrete rules are followable; platitudes aren't.
- **Structure for scanning.** Group with short headers and bullets (Commands,
  Code style, Conventions, Do / Don't). Dense prose is harder to follow.
- **Explain the *why* for non-obvious rules** — a rule the assistant understands
  is followed more reliably than a bare "ALWAYS/NEVER."
- **Only durable, always-true facts.** One-off task details belong in the
  conversation; multi-step procedures belong in a *skill*; learned facts the
  agent should track itself belong in MEMORY.md. Don't duplicate those here.
- **Keep it lean.** If a section grows large (full API docs, long style guides),
  summarize the rule here and point to the file that has the detail.

## Suggested structure

Adapt to the project; drop sections that don't apply.

```markdown
# Project: <name>

## Commands
- Build: <cmd>
- Test: <cmd>
- Lint/format: <cmd>

## Code style
- <language/version, indentation, naming, imports, typing…>

## Conventions
- <branch/commit rules, PR process, where things go>

## Do / Don't
- Do: <the few rules that matter most>
- Don't: <common mistakes to avoid here>
```

## Steps

1. **Confirm scope** — global vs. project, and (for a project) the repo root path.
2. **Investigate** (init-style requests) — read README/build config/layout so the
   content is accurate; incorporate any conventions already stated in this
   conversation.
3. **Draft** the file following the principles above, keeping it under ~200 lines.
4. **Write** it to the chosen path (`~/.mnemoai/STEERING.md` or `<repo-root>/STEERING.md`).
   If a STEERING.md already exists, read it first and merge rather than clobber.
5. **Tell the user** it applies immediately (STEERING.md is re-read every turn, no
   restart) and that they can view the resolved instructions any time, and edit
   the file directly to change them.

## Improving an existing STEERING.md

Read the current file, then refine to the user's feedback: tighten vague rules
into specific ones, remove stale or redundant guidance, split bulky detail into a
referenced file, and cut anything that isn't an always-on instruction. Keep it
general enough to serve many future sessions, not just the request in front of you.
