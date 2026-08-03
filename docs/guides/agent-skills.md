# Agent Skills

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
argument_hint: the staged changes to describe
---

# Conventional Commit Message

Step-by-step instructions the model follows once this skill is loaded...
```

- **`name`** and **`description`** are required. The **directory name** is the id
  the model uses to load the skill (`commit-message` above).
- Write the **description "pushy"** — start with _"Use when the user…"_ and include
  the phrases a user would actually say. The model decides whether to trigger a
  skill from this description, and tends to under-trigger if it's vague.
- **`argument_hint`** (optional) is a short phrase naming what the skill expects
  the model to gather before invoking it; it shows in the always-on skill listing
  as `(expects: …)`. Omit it for skills that need no specific input. Other
  frontmatter keys are tolerated but ignored. The whole listing is size-bounded, so
  with many skills installed the overflow collapses into a `+N more` line (all
  skills remain loadable — the listing is discovery only).
- Skills are seeded on first run: a `commit-message` example to copy, a
  **`skill-creator`** skill (just ask the assistant to "create a skill for X" and
  it writes a well-formed `SKILL.md` for you), and a **`steering-creator`** skill
  (ask it to "create a STEERING.md", "improve my CLAUDE.md", or "document how to
  work in this project" and it investigates the repo and writes a well-formed
  instructions file following best practices).

### Using and managing skills

- The assistant triggers a matching skill automatically — no special syntax needed.
- `/skills` lists installed skills; if a skill is malformed it's shown under
  **Skipped** with the reason (e.g. missing `description`) so you can fix it.
- `/skills <name>` previews a skill's full body.
- Toggle the whole feature with `ENABLE_SKILLS` (default `true`) in `config.yaml`.
