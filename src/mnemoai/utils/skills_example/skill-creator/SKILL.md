---
name: Skill Creator
description: Use when the user asks to create, write, author, add, or improve a mnemoai skill — e.g. "make a skill for X", "turn this workflow into a skill", "add a skill that…", "fix my skill". Guides authoring a well-formed SKILL.md under the skills directory.
version: 1
---

# Skill Creator

Author a new mnemoai **skill** (or improve an existing one). A skill is an
authored, reusable procedure the assistant loads on demand when a task matches —
ideal for multi-step work the user does repeatedly and wants done a specific way.

## What a skill is (mechanically)

Each skill is a directory under the skills root with a `SKILL.md` inside:

```
<skills-dir>/<skill-name>/
├── SKILL.md            # required: YAML frontmatter + markdown body
├── reference.md        # optional — detailed docs, read on demand
└── scripts/            # optional — executables, run on demand
```

Skills use **three-tier progressive disclosure**, so keep each tier lean:

1. **Metadata** — only `name` + `description` is always in context. This is what
   the model matches against to decide whether to load the skill.
2. **Body** — the full `SKILL.md` body enters context only when the skill is
   loaded (via the `use_skill` tool). Keep it focused; aim for under ~200 lines.
3. **Resources** — `reference.md` / `scripts/` are read or run only when the body
   tells the model to. Push bulky detail here, not into the body.

## Steps to create a skill

1. **Capture intent.** Confirm with the user: what should the skill let the
   assistant do, *when* should it trigger (the phrases a user would actually
   say), and what's the expected output? If a workflow already happened in this
   conversation, extract the steps, tools, and corrections from it.

2. **Choose a directory name** — lowercase kebab-case (`commit-message`,
   `release-checklist`). This is the canonical id the assistant loads it by.

3. **Write the frontmatter.** Required keys: `name` and `description`.

   ```
   ---
   name: Release Checklist
   description: Use when the user asks to cut a release, publish a version, or
     tag a build. Runs the project's release steps in order.
   ---
   ```

   - **The `description` is the entire trigger** — the model decides whether to
     load the skill from this alone. Write it in the **third person** and make it
     **pushy**: start with "Use when the user…" and include concrete phrases and
     synonyms they'd actually type. Models tend to *under*-trigger skills, so err
     toward broad, explicit triggers. Keep it under ~1024 characters and avoid
     `<`/`>` angle brackets.
   - Avoid putting "when to use" guidance in the body — it belongs in the
     description, the only part always in context.

4. **Write the body** as instructions for *another assistant instance* to follow:
   - Use **imperative/infinitive** form ("Run the tests", not "You should run").
   - Explain the **why** behind steps rather than piling on rigid "ALWAYS/NEVER"
     rules — a model that understands the intent follows it better.
   - Define output formats explicitly with a template when the shape matters.
   - Include a short worked example (input → output) when it helps.
   - If the procedure is long or has variants, move detail into `reference.md`
     and point at it from the body ("for the full argument list, read
     reference.md") rather than inlining everything.
   - If the procedure repeatedly needs the same computation, put a script in
     `scripts/` and tell the body to run it (it executes without loading its
     source into context).

5. **Write the file** to `<skills-dir>/<name>/SKILL.md` (create the directory).
   The skills directory is shown by the `/skills` command; ask the user if you
   don't know its path. Create any `reference.md` / `scripts/` alongside it.

6. **Verify.** Tell the user to run `/skills` — the new skill should appear with
   its description. If it shows under "Skipped", fix the reported reason (almost
   always a missing `description` or malformed frontmatter) and check again.

## Improving an existing skill

Read the current `SKILL.md`, then refine based on the user's feedback. Common
wins: tighten or broaden the `description` if the skill triggers at the wrong
times; remove instructions that make the model waste effort; move bulky detail
out of the body into a reference file; replace repeated ad-hoc steps with a
bundled script. Keep changes general — a skill should work across many future
requests, not just the one example in front of you.
