---
name: Conventional Commit Message
description: Use when the user asks to write, draft, or improve a git commit message, or to commit changes. Produces a well-structured Conventional Commits message from the staged diff.
version: 1
---

# Conventional Commit Message

Write a clear commit message in the **Conventional Commits** style for the
currently staged changes.

## Steps

1. Inspect what is staged. Run:
   - `git diff --staged --stat` (overview of files changed)
   - `git diff --staged` (the actual changes — read enough to summarize intent)
     If nothing is staged, tell the user to `git add` first and stop.
2. Decide the **type** from the change's intent:
   `feat` (new capability), `fix` (bug fix), `docs`, `refactor`, `test`,
   `chore`, `perf`, `build`, `ci`.
3. Write the message in this exact shape:

   ```
   <type>(<optional scope>): <imperative summary, ≤ 72 chars>

   <body: WHY the change was made and any notable consequence — wrap at ~72 cols>
   ```

   - Summary line: imperative mood ("add", not "added"/"adds"), no trailing period.
   - Body: explain the motivation, not a restatement of the diff. Omit the body
     for a trivial one-line change.

4. Present the message to the user for approval. Do **not** run `git commit`
   yourself unless the user explicitly asks you to.

## Notes

- One logical change per commit. If the staged diff spans unrelated concerns,
  say so and suggest splitting it.
- Keep scope short and lowercase (e.g. `feat(auth):`).
