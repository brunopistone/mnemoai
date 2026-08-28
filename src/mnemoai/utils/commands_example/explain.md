---
description: Explain how something works, citing the code
argument_hint: <topic or file>
---

Explain how $ARGUMENTS works in this codebase.

Ground every claim in the code rather than in what the name suggests:

- Find the relevant files first (`glob_search`, `grep_search`), then read them.
- Cite each fact as `path:line` so I can jump straight to it.
- Cover the entry point, how the pieces hand off to each other, and where state lives.
- Call out anything surprising or easy to get wrong — that part is worth writing down.

Don't change any files: this is a read-only explanation.
