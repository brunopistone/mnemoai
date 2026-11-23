# File attachments { #attaching-a-file-with }

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
  [`STEERING.md`](../guides/memory.md#steering-steeringmd) work the same way, and a mentioned file's own
  `@`-references are followed too.
- **Anything that isn't a readable path is left alone**, so `@staticmethod`,
  `@someone` and an email address keep meaning what they say. A path-looking
  mention that doesn't exist gets the gray `no such file` line above.
- **Binary files aren't inlined** (they're reported as `not a text file`) — ask
  about an image and the vision tool handles it instead.
- **Attaching isn't reading.** Before editing a file, the assistant still opens it
  with its own read tool; a mention doesn't stand in for that.
