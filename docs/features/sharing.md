# Copy and export

## Copying an answer { #copying-an-answer }

Selecting a streamed answer with the mouse takes the terminal's line wrapping with
it, which is why a copied code block so often has to be reflowed by hand. `/copy`
takes the message as the model wrote it:

```
/copy          # the last answer
/copy code     # just its last fenced code block
/copy 2        # the answer before last
```

Two transports, tried in the order that's right for where you are. A **local helper**
(`pbcopy`, `wl-copy`, `xclip`, `xsel`, `clip.exe`) when one is installed, and
**OSC 52** — the escape sequence that asks the terminal itself to set the clipboard —
otherwise, so a machine with no helper still works. **Over SSH the terminal goes
first**, because a helper running on the far end would set the clipboard of a machine
nobody is sitting at; that's the case OSC 52 exists for. Inside tmux or screen the
sequence is wrapped in a passthrough so it reaches the outer terminal.

The notice says what was copied, how big it was and which transport carried it
(`Copied the last sh block (1 line, 9 chars) via pbcopy.`) — a clipboard write is
otherwise invisible until you paste. If neither transport is available it says so and
points at `/export`, which writes to a file instead.

## Exporting a transcript { #exporting-a-transcript }

`/export` writes the conversation as a **readable transcript** — the thing you paste
into a bug report, a PR description, or a message to someone else:

```
/export                  # Markdown, into the current directory
/export txt              # plain text instead
/export ~/notes/         # into a directory (name generated from your first prompt)
/export bug-report.md    # to an exact file
/export reasoning        # include the assistant's thinking blocks
```

This is **not** `/save`. `/save` writes JSON that `/load` can read back; an export is
one-way and optimized for a human reader:

- **Tool calls** appear as one-line summaries (`fs_read(path=…, mode=Search)`); a
  file body passed as an argument is replaced by its size rather than pasted in.
- **Tool results are left out.** A few thousand lines of file content is the single
  biggest source of noise in a transcript, and the answer already says what mattered.
- **Injected context is stripped** — retrieved memories and steering instructions
  were never something you typed, and unstripped they dominate the export.
- **Reasoning is off by default** (thinking blocks usually dwarf the conversation);
  `/export reasoning` includes them, collapsed in Markdown.

The filename is derived from your opening prompt
(`conversation_20260730_161738_reply-with-exactly-red.md`), so exports are
identifiable in a directory listing instead of being one of many timestamps.
