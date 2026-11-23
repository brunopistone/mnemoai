# Track file changes

## What this session touched { #what-this-session-touched }

`/files` answers the question a long session makes hard: which files did it actually
open, and which did it change?

```
Files this session

  Changed (2)
    ✎ tests/unit/test_diff_report.py      1 edit
    ✎ src/mnemoai/client/diff_report.py   2 edits · 1 read

  Attached with @ (1)
    @ docs/guides/usage.md                attached

  Read (2)
    · src/mnemoai/client/ui/turn_view.py  2 reads
    · src/mnemoai/client/agent/agent.py   1 read
```

Most recently touched first, **changed files first** — that's the group with
consequences on disk. Worth knowing:

- **It counts the work you don't see.** A sub-agent's edits and a parallel wave's
  reads are in the list, because the record is taken where every tool call passes
  through rather than from what got printed.
- **Two spellings are one row.** `./src/x.py`, `src/x.py` and `~/proj/src/x.py`
  resolve to the same file, so the counts are the real counts.
- **It survives compaction.** A file read an hour ago may have had its content
  summarized out of the context since; the touch still counts, and the report says
  so rather than implying the content is still in the window.
- **A tree-wide search isn't a touch.** `grep_search` over a directory produces no
  file you can go and look at, so only the tools that name a single file are recorded.
- `/clear` resets it, since a cleared conversation is a new session.

## Seeing what changed { #seeing-what-changed }

`/diff` is `git status` with the one column git can't give you — **which of these
files this conversation wrote**:

```
Changes in ~/development/mnemoai (dev)

  ✎ src/mnemoai/client/diff_report.py        +402    new
  ✎ src/mnemoai/client/ui/chat_interface.py  +24 -1
  ✎ CHANGELOG.md                             +31
    docs/guides/usage.md                     +6 -2
    assets/logo.png                          binary

  5 files · +463 -3 · ✎ 3 written this session
  /diff <path> shows one file's diff.
```

Staged, unstaged and untracked files in one list. A `✎` marks the ones the session
wrote (they sort first); **everything unmarked was already dirty when you started**,
which is the distinction that decides what's safe to commit. `/diff <path>` shows
that one file's unified diff, colored, with an untracked file rendered as the
all-additions diff it effectively is.

**It is read-only by construction.** The only git it runs is `rev-parse`, `diff` and
`ls-files` — there is no code path that could stage, stash or check anything out. It
is also bounded (a long list and a long diff both collapse), and when something is
cut it prints the exact `git` command that shows the rest.

## Why a file looks like this { #why-a-file-looks-like-this }

`/files` says what was touched and `/diff` says what is different. Neither answers
the question you actually have when you open a file later and don't recognize a
function in it: **which prompt asked for this?** `/why` does:

```
Why src/mnemoai/client/status_bar.py looks like this

  This session
    ✎ turn 7 · 16:04 · file_edit · +12 -3
        > make the footer stop wrapping on a narrow terminal

  Earlier · session session_20260908_143012_5511_a1b2
    ✎ turn 2 · 09-08 14:36 · fs_write · wrote 397 lines
        > add a pinned status line under the input

  2 recorded changes. The prompt under each is the one that asked for
  it. A change made outside this app, or before the index existed, isn't here.
```

A bare `/why` lists every file with a recorded change instead, each with its count
and the most recent prompt behind it — this session's files when it changed any,
the project's history otherwise. `/why <path>` then drills into one file.

Worth knowing:

- **It reaches past the conversation that made the change.** The record is kept per
  project in a small index (`~/.mnemoai/{profile}/provenance/`), so it survives the
  session ending, and it outlives the transcripts themselves — those are swept at
  `SESSION_MAX_AGE_DAYS`, while a line of code lives for years.
- **It names the session, so the whole turn is still reachable.** The heading of an
  earlier group is a session id you can pass to `mnemoai --resume <id>`; the index
  holds a summary, the transcript holds the conversation.
- **Like the other two, it costs nothing.** No model call, no tool, no write beyond
  its own index — so it's cheap to run on a hunch.
- **It records what this app changed, and only that.** An edit you made in your
  editor, a `git` operation, or anything from before the index existed is not in it,
  and the report says so rather than implying the file has no history.
- **A `/rewind` doesn't withdraw an entry, deliberately.** Rewinding takes back the
  conversation; the file on disk is untouched, so the prompt that caused the change
  is still the true answer to why it's there.
- Setting `SESSION_MAX_AGE_DAYS: 0` turns recording off — the index stores what you
  typed, so the switch that stops recording sessions stops this too.

The turn-end line carries the same pointer for the turn you just ran:

```
· done in 1m12s · 3 files · /why · 16:19
```

The count is the number of files that turn changed, and it's left out entirely when
a turn changed nothing — so seeing it there means something. It survives a cancel
too (`⊘ stopped after 12s · 2 files · /why`), which is exactly when knowing what
already landed on disk matters most.
