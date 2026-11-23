# Sessions and history

## Resuming a Session { #resuming-a-session }

Every session is recorded automatically, **scoped to the directory you launched
from** — so resuming inside a project only ever offers that project's sessions:

```bash
mnemoai --resume              # pick from this directory's recent sessions
mnemoai --resume <session-id> # resume that session directly
mnemoai --continue            # resume the most recent one, no prompt
```

`--resume` with no value lists this directory's sessions newest-first, showing how
long ago each one ran, how many turns it had, and the opening prompt **as you
typed it** — retrieved memory, steering instructions and other context the
assistant adds behind the scenes are left out of the label:

```
   4m ago    6 turns  refactor the FSDP config parser (continued)
  2h ago    2 turns  why does the loader crash on startup?
   3d ago   14 turns  add a --resume flag
```

The turn count describes the **whole conversation you'd restore**, inherited
history included — so a chat you've resumed several times shows its full length,
not just what you typed since the last resume. `(continued)` marks a session that
carried on from an earlier one.

**One row per conversation, not per file.** Resuming records a new session seeded
with everything that came before, so a chat you resumed three times exists as four
files on disk. Only the most recent is offered: the earlier ones are contained in
it, and that's the one you want to continue. Nothing is deleted — and if you want
an earlier point specifically, `mnemoai --resume <session-id>` still restores that
exact one. (A [`/branch`](#branching-a-session) fork is _not_ collapsed, since it
diverges rather than continues.)

Pick one and the conversation replays into the terminal and continues where you
left off, with the restored conversation shown just above the prompt. `Esc`
cancels and **exits** — since you asked to resume, it won't quietly start a new
conversation instead; run `mnemoai` without a flag for that.

!!! note "`--resume` and `/save` are separate things"

    `--resume` is **automatic** — every session is recorded without you asking, and
    old ones expire (30 days by default; set `SESSION_MAX_AGE_DAYS`, or `0` to
    turn session recording off entirely).

    [`/save` and `/load`](../reference/commands.md#commands) are **yours to curate** — a conversation you
    deliberately keep, under a name or path you choose, in
    `~/.mnemoai/{profile}/conversations/`. Saved conversations are **never expired
    or deleted** by session cleanup, and resuming a session never overwrites one
    (after a `--resume`, a bare `/save` writes a new file rather than adopting the
    resumed session).

Sessions live in `~/.mnemoai/{profile}/sessions/{directory}/` as append-only
files, one per session. A turn that was **cancelled or that failed** is recorded
too — so a resumed session shows exactly what the live one did, including a
question you interrupted and one a dropped connection cut short.

**Changing models also preserves the conversation.** If `/model` restarts the app,
the current messages, summary, label and draft input are restored automatically.
For a conversation opened with `/load`, the save-file association survives too,
including any replies you have not saved yet. See
[model-change restarts](../configuration/parameters.md#model-parameters) for failure recovery.

**Resuming carries the whole conversation forward.** When you resume (or `/load`),
the restored history is copied into the new session's file, so that file is a
complete record on its own and resuming _it_ later replays everything — not just
what you added after the restore. The session you resumed from is never modified,
so you can always go back and resume the same point again.

**A compacted session comes back compacted.** If the conversation was
[compacted](../reference/commands.md#commands) — by you, or automatically once it outgrew the window — the
transcript records the summary alongside the turns that stayed verbatim, so
resuming restores the state the session **ended** in: the same summary, the same
recent turns, and roughly the same context size the footer last showed you. The
full text is still in the file and still replays into the terminal, so you can
scroll back and read anything the summary condensed — it just isn't re-sent to the
model, which is what a compaction was for.

The same holds when the context was reclaimed **without** a summary. Before
summarizing anything, mnemoai first trims the bodies of older tool results — a
cheap pass that costs no model call and often frees enough on its own. A resume
brings back those trimmed results too, so the context you come back to matches the
one you left in that case as well.

A launch you never typed into records nothing resumable, so its file is removed
when you exit — only sessions with at least one exchange are offered, which also
means a resume you didn't ask anything in won't appear as a duplicate of the
session it restored. If this directory has no sessions yet, `--resume` says so
and starts a normal session.

## Naming a session { #naming-a-session }

The picker labels every session with its opening prompt, which stops being enough
once a project has a few of them — especially after a `--resume` or a `/branch`,
where several rows legitimately share the same first question. `/rename` gives the
current session a name of your own:

```
/rename hooks phase 1        # name it
/rename                      # show the current name
/rename clear                # back to the first-prompt label
```

```
   4m ago    6 turns  hooks phase 1
  2h ago    2 turns  why does the loader crash on startup?
```

The name is appended to the session file like everything else, so renaming twice
just means the last name wins, and it **survives a resume** — same conversation,
new file, same name. A [`/branch`](#branching-a-session) fork deliberately does
**not** inherit it: the fork and its parent already share an opening prompt, and
sharing the name too would put you back where you started. Naming a session you
never type into doesn't keep its empty file alive.

## Branching a session { #branching-a-session }

`/branch` **forks the current conversation** so you can try a different direction
without losing the one you have. Use it when a conversation has gone somewhere you
don't want — a wrong assumption three turns back, a tangent that ate the context —
and you'd rather rewind than start over and re-explain everything.

```
/branch          # pick the turn to branch after
/branch 3        # branch straight after turn 3
```

With no argument you get a list of this session's turns, labelled by what you
asked:

```
1. refactor the FSDP config parser
2. also handle the sharding edge case
3. now add tests for it  (latest)
```

Branching after turn 2 gives you a session containing turns 1–2, and you continue
from there — turn 3 stays behind in the original. **The original session is copied,
never changed**, so it remains resumable exactly as it was; if the branch turns out
to be a dead end, `--resume` the original and nothing is lost. That also means
branching is cheap: there's no "are you sure", because nothing is destroyed.

If the conversation was [compacted](../reference/commands.md#commands), branching after that point carries
the summary with it, exactly as [resuming](#resuming-a-session) does — while
branching to a turn _before_ it forks the raw history instead, which is a way to get
the uncompacted detail of those turns back into a conversation.

Forks are marked in the `--resume` picker, since a branch inherits its parent's
opening prompt and the two rows would otherwise look identical:

```
   2m ago    1 turn   refactor the FSDP config parser (branch @ turn 2)
   9m ago    3 turns  refactor the FSDP config parser
```

!!! note "Branching moves the conversation, not your files"

    A branch rewinds the **conversation only**. Files the abandoned turns wrote are
    still on disk — mnemoai doesn't snapshot and restore your working tree. So after
    branching past a turn that edited code, the assistant's picture of those files
    comes from the branch's history: say what you want undone, or check `git diff`
    before continuing.

## Taking back your last prompt { #taking-back-your-last-prompt }

Sometimes the prompt itself was the mistake — the wrong file, the wrong framing, a
question that sent the whole turn somewhere you didn't want it to go. `/rewind`
drops that prompt **and everything the turn produced** from the conversation, so
the context is what it was the moment before you pressed Enter:

```
> /rewind

⟲ withdrew your last prompt
  "why is the FSDP config parser slow"
  14 messages dropped from the conversation and transcript.

  Files on disk are untouched — a rewind moves the conversation only.
```

The echoed prompt is there so you can see _which_ turn went, and the count covers
the whole turn: the answer, the tool calls, their results. Run it twice and you walk
back two turns. It looks for the last thing **you** typed, not the last message with
your name on it, so a turn made of tool results or an [auto-delivered sub-agent
report](../guides/orchestration.md#sub-agents-spawn_agent) can't be mistaken for a prompt.

Its scope is deliberately narrow, and it is the one thing to be clear about:

!!! warning "A rewind moves the conversation, not your files"

    Files the withdrawn turn wrote are **still on disk** — exactly as with
    [`/branch`](#branching-a-session). Undoing an edit is `git` territory (or
    [`/diff`](changes.md#seeing-what-changed), to see what there is to undo); a command that
    rolled back half a turn would be worse than one that tells you which half it
    does.

Three more things worth knowing:

- **It refuses after a compaction.** If the conversation was
  [compacted](../reference/commands.md#commands) during or right after that turn, the summary standing in
  for it can't be un-summarized, so `/rewind` says so and changes nothing rather
  than half-applying. Older compactions don't get in the way.
- **The transcript records the withdrawal; it doesn't erase it.** Session
  transcripts are append-only, so the turn is marked withdrawn: it stops being part
  of the conversation, a [`/branch`](#branching-a-session) can no longer fork at it,
  and `--resume` won't bring it back — but the text stays in the file. A rewind is
  an undo, not a redaction.
- **What the session _learned_ from the turn stays**, and the notice names which
  stores that means. [Episodic memory](../guides/memory.md#episodic-memory) keeps entries by
  similarity and the playbook folds a repeat strategy into an existing one, so
  there's nothing precise left to delete; `/memory` is editable if a
  [curated](../guides/memory.md#persistent-memory-memorymd) fact needs to go.
