# Terminal controls

## The prompt { #the-prompt }

On an interactive terminal the input `>` stays **pinned at the bottom** of the
screen while the assistant works; the answer, its reasoning, and tool activity
stream **above** it (native scrollback and copy/paste are preserved). You can
keep typing during a turn — see the shortcuts below.

## The status footer { #the-status-footer }

Under the input sits a dim one-line footer with the three facts that apply to
whatever you type next:

```
claude-opus-5 · bedrock          ~/dev/project          ▓░░░░░░░ 90.1k · 9%
```

the **model** (and provider) the turn will go to, the **directory** the session is
running in — which is what the file and shell tools act on — and the **context
meter**: the last measured prompt size and how much of the model's window that
fills. The meter turns **amber past 70%** and **red past 90%**, which is the point
at which [compaction](../reference/commands.md#commands) is close (it starts at 80% by default), so a long
session tells you it's getting long before a summary interrupts it.

The count is the provider's input-token number for the last request, not the size
of the unsent next request. It stays visible while the next request runs, then
updates when the provider reports the new count. Before a measurement is available,
after resume, `/load` or compaction, and when a request fails or omits usage, it is a local estimate
marked `~`. The estimate includes instructions, tools and structured history;
it can differ from the provider's actual tokenizer.
A narrow terminal drops the path, then the provider, keeping the meter.

Off an interactive terminal (a pipe, CI) there's no footer, so the context size is
printed after each turn as `[Context: N tokens]` instead (`~N` for an estimate).

## When the terminal wants you back { #when-the-terminal-wants-you-back }

A long turn is exactly the turn you stop watching. Three moments therefore ring
the terminal:

- a **turn finishes** after running longer than `NOTIFY.AFTER_SECONDS` (30s by
  default),
- a **confirmation prompt or a question is waiting** on you — the work is stopped
  until it's answered, so this one is sent however short the turn has been,
- a **background sub-agent's report arrives** while you're idle.

Each sends two things: the **terminal bell**, which every terminal has and which
tmux and screen turn into a window-activity flag, and an **OSC 9 desktop
notification** (`mnemoai · done in 4m12s · project`), which iTerm2, WezTerm,
kitty, Windows Terminal and others raise as a real system notification —
terminals that don't understand it ignore it. Inside tmux or screen the sequence
is wrapped so it reaches the outer terminal instead of being swallowed.

Nothing is sent when output isn't going to a terminal (a pipe, CI), nor for a
turn **you** cancelled, and two notifications closer together than 10 seconds
collapse into one — a task confirming eight writes is one interruption, not eight.

```yaml
NOTIFY:
  AFTER_SECONDS: 30 # a turn must run this long for its end to notify (0 = never)
  BELL: true # the terminal bell
  DESKTOP: true # the OSC 9 desktop notification
```

Set both `BELL` and `DESKTOP` to `false` to stay silent entirely; `AFTER_SECONDS: 0`
silences only the turn-end half, keeping the ones that are waiting on you.

## Keyboard Shortcuts { #keyboard-shortcuts }

- `Ctrl+J`: Insert a new line in the input (`Enter` submits)
- `Ctrl+O`: Expand/collapse peer-review details when available; Esc closes the
  focused detail pane. The grey pinned review row is also clickable.
- `Enter`: Submit the message. **While the assistant is working**, a submitted
  message goes to the running turn at its next checkpoint. If that turn can no
  longer accept guidance, the message waits for its own turn. Slash commands
  always wait for their own turn.
- `Tab`: **While the assistant is working**, queue the typed message as a separate
  turn after the current one finishes. It is not sent as guidance to the current
  turn. When a completion menu is open, Tab still completes the selection; when
  idle, normal completion behavior is unchanged.
- `Ctrl+Q`: Open pending messages. Use ↑/↓ to select one and `Delete` or
  `Backspace` to remove it. `Esc`, `Enter` or `Ctrl+Q` closes the view, preserving
  your draft and leaving the running task alone. Guidance can be removed only
  before the running turn takes it; a message already sent cannot be withdrawn.
- `Esc`: Cancel the turn that's currently running when the normal input has
  focus. In the pending-message or review view, Esc closes that view instead.
- `Ctrl+C`: Close a focused pending-message or review view; otherwise cancel the
  running turn. When nothing is running, press twice (or `Ctrl+D`) to exit.
- `/` then a letter: shows a slash-command completion menu (↑/↓ to move,
  `Enter`/`Tab` to accept)
- `@` then part of a path: completes files and directories anywhere in the line —
  see [attaching a file with `@`](files.md#attaching-a-file-with)

## Pasting long text { #pasting-long-text }

Pasting a long block (a transcript, a file, a stack trace) collapses it into a
compact **`[Pasted text #N +M lines]`** placeholder in the input instead of
flooding the prompt — `M` is the number of line breaks. The full text is kept
aside; when you submit, the **assistant receives the complete paste**, and the
scrollback shows it **dimmed (gray)** so it's easy to tell apart from what you
typed. A large paste is shown **capped** in the scrollback — the first and last
few lines with a `… +N lines …` marker in between — so it never floods the
screen (the assistant still gets all of it). The collapsed view is only while
you're composing. Backspace over the placeholder deletes the **whole block at
once** (not character by character). Short pastes insert normally.

In dialogs (the `/load` picker and the `/config`, `/model`, `/params`,
`/features` prompts): arrow keys move, `Enter` confirms, `Esc` cancels, and
`Ctrl+B` steps back to the previous prompt within a model-section flow. These
briefly take over the full screen, then return you to the pinned prompt.

## Verbose Mode { #verbose-mode }

Control thinking process visibility:

```bash
mnemoai              # Verbose mode (shows thinking)
mnemoai --no-verbose # Hide thinking process
# from a checkout: PYTHONPATH=src python -m mnemoai [--no-verbose]
```
