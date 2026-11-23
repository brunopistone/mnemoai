# Manage ongoing work

Keep the active task moving while making room for follow-ups and future sessions.

## Give the task a clear endpoint

```text
Implement the settings change, run the affected tests, and update the user guide.
Continue until those steps are complete. Report any blocker with the evidence.
```

MnemoAI can [continue through execution checkpoints and compaction](../features/long-running-work.md).
Your constraints and the task's completion criteria still matter.

## Steer or queue

- Press **Enter** for guidance that belongs to the task already running.
- Press **Tab** for a separate task that should run afterward.
- Use **Ctrl+Q** to remove a pending message.
- With the normal input focused, press **Esc** to cancel the current turn and
  let the next queued task proceed.

See [Queue, steer, and cancel](../features/queue.md).

## Inspect progress

Use [context and token usage](../features/context.md) to understand the
conversation's size and cumulative model usage.
Use [file-change reports](../features/changes.md) to inspect the actual work.

Independent tasks may benefit from [sub-agents](../guides/orchestration.md).
Their results return through the main conversation; delegation is useful when
work can proceed independently.

## Pick up later

Name the session with `/rename`, then return to the project and run:

```bash
mnemoai --continue
```

Use `mnemoai --resume` to choose a session instead.
For a conversation you want to keep deliberately, use `/save`.
See [Sessions and history](../features/sessions.md) for the distinction.
