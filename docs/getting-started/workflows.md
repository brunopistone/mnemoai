# Working with MnemoAI

MnemoAI works from a conversation in your terminal. Give it a goal, enough
context to find the relevant material, and a clear way to check the result.

## Start with a concrete task

Include the files or project involved, the behavior you want, and any constraints:

```text
Fix the failing parser test in this project. Preserve the public function
signature, explain the cause, and run the relevant tests after the change.
```

For a task that should only investigate, say so or turn on
[plan mode](../guides/safety.md#investigate-before-changing-anything).

## Supply useful context

Launch MnemoAI from the project directory. Mention a path, or use
[`@` attachments](../features/files.md) to include a file directly in your prompt.

Keep reusable project guidance in the supported
[steering files](../guides/memory.md#steering-steeringmd).
Use [skills](../guides/agent-skills.md) for procedures the assistant can select
when they are useful.

## Direct work as it runs

You can keep typing while the assistant works:

- **Enter** sends guidance at the running turn's next checkpoint.
- **Tab** queues a separate task for after the current turn.
- **Ctrl+Q** lets you inspect and remove pending messages.
- **Esc** cancels the running turn when the normal input has focus.

See [Queue, steer, and cancel](../features/queue.md) for focus behavior and
how to prioritize a queued task.

## Define completion

Ask for the result and the checks that establish it. For code, that may be a
reproduced failure and passing tests. For a document, that may be factual
sources, required sections, and a saved file.

Use [`/diff`, `/files`, and `/why`](../features/changes.md) to inspect changes.
Optional [peer review](../guides/peer-review.md) can supply a second assessment;
it does not grant permission or prove a result correct.

## Continue later

Use [sessions and history](../features/sessions.md) to resume a conversation,
save it deliberately, or explore a different direction in a branch.
Long tasks can [continue through compaction](../features/long-running-work.md).
