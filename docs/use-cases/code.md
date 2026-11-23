# Fix and verify code

Use MnemoAI to investigate a failure and make a change you can inspect.

## Open the project

```bash
cd /path/to/your/project
mnemoai
```

Start with a bounded request:

```text
Investigate why the parser accepts an empty name. Read the relevant code and
tests, reproduce the behavior, then make a focused fix and run the affected tests.
Preserve unrelated changes.
```

To investigate before allowing edits, use
[plan mode](../guides/safety.md#investigate-before-changing-anything).

## Add context and constraints

Attach a small relevant file with [`@`](../features/files.md), include the
failure output, or name the test command if you already know it.
While the task runs, press Enter to send new guidance.

## Inspect the result

Use `/diff` to view changes, `/files` to see which files were involved,
and `/why <path>` to see the request behind a recorded change.
See [Track file changes](../features/changes.md).

Ask for the cause, the resulting behavior, and the checks actually run.
Optional [peer review](../guides/peer-review.md) can return findings for the
chat model to address.

## Keep the next task separate

While the fix is running, type a documentation follow-up and press Tab.
It will get a separate turn after the current work.
See [Queue, steer, and cancel](../features/queue.md).
