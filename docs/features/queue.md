# Queue, steer, and cancel

The input stays available while MnemoAI works. Choose whether your next message
changes the current task or starts a separate one.

## Steer the running task

Type guidance and press **Enter**:

```text
Also cover the empty-input case, and keep the existing public API.
```

The running turn receives the message at its next checkpoint. If it can no
longer accept guidance, the message waits for its own turn. Slash commands
always wait for their own turn.

## Queue a separate task

Type a new task and press **Tab** while the assistant is working:

```text
After this, update the README with the new behavior.
```

The queued message starts its own turn after the current one finishes.
When a completion menu is open, Tab completes the selected entry first;
when idle, Tab retains its normal completion behavior.

## Remove a pending message

1. Press **Ctrl+Q** to open pending messages.
2. Select a message with **↑/↓**.
3. Press **Delete** or **Backspace** to remove it.
4. Close the view with **Esc**, **Enter**, or **Ctrl+Q**.

This leaves the running task and your draft alone. Pending guidance can be
removed before the running turn takes it; a message already delivered to the
model cannot be withdrawn from this view.

## Prioritize a queued task

Queue the task you want next. Return to the normal input if a pending-message
or review view is open, then press **Esc** to cancel the current turn.
The next queued message can then start.

Esc closes a focused pending-message or review view; it cancels the running
turn only when the normal input has focus. Ctrl+C also closes those focused
views before acting on the running turn.

Cancelling a turn does not undo file changes already made. Use
[file-change reports](changes.md) to inspect them.

For all keys and paste behavior, see [Terminal controls](terminal.md#keyboard-shortcuts).
