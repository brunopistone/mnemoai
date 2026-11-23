# Your first conversation { #basic-chat }

Start MnemoAI from the project directory so relative paths refer to that project.
Type a question and press Enter. The assistant uses available tools when needed.

```
You: What files are in the current directory?
Assistant: [Uses fs_read tool to list directory contents]

You: Read the README.md file
Assistant: [Uses fs_read tool and displays content]
```

## Add context

Use [`@` to attach a text file](../features/files.md), or give the path to a file
the assistant should inspect. State the expected result and any constraints,
such as keeping unrelated files unchanged.

## Review an action

When an action requires approval, the terminal asks before it runs.
Use [plan mode](../guides/safety.md#investigate-before-changing-anything) to
investigate before permitting edits.

## Continue the conversation

Ask a follow-up after the answer, or [send guidance while work is running](../features/queue.md).
Use `/help` to find commands, and `/quit` to leave the application.

See [Working with MnemoAI](workflows.md) for longer tasks.
