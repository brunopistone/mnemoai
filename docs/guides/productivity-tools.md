# Everyday tools

The tools the assistant reaches for on most tasks: tracking multi-step work,
finding code, editing precisely, and running long jobs without blocking.

For exact parameters and defaults, see the [Tools reference](../reference/tools.md).
For what asks before it runs and what is refused, see
[Safety & permissions](safety.md).

## 📋 Todo List Management

Track multi-step tasks with automatic status management:

**Tools:**

- `todo_write(todos)`: Update the todo list
- `todo_read()`: View current todos
- `todo_clear()`: Clear all todos

**Features:**

- Three states: `pending`, `in_progress`, `completed`
- Enforces exactly ONE task in progress at a time
- Real-time progress tracking
- Stored in `~/.mnemoai/{profile}/todos/current_todos.json`

**Usage Example:**

```
You: Implement user authentication
Assistant: [Creates todos for: database setup, API endpoints, frontend integration, testing]
Assistant: [Marks first todo as in_progress]
Assistant: [Completes each step, updating todos in real-time]
```

## 🔎 Fast Search Tools

High-performance file and content searching:

### Glob Search (File Names)

Find files by name patterns:

```python
glob_search(pattern="**/*.py")  # All Python files recursively
glob_search(pattern="src/**/*.ts", max_results=100)  # TypeScript in src/
glob_search(pattern="test_*.py", sort_by_mtime=False)  # Unsorted for speed
```

**Parameters:**

- `pattern`: Glob pattern (e.g., `**/*.py`, `*.{yaml,json}`)
- `path`: Directory to search (default: current directory)
- `max_results`: Limit results (default: 1000, use 0 for unlimited)
- `sort_by_mtime`: Sort by modification time (default: True)

**Performance:** Best for project/codebase searches. For system-wide searches (entire home directory), the assistant automatically uses `find` command instead.

### Grep Search (File Content)

Search within file contents using ripgrep:

```python
grep_search(pattern="class Foo")  # Find class definitions
grep_search(pattern="TODO|FIXME", file_pattern="*.py", case_insensitive=True)
grep_search(pattern="import React", output_mode="content")  # Show matched lines
```

**Parameters:**

- `pattern`: Regex pattern to search for
- `path`: Directory to search (default: current directory)
- `file_pattern`: Filter by file type (e.g., `*.py`, `*.{ts,tsx}`)
- `case_insensitive`: Case-insensitive search (default: False)
- `output_mode`: `files_with_matches` (default), `content`, or `count`
- `context_lines`: Lines of context around matches
- `max_results`: Total cap on matches returned across all files (default: 100; `0` = unlimited)

**Requirements:** [ripgrep](../getting-started/installation.md#7-recommended-optional-tools) is **required**. Without it `grep_search` returns `ripgrep (rg) not installed` — there is no fallback.

**Performance:** 10-100x faster than traditional grep for large codebases.

## ✏️ Precise File Editing

Safe string replacement with validation:

```python
file_edit(
    file_path="/path/to/file.py",
    old_string="def old_function():\n    pass",
    new_string="def new_function():\n    return True",
    replace_all=False  # Requires uniqueness (default)
)
```

**Safety Features:**

- Validates file exists before editing
- Checks that `old_string` exists in file
- Enforces uniqueness (prevents accidental multiple replacements)
- Provides detailed error messages with troubleshooting steps
- Returns line count changes

**Best Practice Workflow:**

1. Read the file first with `fs_read`
2. Copy the EXACT text you want to replace (including whitespace)
3. Create the new version with your changes
4. Call `file_edit` with exact strings

**Error Handling:** If the string isn't unique, the tool provides the line numbers where it appears so you can add more context.

## 🛡️ Enhanced Error Handling

All tools now provide intelligent error messages with troubleshooting guidance:

**Example Error Response:**

```json
{
  "error": true,
  "error_type": "FileNotFoundError",
  "message": "File or directory not found: /path/to/file.txt",
  "next_steps": [
    "Verify the file path is correct",
    "Use glob_search to find files by pattern",
    "Check with execute_bash('ls -la /parent/dir')",
    "Ensure you have read permissions"
  ],
  "original_error": "..."
}
```

**Handled Error Types:**

- FileNotFoundError
- PermissionError
- IsADirectoryError
- JSONDecodeError
- Encoding errors
- Command execution errors
- Timeout errors

## ❓ Questions the Assistant Asks You

When the assistant is blocked on a decision that's genuinely **yours** — a real fork
with different tradeoffs, which it can't settle from your request, the code, or a
sensible default — it puts the question to you as a short list instead of guessing or
writing out every alternative for you to sort through:

```
┌──────────| ? Which caching layer should the new store use? |──────────┐
│ ↑/↓ to choose · Tab for a note · Enter to confirm · Esc to dismiss    │
│                                                                       │
│ (*) In-process LRU (recommended)                                      │
│ ( ) Redis                                                             │
│ ( ) Memcached                                                         │
│ ( ) None of these — let's talk about it                               │
│                                                                       │
│ note ▸ fine for now, but it has to survive a restart later            │
│                                                                       │
│                       <    OK    > <  Cancel  >                       │
└───────────────────────────────────────────────────────────────────────┘
```

Arrow keys move, `Tab` reaches the note, `Enter` confirms from either field, `Esc`
dismisses. What you answered is echoed above the prompt, so the conversation keeps a
record of the question and your reply.

**The options are never the whole answer.** They were guessed by the assistant, so two
things are always there that it didn't offer:

- **A free-text note.** It rides along with whatever you pick, which makes "that one,
  but only for local runs" a single answer instead of a choice you then have to correct.
- **A "None of these" row.** Take it when you disagree with every option: the assistant
  drops all of them and answers you in prose — with your note, if you left one, as the
  thing it responds to.

**And there are three ways out, not two.** Choosing settles the question; declining every
option turns it into a conversation; dismissing hands the decision back:

| You do this          | The assistant then                                             |
| -------------------- | -------------------------------------------------------------- |
| Pick an option       | Proceeds on it, reading your note as a qualification.          |
| Pick _None of these_ | Drops every option and replies about the real tradeoff.        |
| Press `Esc`          | Carries on with its own best judgment and says what it assumed. |

That last one is deliberate: a question you didn't want to answer shouldn't stall the
work. But it also means `Esc` is **not** how you disagree with the options — it's how you
tell the assistant to decide for you. Use the "None of these" row for the other case.
Either way it won't ask the same question again.

This is meant to be **rare**. The assistant is instructed to prefer acting on a
reasonable default and saying what it assumed, because one question you have to answer
costs you more than a choice you can correct. It asks at most one at a time.

!!! note "Sub-agents can't ask"

    A [sub-agent](orchestration.md) has no direct user — a background one runs with no
    terminal at all — so it never raises this prompt. Instead it decides for itself and
    reports the assumption it made, which is why a background task can't quietly stall
    waiting on a prompt nobody would see.

## 🔄 Background Tasks

Run long operations in parallel without blocking:

**Tools:**

- `start_background_task(command="...", description="...")` - Start task
- `get_task_status(task_id="...")` - Check progress
- `get_task_output(task_id="...")` - Get output
- `list_background_tasks()` - See all tasks
- `cancel_background_task(task_id="...")` - Stop task
- `wait_for_task(task_id="...", timeout_seconds=300)` - Wait for completion

**When to Use:**

- Running full test suites
- Building large projects
- Installing dependencies
- Running linters on entire codebase
- Any command > 30 seconds

**Example:**

```python
# Start tests in background
result = start_background_task(command="pytest", description="Running tests")
# Returns: {"task_id": "abc123", ...}

# Check status later
get_task_status(task_id="abc123")

# Get output when done
get_task_output(task_id="abc123", tail_lines=50)
```

**Task Storage:** Output logs saved to `~/.mnemoai/tasks/`
