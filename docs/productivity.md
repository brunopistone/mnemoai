# Productivity Tools

## 🚀 Productivity Tools

The assistant includes specialized tools for efficient code and file manipulation:

### 📋 Todo List Management

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

### 🔎 Fast Search Tools

High-performance file and content searching:

#### Glob Search (File Names)

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

#### Grep Search (File Content)

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
- `max_results`: Maximum matches per file (default: 100)

**Requirements:** [ripgrep](getting-started.md#7-recommended-optional-tools) is recommended for full speed; without it, `grep_search` automatically falls back to a slower built-in search.

**Performance:** 10-100x faster than traditional grep for large codebases.

### ✏️ Precise File Editing

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

### 🛡️ Enhanced Error Handling

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

### 🔐 Action Confirmation (bash & file writes)

Destructive tools ask for explicit confirmation before they run — shell commands (`execute_bash`) and file modifications (`fs_write`, `file_edit`):

```
▶ Run shell command?
  rm -rf build/
  Proceed? (y/N):

▶ Write to file?
  create ~/script.py
  Proceed? (y/N):
```

Only an explicit `y`/`yes` proceeds; anything else (including Enter) declines, and the model is told the user declined. This is a **hard gate enforced client-side** — the prompt always fires regardless of what the model does, because the client owns the terminal (the MCP server is a piped subprocess and can't prompt).

- Toggles: `REQUIRE_BASH_CONFIRMATION` and `REQUIRE_WRITE_CONFIRMATION` (both default `true`). Set either to `false` for trusted/automation setups.
- Non-interactive runs (no TTY — tests, pipes, CI) auto-proceed so they don't hang.

### 🧱 Server-Side Safety Floor

The confirmation prompt above is a **convenience layer** you can turn off. Below it sits a **hard floor** that is always on and lives inside the MCP server itself — so it holds even when confirmations are disabled, in non-interactive runs, or if the server is driven by another client. It blocks only **catastrophic, irreversible** actions; ordinary work is never affected.

| Layer                        | What it does                                             | Can be disabled?          |
| ---------------------------- | -------------------------------------------------------- | ------------------------- |
| Client confirmation prompt   | Asks `Proceed? (y/N/a)` before each shell/file/memory op | Yes (`REQUIRE_*` toggles) |
| **Server-side safety floor** | Refuses system-destroying commands and system-dir writes | **No** (always enforced)  |

**Shell commands** (`execute_bash`, `start_background_task`) are refused when they would:

- recursively force-delete a root or home target — `rm -rf /`, `rm -rf ~`, `rm -rf /*`
- create a filesystem or overwrite a raw device — `mkfs…`, `dd of=/dev/…`, `shred`/`wipefs` on a device
- change the machine power state — `shutdown`, `reboot`, `halt`, `poweroff`, `init 0/6`
- fork-bomb the machine — `:(){ :|:& };:`

**File writes** (`fs_write`, `file_edit`) are refused when the target is inside a critical system directory — `/`, `/etc`, `/bin`, `/usr`, `/boot`, `/dev`, `/System`, `/Library`, … (`..` and `~` are resolved first). Your home, project trees, temp directories, and the app home stay fully writable.

This floor is intentionally **narrow**: scoped, everyday-destructive commands like `rm -rf build/` or `git reset --hard` are _not_ blocked here — they remain gated by the confirmation prompt. The goal is to make irreversible system damage impossible, not to second-guess normal edits.

### 🛡️ Git Safety

Safe git operations with protection against common mistakes:

**Tools:**

- `git_safe(command="...")` - Execute git commands with safety checks
- `git_status_safe()` - Comprehensive status with warnings
- `git_commit_safe(message="...", add_all=True)` - Safe commits with staging

**Protected Operations:**

| Operation                  | Protection                         |
| -------------------------- | ---------------------------------- |
| Force push to main/master  | Blocked                            |
| `git reset --hard`         | Warning + confirmation required    |
| `git push --force`         | Warning (use `--force-with-lease`) |
| `git commit --amend`       | Checks if already pushed           |
| Skip hooks (`--no-verify`) | Warning                            |
| Force delete branch (`-D`) | Warning                            |

**Example:**

```python
# Safe - uses git_safe with protections
git_safe(command="push origin feature-branch")

# Dangerous - requires confirmation
git_safe(command="reset --hard HEAD~1", allow_dangerous=True, reason="Discarding failed experiment")
```

### 📝 Plan Mode

**Enforced read-only plan mode (`/plan`).** Toggle it with the **`/plan`** command
before asking for a complex change — the agent investigates and drafts a plan
without touching anything, then you approve and it executes.

**When to Use:**

- New feature with multiple files
- Architectural decisions needed
- Multi-step refactoring
- Unclear requirements

**How it works:**

- While ON, the agent can only use **read-only** tools (file reads, glob/grep
  search, read-only shell like `ls`/`cat`/`git status`, web search, document
  readers) and its own memory notebook. Any attempt to edit files
  (`fs_write`/`file_edit`), run mutating shell commands (`execute_bash`), perform
  git writes, or start background tasks is **hard-blocked client-side** — regardless
  of what the model tries to do.
- **Approve → execute in one step.** When the plan is ready the agent calls the
  `exit_plan_mode` tool, which presents the plan and asks how to proceed:
  - **y** — approve and run: plan mode turns off and the agent **executes the
    approved plan immediately, in the same turn** (no need to `/plan` off and re-ask).
  - **e** — open the plan in your `$EDITOR` to tweak it, then re-review.
  - **n** — keep planning: stays read-only and refines the plan.
- The approved plan is saved to `~/.mnemoai/plans/plan_<timestamp>.md` so it
  survives compaction and can be re-read later.
- **Pre-approved commands.** A plan can declare the shell commands it will run
  during execution (tests, builds, installs). Approving the plan pre-approves
  them, so they run **without** a per-command `Proceed?` prompt while the plan
  executes; anything not on the list still prompts. These pre-approvals are scoped
  to that plan — they're cleared when you `/clear` or re-enter plan mode.
- You can still toggle plan mode off manually with `/plan` at any time.

This is enforced at the same client-side chokepoint as the action-confirmation gate,
so it holds across both the normal loop and the orchestrator workers, and even a
misbehaving local model cannot mutate anything while plan mode is on.

### 🔄 Background Tasks

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
