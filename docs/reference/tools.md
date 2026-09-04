# Tools reference

Every tool Mnemo AI can call, with its parameters, defaults, and limits.

You never call these yourself — the model does, and the assistant prints each
call as it happens. This page is for reading those calls, understanding why one
was refused, and knowing which parameters exist. For the toggles and prompts that
govern whether a call runs at all, see [Control what Mnemo AI can
do](../guides/safety.md).

**31 tools in 17 groups.** Twenty-three are always available; the remaining
eight appear only when a feature is enabled.

| I want to know                    | Go to                                                           |
| --------------------------------- | --------------------------------------------------------------- |
| Which file-reading modes exist    | [`fs_read` modes](#fs_read-modes)                               |
| How an edit is applied            | [Write and edit files](#write-and-edit-files)                   |
| Why a write was refused           | [Read-before-write](#read-before-write)                         |
| What a search flag does           | [Search](#search)                                               |
| Which tools need a feature toggle | [Conditional tools](#conditional-tools)                         |
| What is refused outright          | [Safety floor](../guides/safety.md#understand-the-safety-floor) |

## Read files

| Tool      | Parameters                                                                                                                              | Purpose                                                    |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `fs_read` | `path` **(required)**, `mode` = `"Line"`, `start_line` = `1`, `end_line` = `-1`, `pattern` = `""`, `context_lines` = `2`, `depth` = `0` | Read a file — or list a directory — in one of eight modes. |

A successful read also _records_ that the file was read, which is what later
permits a write to it. Listing a directory records nothing.

### `fs_read` modes

`mode` selects the reader. Each mode uses only some parameters; the rest are
ignored.

| `mode`        | What it does                                                                                                                                                      | Uses                       |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `"Line"`      | Streams a line range with a `cat -n` gutter. Negative values count from the end of the file. Binary and image files fail fast with a pointer to `describe_image`. | `start_line`, `end_line`   |
| `"Search"`    | Literal, **case-insensitive** substring search with N lines of context per hit. Not a regex.                                                                      | `pattern`, `context_lines` |
| `"Directory"` | Recursive listing. Skips dot-entries. `depth: 0` is the current level only.                                                                                       | `depth`                    |
| `"CSV"`       | Sniffs the delimiter — `,` first, then `;`, tab, or pipe — and returns columns plus rows up to the token cap.                                                     | —                          |
| `"JSON"`      | Reads the file, applies the line range, then **validates the syntax** of what is left.                                                                            | `start_line`, `end_line`   |
| `"JSONL"`     | Same reader as `"JSON"`; per-line validation is chosen by the `.jsonl` **file extension**, not by this value.                                                     | `start_line`, `end_line`   |
| `"PDF"`       | Page-marked text extraction. Large documents are offloaded to RAG or chunk-summarized.                                                                            | —                          |
| `"DOCX"`      | Paragraph text. Rejects any file not ending in `.docx`. Tables, headers, and footnotes are not extracted.                                                         | —                          |

Any other value returns an error listing all eight. One thing to know: a partial
line range on a `.json` file fails validation, because the syntax check runs
_after_ the slice — half an object is not valid JSON.

**Limits.** Output is capped by `DOC_MAX_TOKENS` (`16384` in the shipped config;
`8192` when the key is absent) and truncation appends `[TRUNCATED - Content
exceeds token limit]`. PDF and DOCX offload to RAG above `RAG.MAX_TOKENS`
(`8192`), chunking at `RAG.CHUNK_TOKENS` (`1024`).

## Write and edit files

| Tool        | Parameters                                                                                                                                     | Purpose                                                           |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `fs_write`  | `path` **(required)**, `command` **(required)**, `file_text` = `""`, `old_str` = `""`, `new_str` = `""`, `insert_line` = `0`, `summary` = `""` | Create a file, or modify one via a four-command interface.        |
| `file_edit` | `file_path` **(required)**, `old_string` **(required)**, `new_string` **(required)**, `replace_all` = `False`                                  | Exact-string replacement, validated for existence and uniqueness. |

Both preserve the file's existing encoding, BOM, and line endings. Note the
parameter names differ: `fs_write` takes `path`, `file_edit` takes `file_path`.

### `fs_write` commands

| `command`       | Requires                 | Behavior                                                                                             |
| --------------- | ------------------------ | ---------------------------------------------------------------------------------------------------- |
| `"create"`      | `file_text`              | Creates parent directories and **overwrites** an existing file.                                      |
| `"str_replace"` | `old_str`, `new_str`     | Fails if the text is absent, and fails if it appears more than once. There is no `replace_all` here. |
| `"insert"`      | `insert_line`, `new_str` | Inserts **after** the given line. `0` prepends; the maximum is the file's line count.                |
| `"append"`      | `new_str`                | Appends, inserting a leading newline if the file doesn't end with one.                               |

### `file_edit` matching

`old_string` must match the file byte-for-byte. Two behaviors are worth knowing
because they change what you should paste:

- **Line-number gutters are tolerated.** If the match fails, the tool strips a
  leading `cat -n` gutter from `old_string` and retries — so text copied
  straight out of an `fs_read` result still matches. `new_string` is never
  altered.
- **Ambiguity is an error, not a guess.** More than one occurrence with
  `replace_all=False` fails and returns up to three sample locations with line
  numbers. `replace_all=True` replaces all of them; `False` replaces the first.

A replacement that would change nothing is also refused.

### Read-before-write

An existing file must be read before it can be modified. This is enforced
server-side, so it applies however the tool is driven, and it produces one of two
errors:

| Error field       | Meaning                               | Fix                                |
| ----------------- | ------------------------------------- | ---------------------------------- |
| `must_read_first` | The file was never read this session. | Read it, then retry.               |
| `stale_read`      | It changed on disk since it was read. | Read it again, re-derive the edit. |

Creating a _new_ file needs no prior read. A successful write counts as a read,
so consecutive edits to the same file don't trip the gate. The record is keyed by
resolved path and modification time, and lasts for the session.

## Search

| Tool          | Parameters                                                                                                                                                                                                                                            | Purpose                                      |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| `glob_search` | `pattern` **(required)**, `path` = `None`, `max_results` = `1000`, `sort_by_mtime` = `True`, `include_ignored` = `False`                                                                                                                              | Find files by name. Newest first by default. |
| `grep_search` | `pattern` **(required)**, `path` = `None`, `file_pattern` = `None`, `case_insensitive` = `False`, `output_mode` = `"files_with_matches"`, `context_lines` = `0`, `context_before` = `0`, `context_after` = `0`, `max_results` = `100`, `offset` = `0` | Search file **contents** via ripgrep.        |

`path` defaults to the current working directory. `max_results: 0` means
unlimited.

**`glob_search`** skips eleven noise directories unless `include_ignored=True`:
`.git`, `.idea`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `.tox`, `.venv`,
`__pycache__`, `build`, `dist`, `node_modules`. They are skipped **before** being
walked, so a pattern that names one explicitly (`node_modules/**/*.js`) still
works. Hidden files and directories need a pattern segment that starts with a dot
(`.*rc`), as with any glob. With `sort_by_mtime=True` it collects then sorts, so a
capped result really is the newest N — at the cost of scanning everything, up to a
ceiling of 100,000 matches. `sort_by_mtime=False` stops early and is faster.

The scan is **bounded at 30 seconds**: past that it returns the matches it has
with `truncated` and `timed_out` set, rather than running until the caller's own
timeout fires. Symlinked **directories** are not traversed — a link back to a
parent would make the walk endless — while symlinks to files still match.

**`grep_search`** takes three `output_mode` values: `"files_with_matches"`
(default), `"content"`, and `"count"`. Context flags apply **only** in `content`
mode, where `context_before`/`context_after` override the symmetric
`context_lines`.

!!! note "`max_results` counts different things per mode"

    In `content` mode it caps **matches** (context lines ride along, extra). In
    `files_with_matches` and `count` mode it caps **files**. `offset` pages
    through a stable order in both cases — results are sorted by path so paging
    is meaningful.

**Limits.** Lines are truncated at 500 characters with a `… [+N chars]` marker.
The ripgrep subprocess times out after 30 seconds. ripgrep is **required** — it
is not optional and there is no fallback; without it `grep_search` returns an
install hint.

## Run commands

| Tool           | Parameters                                 | Purpose                                                         |
| -------------- | ------------------------------------------ | --------------------------------------------------------------- |
| `execute_bash` | `command` **(required)**, `timeout` = `30` | Run a shell command, returning stdout, stderr, and exit status. |

Commands run under **bash** (not `/bin/sh`, which is dash on some systems), so
`[[ … ]]`, arrays, `source` and process substitution behave as written.

The shell also **remembers where it is**: the directory a command ends in becomes
the starting directory of the next one, so a `cd` carries over between calls like
in an interactive session, and the directory in force is returned as `cwd`
alongside the output. A command killed by its timeout doesn't move it (it never
finished anywhere), and a tracked directory that later disappears falls back to the
directory mnemoai was launched from.

The command runs in its own process group, so a timeout kills the whole tree
rather than just the shell, and its stdin is closed — a command that waits for
input fails fast instead of burning the whole timeout. Output is capped at 30,000
characters, middle-truncated to keep the beginning and the end.

Every command passes the [safety floor](../guides/safety.md#understand-the-safety-floor)
first, and — unless you turn it off — asks you to confirm.

## Run work in the background

Seven tools, for commands too slow to wait on.

| Tool                     | Parameters                                                                 | Purpose                                                            |
| ------------------------ | -------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `start_background_task`  | `command` **(required)**, `description` = `""`, `working_directory` = `""` | Launch a detached command; returns an 8-character task id.         |
| `get_task_status`        | `task_id` **(required)**                                                   | Status, timings, return code, log path.                            |
| `get_task_output`        | `task_id` **(required)**, `tail_lines` = `50`                              | Captured output. `0` or less returns the whole log.                |
| `list_background_tasks`  | `include_completed` = `True`                                               | All tasks, newest first, plus a summary count.                     |
| `cancel_background_task` | `task_id` **(required)**                                                   | Kills the process group. Only works on a `running` task.           |
| `wait_for_task`          | `task_id` **(required)**, `timeout_seconds` = `300`                        | Blocks until the task finishes.                                    |
| `clear_completed_tasks`  | _(none)_                                                                   | Drops finished tasks from the registry **and deletes their logs**. |

Statuses are `pending`, `running`, `completed`, `failed`, `error`, and
`cancelled`. Output goes to `~/.mnemoai/tasks/<task_id>.log`; logs older than
**7 days** are swept at startup. The working directory is checked against the
same path policy as a file write, and with none given a task starts in the shell's
current directory — wherever the last `execute_bash` left off, not necessarily
where mnemoai was launched.

## Track multi-step work

| Tool         | Parameters                                                | Purpose                                    |
| ------------ | --------------------------------------------------------- | ------------------------------------------ |
| `todo_write` | `todos` **(required, JSON array)**, `scope` = `"default"` | Validate and replace the list for a scope. |
| `todo_read`  | `scope` = `"default"`                                     | Read the current list.                     |
| `todo_clear` | `scope` = `"default"`                                     | Delete the list.                           |

Each item needs all three keys:

```json
{
  "content": "Task description (imperative form)",
  "status": "pending|in_progress|completed",
  "activeForm": "Running tests"
}
```

`status` must be one of `pending`, `in_progress`, `completed`. Exactly one item
should be `in_progress`; more than one logs a warning but is not rejected.

`scope` lets separate workstreams keep separate lists. It is sanitized to
`[A-Za-z0-9_-]` and truncated to 64 characters, so two labels differing only in
punctuation can collide. Lists are also namespaced per running instance, so two
terminal tabs don't overwrite each other — and `todo_read` must use the same
`scope` string as the write.

## Work with git

| Tool              | Parameters                                                                                                                                                  | Purpose                                          |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `git_safe`        | `command` **(required, no `git` prefix)**, `allow_dangerous` = `False`, `reason` = `""`                                                                     | Run a git command through the safety classifier. |
| `git_status_safe` | _(none)_                                                                                                                                                    | Branch, change count, ahead/behind, warnings.    |
| `git_commit_safe` | `message` **(required)**, `add_all` = `False`, `add_files` = `""`, `amend` = `False`, `allow_empty` = `False`, `allow_dangerous` = `False`, `reason` = `""` | Stage and commit, with amend-safety checks.      |

`allow_dangerous=True` always requires a `reason`. Arguments are split with
`shlex`, so quoted commit messages survive intact. See [Keep git operations
recoverable](../guides/safety.md#keep-git-operations-recoverable) for what is
blocked outright versus what asks first.

## Ask, plan, and delegate

| Tool                | Parameters                                                                                               | Purpose                                                              |
| ------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `ask_user_question` | `question` **(required)**, `options` **(required)**                                                      | Hand a decision back to you as a picker.                             |
| `exit_plan_mode`    | `plan` **(required, markdown)**, `allowed_bash` = `None`                                                 | Present a finished plan for approval and leave plan mode.            |
| `spawn_agent`       | `agent_type` **(required)**, `prompt` **(required)**, `description` = `""`, `run_in_background` = `True` | Delegate a self-contained task to a fresh sub-agent.                 |
| `resume_agent`      | `agent_id` **(required)**, `prompt` **(required)**, `run_in_background` = `True`                         | Continue a previous sub-agent with a follow-up, keeping its context. |

The `ask_user_question` picker always offers more than the options it was given:
a free-text **note** field (Tab to reach it, Enter to submit from either field)
and a final **"None of these — let's talk about it"** row. The note rides along
with whichever option you pick, so "that one, but only for local runs" is a
single answer. The three ways out mean three different things:

| You do this          | The model is told                                   |
| -------------------- | --------------------------------------------------- |
| Pick an option       | Proceed on it; any note qualifies the choice.       |
| Pick _None of these_ | Don't act on any option — answer in prose instead.  |
| Press Esc            | Decide for itself and say which assumption it made. |

So dismissing the picker is not the way to disagree with every option — the
escape row is. Off a TTY the same three outcomes are reachable: the numbered
rows are printed, Enter alone dismisses, and a second prompt takes the note.

`agent_type` is `general-purpose` (full toolset), `explore` (read-only), or
`plan` (read-only) — plus any custom agent in `~/.mnemoai/agents/`. Entries in
`allowed_bash` become pre-approved commands, matched by prefix, so an approved
plan's `pytest` also covers `pytest tests/unit`.

These four are **handled by the client**, not the tool server — they need your
terminal. Concurrency is capped by `LLM.SUBAGENT_MAX_CONCURRENCY` (`4`).

## Conditional tools

These register only when their feature is on. A tool that isn't registered is
invisible to the model.

| Tool                                                       | Enabled by                                               | Default |
| ---------------------------------------------------------- | -------------------------------------------------------- | ------- |
| `memory`                                                   | `ENABLE_MEMORY`                                          | **on**  |
| `use_skill`                                                | `ENABLE_SKILLS`                                          | **on**  |
| `list_documents`, `search_in_documents`, `clear_documents` | `ENABLE_RAG`                                             | **off** |
| `web_search`                                               | `ENABLE_WEB_SEARCH` **and** a `BRAVE_API_KEY`            | off     |
| `web_crawler`                                              | `ENABLE_WEB_CRAWL`                                       | off     |
| `describe_image`                                           | a working `VISION_MODEL_ID` — there is no `ENABLE_` flag | off     |

The defaults above are what the **code** does when a key is absent from
`config.yaml`. The setup wizard writes a different, more generous set — see the
toggle table in [Usage](../guides/usage.md).

| Tool                  | Parameters                                                                                                                   | Purpose                                                    |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `memory`              | `action` **(required)**, `text` = `""`, `old_text` = `""`                                                                    | Maintain `MEMORY.md`: `add`, `replace`, or `remove`.       |
| `use_skill`           | `name` **(required)**, `arguments` = `""`                                                                                    | Load a skill's full instructions into context.             |
| `search_in_documents` | `query` **(required)**, `top_k` = `8`                                                                                        | Hybrid semantic + keyword search over indexed documents.   |
| `list_documents`      | _(none)_                                                                                                                     | Indexed document ids and chunk counts.                     |
| `clear_documents`     | _(none)_                                                                                                                     | Drop all vectors and the keyword index.                    |
| `web_search`          | `query` **(required)**, `search_lang` = `"en"`, `num_results` = `10`, `freshness` = `""`, `country` = `""`, `ui_lang` = `""` | Brave Search results.                                      |
| `web_crawler`         | `url` **(required)**                                                                                                         | Fetch a page as markdown, or index it into RAG when large. |
| `describe_image`      | `image_path` **(required)**, `question` = `"Describe this image in detail."`                                                 | Answer a question about an image.                          |

Notes on the ones with sharp edges:

- **`memory`** replaces and removes by _substring_: `old_text` need only
  uniquely identify one entry. Ambiguity is an error. `MEMORY.MAX_CHARS` (`2200`)
  is a hard cap — an add that would overflow it fails and asks for consolidation.
- **`web_search`** clamps `num_results` to **1–20**. `freshness` accepts `pd`,
  `pw`, `pm`, `py`, or a `YYYY-MM-DDtoYYYY-MM-DD` range; it, `country`, and
  `ui_lang` are only sent when non-empty. Safe search is off.
- **`web_crawler`** checks the URL against the [SSRF
  policy](../guides/safety.md#understand-the-safety-floor) _before_ fetching.
  Large pages return no `content` at all — instead you get a chunk count and a
  pointer to `search_in_documents`. Inline results are capped at 100,000
  characters; the page timeout is `WEB_CRAWL.PAGE_TIMEOUT_MS` (`60000`).
- **`describe_image`** accepts `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.webp`.
- **`use_skill`** substitutes `$ARGUMENTS` and `${SKILL_DIR}` into the skill body.

## See also

- [Control what Mnemo AI can do](../guides/safety.md): the confirmation prompt, plan mode, and what is refused outright
- [Configuration](../configuration.md): every key these tools read, including `DOC_MAX_TOKENS`, `RAG`, and `WEB_CRAWL`
- [The `~/.mnemoai` directory](../getting-started/mnemoai-directory.md): where memory, plans, skills, and task logs are written
- [Orchestration & sub-agents](../guides/orchestration.md): choosing an agent type and writing a custom one
- [Everyday tools](../guides/productivity-tools.md): the same tools framed as tasks rather than signatures
