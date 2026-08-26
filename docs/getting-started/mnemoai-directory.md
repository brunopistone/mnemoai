# The `~/.mnemoai` directory

Everything Mnemo AI keeps between sessions lives in one directory. This page is
the map: what each file is for, which ones you edit by hand, and which ones the
assistant writes for itself.

Set `MNEMOAI_HOME` to move the whole tree somewhere else.

## Choose the right file

Start here if you know what you want to change but not where.

| You want to                                 | Edit                        | Scope               | Reference                                                           |
| ------------------------------------------- | --------------------------- | ------------------- | ------------------------------------------------------------------- |
| Change model, provider, or a feature toggle | `config/config.yaml`        | Everything          | [Configuration](../configuration.md)                                |
| Change how the assistant behaves or speaks  | `config/prompts.yaml`       | Everything          | [Prompts](../configuration.md#prompts-promptsyaml)                  |
| State a rule for **one project**            | `./STEERING.md` in the repo | That directory tree | [Steering](../guides/memory.md#steering-steeringmd)                 |
| State a rule for **every** project          | `STEERING.md` (this folder) | Everything          | [Steering](../guides/memory.md#steering-steeringmd)                 |
| Record a durable fact about you             | `<profile>/MEMORY.md`       | One profile         | [Persistent memory](../guides/memory.md#persistent-memory-memorymd) |
| Add a repeatable procedure                  | `skills/<name>/SKILL.md`    | Everything          | [Agent skills](../guides/agent-skills.md)                           |
| Define a custom sub-agent                   | `agents/<name>.md`          | Everything          | [Sub-agents](../guides/orchestration.md)                            |
| Connect an external tool server             | `mcp/mcp.json`              | Everything          | [External MCP servers](../guides/mcp.md)                            |
| Run a command around every tool call        | `hooks/hooks.json`          | Everything          | [Tool hooks](../guides/hooks.md)                                    |

For the two steering rows, `CLAUDE.md` is read as an equivalent filename, so a repo
that already keeps its instructions under that name needs no second file. Within a
single directory `STEERING.md` wins and its sibling `CLAUDE.md` is skipped — which is
what lets one repo hold both and give this assistant different instructions. The
global tier is **this folder only** (`STEERING.md`, else `CLAUDE.md`): no other
tool's instructions file is picked up as your always-on rules.

## The tree

```text
~/.mnemoai/
├── config/
│   ├── config.yaml           # ← you edit (or use /config, /model, /params, /features)
│   └── prompts.yaml          # ← you edit; model-facing prompts
├── mcp/
│   └── mcp.json              # ← you edit; external MCP servers
├── hooks/
│   └── hooks.json            # ← you edit; commands run around tool calls
├── STEERING.md               # ← you edit; global project rules (or CLAUDE.md)
├── skills/
│   └── <name>/SKILL.md       # ← you edit; on-demand procedures
├── agents/
│   └── <name>.md             # ← you edit; custom sub-agent definitions
├── plans/
│   └── plan_<ts>.md          # written on plan approval
├── tasks/
│   └── <id>.log              # background-task output
├── logs/
│   ├── mcp.log               # MCP subprocess stderr (survives restarts)
│   └── mcp.log.1             # one rotated generation
└── <profile>/                # "default" unless PROFILE.NAME is set
    ├── MEMORY.md             # ← you may edit; curated persistent memory
    ├── <profile>.json        # learned user profile (do not hand-edit)
    ├── sessions/
    │   └── <sanitized-cwd>/  # append-only JSONL transcripts, per directory
    ├── conversations/        # /save and /load targets
    └── models/
        └── <model-name>/
            ├── episodic_memory/   # vector store of past turns
            └── playbook/          # playbook.json + metrics.json
```

## What you edit vs. what the assistant writes

The distinction matters: hand-editing a learned file is usually pointless,
because the next turn overwrites it.

| Path                                                        | Written by                           | Safe to hand-edit?                          |
| ----------------------------------------------------------- | ------------------------------------ | ------------------------------------------- |
| `config/config.yaml`, `config/prompts.yaml`, `mcp/mcp.json` | You                                  | **Yes** — this is the intended interface    |
| `hooks/hooks.json`                                          | You                                  | **Yes** — read at startup, so restart after |
| `STEERING.md` (or `CLAUDE.md`), `skills/`, `agents/`        | You                                  | **Yes**                                     |
| `<profile>/MEMORY.md`                                       | The assistant, via the `memory` tool | Yes — it's Markdown, and `/memory` shows it |
| `<profile>/<profile>.json`                                  | The assistant, every turn            | No — EMAs and counters are recomputed       |
| `models/*/episodic_memory/`, `models/*/playbook/`           | The assistant, every turn            | No — delete the directory to reset instead  |
| `sessions/`, `plans/`, `tasks/`, `logs/`                    | The runtime                          | No — these are records, not inputs          |

## Why memory is scoped per model

`episodic_memory/` and `playbook/` sit under `models/<model-name>/` because both
are keyed to the embedding model that produced their vectors. Switching model
starts a fresh store rather than searching vectors from a different embedding
space, which would return nonsense. `MEMORY.md` and `STEERING.md` (or
`CLAUDE.md`) are plain text and so are shared across models.

## What is cleaned up automatically

At startup, Mnemo AI sweeps its own records so the directory doesn't grow
forever:

| Path                          | Retention                            | Setting                |
| ----------------------------- | ------------------------------------ | ---------------------- |
| `<profile>/sessions/`         | 30 days                              | `SESSION_MAX_AGE_DAYS` |
| `plans/`                      | 7 days                               | —                      |
| `tasks/`                      | 7 days                               | —                      |
| RAG stores under `<profile>/` | 7 days                               | —                      |
| `logs/mcp.log`                | one rotated generation (`mcp.log.1`) | —                      |

`conversations/` (from `/save`) is **never** swept — an explicitly saved
conversation is yours until you delete it.

## Profiles

Set `PROFILE.NAME` in `config.yaml` to keep separate memory, sessions, and
learned state under one install:

```yaml
PROFILE:
  NAME: work
```

Everything above the `<profile>/` line in the tree is shared; everything inside
it is per-profile. Sessions are additionally partitioned by working directory, so
resuming in one repo never offers you another repo's history.

## See also

- [Configuration](../configuration.md): every key, with defaults
- [Memory & learning](../guides/memory.md): what each memory kind is for
- [Troubleshooting](../development/troubleshooting.md): permission and startup failures
