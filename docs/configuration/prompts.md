# Prompt templates { #prompts-promptsyaml }

All model-facing prompts live in **`prompts.yaml`** — a sibling of `config.yaml` in the same `config/` directory, kept separate from settings. `config.yaml` is never consulted for prompts — prompt keys left there are ignored with a migration warning.

Resolution order (first match wins): `$MNEMOAI_PROMPTS` → `~/.mnemoai/config/prompts.yaml` → the bundled package defaults.

| Prompt key              | Purpose                                      | Required?                            |
| ----------------------- | -------------------------------------------- | ------------------------------------ |
| `SYSTEM_PROMPT`         | Core identity, tool-usage rules, behavior    | Always                               |
| `SUMMARY_SYSTEM_PROMPT` | System prompt used during context compaction | Always                               |
| `SUMMARY_TASK_PROMPT`   | Task instructions for summarization          | Always                               |
| `ROUTING_PROMPT`        | Query classifier prompt                      | Only if `ENABLE_ROUTING: true`       |
| `ORCHESTRATOR_PROMPT`   | Task-decomposition prompt                    | Only if `ENABLE_ORCHESTRATION: true` |
| `AGGREGATOR_PROMPT`     | Worker-result synthesis prompt               | Only if `ENABLE_ORCHESTRATION: true` |

A missing _required_ prompt is a hard startup error (there are no in-code fallbacks) — copy the default from the bundled `prompts.yaml`. Customize `SYSTEM_PROMPT` to change the assistant's personality, instructions, and tool-usage patterns. Key sections in the default prompt:

- `<identity>`: Basic identity and core principles
- `<reasoning_discipline>`: Thinking rules and loop detection
- `<output_format>`: Response formatting requirements
- `<information_sources>`: RAG vs web vs internal knowledge decision tree
- `<file_operations>`: Read/write/edit workflow rules
- `<search_tools>`: Glob and grep usage guidance
- `<git_operations>`: Git safety rules
- `<task_management>`: Todo, plan mode, and background task rules
- `<error_handling>`: Error response guidelines
- `<communication>`: Style and security rules
