# Long-running work { #long-running-tasks }

MnemoAI keeps the active request and your follow-up guidance while it works.
Compaction preserves completed work and the next action; it does not end the task.
A status question can receive a brief answer while the original work continues.
Use **Esc** to cancel the running turn.

By default, the 200-step limit is an execution checkpoint. The assistant continues
when completed tool activity has changed, and stops when activity repeats without
new results. Completed tools are not replayed when execution resumes.
Delegated workers can also continue and summarize their own context without
changing the parent conversation. These summaries use the configured summary
model and are included in usage accounting.

For a strict step limit, set the following in your configuration and restart:

```yaml
LLM:
  RECURSION_LIMIT: 200
  AUTO_CONTINUE_TASKS: false
```

`AUTO_CONTINUE_TASKS` defaults to `true` when omitted. Cancellation and explicit
peer-review budgets remain enforced in either mode. A stopped or incomplete task
is labelled accordingly rather than shown as `done`.

**Supervisor peer review:** `/review on|off|last` controls a bounded
reviewer → chat-model correction → recheck dialogue without changing permissions.
`/config review` edits its shared budgets and per-review limits.
Existing `/review` user macros retain priority; use `/config review on|off|last`
for the built-in controls in that case. See [Peer review](../guides/peer-review.md).
