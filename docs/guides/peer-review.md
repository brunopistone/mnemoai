# Peer review

Peer review is an optional supervisor dialogue. The chat model does the work;
a reviewer examines it and sends findings or questions back. The chat model
can correct a supported problem, run authorized checks, or explain counterevidence.
The reviewer then rechecks the updated result. You do not need to relay their
messages manually.

Only the chat model executes tools or edits files. The reviewer grants no
permission. The loop ends with **pass**, **revise** (unresolved findings), or
**inconclusive**, and stops at its shared round/time/step limits. Automatic
pre-execution plan review and per-tool/per-token supervision are not implemented.

## Use it

```text
/review on
Implement the change and run the relevant checks.
/review last
/review off
```

It is off by default. When on, completion of a nonempty user request starts the
dialogue, whether execution was direct, atomic, or orchestrated. Corrections use
the foreground chat model and its normal tools. A background-result delivery
without a new user request is explicitly unreviewed.

**Existing `/review` macros keep working.** If you have a user-authored
`commands/review.md`, that macro retains priority. Use the always-available
aliases `/config review on`, `/config review off`, and `/config review last`
for peer-review controls. No command file is renamed or removed.

The first answer streams, followed by labelled reviewer feedback and any chat-model
corrections. `/review last` shows the exchange and its unresolved findings.
Review feedback is labelled as automated advice, not a user request or an approval.
It remains in the task's history so follow-ups make sense, but it does not create
additional user turns or replace the original prompt in file provenance.
Reviewer suggestions have **not** been executed until actual tool evidence says so.

## Configuration

| Change                                    | Interactive path                                                 |
| ----------------------------------------- | ---------------------------------------------------------------- |
| Enable/disable for this session           | `/review on` / `/review off`                                     |
| Startup default                           | `/features` → Peer review                                        |
| Reviewer model/provider                   | `/model` → Reviewer                                              |
| Reviewer generation parameters            | `/params` → Reviewer, after selecting an explicit model override |
| Time and input limits, without restarting | `/config review`                                                 |
| Configuration diagnostics                 | `/doctor`                                                        |

Full `/config` setup also offers reviewer selection and limits.
Selecting or enabling the reviewer through `/model` preserves the conversation.

```yaml
ENABLE_REVIEW: false
REVIEW:
  TIMEOUT: 45
  MAX_INPUT_TOKENS: 6000
  MAX_ROUNDS: 2
  TOTAL_TIMEOUT: 180
# Optional, using the normal partial-model configuration:
# AREA_MODELS:
#   REVIEWER:
#     TYPE: ollama
#     NAME: your-reviewer-model
#     MAX_TOKENS: 2048
```

`TIMEOUT` is 1–120 seconds per reviewer call, including evidence preparation,
model initialization and the response. `MAX_ROUNDS` allows 0–4 chat-model
correction rounds (default 2); zero retains one-shot, report-only review.
There can be at most one initial review plus one recheck per correction round.
`TOTAL_TIMEOUT` is a shared 1–1800-second supervision budget (default 180),
starting after the original actor execution. Retries and rounds never reset it.
Additional actor steps also consume what remains of the original agent step
budget. A started tool or approval dialog may finish after the deadline; no
new action starts afterward. Expiry is not rollback and never abandons an
editing actor on a detached thread.

`MAX_INPUT_TOKENS` is a conservative estimate (1000–32000) per reviewer request,
not a provider billing quota. The reviewer's output
limit defaults to 2048 tokens; an explicit reviewer `MAX_TOKENS` overrides it.
Actual reported usage is attributed to the model in `/usage`; there are no
invented dollar totals.

Without an override, the reviewer uses a separate, callback-free instance of
the chat model, with reasoning disabled where supported. That is a separate
context, not an independent model. An explicit reviewer that fails to initialize
is reported unavailable, never silently replaced by a supposedly different judge.

## Evidence and limits

The reviewer receives the current request, bounded recent visible conversation
and steering, delivered mid-turn instructions, captured tool outcomes, current
named-file snapshots, and a bounded workspace diff relative to Git HEAD when
available. Before/after diff phases distinguish pre-existing dirty changes from
new changes. Foreground and joined orchestrator/spawned-worker calls share the
capture; detached workers cannot contaminate a later task's review.

Automatic reviewer evidence inspection stays inside the starting working directory and
excludes symlinks, nonregular/binary files, and credential-like paths. Git
inspection disables external diff/text conversion, hooks, optional index writes
and lazy fetching. Shell/external changes to untracked or out-of-directory files
are not fully enumerated. This is not a whole-filesystem audit.

Capture is bounded: up to 24 tool observations and 12 named files, limited text
excerpts, and 12 KB per workspace diff. Missing/truncated evidence is disclosed;
a proposed pass becomes inconclusive when coverage has gaps. A failed shell
command cannot become a pass without an observed successful rerun of that command
in the same directory. This does not prove that the chosen checks are sufficient.

Findings must cite supplied evidence IDs. Malformed output, invented IDs, tool
requests, provider errors, cancellation and timeout produce an inconclusive
report. An actionable, well-formed inconclusive finding can be sent to the chat
model for clarification; unavailable or stale review is not acted on. A timeout
may leave a reviewer provider request running, but it cannot publish a late
verdict or start an accumulation of reviewer requests. Ordinary chat remains
available; turn review off to continue without it.

Corrective rounds do not launch detached jobs. Foreground sub-agents can run
within the same budget and their existing restrictions. Normal background-task
support in the original task is unchanged. If an additional correction exceeds
the model context, supervision stops without wiping history; `/compact` and a
new user request remain available.

Reports are tied to the inspected file/diff revisions. `/review last` marks a
changed artifact as historical/inconclusive. A new reviewed task, model/settings
reload, clear, rewind, load, branch or resume clears the current review state.

Bounded round reports and evidence fingerprints are recorded alongside the session
when session recording is enabled. The separate audit record is not replayed as
instructions. Explicitly labelled automated feedback in the conversation is
kept as context, not counted as something the user requested.
Known credential fields and common token formats are redacted, but redaction is
not a comprehensive secret detector. Enabling review sends the supplied evidence
to the selected reviewer provider; use an appropriate model and scope.

## Permissions are separate

Review never changes `/auto`, plan mode, hooks, session trust, approval prompts or
the server safety floor. The foreground actor retains its approval channel.
Reviewers never ask for approval because they execute no tools. The chat model
may ask through its existing foreground approval channel. A pass means
only “no blocking finding in the inspected scope”—not proof of correctness,
authorization, or successful completion of a suggested check.
