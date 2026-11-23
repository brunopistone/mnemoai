# Models for specific tasks { #per-area-models-area_models }

`REVIEWER` selects the optional coding/document supervisor model. It exchanges feedback with the
chat model, which alone executes authorized corrections. Enable it for
this session with `/review on`, or set its startup default in `/features`.
Choose its provider/model with `/model` → Reviewer and tune an explicit override
with `/params`. `/config review` edits `REVIEW.TIMEOUT` (45 seconds by default)
and `REVIEW.MAX_INPUT_TOKENS` (6000 estimated input tokens), plus shared
`REVIEW.MAX_STRATEGY_ROUNDS` (1 strategy revision), `REVIEW.MAX_CHANGE_REVIEWS`
(2 intermediate checks), `REVIEW.MAX_ROUNDS` (2 final corrections; 0 means
report-only final review) and `REVIEW.TOTAL_TIMEOUT` (180 seconds of total
supervision overhead, excluding normal implementation), without restarting.
No reviewer override means a separate main-model invocation; an explicit build
failure is inconclusive, not silent model fallback or permission approval.
Unavailable review produces one notice per outage and does not block authorized
work; later tasks can retry without repeated warnings.
See [Peer review](../guides/peer-review.md) for scope, limits and macro compatibility.

For learning configuration, use `/features` to enable/disable the playbook,
or `/config playbook` to toggle it and tune capacity/candidate limits in place.
Learning is local-only and enabled by default (`ENABLE_PLAYBOOK: true`); an
explicit `false` disables it. It records observed tool outcomes, not generated
lessons, and makes no LLM or embedding requests. There is no reflector model,
timeout setting or LLM-learning mode. Old `AREA_MODELS.REFLECTOR`,
`PLAYBOOK.REFLECTION_TIMEOUT`, and `PLAYBOOK.SIMILARITY_THRESHOLD` keys are
accepted but ignored, so existing configs still load without restoring model
reflection. Peer review remains a separate opt-in feature.
See [Inspect and control learning](../guides/memory.md#inspect-and-control-learning).

A turn is not one model call. Before the answer is streamed, the query is
**classified** by the router and — with orchestration on — **split into subtasks**
by the decomposer; afterwards a **summarizer** may compact the history. Those
calls are short, structured and invisible, and they don't have to run on the model
that writes the answer.

`AREA_MODELS` gives each of them its own model. Nothing here is required: an area
you don't list uses `MODEL_ID`, which is what happens today.

When an override changes provider, inherited connection/authentication fields
and `EXTRA_PARAMS` are cleared before applying the override. A local auxiliary
model therefore cannot inherit the chat model's remote endpoint or API key.
Set connection fields explicitly for the new provider. Within one provider,
`null` or `""` on a connection field clears its inherited value.
Internal non-reasoning variants also remove reasoning overrides from nested
`EXTRA_PARAMS`; the original model configuration is not mutated.

```yaml
AREA_MODELS:
  ROUTER: qwen3.5:1.7b # shorthand — just a model name
  ORCHESTRATOR: qwen3.5:32b
  SUMMARY:
    NAME: qwen3.5:7b
    TEMPERATURE: 0.3
```

**You don't have to write it by hand.** The first-run configurator offers a
router model when `ENABLE_ROUTING` is on and an orchestrator model when
`ENABLE_ORCHESTRATION` is on, and `/model` lists all areas beside the chat,
vision and embeddings sections (`/params` tunes an area you've already
configured). Each starts with the same question — _use the same model as chat?_ —
and answering yes writes **nothing**, because that is exactly what "no entry"
means: the area follows `MODEL_ID`, including whenever you change it later.
Answering no runs the ordinary provider flow, so an area can be set up on any
provider the chat model could use. Picking an area whose feature is switched off
offers to enable it first (compaction always runs, so `SUMMARY` is never gated).

| Area           | The call it covers                                     | Why change it                                                                                     |
| -------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `ROUTER`       | Query classification (which tools to bind this turn)   | It produces a one-word label. A large reasoning model spends real latency on it — on _every_ turn |
| `ORCHESTRATOR` | Task decomposition (splitting a request into subtasks) | The opposite trade-off: a bad split wastes the whole task, so this one may deserve a bigger model |
| `SUMMARY`      | Conversation compaction                                | A long, mechanical rewrite — throughput matters more than depth                                   |

Each entry is a **partial `MODEL_ID`, merged over the main one**, so you only
write what differs. That includes `TYPE`: an area can run on a different provider
entirely — a local model classifying while the answer comes from a hosted one.

```yaml
AREA_MODELS:
  ROUTER:
    TYPE: ollama
    NAME: qwen3.5:1.7b
    TEMPERATURE: 0
```

Notes:

- **A misconfigured area falls back to the main model.** An unreachable endpoint
  or a bad model name is logged once at startup and that area keeps using
  `MODEL_ID` — a side model must not break your turns.
- **On one local server, an area model has to be served too.** Ollama loads a
  model on demand, so naming a second one just works. A single-process server
  (MLX, `llama-server`, vLLM) only answers for the models it was launched with, so
  an unserved name is the fallback case above — the area silently keeps using
  `MODEL_ID`, and the only sign is the one startup line. Launch the server with
  every model you name (`--config <file>` for `mlx-openai-server`), and keep in
  mind that two resident models share the machine's RAM and that requests to one
  server queue behind each other, so a "cheaper" router can cost wall-clock time
  rather than save it.
- **`/doctor` lists what's active**, one row per configured area, and warns about
  an area name it doesn't recognize (a typo is otherwise silent — the area simply
  keeps the main model).
- **`/usage` attributes each call to the model that ran it**, so a smaller router
  shows up as its own row.
- **Changing one applies without restarting**, from either command: `/params` for
  its inference knobs, `/model` for the model itself. Changing the chat, vision or
  embeddings model restarts the app and automatically restores the conversation. (The one
  exception: if that same edit also switches routing or orchestration **on**, the
  app restarts, because the router and the orchestrator are built at startup.)
- `LLM.SUMMARIZATION_THINK: true` takes precedence over the `SUMMARY` area: it
  means "summarize with the full model, thinking and all".
- The **aggregator is deliberately not an area.** Its output is the answer you
  read, so a different model there would change the voice of the reply rather than
  the cost of an internal step.
