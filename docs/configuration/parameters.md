# Model parameters { #model-parameters }

After `/model`, the confirmation names the selected model without repeating
credential instructions or the config-file path. An in-place update reports
“changed to”; a change requiring the existing restart reports “saved” and
“Restarting to apply…”. A short `/params` hint appears only when configured
inference parameters were reset. Switching an area back to the chat model is
reported as “now follows Chat”. `/doctor` and the application
log retain configuration-path diagnostics.

When a model change requires a restart, MnemoAI restores the current conversation,
including unsaved replies, the compaction summary, label and unfinished input.
If you used `/load`, a later `/save` still targets that file; switching models
does not overwrite it. An empty chat remains empty. Queued messages and pending
agent reports are handled before the restart.

Restoration uses a private temporary file under `~/.mnemoai/restarts` (or
`$MNEMOAI_HOME/restarts`), including when automatic session recording is disabled.
The file is removed after successful restoration. If creating it fails, the
current chat stays open with its existing model; the new settings are saved.
If the new process cannot start or restore, the file is retained for recovery
and the app shows its path. A restore failure offers the session picker.

Auto-approval tier and plan mode carry across this internal restart. Individual
tool approvals reset, and context tokens are estimated again until the new
model reports usage. `/usage` starts a new process total, as on manual resume.
Restoring the conversation itself makes no model or embedding request.

`/params` applies chat and auxiliary-chat changes in place. Vision and embedding
models are also cached by the MCP server: their edits are saved, and the command
explicitly asks you to restart to apply them. It does not silently restart and
discard your current conversation, or claim those saved values are already live.

This is the full reference for what you can put under `MODEL_ID`,
`VISION_MODEL_ID`, and `RAG.EMBED_MODEL_ID`. Only `NAME` and `TYPE` are
required; everything else is optional and omitted keys fall back to the
documented MnemoAI default or the provider/model default. The interactive configurator (`/config`, `/model`)
sets the common ones — use this reference to hand-tune `config.yaml` for
anything else a provider or model supports.

## Identity, connection & auth { #identity-connection-auth }

| Parameter      | Applies to `TYPE`                       | Description                                                                                                                                                                    |
| -------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `NAME`         | all (**required**)                      | Model id / Ollama model / Bedrock model id / Mantle bare id / SageMaker endpoint name / MLX served name                                                                        |
| `TYPE`         | all (**required**)                      | `ollama`, `bedrock`, `mantle`, `openai`, `anthropic`, `sagemaker`, `litellm`, `mlx` (embeddings: the same list minus `mantle`/`anthropic`)                                     |
| `HOST`         | `ollama`, `mlx`                         | Server host (default `localhost` for ollama, `127.0.0.1` for mlx)                                                                                                              |
| `PORT`         | `ollama`, `mlx`                         | Server port (default `11434` for ollama, `8000` for mlx)                                                                                                                       |
| `REGION`       | `bedrock`, `mantle`, `sagemaker`        | AWS region (default `us-east-1`)                                                                                                                                               |
| `API_PROTOCOL` | `mantle`                                | `chat_completions` (default), `responses`, or `anthropic`                                                                                                                      |
| `ENDPOINT_URL` | `bedrock`, `mantle`, `anthropic`        | Override the default endpoint URL (Anthropic: custom base URL)                                                                                                                 |
| `API_KEY`      | `mantle`, `anthropic`, `litellm`, `mlx` | Mantle: Bedrock API key (else `BEDROCK_API_KEY` env / minted token). Anthropic: else `ANTHROPIC_API_KEY` env. LiteLLM: provider key. MLX: optional — a local server needs none |
| `API_BASE`     | `litellm`, `mlx`                        | API base URL — LiteLLM proxy, or an MLX server behind a proxy/path prefix (wins over `HOST`/`PORT`)                                                                            |
| `INPUT_FORMAT` | `sagemaker`                             | `openai_chat` (default) or `huggingface`                                                                                                                                       |

> Standard Bedrock also reads the `AWS_BEARER_TOKEN_BEDROCK` env var, and all AWS
> providers honor `AWS_PROFILE` — see the API-key/profile notes under Amazon Bedrock.

## Inference parameters { #inference-parameters }

Optional generation and runtime settings. The **Honored by** column lists the providers that
actually send each one (others ignore it). These apply to `MODEL_ID` and
`VISION_MODEL_ID`; **`EMBED_MODEL_ID` takes no generation params** (embeddings
only use `NAME`/`TYPE` + connection, plus `DIMENSION`, and `KEEP_ALIVE` on
`mlx`, or `NUM_BATCH` on `ollama`).

This table is derived from `models/provider_params.py` — the single source of
truth that the controllers build their client kwargs from — so it reflects
exactly what each provider's init path forwards. (`mantle` reads
`TEMPERATURE`/`MAX_TOKENS`/`TOP_P` via the Mantle factory.)

| Parameter            | Description                                                                                                    | Honored by (`MODEL_ID`)                                             |
| -------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `MAX_TOKENS`         | Max output tokens to generate                                                                                  | ollama, bedrock, mantle, openai, anthropic, sagemaker, litellm, mlx |
| `NUM_BATCH`          | Positive integer: maximum tokens processed per prompt-evaluation batch; default `2048`                         | ollama (chat, vision and embeddings)                                |
| `TEMPERATURE`        | Sampling temperature                                                                                           | ollama, bedrock, mantle, openai, anthropic, sagemaker, litellm, mlx |
| `TOP_P`              | Top-p (nucleus) sampling                                                                                       | ollama, bedrock, mantle, openai, anthropic, sagemaker, litellm, mlx |
| `TOP_K`              | Top-k sampling                                                                                                 | ollama, anthropic, sagemaker, mlx                                   |
| `MIN_P`              | Min-p sampling: drop tokens under this share of the top token's probability (`0` disables)                     | mlx                                                                 |
| `STOP`               | Stop sequences (YAML list)                                                                                     | ollama, bedrock, anthropic, sagemaker, litellm, mlx                 |
| `STREAM`             | Stream tokens (default `true`)                                                                                 | mantle, openai, anthropic, litellm, mlx                             |
| `PRESENCE_PENALTY`   | Presence penalty                                                                                               | ollama, openai, mlx                                                 |
| `FREQUENCY_PENALTY`  | Frequency penalty                                                                                              | ollama, mlx                                                         |
| `REPETITION_PENALTY` | Repetition penalty                                                                                             | ollama, litellm, mlx                                                |
| `REASONING`          | Enable extended thinking (boolean)                                                                             | bedrock, anthropic                                                  |
| `THINKING_TOKENS`    | Thinking token budget (default `2048`)                                                                         | bedrock, anthropic                                                  |
| `REASONING_EFFORT`   | reasoning effort (provider-dependent: `none`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max`)                    | openai, anthropic, bedrock, mantle, litellm                         |
| `PROMPT_CACHE`       | Cache the stable prompt prefix (boolean, default **on** where supported)                                       | bedrock, anthropic, mantle (`API_PROTOCOL: anthropic`)              |
| `PROMPT_CACHE_TTL`   | How long a cached prefix lives: `5m` (default) or `1h`                                                         | bedrock, anthropic, mantle (`API_PROTOCOL: anthropic`)              |
| `KEEP_ALIVE`         | How long the server keeps the model loaded: `30m`, `1h30m`, `600` (bare seconds), `0` (unload now), `-1` (pin) | mlx (also on `EMBED_MODEL_ID`)                                      |

`VISION_MODEL_ID` supports the same eight providers as `MODEL_ID`. It accepts a
subset of params: `MAX_TOKENS`/`TEMPERATURE`/`TOP_P` across providers, plus
`TOP_K` on ollama/anthropic/sagemaker, `STOP` on ollama/sagemaker/mlx, and
`NUM_BATCH` on ollama or `KEEP_ALIVE` on mlx. Connection keys follow the provider (host/port, region,
Mantle protocol, SageMaker `INPUT_FORMAT`, LiteLLM/Anthropic/MLX
`API_BASE`/`API_KEY`/base URL). `TOP_K` is deliberately **not** offered for
`mlx` vision: that server's multimodal path never passes it to the sampler, so
setting it would look effective and do nothing.

> **`/params` only offers what the provider supports.** The set of tunable
> params is taken per-provider from the registry, so `/params` never prompts for
> — and never writes — a key the model ignores (e.g. Anthropic has no
> `PRESENCE_PENALTY`/`FREQUENCY_PENALTY`; only the params it honors are offered).
>
> **`REASONING_EFFORT` is a single, first-class knob translated per provider.**
> Set one effort value and mnemoai maps it to each provider's mechanism:
> forwarded as `reasoning_effort` on OpenAI and Mantle's `responses` protocol;
> mapped to a `thinking` token budget on Anthropic, standard Bedrock, and
> Mantle's `anthropic` protocol; passed through LiteLLM (which translates it per
> backend). When thinking is enabled this way, `temperature`/`top_p`/`top_k` are
> dropped automatically (the providers reject them). For finer control, set the
> raw provider parameter via `EXTRA_PARAMS` (below), which overrides this.
>
> **Bedrock is one endpoint for many model families**, so `REASONING_EFFORT` maps
> to a thinking budget only for Claude there. A non-Claude family gets no
> injection — several reason automatically, and the field name differs per family
> — so use `EXTRA_PARAMS` for those. For an OpenAI GPT model on Bedrock
> (`openai.gpt-*`), the working form is:
>
> ```yaml
> EXTRA_PARAMS:
>   additional_model_request_fields:
>     reasoning:
>       effort: high # none | low | medium | high | xhigh
> ```
>
> (a bare `reasoning_effort` at that level is rejected as an unknown parameter).
> Those models **reason but never show it**: Bedrock returns their thinking only
> in encrypted form, at every effort and summary setting, so you get the benefit of
> the reasoning without a `Thought for Ns…` block. That is the provider's policy,
> not a missing feature — Claude on Bedrock does show its thinking.

### Ollama prompt-processing batch size

Use `/params` → select an Ollama model → `NUM_BATCH`, or set it during
`/config` model setup. `/model` also offers it when configuring an Ollama model.
Enter a positive integer such as `512`; enter `none` in `/params` to remove the
override and restore MnemoAI's default of **2048**. Omitted or YAML `null`
values also use 2048. MnemoAI sends this value in the request, overriding a
different Modelfile batch size.

```yaml
MODEL_ID:
  TYPE: ollama
  NAME: your-chat-model
  NUM_BATCH: 2048
VISION_MODEL_ID:
  TYPE: ollama
  NAME: your-vision-model
  NUM_BATCH: 2048
RAG:
  EMBED_MODEL_ID:
    TYPE: ollama
    NAME: qwen3-embedding:0.6b
    NUM_BATCH: 2048
```

Each model section has its own setting. Auxiliary chat models inherit the chat
setting unless their `AREA_MODELS` entry overrides it. Other providers ignore this
Ollama-only option.

`NUM_BATCH` controls prompt-processing batches. It does not change the context
window (`MAX_CONVERSATION_TOKENS`), output-token limit (`MAX_TOKENS`), number of
agents, or number of documents sent for embedding. Smaller values can reduce
temporary memory use; larger values can improve prompt-processing throughput
when the server has sufficient memory.

Chat and auxiliary-chat changes apply to subsequent requests without restarting
MnemoAI. Vision and embedding changes are saved and require a restart, as described
above. Ollama may reload a model when its runtime options change.

### `PROMPT_CACHE` — reuse the prompt prefix instead of re-paying for it { #prompt_cache-reuse-the-prompt-prefix-instead-of-re-paying-for-it }

Every call in a turn re-sends the same opening: the system prompt, the tool
definitions, and the conversation so far. Where the provider supports prompt
caching, mnemoai marks that prefix so it is stored server-side and **read** on the
next call at a fraction of the input price — and, just as usefully, without being
re-processed, which is what a long prompt spends most of its time-to-first-token
on. An agentic turn makes one call per tool round, so the prefix is re-sent many
times before you see an answer.

It is **on by default** wherever it applies, needs no config change on an existing
install, and appears in `/usage` as the `cache: N read · N written` line. Two keys
tune it, under `MODEL_ID`:

```yaml
MODEL_ID:
  NAME: global.anthropic.claude-opus-4-8
  TYPE: bedrock
  # PROMPT_CACHE: false     # opt out entirely
  # PROMPT_CACHE_TTL: 1h    # 5m (default) | 1h — 1h costs more to write
```

What it applies to is deliberately narrow, because a cache marker sent where it
isn't understood is an error on **every** call rather than a missed saving: `TYPE:
bedrock`, `TYPE: anthropic`, and `TYPE: mantle` **only** on `API_PROTOCOL:
anthropic`, and only for model families that cache (Claude, Nova). Everything else
ignores the setting — `PROMPT_CACHE: true` cannot force it on, since Ollama has no
such concept and OpenAI-compatible servers do it automatically. A prompt shorter
than the model's own minimum (1024 tokens for most Claude models) simply isn't
cached; nothing fails.

Caching pays off when the prefix is **stable**, which is why mnemoai keeps it that
way: the system prompt is assembled once per session, and the per-turn injections
(steering, episodic memory, plan reminders) ride the newest message rather than
being spliced into the middle of the history. A `/compact`, a `/params` reload, or
a `/model` switch rewrites the prefix and the next call re-writes the cache entry
once.

### `EXTRA_PARAMS` — generic passthrough for anything else { #extra_params-generic-passthrough-for-anything-else }

The table above is the curated set. For provider-specific knobs it doesn't model
— or new ones that ship after a release — add an `EXTRA_PARAMS` dict to any
`MODEL_ID` / `VISION_MODEL_ID`. Its contents are forwarded **verbatim** to the
underlying model's request body, with **no interpretation** by mnemoai, so you
use the **provider's own parameter names**. This means new parameters need no
code change. Works for every provider; it's the right place for reasoning
controls on Mantle, which the curated columns don't cover.

```yaml
# OpenAI / GPT-5.x (TYPE: openai, or Mantle API_PROTOCOL: responses)
MODEL_ID:
  NAME: openai.gpt-5.5
  TYPE: mantle
  API_PROTOCOL: responses
  EXTRA_PARAMS:
    reasoning_effort: high      # none | low | medium | high | xhigh
    # verbosity: low

# Anthropic / Claude (TYPE: anthropic, or Mantle API_PROTOCOL: anthropic)
MODEL_ID:
  NAME: anthropic.claude-opus-4-8
  TYPE: mantle
  API_PROTOCOL: anthropic
  EXTRA_PARAMS:
    thinking: { type: enabled, budget_tokens: 10000 }
```

Notes: `reasoning_effort` is lifted to a first-class argument on OpenAI-family
clients (so it isn't double-specified); everything else is merged into the
request body (`model_kwargs`, or `extra_body` on `mlx`, whose extra fields aren't
part of the OpenAI API). A non-dict `EXTRA_PARAMS` is ignored rather than crashing. It is
not offered by the `/params` interactive tuner (it's a free-form dict, not a
scalar) — set it in `config.yaml` directly.

> **Provider-appropriate tuning matters.** Newer Claude and GPT models reject
> `TEMPERATURE` outright; `STOP`, penalties, and `TOP_K` are largely
> Ollama/SageMaker concepts. When `/model` switches a section's provider it
> drops the keys the new provider doesn't consume for you, but for everything
> else edit `config.yaml` to match what your specific provider/model accepts.

The context window is set separately, at the top level (it's not part of a model
section): `MAX_CONVERSATION_TOKENS` (see General Parameters below).
