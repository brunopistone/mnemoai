# Configuration

## 🔧 Configuration

### Complete example config

The fragments below show one section at a time. For a full, coherent file you can copy and trim, use the annotated examples that ship in the repo (and are seeded to `~/.mnemoai/config/` on first run):

- **[`config.yaml.example`](https://github.com/brunopistone/mnemoai/blob/main/src/mnemoai/utils/config.yaml.example)** — local Ollama setup (chat + vision + embeddings + RAG + memory), the default the wizard is modeled on.
- **[`config.yaml.bedrock.example`](https://github.com/brunopistone/mnemoai/blob/main/src/mnemoai/utils/config.yaml.bedrock.example)** — Amazon Bedrock setup.
- **[`config.yaml.bedrock.mantle.example`](https://github.com/brunopistone/mnemoai/blob/main/src/mnemoai/utils/config.yaml.bedrock.mantle.example)** — Bedrock Mantle setup.

Copy one and edit it:

```bash
cp ~/.mnemoai/config/config.yaml.example ~/.mnemoai/config/config.yaml
```

The rest of this page is the per-section reference for tuning that file.

### Model Configuration

The assistant supports multiple model types:

#### Amazon Bedrock

```yaml
MODEL_ID:
  NAME: us.amazon.nova-pro-v1:0
  TYPE: bedrock
  REGION: us-east-1
  TEMPERATURE: 0.1
```

> **Note:** Newer Claude models on Bedrock reject `temperature` as deprecated. Omit `TEMPERATURE` for those — it is only sent when explicitly configured.

> **Using a named AWS profile (Bedrock, SageMaker, Mantle).** These providers use the standard boto3 credential chain (default profile / env vars / instance role). To select a specific named profile instead, set `AWS_PROFILE` via the config `ENV:` section — values there are exported as environment variables at startup, and boto3 picks them up automatically. No model-level config key is needed:
>
> ```yaml
> ENV:
>   AWS_PROFILE: my-bedrock-profile
>   # AWS_REGION: us-east-1   # any AWS env var works here too
> ```

> **Using a Bedrock API key (instead of AWS credentials).** Bedrock supports short-term API keys (a `bedrock-api-key-...` value from the console). For **standard Bedrock** (`TYPE: bedrock`), set it as `AWS_BEARER_TOKEN_BEDROCK` — `langchain-aws` reads it automatically, no model config needed:
>
> ```yaml
> ENV:
>   AWS_BEARER_TOKEN_BEDROCK: bedrock-api-key-XXXXXXXX
> ```
>
> (For **Mantle**, the same key is supplied differently — see the Mantle section below.)

#### Amazon Bedrock Mantle

Bedrock Mantle is an **OpenAI-compatible** API (not the Bedrock Converse API). By default it authenticates with a short-lived bearer token minted from your standard AWS credentials via [`aws-bedrock-token-generator`](https://pypi.org/project/aws-bedrock-token-generator/), so your normal `aws configure` / SSO setup works — no extra keys to manage. Use `TYPE: mantle` and a bare model ID from the Mantle catalog.

```yaml
MODEL_ID:
  NAME: qwen.qwen3-32b # bare Mantle model id (e.g. anthropic.claude-opus-4-8)
  TYPE: mantle
  REGION: us-east-1
  MAX_TOKENS: 8192
```

**Authenticating with a Bedrock API key (no AWS credentials).** Instead of minting a token, you can supply a short-term Bedrock API key directly. Mantle reads it from the `BEDROCK_API_KEY` environment variable (set it via the config `ENV:` section), or from a per-model `API_KEY` field. When a key is present it's used as-is; otherwise the app falls back to minting from AWS credentials. (Note: standard Bedrock uses `AWS_BEARER_TOKEN_BEDROCK` for the same key — Mantle uses `BEDROCK_API_KEY`.)

```yaml
# Option A — environment variable (applies to all Mantle calls)
ENV:
  BEDROCK_API_KEY: bedrock-api-key-XXXXXXXX

# Option B — per-model key
MODEL_ID:
  NAME: qwen.qwen3-32b
  TYPE: mantle
  REGION: us-east-1
  API_KEY: bedrock-api-key-XXXXXXXX
```

**API protocols.** Mantle serves models under three protocols. Select with `API_PROTOCOL` (works for both chat and vision):

- `chat_completions` (default) — base `/v1`, OpenAI Chat Completions API. Most models (Qwen, Gemma, GPT-OSS, DeepSeek, …).
- `responses` — base `/openai/v1`, OpenAI Responses API. Required by models that only expose Responses, such as `openai.gpt-5.4`.
- `anthropic` — base `/anthropic`, Anthropic Messages API. For Claude models (e.g. `anthropic.claude-haiku-4-5`).

```yaml
# OpenAI Responses model (e.g. GPT-5.4)
MODEL_ID:
  NAME: openai.gpt-5.4
  TYPE: mantle
  REGION: us-west-2 # gpt-5.4 is in us-west-2, not us-east-1
  API_PROTOCOL: responses
  MAX_TOKENS: 8192

# Anthropic Claude model
MODEL_ID:
  NAME: anthropic.claude-haiku-4-5
  TYPE: mantle
  REGION: us-east-1
  API_PROTOCOL: anthropic
  MAX_TOKENS: 8192
```

- `ENDPOINT_URL` is optional; it defaults to `https://bedrock-mantle.<REGION>.api.aws/{v1 | openai/v1 | anthropic}` depending on the protocol.
- The Mantle catalog (Qwen, Mistral, DeepSeek, GLM, Gemma, Claude, GPT-5.4, …) differs from standard Bedrock and varies by account/region.
- `TYPE: mantle` works for both `MODEL_ID` (chat) and `VISION_MODEL_ID` (image description) — vision-capable models like `qwen.qwen3-vl-235b-a22b-instruct` are supported.
- **Caveats:** Pick the right `API_PROTOCOL` per model (using the wrong one returns a 400 "does not support the '/v1/…' API" error). `anthropic` requires the `langchain-anthropic` package (in `requirements.txt`). Models like `anthropic.claude-fable-5` also require the account's data-retention mode to be `provider_data_share`, otherwise they report `unavailable`.
- **Reasoning models need a generous `MAX_TOKENS`.** Reasoning models (e.g. Grok, GPT-5, Claude with `REASONING_EFFORT`) spend output tokens _reasoning_ before they answer. If a turn is cut off by `MAX_TOKENS` mid-response, the agent **auto-continues** — it feeds the partial turn back and resumes, up to `LLM.MAX_OUTPUT_CONTINUE_RETRIES` times (default 3), so you never have to type "continue" (see [LLM Interaction Configuration](#llm-interaction-configuration)). Still, give reasoning models real headroom — set `MAX_TOKENS` to a few thousand (e.g. `8192`), or higher for `REASONING_EFFORT: high`/`max` on a large context — so a turn can reason _and_ answer without repeatedly hitting the limit.

> For **standard** Bedrock (Converse API), `ENDPOINT_URL` is also accepted on `MODEL_ID`/`VISION_MODEL_ID` with `TYPE: bedrock` to override the default endpoint.

#### Ollama (Local)

```yaml
MODEL_ID:
  NAME: qwen3-4b-thinking-2507-q6-k:latest
  TYPE: ollama
  HOST: localhost
  PORT: 11434
  REPETITION_PENALTY: 1.1
  PRESENCE_PENALTY: 1.5
  TEMPERATURE: 0.1
  TOP_P: 0.95
```

#### OpenAI

```yaml
MODEL_ID:
  NAME: gpt-5-mini-2025-08-07
  TYPE: openai
  STREAM: true
  REASONING_EFFORT: medium
# Requires OPENAI_API_KEY environment variable
```

#### Local OpenAI-compatible servers (llama.cpp / LM Studio / vLLM)

`TYPE: openai` can point at **any OpenAI-compatible endpoint** via `API_BASE`
(alias `ENDPOINT_URL`) — so a local [`llama-server`](https://github.com/ggml-org/llama.cpp)
(llama.cpp), [LM Studio](https://lmstudio.ai), [vLLM](https://docs.vllm.ai), or
`llama-swap` works as a drop-in alternative to Ollama, no extra provider needed.
Local servers usually ignore auth, so `API_KEY` is optional (a placeholder is
sent when a custom `API_BASE` is set and no key is given).

**llama.cpp (`llama-server`)** — `brew install llama.cpp`, then
`llama-server -hf bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M --port 8080 --ctx-size 8192`
(pulls the GGUF straight from Hugging Face; OpenAI API at `:8080/v1`):

```yaml
MODEL_ID:
  NAME: qwen2.5-7b-instruct # the model name your server reports
  TYPE: openai
  API_BASE: http://localhost:8080/v1
  STREAM: true
# No API key needed for a local server.
```

**LM Studio** — start its local server (Developer tab), default port `1234`:

```yaml
MODEL_ID:
  NAME: your-loaded-model
  TYPE: openai
  API_BASE: http://localhost:1234/v1
```

The same `API_BASE`/`API_KEY` keys work for `VISION_MODEL_ID` when the local
server hosts a vision-capable model, and for `RAG.EMBED_MODEL_ID` to serve
embeddings from the local server (`TYPE: openai` + `API_BASE`). For a
multi-model setup like Ollama's (one endpoint, hot-swap by name), put
[`llama-swap`](https://github.com/mostlygeek/llama-swap) in front of several
`llama-server` instances and point every section's `API_BASE` at it.

```yaml
RAG:
  EMBED_MODEL_ID:
    NAME: qwen3-embedding # name your server reports
    TYPE: openai
    API_BASE: http://localhost:8080/v1
    DIMENSION:
      1024 # optional: match your embedder's real
      # size so the SHA256 fallback (used only
      # when the server is unreachable) stays
      # dimension-consistent with the index
```

(Ollama remains fully supported via `TYPE: ollama`; this is an alternative, not
a replacement.)

#### Anthropic (Claude API)

The direct Anthropic API (`api.anthropic.com`) via `langchain-anthropic`. This is **distinct from the Bedrock Mantle `anthropic` protocol** (which reaches Claude through Bedrock) — `TYPE: anthropic` talks to Anthropic directly. `STOP` maps to Anthropic's `stop_sequences`, and extended thinking is enabled with `REASONING` (+ optional `REASONING_EFFORT` / `THINKING_TOKENS`).

```yaml
MODEL_ID:
  NAME: claude-opus-4-8
  TYPE: anthropic
  MAX_TOKENS: 4096
  TEMPERATURE: 0.4
  # REASONING: true          # enable extended thinking
  # REASONING_EFFORT: high   # low | medium | high | max
  # ENDPOINT_URL: https://...  # optional custom base URL
# Requires ANTHROPIC_API_KEY env var, or set MODEL_ID.API_KEY
```

#### Amazon SageMaker AI

```yaml
MODEL_ID:
  NAME: your-endpoint-name
  TYPE: sagemaker
  REGION: us-east-1
  REPETITION_PENALTY: 1.1
  PRESENCE_PENALTY: 1.5
  TEMPERATURE: 0.1
  MAX_TOKENS: 4096
```

#### LiteLLM (100+ Providers)

```yaml
MODEL_ID:
  NAME: openai/your-model-name
  TYPE: litellm
  API_BASE: http://localhost:8000/v1
  API_KEY: your-api-key
  TEMPERATURE: 0.1
  MAX_TOKENS: 4096
```

### Vision Model Configuration

For Bedrock:

```yaml
VISION_MODEL_ID:
  NAME: global.anthropic.claude-haiku-4-5-20251001-v1:0
  TYPE: bedrock
  REGION: us-east-1
  TEMPERATURE: 0.3
```

For Ollama:

```yaml
VISION_MODEL_ID:
  NAME: qwen3-vl:2b
  TYPE: ollama
  HOST: localhost
  PORT: 11434
  TEMPERATURE: 0.3
```

For OpenAI:

```yaml
VISION_MODEL_ID:
  NAME: gpt-5-mini-2025-08-07
  TYPE: openai
  STREAM: true
  REASONING_EFFORT: medium
```

For Anthropic (Claude is multimodal):

```yaml
VISION_MODEL_ID:
  NAME: claude-opus-4-8
  TYPE: anthropic
  MAX_TOKENS: 1500
  TEMPERATURE: 0.3
# Requires ANTHROPIC_API_KEY env var, or set VISION_MODEL_ID.API_KEY
```

For SageMaker AI (endpoint must serve a vision-capable model accepting the OpenAI image format):

```yaml
VISION_MODEL_ID:
  NAME: your-endpoint-name
  TYPE: sagemaker
  REGION: us-east-1
  INPUT_FORMAT: openai_chat
  TEMPERATURE: 0.3
```

For LiteLLM (any of its vision-capable models):

```yaml
VISION_MODEL_ID:
  NAME: openai/gpt-4o # provider-prefixed model id
  TYPE: litellm
  API_BASE: http://localhost:4000 # optional (proxy / self-hosted)
  API_KEY: your-api-key # optional (else the provider's env var)
```

### Model Parameters

This is the full reference for what you can put under `MODEL_ID`,
`VISION_MODEL_ID`, and `RAG.EMBED_MODEL_ID`. Only `NAME` and `TYPE` are
required; everything else is optional and omitted keys fall back to the
provider/model default. The interactive configurator (`/config`, `/model`)
sets the common ones — use this reference to hand-tune `config.yaml` for
anything else a provider or model supports.

#### Identity, connection & auth

| Parameter      | Applies to `TYPE`                | Description                                                                                                                                      |
| -------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `NAME`         | all (**required**)               | Model id / Ollama model / Bedrock model id / Mantle bare id / SageMaker endpoint name                                                            |
| `TYPE`         | all (**required**)               | `ollama`, `bedrock`, `mantle`, `openai`, `anthropic`, `sagemaker`, `litellm` (embeddings: `ollama`, `bedrock`, `openai`, `sagemaker`, `litellm`) |
| `HOST`         | `ollama`                         | Ollama host (default `localhost`)                                                                                                                |
| `PORT`         | `ollama`                         | Ollama port (default `11434`)                                                                                                                    |
| `REGION`       | `bedrock`, `mantle`, `sagemaker` | AWS region (default `us-east-1`)                                                                                                                 |
| `API_PROTOCOL` | `mantle`                         | `chat_completions` (default), `responses`, or `anthropic`                                                                                        |
| `ENDPOINT_URL` | `bedrock`, `mantle`, `anthropic` | Override the default endpoint URL (Anthropic: custom base URL)                                                                                   |
| `API_KEY`      | `mantle`, `anthropic`, `litellm` | Mantle: Bedrock API key (else `BEDROCK_API_KEY` env / minted token). Anthropic: else `ANTHROPIC_API_KEY` env. LiteLLM: provider key              |
| `API_BASE`     | `litellm`                        | LiteLLM API base URL                                                                                                                             |
| `INPUT_FORMAT` | `sagemaker`                      | `openai_chat` (default) or `huggingface`                                                                                                         |

> Standard Bedrock also reads the `AWS_BEARER_TOKEN_BEDROCK` env var, and all AWS
> providers honor `AWS_PROFILE` — see the API-key/profile notes under Amazon Bedrock.

#### Inference parameters

Optional generation settings. The **Honored by** column lists the providers that
actually send each one (others ignore it). These apply to `MODEL_ID` and
`VISION_MODEL_ID`; **`EMBED_MODEL_ID` takes none of them** (embeddings only use
`NAME`/`TYPE` + connection).

This table is derived from `models/provider_params.py` — the single source of
truth that the controllers build their client kwargs from — so it reflects
exactly what each provider's init path forwards. (`mantle` reads
`TEMPERATURE`/`MAX_TOKENS`/`TOP_P` via the Mantle factory.)

| Parameter            | Description                                                                                 | Honored by (`MODEL_ID`)                                        |
| -------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `MAX_TOKENS`         | Max output tokens to generate                                                               | ollama, bedrock, mantle, openai, anthropic, sagemaker, litellm |
| `TEMPERATURE`        | Sampling temperature                                                                        | ollama, bedrock, mantle, openai, anthropic, sagemaker, litellm |
| `TOP_P`              | Top-p (nucleus) sampling                                                                    | ollama, bedrock, mantle, openai, anthropic, sagemaker, litellm |
| `TOP_K`              | Top-k sampling                                                                              | ollama, anthropic, sagemaker                                   |
| `STOP`               | Stop sequences (YAML list)                                                                  | ollama, bedrock, anthropic, sagemaker, litellm                 |
| `STREAM`             | Stream tokens (default `true`)                                                              | mantle, openai, anthropic, litellm                             |
| `PRESENCE_PENALTY`   | Presence penalty                                                                            | ollama, openai                                                 |
| `FREQUENCY_PENALTY`  | Frequency penalty                                                                           | ollama                                                         |
| `REPETITION_PENALTY` | Repetition penalty                                                                          | ollama, litellm                                                |
| `REASONING`          | Enable extended thinking (boolean)                                                          | bedrock, anthropic                                             |
| `THINKING_TOKENS`    | Thinking token budget (default `2048`)                                                      | bedrock, anthropic                                             |
| `REASONING_EFFORT`   | reasoning effort (provider-dependent: `none`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max`) | openai, anthropic, bedrock, mantle, litellm                    |
| `PROMPT_CACHE`       | Cache the stable prompt prefix (boolean, default **on** where supported)                    | bedrock, anthropic, mantle (`API_PROTOCOL: anthropic`)         |
| `PROMPT_CACHE_TTL`   | How long a cached prefix lives: `5m` (default) or `1h`                                      | bedrock, anthropic, mantle (`API_PROTOCOL: anthropic`)         |

`VISION_MODEL_ID` supports the same seven providers as `MODEL_ID`. It accepts a
subset of params: `MAX_TOKENS`/`TEMPERATURE`/`TOP_P` across providers, plus
`TOP_K` on ollama/anthropic/sagemaker and `STOP` on ollama/sagemaker. Connection
keys follow the provider (host/port, region, Mantle protocol, SageMaker
`INPUT_FORMAT`, LiteLLM/Anthropic `API_BASE`/`API_KEY`/base URL).

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

##### `PROMPT_CACHE` — reuse the prompt prefix instead of re-paying for it

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

##### `EXTRA_PARAMS` — generic passthrough for anything else

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
clients (so it isn't double-specified); everything else is merged into
`model_kwargs`. A non-dict `EXTRA_PARAMS` is ignored rather than crashing. It is
not offered by the `/params` interactive tuner (it's a free-form dict, not a
scalar) — set it in `config.yaml` directly.

> **Provider-appropriate tuning matters.** Newer Claude and GPT models reject
> `TEMPERATURE` outright; `STOP`, penalties, and `TOP_K` are largely
> Ollama/SageMaker concepts. When `/model` switches a section's provider it
> drops the keys the new provider doesn't consume for you, but for everything
> else edit `config.yaml` to match what your specific provider/model accepts.

The context window is set separately, at the top level (it's not part of a model
section): `MAX_CONVERSATION_TOKENS` (see General Parameters below).

### General Parameters

```yaml
# Context window size (passed to model as num_ctx for Ollama)
MAX_CONVERSATION_TOKENS: 65536

# Maximum tokens when reading documents (CSV, JSON, text files)
DOC_MAX_TOKENS: 16384

# Days a recorded session is kept for `--resume` (0 disables recording entirely).
# Saved conversations (/save) are separate and never expire.
SESSION_MAX_AGE_DAYS: 30

# Days a log file under ~/.mnemoai/logs/ is kept (0 disables the sweep).
LOG_MAX_AGE_DAYS: 7

# Profile configuration
PROFILE:
  NAME: default # Used for session data isolation (~/.mnemoai/{NAME}/)
  USE_PROFILING: true # Enable automatic user profiling
```

`SESSION_MAX_AGE_DAYS` controls the automatic session transcripts that
`mnemoai --resume` restores (see [Resuming a session](guides/usage.md#resuming-a-session)).
Sessions are grouped by the directory you launched from, so resuming in a project
only offers that project's sessions. Set it to `0` to stop recording sessions
altogether; this never affects `/save` / `/load`.

`LOG_MAX_AGE_DAYS` expires everything under `~/.mnemoai/logs/` — the app log
(`mnemoai.log`, where the tracebacks the terminal deliberately doesn't print end
up) and the MCP subprocess log. The app log also rotates at 2 MB, keeping two
generations, so one very noisy `LOG_LEVEL=DEBUG` run can't fill the disk before
the next sweep. Set it to `0` to keep logs indefinitely. See
[Read the logs](development/troubleshooting.md#read-the-logs).

### Environment variables

Mnemo AI reads a handful of environment variables. Provider API keys can be set either in your shell or, more conveniently, under the config `ENV:` block — every key there is exported as an environment variable at startup.

| Variable                     | Purpose                                                    | Notes                                                                                                                                                           |
| ---------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MNEMOAI_CONFIG`             | Explicit path to `config.yaml`                             | Highest-priority config location; overrides the normal resolution order                                                                                         |
| `MNEMOAI_HOME`               | Override the app home (default `~/.mnemoai`)               | Moves config, prompts, plans, tasks, and per-profile/per-model state together                                                                                   |
| `MNEMOAI_PROMPTS`            | Explicit path to `prompts.yaml`                            | Overrides the normal prompts resolution order                                                                                                                   |
| `LOG_LEVEL`                  | Log verbosity: `DEBUG` / `INFO` / `WARNING`                | Default `WARNING`; one line per record on stderr, full records (tracebacks included) in `~/.mnemoai/logs/mnemoai.log`. `DEBUG` also prints tracebacks on screen |
| `OPENAI_API_KEY`             | OpenAI auth (`TYPE: openai`)                               | Or set `MODEL_ID.API_KEY`                                                                                                                                       |
| `ANTHROPIC_API_KEY`          | Anthropic auth (`TYPE: anthropic`)                         | Or set `MODEL_ID.API_KEY`                                                                                                                                       |
| `AWS_PROFILE` / `AWS_REGION` | AWS profile / region (Bedrock, Mantle, SageMaker)          | Standard boto3 chain; often set via the `ENV:` block                                                                                                            |
| `AWS_BEARER_TOKEN_BEDROCK`   | Bedrock API key for **standard** Bedrock (`TYPE: bedrock`) | Read automatically by `langchain-aws`                                                                                                                           |
| `BEDROCK_API_KEY`            | Bedrock API key for **Mantle** (`TYPE: mantle`)            | Or `MODEL_ID.API_KEY`; else a token is minted from AWS creds                                                                                                    |
| `BRAVE_API_KEY`              | Brave Search API key for web search                        | Can also be the top-level `BRAVE_API_KEY` config key                                                                                                            |

```yaml
# Set provider keys/vars without touching your shell:
ENV:
  AWS_PROFILE: my-bedrock-profile
  BEDROCK_API_KEY: bedrock-api-key-XXXXXXXX
```

### Embeddings Configuration

Embeddings settings are nested under the `RAG` section:

```yaml
RAG:
  EMBEDDINGS:
    CACHE_ENABLED: true # LRU cache for embedding vectors (avoids re-embedding same text)
    CACHE_SIZE: 1000 # Maximum cached embeddings
    FALLBACK_ENABLED: true # Fall back to SHA256 if embedding model unavailable
    FALLBACK_TYPE: "sha256" # Fallback type (sha256, random, zeros)
```

### LLM Interaction Configuration

```yaml
LLM:
  ENABLE_THINKING: true # Enable thinking tags (verbose mode)
  RETRY_ENABLED: true # Retry failed LLM calls
  MAX_RETRIES: 5 # Maximum retry attempts; also caps retries of a
  # transient *empty* model response
  RETRY_DELAY: 1.0 # Seconds between retries
  RETRY_BACKOFF: 2.0 # Exponential backoff multiplier
  MAX_OUTPUT_CONTINUE_RETRIES: 3 # Auto-continue a turn cut off by MAX_TOKENS
  # (reasoning + answer exceeded the output
  # budget); 0 disables. See below.
  STREAM_IDLE_TIMEOUT: 120 # Abandon a streaming read that goes silent this
  # long (dead socket, e.g. laptop sleep) and
  # re-run the turn on a fresh connection;
  # 0 disables. See below.
  SUMMARIZATION_THINK: false # Include thinking in summarization
  TOKEN_COUNTING:
    OLLAMA_CHARS_PER_TOKEN: 3.0 # Ollama: chars per token (no tokenizer available)
    ANTHROPIC_MULTIPLIER: 1.5 # Per-provider safety multiplier; see below
  # --- Context management (compaction + overflow protection) ---
  KEEP_RECENT_MESSAGES: 6 # Turns kept verbatim on auto-compaction
  MANUAL_COMPACT_KEEP_RECENT: 2 # Smaller window for the manual /compact command
  KEEP_RECENT_TOKEN_BUDGET: 16384 # Also bound the kept window by tokens
  # COMPACT_HIGH_WATER_TOKENS  # Proactively compact before a turn when history
  # exceeds this. Auto-derives to 80% of
  # MAX_CONVERSATION_TOKENS when unset; 0 disables.
  # MAX_TOOL_RESULT_CHARS      # Cap one tool result (~4 chars/token) so a
  # runaway result can't overflow the window
  # (head+tail kept with a note). Auto-derives to
  # 10% of the window (in chars) when unset;
  # 0 disables.
  # TOOL_EVICTION_KEEP_RECENT: 8  # Messages kept verbatim by the tool-result
  # eviction layer (below).
  # EVICTED_TOOL_RESULT_CHARS: 500 # Char cap an OLD tool result is shrunk to
  # before any LLM summary. 0 disables the layer.
  RECURSION_LIMIT: 200 # Max model<->tool steps per query (runaway guard)
  MCP_CALL_TIMEOUT: 300 # Transport-layer timeout for one MCP tool call (s)
```

**Token counting.** `TOKEN_COUNTING` only tunes the _pre-flight estimate_ for a
prompt that hasn't been sent yet. Once a turn completes, the size comes from the
provider's own `usage_metadata`, which is ground truth and needs no estimate —
that is what `/usage` and the footer's context meter report (an estimate there is
marked with a `~`).

The estimate is deliberately conservative, because undercounting overflows the
window while overcounting only compacts a little early. Text is tokenized with
tiktoken's `o200k_base`, then scaled per provider family:

| Provider `TYPE`                   | Multiplier              | Override key                                |
| --------------------------------- | ----------------------- | ------------------------------------------- |
| `openai`                          | 1.0 (tiktoken is exact) | `OPENAI_MULTIPLIER`                         |
| `anthropic`, `mantle`             | 1.5                     | `ANTHROPIC_MULTIPLIER`, `MANTLE_MULTIPLIER` |
| `bedrock`, `sagemaker`, `litellm` | 1.35                    | `BEDROCK_MULTIPLIER`, …                     |
| anything else                     | 1.35                    | `<TYPE>_MULTIPLIER`                         |

`ollama` is the exception: no tokenizer is available, so the count is
`len(text) / OLLAMA_CHARS_PER_TOKEN` (default `3.0`).

!!! note "Two vestigial keys under `TOKEN_COUNTING`"

    Older config templates list `FALLBACK_MODEL` and `OLLAMA_APPROXIMATION`.
    `FALLBACK_MODEL` is no longer read by anything and can be deleted.
    `OLLAMA_APPROXIMATION` (default `1.3`) still applies, but only to the
    episodic-memory size budget — not to conversation token counting.

**Context management.** The conversation is kept under `MAX_CONVERSATION_TOKENS`
by summarizing older turns into the system prompt while keeping recent ones
verbatim — automatically when over budget, or manually via `/compact`. Several
layers prevent a single oversized turn from breaking the loop:

1. **Tool-result cap** (`MAX_TOOL_RESULT_CHARS`) — one runaway result (e.g. a
   `grep_search` with a huge `max_results`) is truncated head+tail with a note,
   so it can never alone exceed the context window. Auto-derives to 10% of the
   window (in chars) when unset, scaling with the model.
2. **Pre-flight compaction, layered** (`COMPACT_HIGH_WATER_TOKENS`) — before a
   turn, if the accumulated history is over the high-water mark, it is compacted.
   The mark auto-derives to 80% of `MAX_CONVERSATION_TOKENS` when unset. The
   cheapest layer runs first: **tool-result eviction** shrinks the bodies of
   _old_ tool results (grep/read/web dumps outside the recent window, which carry
   most of the context but are rarely needed verbatim once acted on) to a short
   head plus a marker, with **no model call** — recent turns stay verbatim and no
   message is dropped. If that alone gets back under budget, the expensive
   summary is skipped; otherwise it falls through to the full LLM summary. Either
   way the reduced state is checkpointed in the session transcript, so a resume
   comes back to it rather than to the full-size history. Tune
   with `TOOL_EVICTION_KEEP_RECENT` (messages kept verbatim, default 8) and
   `EVICTED_TOOL_RESULT_CHARS` (shrink target, default 500; 0 disables the layer).
3. **Overflow backstop** — if a request still exceeds the window, the turn ends
   with a clear message and compacts for the next turn instead of retrying the
   same oversized prompt in a loop.

Those three guard the _input_ side (prompt too large). The _output_ side has its
own recovery:

4. **Output-token auto-continue** (`MAX_OUTPUT_CONTINUE_RETRIES`) — when the
   model's response is cut off by `MODEL_ID.MAX_TOKENS` mid-turn (common with
   `REASONING_EFFORT: high`/`max` on a large context — reasoning plus a partial
   answer or tool call exhaust the output budget), the agent feeds the partial
   turn back and resumes ("continue where you left off"), accumulating the answer,
   up to this many attempts (default 3; 0 disables). It stops early when a
   continuation finishes cleanly or emits a tool call. You never have to type
   "continue"; if the retries are exhausted it surfaces a message to raise
   `MAX_TOKENS`.
5. **Stalled-stream recovery** (`STREAM_IDLE_TIMEOUT`) — a streaming read is a
   blocking socket read; if the connection dies silently (e.g. the laptop sleeps),
   it would otherwise park the worker thread forever and freeze the whole UI. A
   per-chunk idle-timeout watchdog abandons a stream that goes quiet for this long
   (default 120s; 0 disables) and the turn is **re-run on a fresh connection with
   exponential backoff** — the same recovery path used for a transient network
   drop (reset/timeout/5xx). The partial response is discarded (a dropped stream
   can't resume mid-generation), but the conversation continues; if reconnection
   keeps failing it ends with a "lost the connection — send your message again"
   message rather than hanging.

### Prompts (`prompts.yaml`)

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

### RAG Configuration

```yaml
ENABLE_RAG: true # Master toggle for RAG system
RAG:
  MAX_TOKENS: 8192 # Threshold: documents above this are ingested into RAG
  CHUNK_TOKENS: 1024 # Chunk size in tokens (recommended: 512-2048)
  SEARCH:
    SEMANTIC_WEIGHT: 0.5 # Semantic similarity weight (0-1)
    KEYWORD_WEIGHT: 0.5 # BM25 keyword weight (0-1)
  VECTOR_STORE:
    TYPE: chromadb # Vector store backend: "faiss" or "chromadb"
  EMBEDDINGS:
    CACHE_ENABLED: true
    CACHE_SIZE: 1000
    FALLBACK_ENABLED: true
    FALLBACK_TYPE: "sha256"
```

**Requires:** An embedding model configured via `RAG.EMBED_MODEL_ID` (see [Embeddings Model](#embeddings-model)).

### Episodic Memory Configuration

```yaml
ENABLE_EPISODIC_MEMORY: true
EPISODIC_MEMORY:
  STORE_TYPE: chromadb # or faiss
  # Similarity Thresholds
  DUPLICATE_THRESHOLD: 0.95 # Higher = stricter duplicate detection
  RETRIEVAL_THRESHOLD: 0.7 # Minimum similarity to retrieve episodes
  FOLLOW_UP_THRESHOLD: 0.4 # Similarity to detect follow-up questions (skips injection)
  REDUNDANCY_THRESHOLD: 0.5 # Filter episodes redundant with conversation
  # Hybrid Search Weights
  SEMANTIC_WEIGHT: 0.7 # Semantic similarity weight (0-1)
  KEYWORD_WEIGHT: 0.3 # Keyword matching weight (0-1)
  # Token and Size Limits
  MAX_TOKENS_PER_EPISODE: 400 # Max tokens for episode text
  MAX_EPISODES: 1000 # Maximum stored episodes
  MAX_AGE_DAYS: 90 # Maximum episode age in days
  # Success Detection
  SUCCESS_MARKERS: # Phrases that indicate task success
    - thanks
    - perfect
    - great
    - worked
  CORRECTION_MARKERS: # Phrases that indicate errors
    - wrong
    - error
    - fix
    - actually
  # Storage Behavior
  IMMEDIATE_STORAGE: true # Store episodes immediately
  MIN_TOOLS_OR_LENGTH: 300 # Min response length if no tools used
  # Query Enhancement
  ENABLE_QUERY_EXPANSION: true # Expand queries with synonyms
  QUERY_EXPANSION_TERMS: 3 # Max terms to add per query
```

**Requires:** An embedding model configured via `RAG.EMBED_MODEL_ID` (see [Embeddings Model](#embeddings-model)).

**How it works, success detection, storage paths, and a worked example** of
the injected context are covered on the conceptual page: see
[Episodic Memory](guides/memory.md#episodic-memory).

#### Embeddings Model

All embedding configuration is nested under `RAG:`:

For Bedrock:

```yaml
RAG:
  EMBED_MODEL_ID:
    NAME: amazon.titan-embed-text-v2:0
    TYPE: bedrock
    REGION: us-east-1
```

For Ollama:

```yaml
RAG:
  EMBED_MODEL_ID:
    NAME: qwen3-embedding:0.6b
    TYPE: ollama
    HOST: localhost
    PORT: 11434
```

For OpenAI:

```yaml
RAG:
  EMBED_MODEL_ID:
    NAME: text-embedding-ada-002
    TYPE: openai
```

For SageMaker:

```yaml
RAG:
  EMBED_MODEL_ID:
    NAME: your-endpoint-name
    TYPE: sagemaker
    REGION: us-east-1
```

For LiteLLM (any of its 100+ providers via one OpenAI-style API):

```yaml
RAG:
  EMBED_MODEL_ID:
    NAME: openai/text-embedding-3-small # provider-prefixed model id
    TYPE: litellm
    API_BASE: http://localhost:4000 # optional (proxy / self-hosted)
    API_KEY: your-api-key # optional (else the provider's env var)
```

**Vector Store Options:**

- **ChromaDB** (default): Persistent vector database with built-in metadata support
- **FAISS**: Fast, in-memory vector search with disk persistence

Switch between stores by changing `RAG.VECTOR_STORE.TYPE` in config. The system uses a controller pattern, so all RAG functionality works identically regardless of the store.
