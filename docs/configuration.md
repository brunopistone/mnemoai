# Configuration

## 🔧 Configuration

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
- **Reasoning models need a generous `MAX_TOKENS`.** Reasoning models (e.g. Grok, GPT-5) on the `responses` protocol spend output tokens _reasoning_ before they answer. A small `MAX_TOKENS` can be consumed entirely by reasoning, leaving no answer — the agent detects this and tells you to raise `MAX_TOKENS` rather than returning an empty reply. Set it to a few thousand (e.g. `8192`) for these models.

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

# Profile configuration
PROFILE:
  NAME: default # Used for session data isolation (~/.mnemoai/{NAME}/)
  USE_PROFILING: true # Enable automatic user profiling
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
  MAX_RETRIES: 3 # Maximum retry attempts
  RETRY_DELAY: 1.0 # Seconds between retries
  RETRY_BACKOFF: 2.0 # Exponential backoff multiplier
  SUMMARIZATION_THINK: false # Include thinking in summarization
  TOKEN_COUNTING:
    OLLAMA_APPROXIMATION: 1.3 # Chars-to-tokens multiplier for Ollama
    FALLBACK_MODEL: "gpt-4" # Tiktoken model for fallback counting
```

### System Prompt

The system prompt lives in **`prompts.yaml`** (a sibling of `config.yaml`, in the same `config/` directory — all model-facing prompts live there, separate from configuration). Customize the `SYSTEM_PROMPT` field to change the assistant's personality, instructions, and tool usage patterns. Key sections in the default prompt:

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

**How it works:**

- Automatically stores successful task completions with full conversation context
- Uses hybrid search (70% semantic + 30% BM25) to find similar past tasks
- **Conversation-aware injection**: Only injects episodic memory when relevant
  - Detects follow-up questions and skips injection (uses conversation context instead)
  - Filters out episodes redundant with current conversation
  - Uses semantic similarity (with embeddings) or Jaccard similarity (fallback)
- Injects compact context showing: task → tools used → outcome
- Automatic cleanup: keeps max 1000 episodes, removes entries older than 90 days

**Success detection:**

- User feedback: "thanks", "perfect", "great"
- No error markers in response
- All tools executed successfully
- Filters out simple greetings and short responses

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
    NAME: mxbai-embed-large
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
