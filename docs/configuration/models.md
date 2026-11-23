# Model providers { #model-configuration }

The assistant supports multiple model types:

## Amazon Bedrock { #amazon-bedrock }

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

## Amazon Bedrock Mantle { #amazon-bedrock-mantle }

Bedrock Mantle is an **OpenAI-compatible** API (not the Bedrock Converse API). By default it authenticates with a short-lived bearer token minted from your standard AWS credentials via [`aws-bedrock-token-generator`](https://pypi.org/project/aws-bedrock-token-generator/), so your normal `aws configure` / SSO setup works — no extra keys to manage. Use `TYPE: mantle` and a bare model ID from the Mantle catalog.

Generated tokens are refreshed for each HTTP request, including async requests;
cached models do not retain an expired generated token indefinitely. This does
not extend an expired SSO login or other underlying AWS credentials. Explicit
API keys remain user-managed and are used as supplied.

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
- **Reasoning models need a generous `MAX_TOKENS`.** Reasoning models (e.g. Grok, GPT-5, Claude with `REASONING_EFFORT`) spend output tokens _reasoning_ before they answer. If a turn is cut off by `MAX_TOKENS` mid-response, the agent **auto-continues** — it feeds the partial turn back and resumes, up to `LLM.MAX_OUTPUT_CONTINUE_RETRIES` times (default 3), so you never have to type "continue" (see [LLM Interaction Configuration](runtime.md#llm-interaction-configuration)). Still, give reasoning models real headroom — set `MAX_TOKENS` to a few thousand (e.g. `8192`), or higher for `REASONING_EFFORT: high`/`max` on a large context — so a turn can reason _and_ answer without repeatedly hitting the limit.

> For **standard** Bedrock (Converse API), `ENDPOINT_URL` is also accepted on `MODEL_ID`/`VISION_MODEL_ID` with `TYPE: bedrock` to override the default endpoint.

## Ollama (Local) { #ollama-local }

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

## OpenAI { #openai }

```yaml
MODEL_ID:
  NAME: gpt-5-mini-2025-08-07
  TYPE: openai
  STREAM: true
  REASONING_EFFORT: medium
# Requires OPENAI_API_KEY environment variable
```

## Nebius Token Factory { #nebius-token-factory }

MnemoAI supports Nebius Token Factory for chat, vision, embeddings and per-area
models. The provider uses Nebius's OpenAI-compatible Chat Completions API.

### Configure { #configure }

Set `NEBIUS_API_KEY` in your environment, then select **Nebius Token Factory**
in `/config` or use `/model` to change an existing model. The bundled
`config.yaml.nebius.example` contains a complete configuration.

```yaml
MODEL_ID:
  TYPE: nebius
  NAME: Qwen/Qwen3-30B-A3B-Instruct-2507
  MAX_TOKENS: 8192
  STREAM: true

VISION_MODEL_ID:
  TYPE: nebius
  NAME: openbmb/MiniCPM-V-4_5
  MAX_TOKENS: 2048

MAX_CONVERSATION_TOKENS: 262144

RAG:
  EMBED_MODEL_ID:
    TYPE: nebius
    NAME: Qwen/Qwen3-Embedding-8B
    MAX_INPUT_TOKENS: 32768
    # DIMENSION: 4096

# Optional; otherwise these areas follow Chat.
AREA_MODELS:
  ROUTER: Qwen/Qwen3-30B-A3B-Instruct-2507
  ORCHESTRATOR: Qwen/Qwen3-30B-A3B-Instruct-2507
  SUMMARY: Qwen/Qwen3-30B-A3B-Instruct-2507
  REVIEWER:
    TYPE: nebius
    NAME: openai/gpt-oss-120b
    REASONING_EFFORT: low
```

The default endpoint is `https://api.tokenfactory.nebius.com/v1`. Set `API_BASE`
(or its alias `ENDPOINT_URL`) in a model section to override it. An explicit
`API_KEY` in that section takes precedence over `NEBIUS_API_KEY`.
`OPENAI_API_KEY` is not used for this provider.

| Configuration task                                            | Interface                        |
| ------------------------------------------------------------- | -------------------------------- |
| Initial setup                                                 | `/config` → Nebius Token Factory |
| Chat, vision, embeddings and area models                      | `/model`                         |
| Sampling, output budget, streaming and reasoning effort       | `/params`                        |
| Enable routing, orchestration, review, RAG or episodic memory | `/features`                      |
| Check the active configuration and credential presence        | `/doctor`                        |

Changing providers to or from Nebius clears the previous provider's API key and
endpoint from the edited section. Models within Nebius retain their connection
settings. Per-area overrides also keep credentials scoped to their provider.

### Reasoning and tools { #reasoning-and-tools }

Reasoning-capable models return reasoning separately from the final answer.
MnemoAI preserves both `reasoning` and `reasoning_content` response formats,
including streamed responses and assistant history around tool calls.

Start normally to display reasoning. `mnemoai --no-verbose` hides reasoning
without disabling the model's reasoning or changing its answer request.
`STREAM: false` selects non-streamed requests; reasoning and usage are still
retained when returned by the model.

Use `/params` → `REASONING_EFFORT` for models that support it. Values are
model-specific: GPT-OSS accepts `low`, `medium` and `high`; it rejects the literal
API value `none`. In the parameter editor, entering `none` removes the setting
and restores the provider default.

Some models support additional controls, for example Nemotron's thinking switch:

```yaml
MODEL_ID:
  TYPE: nebius
  NAME: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B
  EXTRA_PARAMS:
    chat_template_kwargs:
      enable_thinking: false
```

This changes inference behavior and is distinct from `--no-verbose`. Leave
model-specific parameters unset unless the selected model supports them.
`TOP_K` and `REPETITION_PENALTY` are forwarded through the SDK's extra request
body, alongside other `EXTRA_PARAMS`.

Capabilities vary by model. Choose a chat model with tool support for agent work,
and a vision-capable model for `VISION_MODEL_ID`; copying a text-only chat model
into the vision section does not add vision capabilities.

### Model catalog and usage { #model-catalog-and-usage }

Nebius's authenticated `GET /v1/models?verbose=true` catalog reports model IDs,
context lengths, modalities, supported features and sampling parameters. Use the
selected model's actual context limit when setting `MAX_CONVERSATION_TOKENS`.
The example above uses the Qwen Instruct model's 262,144-token window.

Streaming requests explicitly request usage metadata. `/usage` records reported
input/output and cached-input tokens; the context meter uses the last measured
input count when available and marks local estimates with `~`.
Nebius's cumulative per-chunk counters are normalized before aggregation, so the
same input tokens are not counted again for each streamed fragment.

Embeddings use `/v1/embeddings`. `DIMENSION`, when configured, is sent as the
requested output dimension. Input limits, caching, retries and outage behavior
use MnemoAI's existing embedding machinery. Failed writes never fabricate
vectors, and existing stores retain their normal keyword-search fallback.
Local playbook learning continues to make no model or embedding calls.

The embedding model's `MAX_INPUT_CHARS` setting (also exposed in `/params`)
overrides `RAG.EMBEDDINGS.MAX_INPUT_CHARS`. When omitted or null, the global setting
applies, defaulting to 200,000 characters. Set it to `0` to disable the character
cap; the independent `MAX_INPUT_TOKENS` limit still applies.

Official references:

- [Quickstart and authentication](https://docs.tokenfactory.nebius.com/quickstart)
- [Chat Completions API](https://docs.tokenfactory.nebius.com/api-reference/inference/create-chat-completion)
- [Function calling](https://docs.tokenfactory.nebius.com/ai-models-inference/function-calling)
- [Model discovery and capabilities](https://docs.tokenfactory.nebius.com/api-reference/models/list-models)
- [Vision inputs](https://docs.tokenfactory.nebius.com/api-reference/examples/vision-capabilities)
- [Embeddings API](https://docs.tokenfactory.nebius.com/api-reference/inference/create-embeddings)

## Local OpenAI-compatible servers (llama.cpp / LM Studio / vLLM) { #local-openai-compatible-servers-llamacpp-lm-studio-vllm }

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
    DIMENSION: 1024 # optional: expected real size; otherwise probed from the provider
```

(Ollama remains fully supported via `TYPE: ollama`; this is an alternative, not
a replacement.)

## MLX server (Local, Apple Silicon) { #mlx-server-local-apple-silicon }

`TYPE: mlx` targets a local [MLX](https://github.com/ml-explore/mlx) server such
as [`mlx-openai-server`](https://github.com/cubist38/mlx-openai-server), which
runs quantized models natively on Apple Silicon. Its API is OpenAI-compatible,
so this is `TYPE: openai` with the local ergonomics filled in: the connection is
`HOST`/`PORT` (no hand-written `/v1` suffix), no key is required, and it accepts
MLX's own sampling knobs — `TOP_K`, `MIN_P`, `REPETITION_PENALTY` — plus
`KEEP_ALIVE`.

```yaml
MODEL_ID:
  NAME: qwen-agentcoder # the name the server serves it under
  TYPE: mlx
  HOST: 127.0.0.1 # default
  PORT: 8000 # default
  TEMPERATURE: 0.6
  TOP_P: 0.95
  TOP_K: 20
  MIN_P: 0.05
  REPETITION_PENALTY: 1.15
  MAX_TOKENS: 32768
  KEEP_ALIVE: 30m # stay resident 30 min after each request
# No API key needed for a local server.
```

Start the server before use — e.g. `mlx-openai-server launch --model-path
mlx-community/Qwen3-4B-4bit --model-type lm`, or `--config <file>` for a server
that declares several models. `NAME` is whatever that server answers to: its
served model name, a version or alias of it, or the model path/repo id when no
served name is set.

**Set a tool-call parser on the server** — in the server's own config, not here.
Mnemo AI is tool-driven, and models emit tool calls in their own dialect (Qwen3
writes Hermes-style `<tool_call>{…}</tool_call>`). With no parser configured the
server hands that straight back as assistant text, so the agent talks _about_ the
tool it meant to call instead of calling it — and nothing in the config above
looks wrong. For `mlx-openai-server` that's `tool_call_parser` per model entry
(`hermes` for Qwen3 Instruct, `qwen3_coder` for the Coder line), plus
`reasoning_parser` for a thinking model.

**A `reasoning_parser` gets you the `Thought for Ns…` block.** With one set the
server reports the model's thinking as its own field instead of leaving it in the
answer, and Mnemo AI renders it as a collapsed reasoning block (verbose mode) or
dims it inline. Which field it lands in differs per server and per parser — every
known shape is read, so any parser your server offers works.

**`KEEP_ALIVE` — how long the model stays loaded.** It rides in the request, so
each section sets its own: `30m`, `1h30m`, `500ms`, a bare number of seconds
(`600`), `0` to unload as soon as the request finishes, or `-1` to pin the model
in memory. Omit it to leave the server's own default in charge. It earns its keep
when one server hosts several models — pin the chat model you talk to constantly
and let a rarely-used vision model fall out of memory instead of holding RAM.

`API_BASE` and `API_KEY` are optional, for the two exceptions: a server behind a
reverse proxy or a path prefix (`API_BASE: https://mac.internal/mlx/v1`, which
takes precedence over `HOST`/`PORT`; `ENDPOINT_URL` is accepted as an alias, as
with `TYPE: openai`), and a server you have put an auth layer in front of. With
neither set a placeholder key is sent, so a real `OPENAI_API_KEY` in your
environment is never forwarded to a local server.

The same keys work for `VISION_MODEL_ID` (an MLX `multimodal` model) and
`RAG.EMBED_MODEL_ID` (its `/v1/embeddings`) — one server can back all three
sections, on the same port, provided it has each model loaded.

## Anthropic (Claude API) { #anthropic-claude-api }

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

## Amazon SageMaker AI { #amazon-sagemaker-ai }

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

## LiteLLM (100+ Providers) { #litellm-100-providers }

```yaml
MODEL_ID:
  NAME: openai/your-model-name
  TYPE: litellm
  API_BASE: http://localhost:8000/v1
  API_KEY: your-api-key
  TEMPERATURE: 0.1
  MAX_TOKENS: 4096
```
