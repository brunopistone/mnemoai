# Nebius Token Factory

MnemoAI supports Nebius Token Factory for chat, vision, embeddings and per-area
models. The provider uses Nebius's OpenAI-compatible Chat Completions API.

## Configure

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

| Configuration task | Interface |
|---|---|
| Initial setup | `/config` → Nebius Token Factory |
| Chat, vision, embeddings and area models | `/model` |
| Sampling, output budget, streaming and reasoning effort | `/params` |
| Enable routing, orchestration, review, RAG or episodic memory | `/features` |
| Check the active configuration and credential presence | `/doctor` |

Changing providers to or from Nebius clears the previous provider's API key and
endpoint from the edited section. Models within Nebius retain their connection
settings. Per-area overrides also keep credentials scoped to their provider.

## Reasoning and tools

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

## Model catalog and usage

Nebius's authenticated `GET /v1/models?verbose=true` catalog reports model IDs,
context lengths, modalities, supported features and sampling parameters. Use the
selected model's actual context limit when setting `MAX_CONVERSATION_TOKENS`.
The example above uses the Qwen Instruct model's 262,144-token window.

Streaming requests explicitly request usage metadata. `/usage` records reported
input/output and cached-input tokens; the context meter uses the last measured
input count when available and marks local estimates with `~`.

Embeddings use `/v1/embeddings`. `DIMENSION`, when configured, is sent as the
requested output dimension. Input limits, caching, retries and outage behavior
use MnemoAI's existing embedding machinery. Failed writes never fabricate
vectors, and existing stores retain their normal keyword-search fallback.
Local playbook learning continues to make no model or embedding calls.

Official references:

- [Quickstart and authentication](https://docs.tokenfactory.nebius.com/quickstart)
- [Chat Completions API](https://docs.tokenfactory.nebius.com/api-reference/inference/create-chat-completion)
- [Function calling](https://docs.tokenfactory.nebius.com/ai-models-inference/function-calling)
- [Model discovery and capabilities](https://docs.tokenfactory.nebius.com/api-reference/models/list-models)
- [Vision inputs](https://docs.tokenfactory.nebius.com/api-reference/examples/vision-capabilities)
- [Embeddings API](https://docs.tokenfactory.nebius.com/api-reference/inference/create-embeddings)
