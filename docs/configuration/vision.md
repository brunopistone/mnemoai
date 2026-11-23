# Vision models { #vision-model-configuration }

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

For a local MLX server (the model must be loaded as `multimodal`):

```yaml
VISION_MODEL_ID:
  NAME: qwen3-vl-4bit
  TYPE: mlx
  HOST: 127.0.0.1
  PORT: 8000
  TEMPERATURE: 0.3
  KEEP_ALIVE: 5m # a vision model is called rarely; don't hold RAM for long
# TOP_K is not offered here: the multimodal path never passes it to the sampler.
```
