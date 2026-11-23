---
hide:
  - toc
---

# Configuration

Choose models, set defaults, and customize how MnemoAI works across conversations.
Most settings can be changed from the terminal with `/config`, `/model`,
`/params`, and `/features`.

Your normal configuration file is `~/.mnemoai/config/config.yaml`.
To use another configuration:

```bash
MNEMOAI_CONFIG=/path/to/config.yaml mnemoai
```

## Get configured

<div class="grid cards" markdown>

- **Models and providers**

    Connect a local or hosted chat model and configure vision or task-specific models.

    [Choose model providers →](configuration/models.md)

- **Features and defaults**

    Enable the capabilities you need and configure runtime limits.

    [Set feature controls →](configuration/features.md)

- **Knowledge and memory**

    Configure embedding models, document retrieval, and episodic memory.

    [Configure embeddings →](configuration/embeddings.md)

- **Customization and connections**

    Add prompt templates, reusable commands, skills, hooks, and external tools.

    [Customize prompt templates →](configuration/prompts.md)

</div>

## Models

- [Chat model providers](configuration/models.md)
- [Vision models](configuration/vision.md)
- [Models for specific tasks](configuration/area-models.md)
- [Model parameters and restart behavior](configuration/parameters.md)
- [Embedding models](configuration/embeddings.md)

## Settings

- [Runtime settings](configuration/runtime.md)
- [Environment variables](configuration/environment.md)
- [Document retrieval](configuration/retrieval.md)
- [Episodic memory](configuration/episodic-memory.md)
- [Files, profiles, and storage locations](getting-started/mnemoai-directory.md)
- [Example configuration](configuration/sample.md)

## Customize and extend

- [Custom slash commands](configuration/slash-commands.md)
- [Agent skills](guides/agent-skills.md)
- [External MCP servers](guides/mcp.md)
- [Tool hooks](guides/hooks.md)

## Find a previous section

Configuration topics now have dedicated pages. Existing section links below
lead to the corresponding topic.

<a id="configuration_1"></a>

<a id="complete-example-config"></a>
- [Complete example config](configuration/sample.md#complete-example-config)

<a id="model-configuration"></a>
- [Model Configuration](configuration/models.md#model-configuration)

<a id="amazon-bedrock"></a>
- [Amazon Bedrock](configuration/models.md#amazon-bedrock)

<a id="amazon-bedrock-mantle"></a>
- [Amazon Bedrock Mantle](configuration/models.md#amazon-bedrock-mantle)

<a id="ollama-local"></a>
- [Ollama (Local)](configuration/models.md#ollama-local)

<a id="openai"></a>
- [OpenAI](configuration/models.md#openai)

<a id="nebius-token-factory"></a>
- [Nebius Token Factory](configuration/models.md#nebius-token-factory)

<a id="local-openai-compatible-servers-llamacpp-lm-studio-vllm"></a>
- [Local OpenAI-compatible servers (llama.cpp / LM Studio / vLLM)](configuration/models.md#local-openai-compatible-servers-llamacpp-lm-studio-vllm)

<a id="mlx-server-local-apple-silicon"></a>
- [MLX server (Local, Apple Silicon)](configuration/models.md#mlx-server-local-apple-silicon)

<a id="anthropic-claude-api"></a>
- [Anthropic (Claude API)](configuration/models.md#anthropic-claude-api)

<a id="amazon-sagemaker-ai"></a>
- [Amazon SageMaker AI](configuration/models.md#amazon-sagemaker-ai)

<a id="litellm-100-providers"></a>
- [LiteLLM (100+ Providers)](configuration/models.md#litellm-100-providers)

<a id="vision-model-configuration"></a>
- [Vision Model Configuration](configuration/vision.md#vision-model-configuration)

<a id="per-area-models-area_models"></a>
- [Per-area models (AREA_MODELS)](configuration/area-models.md#per-area-models-area_models)

<a id="model-parameters"></a>
- [Model Parameters](configuration/parameters.md#model-parameters)

<a id="identity-connection-auth"></a>
- [Identity, connection & auth](configuration/parameters.md#identity-connection-auth)

<a id="inference-parameters"></a>
- [Inference parameters](configuration/parameters.md#inference-parameters)

<a id="general-parameters"></a>
- [General Parameters](configuration/runtime.md#general-parameters)

<a id="environment-variables"></a>
- [Environment variables](configuration/environment.md#environment-variables)

<a id="embeddings-configuration"></a>
- [Embeddings Configuration](configuration/embeddings.md#embeddings-configuration)

<a id="llm-interaction-configuration"></a>
- [LLM Interaction Configuration](configuration/runtime.md#llm-interaction-configuration)

<a id="prompts-promptsyaml"></a>
- [Prompts (prompts.yaml)](configuration/prompts.md#prompts-promptsyaml)

<a id="rag-configuration"></a>
- [RAG Configuration](configuration/retrieval.md#rag-configuration)

<a id="episodic-memory-configuration"></a>
- [Episodic Memory Configuration](configuration/episodic-memory.md#episodic-memory-configuration)

<a id="embeddings-model"></a>
- [Embeddings Model](configuration/embeddings.md#embeddings-model)

<a id="configure"></a>
- [Configure](configuration/models.md#configure)

<a id="reasoning-and-tools"></a>
- [Reasoning and tools](configuration/models.md#reasoning-and-tools)

<a id="model-catalog-and-usage"></a>
- [Model catalog and usage](configuration/models.md#model-catalog-and-usage)

<a id="prompt_cache-reuse-the-prompt-prefix-instead-of-re-paying-for-it"></a>
- [PROMPT_CACHE — reuse the prompt prefix instead of re-paying for it](configuration/parameters.md#prompt_cache-reuse-the-prompt-prefix-instead-of-re-paying-for-it)

<a id="extra_params-generic-passthrough-for-anything-else"></a>
- [EXTRA_PARAMS — generic passthrough for anything else](configuration/parameters.md#extra_params-generic-passthrough-for-anything-else)
