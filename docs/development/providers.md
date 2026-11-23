# Model provider integration

## Switching Model Providers { #switching-model-providers }

The application uses **controller classes** for centralized model management. To switch providers, just update `config.yaml`:

**For LLM:**

```yaml
MODEL_ID:
  NAME: your-model-name
  TYPE: ollama # or bedrock, sagemaker
```

**For Vision:**

```yaml
VISION_MODEL_ID:
  NAME: your-vision-model
  TYPE: ollama # or sagemaker
```

**For Embeddings:**

```yaml
RAG:
  EMBED_MODEL_ID:
    NAME: qwen3-embedding:0.6b
    TYPE: ollama
```

The controllers (`llm_controller.py`, `vision_model_controller.py`, `embeddings_controller.py`) handle all provider-specific initialization automatically.

## Adding New Model Providers { #adding-new-model-providers }

1. Update the appropriate controller in `models/`:

```python
def initialize_model(self):
    if self.model_type == "your_provider":
        # Your provider initialization
        self.model = YourProviderModel(...)
```

2. Add configuration in `config.yaml`
