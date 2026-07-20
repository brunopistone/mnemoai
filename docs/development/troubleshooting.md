# Troubleshooting

### Common Issues

**MCP Connection Errors**

- Verify Python path in `client.py` matches your environment
- Check server path is correct
- Ensure all dependencies are installed (`pip install -r requirements.txt`)

**Model Loading Issues**

- Verify model name and type in `config.yaml`
- For Ollama: Ensure Ollama is running (`ollama serve`) and model is pulled (`ollama pull model-name`)
- For AWS Bedrock: Check credentials (`aws sts get-caller-identity`), region, and model access
- For OpenAI: Ensure `OPENAI_API_KEY` environment variable is set

**RAG / Episodic Memory Not Working**

- Ensure `ENABLE_RAG: true` (or `ENABLE_EPISODIC_MEMORY: true`) in config
- Verify embedding model is configured and available (`RAG.EMBED_MODEL_ID` in config)
- For Ollama embeddings: ensure the embedding model is pulled, for example `ollama pull qwen3-embedding:0.6b`
- Check logs for "fallback embeddings" warnings — this means the real model is unreachable
- Verify documents are being indexed with `list_documents()`

**Permission Errors**

- Ensure write permissions for `~/.mnemoai/`
- Ensure write permissions for `~/.mnemoai/` (the app home: config, plans, tasks, per-profile state)
- Check file paths in configuration

**Import Errors on Startup**

- Some dependencies (chromadb, faiss-cpu, crawl4ai) can be tricky to install. Check platform-specific instructions.
- On Apple Silicon: `faiss-cpu` may require `pip install faiss-cpu --no-cache-dir`

### Logging

Logs are output to stderr with configurable level:

```bash
LOG_LEVEL=DEBUG mnemoai  # Detailed logs
LOG_LEVEL=INFO mnemoai   # Informational logs
mnemoai                  # Default: WARNING-level diagnostics only
```
