# Embedding models

## Embeddings Configuration { #embeddings-configuration }

Embeddings settings are nested under the `RAG` section:

```yaml
RAG:
  EMBEDDINGS:
    CACHE_ENABLED: true # LRU cache for embedding vectors (avoids re-embedding same text)
    CACHE_SIZE: 1000 # Maximum cached embeddings
    FALLBACK_ENABLED: false # legacy, accepted but ignored
    FALLBACK_TYPE: sha256 # legacy, accepted but ignored
```

Provider failures never produce synthetic vectors. Ingestion and episode storage
require real embeddings and fail without adding data when the provider fails.
Recall and `search_in_documents` instead use BM25-only keyword ranking over
previously stored text when semantic search is unavailable. Those results are
labeled as keyword-only; no new content is indexed during the outage.

The two legacy fallback keys still load with any previous value, but cannot
enable fabricated embeddings. This is a runtime integrity correction, not a
configuration-load break. No separate fallback index is created.

On opening an existing episodic or RAG store, a repair identifies the old
nonnegative, normalized, period-32 tiled-byte vectors. It does **not** remove
vectors based on positivity alone. Removed records' metadata is saved first:
`legacy-synthetic-metadata.json` inside a Chroma store, or
`<index-file>.legacy-synthetic-metadata.json` beside a FAISS index. These backups
can contain private episode/document text; keep them private. They are not
searched. Re-ingest the source document or re-embed saved episode text after the
provider recovers; there is no automatic promotion of fabricated records.
Unidentifiable historical random/zero vectors are not purged by this repair.

The scan is skipped after a successful check. Chroma records a version/count
marker in collection metadata; FAISS records a version and file signatures in
`<index-file>.synthetic-checked.json`. Normal writes refresh these markers.
Older append writers or changed FAISS files invalidate the check; arbitrary
manual database edits are not a supported way to maintain these markers.

If repair cannot safely finish (for example, an unwritable/corrupt backup, an
invalid repair journal, or misaligned FAISS metadata), readable text remains
available through BM25. Semantic search, writes, and automatic episodic cleanup
are disabled rather than silently resetting the store or trusting its vectors.
Fix the reported file problem and reopen the store, or explicitly clear/re-ingest
it. If metadata itself cannot be read, there is no text to retrieve.

## Embeddings Model { #embeddings-model }

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

For a local MLX server:

```yaml
RAG:
  EMBED_MODEL_ID:
    NAME: qwen3-embedding-0.6b # the name the server serves it under
    TYPE: mlx
    HOST: 127.0.0.1
    PORT: 8000
    KEEP_ALIVE: 10m # embedding runs come in bursts; keep it warm between them
    DIMENSION: 1024 # optional: expected real size; otherwise probed from the provider
```

**Vector Store Options:**

- **ChromaDB** (default): Persistent vector database with built-in metadata support
- **FAISS**: Fast, in-memory vector search with disk persistence

Switch between stores by changing `RAG.VECTOR_STORE.TYPE` in config. The system uses a controller pattern, so all RAG functionality works identically regardless of the store.
