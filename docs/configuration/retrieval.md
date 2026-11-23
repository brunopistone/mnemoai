# Document retrieval settings { #rag-configuration }

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
    FALLBACK_ENABLED: false # legacy, accepted but ignored
    FALLBACK_TYPE: sha256 # legacy, accepted but ignored
```

**Requires:** An embedding model configured via `RAG.EMBED_MODEL_ID` (see [Embeddings Model](embeddings.md#embeddings-model)).
