# Episodic memory settings { #episodic-memory-configuration }

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

**Requires:** An embedding model configured via `RAG.EMBED_MODEL_ID` (see [Embeddings Model](embeddings.md#embeddings-model)).

**How it works, success detection, storage paths, and a worked example** of
the injected context are covered on the conceptual page: see
[Episodic Memory](../guides/memory.md#episodic-memory).
