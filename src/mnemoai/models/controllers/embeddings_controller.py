import hashlib
import json
from typing import Any, List

import boto3
import numpy as np
import ollama
from openai import OpenAI

from mnemoai.utils.config import config
from mnemoai.utils.logger import logger


class EmbeddingsController:
    """Controller for embedding model operations across different providers."""

    def __init__(self, embed_model_config: dict = None) -> None:
        self.embed_model_config = embed_model_config or config.get("RAG", {}).get(
            "EMBED_MODEL_ID", {}
        )
        self.embed_model_type = self.embed_model_config.get("TYPE", "ollama")
        self.embed_model_name = self.embed_model_config.get("NAME", "mxbai-embed-large")
        self.region = self.embed_model_config.get("REGION", "us-east-1")
        # Optional OpenAI-compatible connection (API_BASE, alias ENDPOINT_URL, +
        # API_KEY) so a local server can serve embeddings.
        self.api_base = self.embed_model_config.get(
            "API_BASE"
        ) or self.embed_model_config.get("ENDPOINT_URL")
        self.api_key = self.embed_model_config.get("API_KEY")

        # Dimension used ONLY for fallback vectors / empty-result shape (real
        # embeddings keep their native size). Explicit DIMENSION wins, else a
        # known-model lookup, else 1024.
        model_dims = {
            "mxbai-embed-large": 1024,
            "nomic-embed-text": 768,
            "all-minilm": 384,
            "qwen3-embedding": 1024,
            "qwen3-embedding:0.6b": 1024,
        }
        configured_dim = self.embed_model_config.get("DIMENSION")
        if configured_dim is not None:
            self.dim = int(configured_dim)
        else:
            self.dim = model_dims.get(self.embed_model_name, 1024)

        embeddings_config = config.get("RAG", {}).get("EMBEDDINGS", {})
        self.cache_enabled = embeddings_config.get("CACHE_ENABLED", True)
        self.cache_size = embeddings_config.get("CACHE_SIZE", 1000)
        self._embedding_cache = {}  # {cache_key: embedding_vector}
        self._cache_order = []  # LRU tracking

        if self.cache_enabled:
            logger.debug(f"Embedding cache enabled with max size: {self.cache_size}")

    def _cache_key(self, text: str) -> str:
        """MD5 cache key for a text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _update_cache(self, text: str, embedding: np.ndarray) -> None:
        """Add an embedding to the cache with LRU eviction."""
        key = self._cache_key(text)
        self._embedding_cache[key] = embedding

        if key in self._cache_order:
            self._cache_order.remove(key)
        self._cache_order.append(key)

        while len(self._embedding_cache) > self.cache_size:
            oldest_key = self._cache_order.pop(0)
            if oldest_key in self._embedding_cache:
                del self._embedding_cache[oldest_key]

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts via the configured provider (with optional caching);
        returns an ``(n, dim)`` array."""
        logger.debug(
            f"Embedding {len(texts)} texts using {self.embed_model_type} model '{self.embed_model_name}'"
        )
        if not texts:
            logger.warning("Empty text list provided to embed()")
            return np.array([], dtype=np.float32).reshape(0, self.dim or 768)

        if not self.cache_enabled:
            return self._embed_uncached(texts)

        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._embedding_cache:
                cached_embeddings.append((i, self._embedding_cache[key]))
                if key in self._cache_order:
                    self._cache_order.remove(key)
                    self._cache_order.append(key)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if cached_embeddings:
            logger.debug(
                f"Cache hit: {len(cached_embeddings)}/{len(texts)} embeddings"
            )

        if not uncached_texts:
            return np.vstack([emb for _, emb in sorted(cached_embeddings)])

        new_embeddings = self._embed_uncached(uncached_texts)
        for text, embedding in zip(uncached_texts, new_embeddings):
            self._update_cache(text, embedding)

        # Reassemble in original order.
        results = [None] * len(texts)
        for i, embedding in cached_embeddings:
            results[i] = embedding
        for i, idx in enumerate(uncached_indices):
            results[idx] = new_embeddings[i]

        return np.vstack(results)

    def _embed_uncached(self, texts: List[str]) -> np.ndarray:
        """Dispatch to the provider-specific embed method (no caching)."""
        if self.embed_model_type == "ollama":
            return self._embed_ollama(texts)
        elif self.embed_model_type == "bedrock":
            return self._embed_bedrock(texts)
        elif self.embed_model_type == "openai":
            return self._embed_openai(texts)
        elif self.embed_model_type == "sagemaker":
            return self._embed_sagemaker(texts)
        elif self.embed_model_type == "litellm":
            return self._embed_litellm(texts)
        else:
            raise ValueError(
                f"Unsupported embedding model type: {self.embed_model_type}"
            )

    def _embed_ollama(self, texts: List[str]) -> np.ndarray:
        """Embed using Ollama at the configured HOST/PORT."""
        # Honor HOST/PORT instead of letting bare ollama.embed() fall back to
        # $OLLAMA_HOST (which could hijack the call to a different server).
        host = "http://{}:{}".format(
            self.embed_model_config.get("HOST", "localhost"),
            self.embed_model_config.get("PORT", 11434),
        )
        try:
            resp = ollama.Client(host=host).embed(
                model=self.embed_model_name, input=texts
            )
            emb = self._extract_embeddings_from_response(resp)
            return np.array(emb, dtype=np.float32)
        except Exception:
            logger.exception(
                "Ollama embed failed, falling back to deterministic embeddings"
            )
            return self._embed_fallback(texts)

    def _embed_bedrock(self, texts: List[str]) -> np.ndarray:
        """Embed using AWS Bedrock."""
        try:
            client = boto3.client("bedrock-runtime", region_name=self.region)

            embeddings = []
            for text in texts:
                response = client.invoke_model(
                    modelId=self.embed_model_name, body=json.dumps({"inputText": text})
                )
                result = json.loads(response["body"].read())
                embeddings.append(result.get("embedding", []))

            return np.array(embeddings, dtype=np.float32)
        except Exception:
            logger.exception(
                "Bedrock embed failed, falling back to deterministic embeddings"
            )
            return self._embed_fallback(texts)

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """Embed via OpenAI or an OpenAI-compatible server.

        Honors ``API_BASE`` (alias ``ENDPOINT_URL``) + ``API_KEY`` for a local
        server; falls back to the OpenAI API (``OPENAI_API_KEY``).
        """
        try:
            client_kwargs = {}
            if self.api_base:
                client_kwargs["base_url"] = self.api_base
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            elif self.api_base:
                # Local servers ignore auth; placeholder key so the client builds.
                client_kwargs["api_key"] = "sk-local"
            client = OpenAI(**client_kwargs)
            response = client.embeddings.create(
                model=self.embed_model_name, input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)
        except Exception:
            logger.exception(
                "OpenAI embed failed, falling back to deterministic embeddings"
            )
            return self._embed_fallback(texts)

    def _embed_sagemaker(self, texts: List[str]) -> np.ndarray:
        """Embed using AWS SageMaker."""
        try:
            client = boto3.client("sagemaker-runtime", region_name=self.region)

            response = client.invoke_endpoint(
                EndpointName=self.embed_model_name,
                ContentType="application/json",
                Body=json.dumps({"inputs": texts}),
            )
            result = json.loads(response["Body"].read())
            embeddings = result.get("embeddings", result)

            return np.array(embeddings, dtype=np.float32)
        except Exception:
            logger.exception(
                "SageMaker embed failed, falling back to deterministic embeddings"
            )
            return self._embed_fallback(texts)

    def _embed_litellm(self, texts: List[str]) -> np.ndarray:
        """Embed via LiteLLM (OpenAI-shaped response; API_BASE/API_KEY optional,
        else the provider's own env vars)."""
        try:
            from litellm import embedding as litellm_embedding

            kwargs = {"model": self.embed_model_name, "input": texts}
            if self.api_base:
                kwargs["api_base"] = self.api_base
            if self.api_key:
                kwargs["api_key"] = self.api_key

            response = litellm_embedding(**kwargs)
            # response.data is ordered by input index; sort defensively.
            items = sorted(response.data, key=lambda d: d.get("index", 0))
            embeddings = [item["embedding"] for item in items]
            return np.array(embeddings, dtype=np.float32)
        except Exception:
            logger.exception(
                "LiteLLM embed failed, falling back to deterministic embeddings"
            )
            return self._embed_fallback(texts)

    def _embed_fallback(self, texts: List[str]) -> np.ndarray:
        """Deterministic SHA256-based embeddings (degraded; logs a warning)."""
        fallback_config = config.get("RAG", {}).get("EMBEDDINGS", {})
        fallback_type = fallback_config.get("FALLBACK_TYPE", "sha256")

        logger.warning(
            f"⚠️  Using fallback embeddings ({fallback_type}) - semantic search will be DEGRADED. "
            f"Embeddings will not capture semantic meaning. "
            f"Please check embedding model availability (Ollama/OpenAI/Bedrock)."
        )

        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            if self.dim and len(v) < self.dim:
                v = np.resize(v, self.dim)
            elif self.dim:
                v = v[: self.dim]
            v = v / (np.linalg.norm(v) + 1e-12)
            out.append(v)
        return np.vstack(out)

    def _extract_embeddings_from_response(self, resp: Any) -> List[List[float]]:
        """Extract embedding vectors from an Ollama response (dict or object)."""
        if isinstance(resp, dict):
            if "embeddings" in resp:
                return resp["embeddings"]
            elif "embedding" in resp:
                emb = resp["embedding"]
                return [emb] if isinstance(emb[0], (int, float)) else emb

        if hasattr(resp, "embeddings"):
            raw = resp.embeddings
            if isinstance(raw, list):
                return raw
        elif hasattr(resp, "embedding"):
            raw = resp.embedding
            if isinstance(raw, list):
                return [raw] if isinstance(raw[0], (int, float)) else raw

        raise ValueError(f"Failed to extract embeddings from response: {type(resp)}")
