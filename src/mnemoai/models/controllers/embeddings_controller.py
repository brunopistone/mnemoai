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
        """Initialize embeddings controller.

        Args:
            embed_model_config: Optional embedding model configuration
        """
        self.embed_model_config = embed_model_config or config.get("RAG", {}).get(
            "EMBED_MODEL_ID", {}
        )
        self.embed_model_type = self.embed_model_config.get("TYPE", "ollama")
        self.embed_model_name = self.embed_model_config.get("NAME", "mxbai-embed-large")
        self.region = self.embed_model_config.get("REGION", "us-east-1")
        # LiteLLM connection (optional): OpenAI-compatible base URL + key.
        self.api_base = self.embed_model_config.get("API_BASE")
        self.api_key = self.embed_model_config.get("API_KEY")

        # Set expected dimension based on model
        model_dims = {
            "mxbai-embed-large": 1024,
            "nomic-embed-text": 768,
            "all-minilm": 384,
        }
        self.dim = model_dims.get(self.embed_model_name, 1024)  # Default to 1024

        # Initialize embedding cache for performance
        embeddings_config = config.get("RAG", {}).get("EMBEDDINGS", {})
        self.cache_enabled = embeddings_config.get("CACHE_ENABLED", True)
        self.cache_size = embeddings_config.get("CACHE_SIZE", 1000)
        self._embedding_cache = {}  # {cache_key: embedding_vector}
        self._cache_order = []  # List of keys for LRU tracking

        if self.cache_enabled:
            logger.debug(f"Embedding cache enabled with max size: {self.cache_size}")

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text using MD5 hash.

        Args:
            text: Text to generate key for

        Returns:
            MD5 hash of text
        """
        return hashlib.md5(text.encode()).hexdigest()

    def _update_cache(self, text: str, embedding: np.ndarray) -> None:
        """Add embedding to cache with LRU eviction.

        Args:
            text: Text that was embedded
            embedding: Embedding vector
        """
        key = self._cache_key(text)

        # Add to cache
        self._embedding_cache[key] = embedding

        # Update LRU order
        if key in self._cache_order:
            self._cache_order.remove(key)
        self._cache_order.append(key)

        # Evict oldest if cache is full
        while len(self._embedding_cache) > self.cache_size:
            oldest_key = self._cache_order.pop(0)
            if oldest_key in self._embedding_cache:
                del self._embedding_cache[oldest_key]

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using configured provider with optional caching.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings with shape (n, dim)
        """
        logger.debug(
            f"Embedding {len(texts)} texts using {self.embed_model_type} model '{self.embed_model_name}'"
        )
        if not texts:
            logger.warning("Empty text list provided to embed()")
            return np.array([], dtype=np.float32).reshape(0, self.dim or 768)

        # Check cache if enabled
        if self.cache_enabled:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                key = self._cache_key(text)
                if key in self._embedding_cache:
                    # Cache hit
                    cached_embeddings.append((i, self._embedding_cache[key]))
                    # Update LRU order
                    if key in self._cache_order:
                        self._cache_order.remove(key)
                        self._cache_order.append(key)
                else:
                    # Cache miss
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Log cache performance
            if cached_embeddings:
                logger.debug(
                    f"Cache hit: {len(cached_embeddings)}/{len(texts)} embeddings"
                )

            # If all cached, return immediately
            if not uncached_texts:
                return np.vstack([emb for _, emb in sorted(cached_embeddings)])

            # Embed uncached texts
            new_embeddings = self._embed_uncached(uncached_texts)

            # Cache the new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                self._update_cache(text, embedding)

            # Reconstruct full result in original order
            results = [None] * len(texts)
            for i, embedding in cached_embeddings:
                results[i] = embedding
            for i, idx in enumerate(uncached_indices):
                results[idx] = new_embeddings[i]

            return np.vstack(results)
        else:
            # Cache disabled, embed directly
            return self._embed_uncached(texts)

    def _embed_uncached(self, texts: List[str]) -> np.ndarray:
        """Embed texts without caching (internal method).

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings
        """
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
        """Embed using Ollama.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings
        """
        try:
            resp = ollama.embed(model=self.embed_model_name, input=texts)
            emb = self._extract_embeddings_from_response(resp)
            return np.array(emb, dtype=np.float32)
        except Exception:
            logger.exception(
                "Ollama embed failed, falling back to deterministic embeddings"
            )
            return self._embed_fallback(texts)

    def _embed_bedrock(self, texts: List[str]) -> np.ndarray:
        """Embed using AWS Bedrock.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings
        """
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
        """Embed using OpenAI.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings
        """
        try:
            client = OpenAI()
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
        """Embed using AWS SageMaker.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings
        """
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
        """Embed using LiteLLM (100+ providers via one OpenAI-style API).

        `litellm.embedding(model, input, api_base, api_key)` returns an
        OpenAI-shaped response: vectors live at `response.data[i]["embedding"]`.
        `API_BASE`/`API_KEY` are optional — omitted keys fall back to the
        provider's own env vars (e.g. OPENAI_API_KEY).

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings
        """
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
        """Fallback to deterministic SHA256-based embeddings with warning.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of deterministic embeddings
        """
        # Get fallback configuration
        fallback_config = config.get("RAG", {}).get("EMBEDDINGS", {})
        fallback_type = fallback_config.get("FALLBACK_TYPE", "sha256")

        # Prominent warning about degraded functionality
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
        """Extract embeddings from Ollama response.

        Args:
            resp: Ollama API response

        Returns:
            List of embedding vectors
        """
        # Handle dict response
        if isinstance(resp, dict):
            if "embeddings" in resp:
                return resp["embeddings"]
            elif "embedding" in resp:
                emb = resp["embedding"]
                return [emb] if isinstance(emb[0], (int, float)) else emb

        # Handle object with attributes
        if hasattr(resp, "embeddings"):
            raw = resp.embeddings
            if isinstance(raw, list):
                return raw
        elif hasattr(resp, "embedding"):
            raw = resp.embedding
            if isinstance(raw, list):
                return [raw] if isinstance(raw[0], (int, float)) else raw

        raise ValueError(f"Failed to extract embeddings from response: {type(resp)}")
