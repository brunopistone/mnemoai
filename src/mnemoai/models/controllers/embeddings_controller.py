import hashlib
import json
import time
from typing import Any, List, Optional

import boto3
import numpy as np
import ollama
from openai import OpenAI

from mnemoai.utils.config import config
from mnemoai.utils.logger import logger
from mnemoai.utils.tokenization import count_tokens

# Conservative token ceiling when the embed model's context can't be resolved.
# Fits common embedding models (OpenAI text-embedding-3 ~8191, Bedrock Titan v2
# 8192, nomic/qwen3 ≥ 8192); a smaller-context model just truncates a bit early.
_DEFAULT_EMBED_TOKEN_LIMIT = 8192

# Floor for the adaptive shrink: never truncate an embed input below this many
# tokens (every real embedding model handles at least this much; going lower
# would mangle the text more than it helps).
_MIN_EMBED_TOKEN_LIMIT = 256

# Substrings that mark an "input too big" embed failure — retrying the IDENTICAL
# input can't help, but a SHORTER input can, so on these we shrink-and-retry
# (adaptive) rather than resend or spam retries. Kept provider-agnostic AND
# runner-agnostic: besides the explicit "context/too long" phrasings, an Ollama/
# llama.cpp embedding runner rejects an over-length batch by dropping the socket
# — surfacing as a bare "EOF" / "status code: 400" / connection error with no
# helpful text — so those count as probable-overflow too (the alternative, a
# blind 3x resend then permanent sha256 fallback, is exactly the bug this fixes).
_OVERFLOW_ERROR_MARKERS = (
    "context length",
    "context window",
    "input length exceeds",
    "maximum context",
    "too large",
    "too long",
    "eof",
    "status code: 400",
    "broken pipe",
    "connection reset",
    "connection aborted",
)

# Learned per-(provider, model, host) embedding token limit, discovered at
# runtime by the adaptive shrink: the largest budget observed to succeed / the
# budget we shrank to after a failure. Shared across controller instances so the
# limit is learned ONCE per process, not re-discovered on every new controller.
_LEARNED_EMBED_LIMITS: dict = {}


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
        # Cap per-text input as defensive hygiene against a genuinely huge text
        # OOM-ing the embed runner (tune via RAG.EMBEDDINGS.MAX_INPUT_CHARS, 0
        # disables). Generous — transient runner EOFs are handled by retry below,
        # not by truncation.
        self.max_input_chars = int(embeddings_config.get("MAX_INPUT_CHARS", 200000))
        # Token cap: the real guard against overflowing the embed model's context
        # (chars alone can't bound tokens). Provider-agnostic — applied in the
        # shared truncate before dispatch. Explicit MAX_INPUT_TOKENS config wins;
        # else a conservative default that the adaptive shrink lowers at runtime
        # to the model's REAL limit the first time an over-length input is
        # rejected (we don't trust a model's reported generation context — an
        # embed runner often accepts far less than the model advertises).
        cfg_tokens = self.embed_model_config.get("MAX_INPUT_TOKENS")
        self._max_input_tokens: Optional[int] = (
            int(cfg_tokens) if cfg_tokens is not None else None
        )
        self._token_limit_resolved = self._max_input_tokens is not None
        # Retry a transient embed-runner failure before degrading to fallback.
        self._embed_retries = int(embeddings_config.get("RETRIES", 3))
        self._embed_retry_delay = float(embeddings_config.get("RETRY_DELAY", 0.5))
        self._embedding_cache = {}  # {cache_key: embedding_vector}
        self._cache_order = []  # LRU tracking

        if self.cache_enabled:
            logger.debug(f"Embedding cache enabled with max size: {self.cache_size}")

    def _learned_key(self) -> str:
        """Cache key for the runtime-learned embed token limit (per provider+model
        +endpoint, since the same model name can be served by runners with
        different real limits)."""
        return "|".join(
            [
                str(self.embed_model_type),
                str(self.embed_model_name),
                str(self.api_base or ""),
                str(self.embed_model_config.get("HOST", "")),
                str(self.embed_model_config.get("PORT", "")),
            ]
        )

    def _resolve_token_limit(self) -> Optional[int]:
        """The per-text token ceiling, resolved once and cached.

        Order: a runtime-**learned** limit (discovered by the adaptive shrink when
        a real embed call rejected an over-length input) → explicit
        ``MAX_INPUT_TOKENS`` config → a conservative default. NOTE we deliberately
        do NOT trust a model's reported *generation* context here: a model can
        advertise 32k while its embedding runner EOFs at a few thousand tokens
        (the reported number is the base-model context, not the embedder's), so a
        probe would over-cap and skip truncation. The learned limit is the only
        model-specific number we trust — and it's provider-agnostic (any runner
        that rejects over-length input teaches us its real limit by failing once).
        """
        if self._token_limit_resolved:
            return self._max_input_tokens

        learned = _LEARNED_EMBED_LIMITS.get(self._learned_key())
        limit = learned or _DEFAULT_EMBED_TOKEN_LIMIT

        self._max_input_tokens = limit
        self._token_limit_resolved = True
        logger.debug(
            "Embed token limit resolved to %d%s",
            limit,
            " (learned)" if learned else " (default)",
        )
        return limit

    def _lower_learned_limit(self, new_limit: int) -> None:
        """Record a smaller working/ceiling embed token limit, learned from a
        failure, so future inputs truncate proactively instead of failing again.
        Shrinks the process-wide cache and this instance's active cap."""
        new_limit = max(_MIN_EMBED_TOKEN_LIMIT, int(new_limit))
        key = self._learned_key()
        current = _LEARNED_EMBED_LIMITS.get(key)
        if current is None or new_limit < current:
            _LEARNED_EMBED_LIMITS[key] = new_limit
        self._max_input_tokens = _LEARNED_EMBED_LIMITS[key]
        self._token_limit_resolved = True

    def _truncate(self, text: str) -> str:
        """Cap a text so it can't overrun the embed model's context (all providers).

        Two guards: a token cap (the real limit — resolved from config/model/default)
        and the coarse char cap (cheap hygiene, ``MAX_INPUT_CHARS``, 0 disables).
        Warns once per oversized text."""
        # Token cap first — it's the guard that actually matches the model.
        limit = self._resolve_token_limit()
        if limit and count_tokens(text) > limit:
            # Trim by chars proportionally to the token overage, then verify.
            approx_chars = max(1, int(len(text) * limit / max(1, count_tokens(text))))
            trimmed = text[:approx_chars]
            while count_tokens(trimmed) > limit and len(trimmed) > 1:
                trimmed = trimmed[: int(len(trimmed) * 0.9)]
            logger.debug(
                "Embed input ~%d tokens > limit %d; truncating to fit the model context.",
                count_tokens(text),
                limit,
            )
            text = trimmed

        if self.max_input_chars and len(text) > self.max_input_chars:
            logger.debug(
                "Embed input %d chars > cap %d; truncating.",
                len(text),
                self.max_input_chars,
            )
            return text[: self.max_input_chars]
        return text

    @staticmethod
    def _is_overflow_error(exc: Exception) -> bool:
        """True if the error means the input was too big for the model context
        (deterministic — not worth retrying the identical input)."""
        msg = str(exc).lower()
        return any(marker in msg for marker in _OVERFLOW_ERROR_MARKERS)

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

        texts = [self._truncate(t) for t in texts]

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
        """Embed ``texts`` with a provider-agnostic retry / shrink / fallback loop.

        This is the single place ALL providers go through, so the recovery
        behavior is identical for every model:

        1. Truncate each text to the current token limit (learned or default).
        2. Call the raw provider embed once.
        3. On an **overflow-type** failure (input too big — including a bare
           runner ``EOF``/400 with no message, which llama.cpp uses to reject an
           over-length batch): HALVE the token limit, remember it (so future
           inputs truncate proactively), re-truncate, and retry. This discovers
           any model's real embedding limit at runtime — no per-model table — so
           it works for every provider and model, exactly the requirement.
        4. On a **transient** failure (a genuine hiccup — first call after a
           (re)load, momentary socket error on normal-size input): brief backoff
           and retry the same input.
        5. Only after exhausting the budget do we degrade to deterministic
           (sha256) embeddings.

        All logging here is DEBUG (expected, self-healing) except the final
        give-up, which is a single ERROR — so a normal recovery is silent.
        """
        attempts = max(1, self._embed_retries)
        last_exc = None
        for attempt in range(attempts):
            capped = [self._truncate(t) for t in texts]
            try:
                emb = self._embed_raw(capped)
                # Success: this size works — remember it as a safe ceiling so a
                # later bigger input truncates to it proactively.
                if self._max_input_tokens:
                    _LEARNED_EMBED_LIMITS.setdefault(
                        self._learned_key(), self._max_input_tokens
                    )
                return emb
            except Exception as e:
                last_exc = e
                is_last = attempt == attempts - 1
                if self._is_overflow_error(e):
                    # Input too big for this runner (or a bare EOF/400 that almost
                    # always means that). Shrink the limit and retry a smaller
                    # input — this is how we learn the model's real ceiling.
                    prev = self._max_input_tokens or _DEFAULT_EMBED_TOKEN_LIMIT
                    if prev > _MIN_EMBED_TOKEN_LIMIT and not is_last:
                        self._lower_learned_limit(prev // 2)
                        logger.debug(
                            "Embed rejected input (%s); shrinking token limit to "
                            "%d and retrying.",
                            type(e).__name__,
                            self._max_input_tokens,
                        )
                        continue
                    # Can't shrink further (already at floor) or out of attempts.
                    break
                if not is_last:
                    logger.debug(
                        "Embed attempt %d/%d failed (%s); retrying",
                        attempt + 1,
                        attempts,
                        type(e).__name__,
                    )
                    time.sleep(self._embed_retry_delay * (attempt + 1))
        logger.error(
            "Embedding failed after %d attempts, using deterministic fallback: %s",
            attempts,
            last_exc,
        )
        return self._embed_fallback(texts)

    def _embed_raw(self, texts: List[str]) -> np.ndarray:
        """One provider embed attempt with NO retry/fallback — raises on failure.

        The generic loop in :meth:`_embed_uncached` owns recovery, so each
        provider method just makes the call and lets exceptions propagate."""
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
        """One Ollama embed call at the configured HOST/PORT; raises on failure.

        Recovery (retry/shrink/fallback) is handled generically by the caller."""
        # Honor HOST/PORT instead of letting bare ollama.embed() fall back to
        # $OLLAMA_HOST (which could hijack the call to a different server).
        host = "http://{}:{}".format(
            self.embed_model_config.get("HOST", "localhost"),
            self.embed_model_config.get("PORT", 11434),
        )
        client = ollama.Client(host=host)
        # truncate=True asks the runner to clamp to context as a backstop; inputs
        # are already token-capped in _truncate (some runners honor this, some
        # EOF instead — the caller's shrink loop covers the latter).
        resp = client.embed(
            model=self.embed_model_name, input=texts, truncate=True
        )
        emb = self._extract_embeddings_from_response(resp)
        return np.array(emb, dtype=np.float32)

    def _embed_bedrock(self, texts: List[str]) -> np.ndarray:
        """One AWS Bedrock embed call; raises on failure (caller handles recovery)."""
        client = boto3.client("bedrock-runtime", region_name=self.region)
        embeddings = []
        for text in texts:
            response = client.invoke_model(
                modelId=self.embed_model_name, body=json.dumps({"inputText": text})
            )
            result = json.loads(response["body"].read())
            embeddings.append(result.get("embedding", []))
        return np.array(embeddings, dtype=np.float32)

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """One OpenAI / OpenAI-compatible embed call; raises on failure.

        Honors ``API_BASE`` (alias ``ENDPOINT_URL``) + ``API_KEY`` for a local
        server; falls back to the OpenAI API (``OPENAI_API_KEY``). Recovery is
        handled by the generic caller."""
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

    def _embed_sagemaker(self, texts: List[str]) -> np.ndarray:
        """One AWS SageMaker embed call; raises on failure (caller handles recovery)."""
        client = boto3.client("sagemaker-runtime", region_name=self.region)
        response = client.invoke_endpoint(
            EndpointName=self.embed_model_name,
            ContentType="application/json",
            Body=json.dumps({"inputs": texts}),
        )
        result = json.loads(response["Body"].read())
        embeddings = result.get("embeddings", result)
        return np.array(embeddings, dtype=np.float32)

    def _embed_litellm(self, texts: List[str]) -> np.ndarray:
        """One LiteLLM embed call (OpenAI-shaped response); raises on failure.

        API_BASE/API_KEY optional, else the provider's own env vars. Recovery is
        handled by the generic caller."""
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

    def _embed_fallback(self, texts: List[str]) -> np.ndarray:
        """Deterministic SHA256-based embeddings (degraded; logs an error).

        Logged at ERROR (not WARNING): reaching the fallback means semantic
        search is genuinely degraded — that's an error the user should see even at
        the default log level. The expected, self-healing retry/shrink steps that
        precede it log at DEBUG, so a normal recovery stays silent."""
        fallback_config = config.get("RAG", {}).get("EMBEDDINGS", {})
        fallback_type = fallback_config.get("FALLBACK_TYPE", "sha256")

        logger.error(
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
