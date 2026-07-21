"""Unit tests for the OpenAI-compatible embeddings path honoring API_BASE.

A local server (llama-server / LM Studio / vLLM, e.g. behind llama-swap) can
provide embeddings via TYPE: openai + API_BASE — matching the chat controller.
These tests capture the OpenAI client kwargs without any network call.
"""

import numpy as np
import pytest

import mnemoai.models.controllers.embeddings_controller as ec
from mnemoai.models.controllers.embeddings_controller import EmbeddingsController


@pytest.fixture(autouse=True)
def _clear_learned_limits():
    """The runtime-learned embed limits are a process-global cache; clear it
    between tests so one test's discovered limit doesn't leak into another."""
    ec._LEARNED_EMBED_LIMITS.clear()
    yield
    ec._LEARNED_EMBED_LIMITS.clear()


class _FakeResp:
    data = [type("D", (), {"embedding": [0.1, 0.2, 0.3, 0.4]})()]


class _FakeClient:
    def __init__(self, **kwargs):
        # Record the constructor kwargs for assertions.
        _FakeClient.captured = kwargs

    class embeddings:
        @staticmethod
        def create(model, input):
            return _FakeResp()


def _controller(cfg, monkeypatch):
    monkeypatch.setattr(ec, "OpenAI", lambda **kw: _FakeClient(**kw))
    c = EmbeddingsController(cfg)
    # disable cache so _embed_openai is actually exercised
    c.cache_enabled = False
    return c


def test_api_base_sets_base_url_and_placeholder_key(monkeypatch):
    c = _controller(
        {"NAME": "qwen3-embedding", "TYPE": "openai", "API_BASE": "http://localhost:8080/v1"},
        monkeypatch,
    )
    out = c._embed_openai(["hello"])
    assert _FakeClient.captured["base_url"] == "http://localhost:8080/v1"
    assert _FakeClient.captured["api_key"] == "sk-local"
    assert out.shape == (1, 4)


def test_endpoint_url_alias_and_explicit_key(monkeypatch):
    c = _controller(
        {
            "NAME": "embed",
            "TYPE": "openai",
            "ENDPOINT_URL": "http://localhost:1234/v1",
            "API_KEY": "lm-studio",
        },
        monkeypatch,
    )
    c._embed_openai(["x"])
    assert _FakeClient.captured["base_url"] == "http://localhost:1234/v1"
    assert _FakeClient.captured["api_key"] == "lm-studio"


def test_plain_openai_passes_no_base_url(monkeypatch):
    # Without API_BASE/ENDPOINT_URL -> the bare OpenAI() client (env key).
    c = _controller({"NAME": "text-embedding-3-small", "TYPE": "openai"}, monkeypatch)
    c._embed_openai(["x"])
    assert "base_url" not in _FakeClient.captured
    assert "api_key" not in _FakeClient.captured


class TestEmbeddingDimension:
    """The embedding dimension is configurable via DIMENSION (used for the
    SHA256/zeros fallback and empty-result shape; real embeddings pass through
    at the provider's native size)."""

    def test_explicit_dimension_wins(self):
        c = EmbeddingsController({"NAME": "anything", "TYPE": "openai", "DIMENSION": 1536})
        assert c.dim == 1536

    def test_known_model_lookup_default(self):
        c = EmbeddingsController({"NAME": "nomic-embed-text", "TYPE": "ollama"})
        assert c.dim == 768

    def test_unknown_model_defaults_1024(self):
        c = EmbeddingsController({"NAME": "some-new-embedder", "TYPE": "ollama"})
        assert c.dim == 1024

    def test_fallback_vector_matches_configured_dimension(self):
        c = EmbeddingsController({"NAME": "x", "TYPE": "openai", "DIMENSION": 512})
        c.cache_enabled = False
        out = c._embed_fallback(["hello"])
        assert out.shape == (1, 512)


class TestOllamaHost:
    """The Ollama embed path must honor the configured HOST/PORT (like the LLM
    and vision controllers) instead of the bare ollama.embed() which silently
    reads $OLLAMA_HOST — a wrong OLLAMA_HOST (e.g. a llama-swap port left over
    from a local-engine experiment) otherwise hijacks the embed and it fails.
    The host is resolved inside _embed_ollama (not __init__), so a non-Ollama
    controller never builds an Ollama URL it doesn't use."""

    @staticmethod
    def _patch_client(monkeypatch, captured):
        class _FakeOllamaClient:
            def __init__(self, host=None):
                captured["host"] = host

            def embed(self, model, input, truncate=None):
                return {"embeddings": [[0.1, 0.2, 0.3]] * len(input)}

        monkeypatch.setattr(ec.ollama, "Client", _FakeOllamaClient)

    def test_configured_host_port_used(self, monkeypatch):
        captured = {}
        self._patch_client(monkeypatch, captured)
        c = EmbeddingsController(
            {"NAME": "qwen3-embedding:0.6b", "TYPE": "ollama", "HOST": "myhost", "PORT": 9999}
        )
        c.cache_enabled = False
        c._embed_ollama(["hello"])
        assert captured["host"] == "http://myhost:9999"

    def test_default_host_port(self, monkeypatch):
        captured = {}
        self._patch_client(monkeypatch, captured)
        c = EmbeddingsController({"NAME": "qwen3-embedding:0.6b", "TYPE": "ollama"})
        c.cache_enabled = False
        c._embed_ollama(["hello"])
        assert captured["host"] == "http://localhost:11434"

    def test_configured_host_wins_over_env(self, monkeypatch):
        captured = {}
        self._patch_client(monkeypatch, captured)
        # Even with OLLAMA_HOST pointing elsewhere, the configured host wins.
        monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:56231")
        c = EmbeddingsController(
            {"NAME": "qwen3-embedding:0.6b", "TYPE": "ollama", "HOST": "localhost", "PORT": 11434}
        )
        c.cache_enabled = False
        out = c._embed_ollama(["hello"])
        assert captured["host"] == "http://localhost:11434"
        assert out.shape == (1, 3)

    def test_non_ollama_controller_has_no_ollama_host_attr(self):
        # Resolving the host inside _embed_ollama means a SageMaker/Bedrock/
        # OpenAI controller never grows an Ollama-specific attribute.
        c = EmbeddingsController({"NAME": "my-endpoint", "TYPE": "sagemaker"})
        assert not hasattr(c, "ollama_host")


class TestOllamaRetryAndTruncation:
    """The Ollama runner can EOF (transiently, OR because the input is too long);
    the generic embed path retries — shrinking the input on a probable-overflow
    EOF — before degrading to deterministic fallback. Recovery lives in
    _embed_uncached (provider-agnostic); _embed_ollama is now a single raw call."""

    def _patch_client(self, monkeypatch, fail_times):
        # Fail the first `fail_times` calls (transient EOF), then succeed.
        state = {"calls": 0}

        class _FlakyClient:
            def __init__(self, host=None):
                pass

            def embed(self, model, input, truncate=None):
                state["calls"] += 1
                if state["calls"] <= fail_times:
                    raise ec.ollama.ResponseError("EOF", 400) if hasattr(
                        ec.ollama, "ResponseError"
                    ) else RuntimeError("EOF")
                return {"embeddings": [[0.1, 0.2, 0.3]] * len(input)}

        monkeypatch.setattr(ec.ollama, "Client", _FlakyClient)
        monkeypatch.setattr(ec.time, "sleep", lambda *a: None)  # no real delay
        return state

    def test_retry_recovers_transient_eof(self, monkeypatch):
        # EOF now triggers shrink-and-retry via the generic loop; a short input
        # that fails twice then succeeds is recovered with a real embedding.
        state = self._patch_client(monkeypatch, fail_times=2)  # fail twice, then OK
        c = EmbeddingsController({"NAME": "qwen3-embedding:0.6b", "TYPE": "ollama"})
        c.cache_enabled = False
        c._embed_retries = 3
        out = c.embed(["hello"])            # go through the generic recovery loop
        assert state["calls"] == 3          # retried (shrinking) until success
        assert out.shape == (1, 3)          # real embedding, not fallback

    def test_falls_back_after_exhausting_retries(self, monkeypatch):
        self._patch_client(monkeypatch, fail_times=99)  # always fail
        c = EmbeddingsController({"NAME": "qwen3-embedding:0.6b", "TYPE": "ollama", "DIMENSION": 3})
        c.cache_enabled = False
        c._embed_retries = 3
        out = c.embed(["hello"])            # generic loop owns the fallback now
        assert out.shape == (1, 3)          # deterministic fallback shape

    def _capture_ai_app(self, level):
        # The ai_app logger has propagate=False, so caplog's root handler won't
        # see it — attach a capturing handler directly at the given level.
        import logging

        logger = logging.getLogger("ai_app")

        class _Cap(logging.Handler):
            def __init__(self):
                super().__init__(level)
                self.records = []

            def emit(self, record):
                self.records.append(record)

        h = _Cap()
        logger.addHandler(h)
        return logger, h

    def test_recovery_is_silent_only_fallback_logs(self, monkeypatch):
        # UX contract: the self-healing retry/shrink steps log at DEBUG (invisible
        # at the default WARNING level); only a genuine degrade-to-fallback is
        # ERROR. So a successful recovery emits NOTHING at WARNING+.
        import logging

        self._patch_client(monkeypatch, fail_times=2)  # fails, shrinks, succeeds
        c = EmbeddingsController({"NAME": "qwen3-embedding:0.6b", "TYPE": "ollama"})
        c.cache_enabled = False
        c._embed_retries = 3
        logger, h = self._capture_ai_app(logging.WARNING)
        try:
            c.embed(["hello"])
        finally:
            logger.removeHandler(h)
        assert h.records == []  # recovered silently — no WARNING/ERROR noise

    def test_fallback_logs_at_error_not_warning(self, monkeypatch):
        import logging

        self._patch_client(monkeypatch, fail_times=99)  # never succeeds → fallback
        c = EmbeddingsController(
            {"NAME": "qwen3-embedding:0.6b", "TYPE": "ollama", "DIMENSION": 3}
        )
        c.cache_enabled = False
        c._embed_retries = 3
        logger, h = self._capture_ai_app(logging.DEBUG)
        try:
            c.embed(["hello"])
        finally:
            logger.removeHandler(h)
        # The give-up + the fallback notice are ERROR; none of it is WARNING.
        assert any(r.levelno == logging.ERROR for r in h.records)
        assert not any(r.levelno == logging.WARNING for r in h.records)

    def test_oversized_input_truncated(self):
        c = EmbeddingsController({"NAME": "x", "TYPE": "ollama"})
        c.max_input_chars = 100
        c._max_input_tokens = 0  # isolate the char-cap behavior
        c._token_limit_resolved = True
        assert len(c._truncate("a" * 500)) == 100
        assert c._truncate("short") == "short"

    def test_char_truncation_disabled_when_zero(self):
        # With both caps off, a short text passes through untouched.
        c = EmbeddingsController({"NAME": "x", "TYPE": "ollama"})
        c.max_input_chars = 0
        c._max_input_tokens = 0
        c._token_limit_resolved = True
        assert len(c._truncate("a" * 500)) == 500


class TestTokenCapAndOverflow:
    """The token cap is the real guard against overflowing the embed model's
    context (provider-agnostic, applied in the shared _truncate); overflow
    errors are classified so deterministic ones aren't retried."""

    def test_explicit_max_input_tokens_config_wins(self):
        c = EmbeddingsController(
            {"NAME": "x", "TYPE": "ollama", "MAX_INPUT_TOKENS": 50}
        )
        assert c._resolve_token_limit() == 50  # no probe, no margin applied

    def test_token_cap_truncates_long_input(self):
        c = EmbeddingsController(
            {"NAME": "x", "TYPE": "openai", "MAX_INPUT_TOKENS": 10}
        )
        c.max_input_chars = 0  # isolate the token cap
        long_text = "word " * 500  # ~500+ tokens
        from mnemoai.utils.tokenization import count_tokens

        out = c._truncate(long_text)
        assert count_tokens(out) <= 10

    def test_default_limit_when_unresolved(self, monkeypatch):
        # Non-Ollama with no config + no probe → conservative default.
        c = EmbeddingsController({"NAME": "text-embedding-3-small", "TYPE": "openai"})
        assert c._resolve_token_limit() == ec._DEFAULT_EMBED_TOKEN_LIMIT

    def test_reported_context_is_not_trusted(self):
        # We deliberately do NOT probe/trust the model's reported context (an
        # embed runner often accepts far less than the model advertises); with no
        # explicit config and nothing learned yet, the conservative default holds.
        c = EmbeddingsController({"NAME": "qwen3-embedding:0.6b", "TYPE": "ollama"})
        assert c._resolve_token_limit() == ec._DEFAULT_EMBED_TOKEN_LIMIT

    def test_eof_is_treated_as_probable_overflow(self):
        # A bare runner EOF / 400 is how llama.cpp rejects an over-length batch —
        # so it counts as probable-overflow and drives shrink-and-retry (the old
        # behavior — resend 3x then permanent fallback — was the reported bug).
        assert EmbeddingsController._is_overflow_error(
            Exception("... EOF (status code: 400)")
        ) is True

    def test_context_length_error_is_overflow(self):
        # An explicit context error IS overflow too.
        assert EmbeddingsController._is_overflow_error(
            Exception("the input length exceeds the context length")
        ) is True
        assert EmbeddingsController._is_overflow_error(
            Exception("maximum context length is 8192 tokens")
        ) is True

    def test_transient_non_overflow_error_is_not_overflow(self):
        # A clearly non-size error (auth, model missing) is not overflow — it gets
        # the plain retry, not a shrink.
        assert EmbeddingsController._is_overflow_error(
            Exception("model 'x' not found")
        ) is False

    def test_overflow_shrinks_limit_then_falls_back(self, monkeypatch):
        # An always-overflowing input SHRINKS the token limit each attempt (down
        # toward the floor) and, only after the budget, degrades to fallback.
        state = {"calls": 0}

        class _OverflowClient:
            def __init__(self, host=None):
                pass

            def embed(self, model, input, truncate=None):
                state["calls"] += 1
                raise RuntimeError("the input length exceeds the context length")

        monkeypatch.setattr(ec.ollama, "Client", _OverflowClient)
        monkeypatch.setattr(ec.time, "sleep", lambda *a: None)
        c = EmbeddingsController({"NAME": "x", "TYPE": "ollama", "DIMENSION": 3})
        c.cache_enabled = False
        c._embed_retries = 3
        start = c._resolve_token_limit()
        out = c.embed(["hello world " * 100])
        assert state["calls"] == 3            # tried, shrinking, across the budget
        assert c._max_input_tokens < start    # limit was lowered by the shrink
        assert out.shape == (1, 3)            # then deterministic fallback
        assert out.shape == (1, 3)       # fell back once
