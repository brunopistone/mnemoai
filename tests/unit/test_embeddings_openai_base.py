"""Unit tests for the OpenAI-compatible embeddings path honoring API_BASE.

A local server (llama-server / LM Studio / vLLM, e.g. behind llama-swap) can
provide embeddings via TYPE: openai + API_BASE — matching the chat controller.
These tests capture the OpenAI client kwargs without any network call.
"""

import numpy as np

import mnemoai.models.controllers.embeddings_controller as ec
from mnemoai.models.controllers.embeddings_controller import EmbeddingsController


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
