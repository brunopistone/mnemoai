"""Unit tests for the `mlx` provider (a local MLX server on Apple Silicon).

The server speaks the OpenAI protocol, so all three controllers reuse the OpenAI
clients; what makes it its own TYPE is the HOST/PORT connection (no hand-written
``/v1`` base URL) and ``KEEP_ALIVE``, its per-request model-residency knob.

No network: the langchain/OpenAI classes are replaced with capturing mocks. The
``TestParamsReachTheRequestBody`` class deliberately builds the REAL
``ChatOpenAI`` and checks the payload against the openai SDK's own signature —
the MLX-only params (``top_k``/``min_p``/``repetition_penalty``/``keep_alive``)
must arrive nested under ``extra_body``. That check exists because the mocked
tests below cannot see the failure it guards: routing those params through
``model_kwargs`` flattens them into the top level, which a capturing mock happily
records and the typed ``client.chat.completions.create()`` then rejects with
``TypeError: unexpected keyword argument 'top_k'`` — client-side, before any
request is sent (found by a live call against a real MLX server).
"""

from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def patch_chat_openai(monkeypatch):
    """Replace ChatOpenAI with a kwarg-capturing mock (used by both controllers)."""
    import langchain_openai

    captured = {}

    def _recorder(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return MagicMock(name="ChatOpenAI")

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _recorder)
    return captured


def _llm(monkeypatch, model_id: dict):
    import mnemoai.models.controllers.llm_controller as mod

    def fake_get(key, default=None):
        if key == "MODEL_ID":
            return model_id
        if key == "MAX_CONVERSATION_TOKENS":
            return 8192
        return default

    monkeypatch.setattr(mod.config, "get", fake_get)
    return mod.LangChainLLMController(verbose=False)


def _vision(monkeypatch, model_id: dict):
    import mnemoai.models.controllers.vision_model_controller as mod

    def fake_get(key, default=None):
        if key == "VISION_MODEL_ID":
            return model_id
        if key == "MAX_CONVERSATION_TOKENS":
            return 8192
        return default

    monkeypatch.setattr(mod.config, "get", fake_get)
    return mod.VisionModelController(verbose=False)


class TestChatConnection:
    """HOST/PORT is the ordinary path; API_BASE covers a proxied server."""

    def test_host_port_becomes_the_v1_base_url(self, patch_chat_openai, monkeypatch):
        ctrl = _llm(
            monkeypatch,
            {"NAME": "qwen-agentcoder", "TYPE": "mlx", "HOST": "192.168.1.9",
             "PORT": 8123},
        )
        ctrl.initialize_model()
        assert patch_chat_openai["base_url"] == "http://192.168.1.9:8123/v1"
        assert patch_chat_openai["model"] == "qwen-agentcoder"

    def test_defaults_to_localhost_8000(self, patch_chat_openai, monkeypatch):
        ctrl = _llm(monkeypatch, {"NAME": "m", "TYPE": "mlx"})
        ctrl.initialize_model()
        assert patch_chat_openai["base_url"] == "http://127.0.0.1:8000/v1"

    def test_api_base_wins_over_host_port(self, patch_chat_openai, monkeypatch):
        ctrl = _llm(
            monkeypatch,
            {"NAME": "m", "TYPE": "mlx", "HOST": "ignored", "PORT": 1,
             "API_BASE": "https://mac.internal/mlx/v1"},
        )
        ctrl.initialize_model()
        assert patch_chat_openai["base_url"] == "https://mac.internal/mlx/v1"

    def test_placeholder_key_so_env_openai_key_cannot_leak_in(
        self, patch_chat_openai, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-account-key")
        ctrl = _llm(monkeypatch, {"NAME": "m", "TYPE": "mlx"})
        ctrl.initialize_model()
        assert patch_chat_openai["api_key"] == "sk-local"

    def test_explicit_api_key_is_honored(self, patch_chat_openai, monkeypatch):
        ctrl = _llm(monkeypatch, {"NAME": "m", "TYPE": "mlx", "API_KEY": "shared-secret"})
        ctrl.initialize_model()
        assert patch_chat_openai["api_key"] == "shared-secret"


class TestChatParams:
    """Params ChatOpenAI knows go top-level; MLX-only ones go via extra_body."""

    def test_params_split_between_main_and_extra_body(
        self, patch_chat_openai, monkeypatch
    ):
        ctrl = _llm(
            monkeypatch,
            {
                "NAME": "m", "TYPE": "mlx",
                "TEMPERATURE": 0.6, "TOP_P": 0.95, "MAX_TOKENS": 4096,
                "STOP": ["<|im_end|>"],
                "PRESENCE_PENALTY": 0.1, "FREQUENCY_PENALTY": 0.2,
                "TOP_K": 20, "MIN_P": 0.05, "REPETITION_PENALTY": 1.15,
            },
        )
        ctrl.initialize_model()
        assert patch_chat_openai["temperature"] == 0.6
        assert patch_chat_openai["top_p"] == 0.95
        assert patch_chat_openai["max_tokens"] == 4096
        assert patch_chat_openai["stop"] == ["<|im_end|>"]
        assert patch_chat_openai["presence_penalty"] == 0.1
        assert patch_chat_openai["frequency_penalty"] == 0.2
        # Not part of the OpenAI API -> nested body passthrough, never
        # model_kwargs (which the typed SDK method would reject).
        assert patch_chat_openai["extra_body"] == {
            "top_k": 20, "min_p": 0.05, "repetition_penalty": 1.15
        }
        assert "model_kwargs" not in patch_chat_openai

    def test_min_p_zero_is_still_sent(self, patch_chat_openai, monkeypatch):
        # 0.0 disables min-p but is a real value: build_kwargs omits None, not
        # falsy, so it must survive (STOP is the only truthiness exception).
        ctrl = _llm(monkeypatch, {"NAME": "m", "TYPE": "mlx", "MIN_P": 0.0})
        ctrl.initialize_model()
        assert patch_chat_openai["extra_body"] == {"min_p": 0.0}

    def test_unset_params_are_omitted_entirely(self, patch_chat_openai, monkeypatch):
        ctrl = _llm(monkeypatch, {"NAME": "m", "TYPE": "mlx"})
        ctrl.initialize_model()
        for absent in ("temperature", "top_p", "max_tokens", "stop", "extra_body"):
            assert absent not in patch_chat_openai

    def test_keep_alive_rides_in_the_body(self, patch_chat_openai, monkeypatch):
        ctrl = _llm(monkeypatch, {"NAME": "m", "TYPE": "mlx", "KEEP_ALIVE": "30m"})
        ctrl.initialize_model()
        assert patch_chat_openai["extra_body"] == {"keep_alive": "30m"}

    def test_keep_alive_zero_unloads_immediately(self, patch_chat_openai, monkeypatch):
        # 0 = unload right after the request; falsy, so it must not be dropped.
        ctrl = _llm(monkeypatch, {"NAME": "m", "TYPE": "mlx", "KEEP_ALIVE": 0})
        ctrl.initialize_model()
        assert patch_chat_openai["extra_body"] == {"keep_alive": 0}

    def test_extra_params_merge_into_the_body(self, patch_chat_openai, monkeypatch):
        ctrl = _llm(
            monkeypatch,
            {"NAME": "m", "TYPE": "mlx", "TOP_K": 20,
             "EXTRA_PARAMS": {"repetition_context_size": 256}},
        )
        ctrl.initialize_model()
        assert patch_chat_openai["extra_body"] == {
            "top_k": 20, "repetition_context_size": 256
        }

    def test_stream_is_honored(self, patch_chat_openai, monkeypatch):
        ctrl = _llm(monkeypatch, {"NAME": "m", "TYPE": "mlx", "STREAM": False})
        ctrl.initialize_model()
        assert patch_chat_openai["streaming"] is False


class TestParamsReachTheRequestBody:
    """The real ChatOpenAI, not a mock: proof that the MLX-only params survive.

    Two things have to hold at once, and each hides the other's failure:

    * the params must be *present* in the payload — a wrong destination is
      otherwise silent, since the server just ignores fields it never receives;
    * they must be present *nested under* ``extra_body`` — the payload is
      splatted into ``client.chat.completions.create(**payload)``, a typed method
      that raises ``TypeError`` on any keyword it doesn't declare. Flattened
      MLX-only keys therefore never reach the socket at all.

    A mock cannot see the second half, which is how ``model_kwargs`` shipped and
    only failed on a live call.
    """

    PAYLOAD_PARAMS = {
        "NAME": "qwen-agentcoder", "TYPE": "mlx",
        "TEMPERATURE": 0.6, "TOP_P": 0.95,
        "TOP_K": 20, "MIN_P": 0.05, "REPETITION_PENALTY": 1.15,
        "KEEP_ALIVE": "30m",
    }

    def test_mlx_only_params_are_in_the_payload(self, monkeypatch):
        ctrl = _llm(monkeypatch, dict(self.PAYLOAD_PARAMS))
        ctrl.initialize_model()  # real ChatOpenAI: unknown kwargs would raise here
        model = ctrl.get_model()
        payload = model._get_request_payload([("human", "hi")])
        assert payload["model"] == "qwen-agentcoder"
        assert payload["temperature"] == 0.6 and payload["top_p"] == 0.95
        assert payload["extra_body"] == {
            "top_k": 20, "min_p": 0.05, "repetition_penalty": 1.15,
            "keep_alive": "30m",
        }

    def test_every_top_level_payload_key_is_one_the_sdk_accepts(self, monkeypatch):
        # The guard the mocked tests structurally cannot provide: compare the
        # payload's top-level keys against the SDK method that receives them.
        # `extra_body` is itself a declared parameter, so the MLX-only knobs pass
        # this check exactly because they are nested inside it.
        import inspect

        from openai.resources.chat.completions import Completions

        ctrl = _llm(monkeypatch, dict(self.PAYLOAD_PARAMS))
        ctrl.initialize_model()
        payload = ctrl.get_model()._get_request_payload([("human", "hi")])

        accepted = set(inspect.signature(Completions.create).parameters)
        unexpected = set(payload) - accepted
        assert not unexpected, (
            f"{sorted(unexpected)} would raise TypeError in "
            f"Completions.create(**payload); non-OpenAI params belong in extra_body"
        )


class TestVision:
    """A vision model on MLX is a `multimodal` entry; same client, fewer knobs."""

    def test_host_port_and_placeholder_key(self, patch_chat_openai, monkeypatch):
        ctrl = _vision(monkeypatch, {"NAME": "qwen-vl", "TYPE": "mlx", "PORT": 8001})
        ctrl.initialize_model()
        assert patch_chat_openai["base_url"] == "http://127.0.0.1:8001/v1"
        assert patch_chat_openai["api_key"] == "sk-local"
        assert patch_chat_openai["model"] == "qwen-vl"

    def test_api_base_wins(self, patch_chat_openai, monkeypatch):
        ctrl = _vision(
            monkeypatch,
            {"NAME": "qwen-vl", "TYPE": "mlx", "API_BASE": "http://box:9000/v1"},
        )
        ctrl.initialize_model()
        assert patch_chat_openai["base_url"] == "http://box:9000/v1"

    def test_top_k_is_not_forwarded(self, patch_chat_openai, monkeypatch):
        # The multimodal handler never passes top_k to its sampler, so the
        # registry omits it — configuring it must not fabricate a body field that
        # silently does nothing.
        ctrl = _vision(
            monkeypatch,
            {"NAME": "qwen-vl", "TYPE": "mlx", "TOP_K": 20, "TEMPERATURE": 0.4},
        )
        ctrl.initialize_model()
        assert patch_chat_openai["temperature"] == 0.4
        assert "top_k" not in patch_chat_openai.get("extra_body", {})

    def test_keep_alive_is_supported(self, patch_chat_openai, monkeypatch):
        ctrl = _vision(
            monkeypatch, {"NAME": "qwen-vl", "TYPE": "mlx", "KEEP_ALIVE": "5m"}
        )
        ctrl.initialize_model()
        assert patch_chat_openai["extra_body"] == {"keep_alive": "5m"}
        assert "model_kwargs" not in patch_chat_openai


class TestEmbeddings:
    """/v1/embeddings on the same server; KEEP_ALIVE goes in extra_body."""

    @staticmethod
    def _controller(monkeypatch, cfg):
        import mnemoai.models.controllers.embeddings_controller as ec

        captured = {}

        class _FakeClient:
            def __init__(self, **kwargs):
                captured["client"] = kwargs

            class embeddings:
                @staticmethod
                def create(model, input, **kwargs):
                    captured["model"] = model
                    captured["input"] = input
                    captured["request"] = kwargs
                    return type(
                        "R", (), {"data": [type("D", (), {"embedding": [0.1, 0.2]})()
                                           for _ in input]}
                    )()

        monkeypatch.setattr(ec, "OpenAI", lambda **kw: _FakeClient(**kw))
        ctrl = ec.EmbeddingsController(cfg)
        ctrl.cache_enabled = False
        return ctrl, captured

    def test_host_port_default_and_placeholder_key(self, monkeypatch):
        ctrl, cap = self._controller(
            monkeypatch, {"NAME": "qwen3-embedding", "TYPE": "mlx"}
        )
        out = ctrl._embed_mlx(["hello"])
        assert cap["client"] == {
            "base_url": "http://127.0.0.1:8000/v1", "api_key": "sk-local"
        }
        assert cap["model"] == "qwen3-embedding"
        assert isinstance(out, np.ndarray) and out.shape == (1, 2)

    def test_configured_host_port(self, monkeypatch):
        ctrl, cap = self._controller(
            monkeypatch,
            {"NAME": "e", "TYPE": "mlx", "HOST": "10.0.0.4", "PORT": 8200},
        )
        ctrl._embed_mlx(["x"])
        assert cap["client"]["base_url"] == "http://10.0.0.4:8200/v1"

    def test_api_base_wins_and_key_is_honored(self, monkeypatch):
        ctrl, cap = self._controller(
            monkeypatch,
            {"NAME": "e", "TYPE": "mlx", "HOST": "ignored",
             "API_BASE": "http://box:9/v1", "API_KEY": "k"},
        )
        ctrl._embed_mlx(["x"])
        assert cap["client"] == {"base_url": "http://box:9/v1", "api_key": "k"}

    def test_keep_alive_goes_in_extra_body(self, monkeypatch):
        ctrl, cap = self._controller(
            monkeypatch, {"NAME": "e", "TYPE": "mlx", "KEEP_ALIVE": "2m"}
        )
        ctrl._embed_mlx(["x"])
        assert cap["request"] == {"extra_body": {"keep_alive": "2m"}}

    def test_no_keep_alive_sends_no_extra_body(self, monkeypatch):
        ctrl, cap = self._controller(monkeypatch, {"NAME": "e", "TYPE": "mlx"})
        ctrl._embed_mlx(["x"])
        assert cap["request"] == {}

    def test_dispatch_routes_embed_to_the_mlx_path(self, monkeypatch):
        # Via the public entry point, so the TYPE -> method wiring is covered too.
        ctrl, cap = self._controller(
            monkeypatch, {"NAME": "qwen3-embedding", "TYPE": "mlx", "DIMENSION": 2}
        )
        out = ctrl.embed(["a", "b"])
        assert out.shape == (2, 2)
        assert cap["client"]["base_url"] == "http://127.0.0.1:8000/v1"


class TestEndpointUrlAlias:
    """ENDPOINT_URL is the accepted alias for API_BASE, as on `openai`.

    All three controllers fall back to it, so it has to be DECLARED in all three
    registry entries too: `supported_keys` drives the `/model` provider-switch
    pruning, which strips any key the chosen provider doesn't claim — so an
    undeclared-but-read key is silently deleted from a working config.
    """

    @pytest.mark.parametrize(
        "section", ["MODEL_ID", "VISION_MODEL_ID", "EMBED_MODEL_ID"]
    )
    def test_declared_so_a_model_switch_cannot_prune_it(self, section):
        from mnemoai.models.provider_params import supported_keys

        assert "ENDPOINT_URL" in supported_keys(section, "mlx")

    def test_chat_uses_it_as_the_base_url(self, patch_chat_openai, monkeypatch):
        ctrl = _llm(
            monkeypatch,
            {"NAME": "m", "TYPE": "mlx", "ENDPOINT_URL": "http://box:9/v1"},
        )
        ctrl.initialize_model()
        assert patch_chat_openai["base_url"] == "http://box:9/v1"

    def test_vision_uses_it_as_the_base_url(self, patch_chat_openai, monkeypatch):
        ctrl = _vision(
            monkeypatch,
            {"NAME": "qwen-vl", "TYPE": "mlx", "ENDPOINT_URL": "http://box:9/v1"},
        )
        ctrl.initialize_model()
        assert patch_chat_openai["base_url"] == "http://box:9/v1"

    def test_embeddings_use_it_as_the_base_url(self, monkeypatch):
        ctrl, cap = TestEmbeddings._controller(
            monkeypatch,
            {"NAME": "e", "TYPE": "mlx", "ENDPOINT_URL": "http://box:9/v1"},
        )
        ctrl._embed_mlx(["x"])
        assert cap["client"]["base_url"] == "http://box:9/v1"

    def test_api_base_wins_over_the_alias(self, patch_chat_openai, monkeypatch):
        ctrl = _llm(
            monkeypatch,
            {"NAME": "m", "TYPE": "mlx", "API_BASE": "http://canonical/v1",
             "ENDPOINT_URL": "http://alias/v1"},
        )
        ctrl.initialize_model()
        assert patch_chat_openai["base_url"] == "http://canonical/v1"
