"""Unit tests for Bedrock endpoint wiring and the Bedrock Mantle model type.

No real AWS calls: langchain classes and the Mantle token generator are
replaced with capturing mocks. Live end-to-end verification of Mantle is done
separately (it requires AWS credentials + a reachable Mantle endpoint).
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def patch_bedrock(monkeypatch):
    """Replace ChatBedrockConverse / ChatBedrock with kwarg-capturing mocks."""
    import langchain_aws

    captured = {}

    def make_recorder(name):
        def _recorder(**kwargs):
            captured[name] = kwargs
            return MagicMock(name=name)

        return _recorder

    monkeypatch.setattr(
        langchain_aws, "ChatBedrockConverse", make_recorder("ChatBedrockConverse")
    )
    monkeypatch.setattr(langchain_aws, "ChatBedrock", make_recorder("ChatBedrock"))
    return captured


@pytest.fixture
def patch_mantle(monkeypatch):
    """Replace ChatOpenAI / ChatAnthropic and the Mantle token generator.

    Returns a dict of the kwargs the constructed model was built with, plus a
    ``_class`` key naming which class was used.
    """
    import aws_bedrock_token_generator
    import langchain_anthropic
    import langchain_openai

    captured = {}

    def make_recorder(cls_name):
        def _recorder(**kwargs):
            captured.clear()
            captured.update(kwargs)
            captured["_class"] = cls_name
            return MagicMock(name=cls_name)

        return _recorder

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", make_recorder("ChatOpenAI"))
    monkeypatch.setattr(
        langchain_anthropic, "ChatAnthropic", make_recorder("ChatAnthropic")
    )
    monkeypatch.setattr(
        aws_bedrock_token_generator,
        "provide_token",
        lambda region=None, **kw: "bedrock-api-fake-token",
    )
    return captured


def _make_llm_controller(monkeypatch, model_id: dict):
    import mnemoai.models.controllers.llm_controller as mod

    def fake_get(key, default=None):
        if key == "MODEL_ID":
            return model_id
        if key == "MAX_CONVERSATION_TOKENS":
            return 8192
        return default

    monkeypatch.setattr(mod.config, "get", fake_get)
    return mod.LangChainLLMController(verbose=False)


def _make_vision_controller(monkeypatch, model_id: dict):
    import mnemoai.models.controllers.vision_model_controller as mod

    def fake_get(key, default=None):
        if key == "VISION_MODEL_ID":
            return model_id
        if key == "MAX_CONVERSATION_TOKENS":
            return 8192
        return default

    monkeypatch.setattr(mod.config, "get", fake_get)
    return mod.VisionModelController(verbose=False)


class TestStandardBedrockEndpoint:
    def test_endpoint_url_passed_when_configured(self, patch_bedrock, monkeypatch):
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "anthropic.claude-opus-4-8",
                "TYPE": "bedrock",
                "ENDPOINT_URL": "https://example.invalid",
            },
        )
        ctrl.initialize_model()
        assert patch_bedrock["ChatBedrockConverse"]["endpoint_url"] == (
            "https://example.invalid"
        )

    def test_endpoint_url_omitted_when_not_configured(
        self, patch_bedrock, monkeypatch
    ):
        ctrl = _make_llm_controller(
            monkeypatch,
            {"NAME": "global.anthropic.claude-opus-4-8", "TYPE": "bedrock"},
        )
        ctrl.initialize_model()
        assert "endpoint_url" not in patch_bedrock["ChatBedrockConverse"]

    def test_reasoning_effort_alone_enables_thinking(self, patch_bedrock, monkeypatch):
        # REASONING_EFFORT without REASONING must still turn thinking on.
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "global.anthropic.claude-sonnet-5",
                "TYPE": "bedrock",
                "REASONING_EFFORT": "max",
            },
        )
        ctrl.initialize_model()
        fields = patch_bedrock["ChatBedrockConverse"]["additional_model_request_fields"]
        assert fields["thinking"]["type"] == "adaptive"
        assert fields["output_config"] == {"effort": "max"}

    def test_newer_claude_uses_adaptive_not_enabled(self, patch_bedrock, monkeypatch):
        # Sonnet 5 rejects thinking.type=enabled; the version-aware form is used.
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "global.anthropic.claude-sonnet-5",
                "TYPE": "bedrock",
                "REASONING": True,
                "REASONING_EFFORT": "max",
            },
        )
        ctrl.initialize_model()
        fields = patch_bedrock["ChatBedrockConverse"]["additional_model_request_fields"]
        assert fields["thinking"]["type"] == "adaptive"

    def test_no_reasoning_sends_no_thinking(self, patch_bedrock, monkeypatch):
        ctrl = _make_llm_controller(
            monkeypatch,
            {"NAME": "global.anthropic.claude-sonnet-5", "TYPE": "bedrock"},
        )
        ctrl.initialize_model()
        assert (
            "additional_model_request_fields"
            not in patch_bedrock["ChatBedrockConverse"]
        )

    def test_non_claude_bedrock_reasoning_effort_injects_nothing(
        self, patch_bedrock, monkeypatch
    ):
        # A non-Claude Bedrock model (Nova) with REASONING_EFFORT must NOT be sent
        # Anthropic-only thinking fields — Converse would reject them.
        for name in (
            "amazon.nova-pro-v1:0",
            "us.deepseek.r1-v1:0",
            "mistral.mistral-large-2407-v1:0",
            "meta.llama3-1-405b-instruct-v1:0",
            "qwen.qwen3-32b-v1:0",
        ):
            ctrl = _make_llm_controller(
                monkeypatch,
                {"NAME": name, "TYPE": "bedrock", "REASONING_EFFORT": "high"},
            )
            ctrl.initialize_model()
            assert (
                "additional_model_request_fields"
                not in patch_bedrock["ChatBedrockConverse"]
            ), name

    def test_non_claude_bedrock_reasoning_flag_injects_nothing(
        self, patch_bedrock, monkeypatch
    ):
        # Bare REASONING: true on a non-Claude Bedrock model also injects nothing.
        ctrl = _make_llm_controller(
            monkeypatch,
            {"NAME": "amazon.nova-pro-v1:0", "TYPE": "bedrock", "REASONING": True},
        )
        ctrl.initialize_model()
        assert (
            "additional_model_request_fields"
            not in patch_bedrock["ChatBedrockConverse"]
        )

    def test_claude_bedrock_still_injects_thinking(self, patch_bedrock, monkeypatch):
        # Regression guard: every Claude id shape still gets the per-effort budget
        # logic (the differentiator that must be preserved for Claude).
        for name in (
            "anthropic.claude-opus-4-8",
            "us.anthropic.claude-opus-5",
            "global.anthropic.claude-sonnet-5",
        ):
            ctrl = _make_llm_controller(
                monkeypatch,
                {"NAME": name, "TYPE": "bedrock", "REASONING_EFFORT": "high"},
            )
            ctrl.initialize_model()
            fields = patch_bedrock["ChatBedrockConverse"][
                "additional_model_request_fields"
            ]
            assert fields["thinking"]["type"] == "adaptive", name
            assert fields["output_config"] == {"effort": "high"}, name

    def test_non_claude_bedrock_extra_params_still_injected(
        self, patch_bedrock, monkeypatch
    ):
        # EXTRA_PARAMS escape hatch survives: an advanced user can still hand-inject
        # a provider-specific reasoning field on a non-Claude Bedrock model.
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "amazon.nova-pro-v1:0",
                "TYPE": "bedrock",
                "REASONING_EFFORT": "high",
                "EXTRA_PARAMS": {
                    "additional_model_request_fields": {"reasoning_config": "x"}
                },
            },
        )
        ctrl.initialize_model()
        assert patch_bedrock["ChatBedrockConverse"][
            "additional_model_request_fields"
        ] == {"reasoning_config": "x"}

    def test_build_non_reasoning_model_disables_thinking(self, patch_bedrock, monkeypatch):
        # Config has reasoning ON, but build_non_reasoning_model() (used for
        # compaction summaries) must produce a model with thinking OFF —
        # provider-agnostic, via clearing REASONING/REASONING_EFFORT.
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "global.anthropic.claude-sonnet-5",
                "TYPE": "bedrock",
                "REASONING_EFFORT": "max",
            },
        )
        ctrl.build_non_reasoning_model()
        assert (
            "additional_model_request_fields"
            not in patch_bedrock["ChatBedrockConverse"]
        )

    def test_build_non_reasoning_model_suppresses_ollama_reasoning(self, monkeypatch):
        # Ollama surfaces reasoning via verbose_mode (sets reasoning=True), NOT
        # via REASONING_EFFORT — so build_non_reasoning_model must also drop
        # verbose on the peer so the summary model has no reasoning flag, even
        # when the MAIN controller is verbose.
        import mnemoai.models.controllers.llm_controller as mod

        captured = {}

        class _FakeOllama:
            def __init__(self, **kw):
                captured.update(kw)

        monkeypatch.setattr(mod, "ChatOllamaWrapper", _FakeOllama)

        def fake_get(key, default=None):
            if key == "MODEL_ID":
                return {"NAME": "qwen3:8b", "TYPE": "ollama", "REASONING_EFFORT": "high"}
            if key == "MAX_CONVERSATION_TOKENS":
                return 8192
            return default

        monkeypatch.setattr(mod.config, "get", fake_get)
        ctrl = mod.LangChainLLMController(verbose=True)  # main is verbose
        ctrl.build_non_reasoning_model()
        assert captured.get("reasoning") is not True  # no reasoning on the summary model


class TestMantleModelType:
    def test_builds_chatopenai_with_token_and_default_endpoint(
        self, patch_mantle, monkeypatch
    ):
        ctrl = _make_llm_controller(
            monkeypatch,
            {"NAME": "qwen.qwen3-32b", "TYPE": "mantle", "REGION": "us-east-1"},
        )
        ctrl.initialize_model()
        assert patch_mantle["model"] == "qwen.qwen3-32b"
        assert patch_mantle["api_key"] == "bedrock-api-fake-token"
        assert patch_mantle["base_url"] == (
            "https://bedrock-mantle.us-east-1.api.aws/v1"
        )

    def test_region_used_in_default_endpoint(self, patch_mantle, monkeypatch):
        ctrl = _make_llm_controller(
            monkeypatch,
            {"NAME": "qwen.qwen3-32b", "TYPE": "mantle", "REGION": "eu-west-1"},
        )
        ctrl.initialize_model()
        assert "eu-west-1" in patch_mantle["base_url"]

    def test_explicit_endpoint_url_overrides_default(self, patch_mantle, monkeypatch):
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "qwen.qwen3-32b",
                "TYPE": "mantle",
                "REGION": "us-east-1",
                "ENDPOINT_URL": "https://custom-mantle.example/v1",
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["base_url"] == "https://custom-mantle.example/v1"

    def test_default_protocol_is_chat_completions(self, patch_mantle, monkeypatch):
        ctrl = _make_llm_controller(
            monkeypatch,
            {"NAME": "qwen.qwen3-32b", "TYPE": "mantle", "REGION": "us-east-1"},
        )
        ctrl.initialize_model()
        # Chat Completions uses the /v1 base and does NOT set use_responses_api.
        assert patch_mantle["base_url"].endswith("/v1")
        assert "use_responses_api" not in patch_mantle

    def test_responses_protocol_uses_openai_v1_and_flag(
        self, patch_mantle, monkeypatch
    ):
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "openai.gpt-5.4",
                "TYPE": "mantle",
                "REGION": "us-west-2",
                "API_PROTOCOL": "responses",
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["base_url"] == (
            "https://bedrock-mantle.us-west-2.api.aws/openai/v1"
        )
        assert patch_mantle["use_responses_api"] is True

    def test_responses_effort_requests_reasoning_summary(
        self, patch_mantle, monkeypatch
    ):
        # On the responses protocol, REASONING_EFFORT must become a `reasoning`
        # object that also asks for a summary — effort alone yields hidden
        # reasoning the user can't see.
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "openai.gpt-5.5",
                "TYPE": "mantle",
                "REGION": "us-west-2",
                "API_PROTOCOL": "responses",
                "REASONING_EFFORT": "high",
            },
        )
        ctrl.initialize_model()
        assert patch_mantle.get("reasoning") == {"effort": "high", "summary": "auto"}
        # The bare effort enum must NOT also be set (would double-specify).
        assert "reasoning_effort" not in patch_mantle

    def test_chat_completions_effort_stays_plain_enum(self, patch_mantle, monkeypatch):
        # Chat Completions takes the plain reasoning_effort enum, no summary obj.
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "qwen.qwen3-32b",
                "TYPE": "mantle",
                "REGION": "us-east-1",
                "API_PROTOCOL": "chat_completions",
                "REASONING_EFFORT": "high",
            },
        )
        ctrl.initialize_model()
        assert patch_mantle.get("reasoning_effort") == "high"
        assert "reasoning" not in patch_mantle

    def test_extra_params_reasoning_overrides_summary_default(
        self, patch_mantle, monkeypatch
    ):
        # An explicit EXTRA_PARAMS.reasoning wins over the auto summary default.
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "openai.gpt-5.5",
                "TYPE": "mantle",
                "REGION": "us-west-2",
                "API_PROTOCOL": "responses",
                "REASONING_EFFORT": "high",
                "EXTRA_PARAMS": {"reasoning": {"effort": "low", "summary": "detailed"}},
            },
        )
        ctrl.initialize_model()
        assert patch_mantle.get("reasoning") == {"effort": "low", "summary": "detailed"}

    def test_anthropic_protocol_uses_chatanthropic(self, patch_mantle, monkeypatch):
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "anthropic.claude-haiku-4-5",
                "TYPE": "mantle",
                "REGION": "us-east-1",
                "API_PROTOCOL": "anthropic",
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["_class"] == "ChatAnthropic"
        assert patch_mantle["anthropic_api_url"] == (
            "https://bedrock-mantle.us-east-1.api.aws/anthropic"
        )
        # Mantle accepts the bearer token supplied as the Anthropic API key.
        assert patch_mantle["anthropic_api_key"] == "bedrock-api-fake-token"


class TestAnthropicModelType:
    """The direct Anthropic API provider (TYPE: anthropic) via ChatAnthropic.

    Distinct from the Mantle 'anthropic' protocol above (Claude via Bedrock):
    this talks to api.anthropic.com using ChatAnthropic directly.
    """

    def test_builds_chatanthropic_with_name_and_key(self, patch_mantle, monkeypatch):
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "claude-opus-4-8",
                "TYPE": "anthropic",
                "API_KEY": "fake-anthropic-key",
                "MAX_TOKENS": 2000,
                "TEMPERATURE": 0.4,
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["_class"] == "ChatAnthropic"
        assert patch_mantle["model"] == "claude-opus-4-8"
        assert patch_mantle["api_key"] == "fake-anthropic-key"
        assert patch_mantle["max_tokens"] == 2000
        assert patch_mantle["temperature"] == 0.4
        # No base_url unless ENDPOINT_URL is set (defaults to api.anthropic.com).
        assert "base_url" not in patch_mantle

    def test_max_tokens_defaults_when_unset(self, patch_mantle, monkeypatch):
        # ChatAnthropic requires max_tokens; controller defaults it to 4096.
        ctrl = _make_llm_controller(
            monkeypatch, {"NAME": "claude-opus-4-8", "TYPE": "anthropic"}
        )
        ctrl.initialize_model()
        assert patch_mantle["max_tokens"] == 4096

    def test_endpoint_url_sets_base_url(self, patch_mantle, monkeypatch):
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "claude-opus-4-8",
                "TYPE": "anthropic",
                "ENDPOINT_URL": "https://proxy.example/v1",
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["base_url"] == "https://proxy.example/v1"

    def test_stop_maps_to_stop_sequences(self, patch_mantle, monkeypatch):
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "claude-opus-4-8",
                "TYPE": "anthropic",
                "STOP": ["</done>"],
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["stop_sequences"] == ["</done>"]

    def test_reasoning_enables_thinking_and_drops_temperature(
        self, patch_mantle, monkeypatch
    ):
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "claude-opus-4-8",
                "TYPE": "anthropic",
                "REASONING": True,
                "REASONING_EFFORT": "high",
                "MAX_TOKENS": 2000,
                "TEMPERATURE": 0.7,
            },
        )
        ctrl.initialize_model()
        # claude-opus-4-8 is a 4.7+ model -> adaptive form with effort, NOT the
        # old enabled+budget (which 4.7+ rejects).
        assert patch_mantle["thinking"] == {
            "type": "adaptive",
            "display": "summarized",
        }
        assert patch_mantle["output_config"] == {"effort": "high"}
        # max_tokens still bumped above the derived budget; sampling dropped.
        assert patch_mantle["max_tokens"] == 16384 + 1024
        assert "temperature" not in patch_mantle
        assert "top_p" not in patch_mantle

    def test_vision_builds_chatanthropic(self, patch_mantle, monkeypatch):
        ctrl = _make_vision_controller(
            monkeypatch,
            {
                "NAME": "claude-opus-4-8",
                "TYPE": "anthropic",
                "API_KEY": "fake-anthropic-vision-key",
                "MAX_TOKENS": 1500,
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["_class"] == "ChatAnthropic"
        assert patch_mantle["model"] == "claude-opus-4-8"
        assert patch_mantle["api_key"] == "fake-anthropic-vision-key"
        assert patch_mantle["max_tokens"] == 1500


class TestMantleVisionModelType:
    def test_vision_builds_chatopenai_with_token_and_endpoint(
        self, patch_mantle, monkeypatch
    ):
        ctrl = _make_vision_controller(
            monkeypatch,
            {
                "NAME": "qwen.qwen3-vl-235b-a22b-instruct",
                "TYPE": "mantle",
                "REGION": "us-east-1",
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["model"] == "qwen.qwen3-vl-235b-a22b-instruct"
        assert patch_mantle["api_key"] == "bedrock-api-fake-token"
        assert patch_mantle["base_url"] == (
            "https://bedrock-mantle.us-east-1.api.aws/v1"
        )

    def test_vision_explicit_endpoint_url_overrides_default(
        self, patch_mantle, monkeypatch
    ):
        ctrl = _make_vision_controller(
            monkeypatch,
            {
                "NAME": "qwen.qwen3-vl-235b-a22b-instruct",
                "TYPE": "mantle",
                "REGION": "us-east-1",
                "ENDPOINT_URL": "https://custom-mantle.example/v1",
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["base_url"] == "https://custom-mantle.example/v1"

    def test_vision_default_protocol_is_chat_completions(
        self, patch_mantle, monkeypatch
    ):
        ctrl = _make_vision_controller(
            monkeypatch,
            {
                "NAME": "qwen.qwen3-vl-235b-a22b-instruct",
                "TYPE": "mantle",
                "REGION": "us-east-1",
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["base_url"].endswith("/v1")
        assert "use_responses_api" not in patch_mantle

    def test_vision_responses_protocol_uses_openai_v1_and_flag(
        self, patch_mantle, monkeypatch
    ):
        ctrl = _make_vision_controller(
            monkeypatch,
            {
                "NAME": "openai.gpt-5.4",
                "TYPE": "mantle",
                "REGION": "us-west-2",
                "API_PROTOCOL": "responses",
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["base_url"] == (
            "https://bedrock-mantle.us-west-2.api.aws/openai/v1"
        )
        assert patch_mantle["use_responses_api"] is True

    def test_vision_anthropic_protocol_uses_chatanthropic(
        self, patch_mantle, monkeypatch
    ):
        ctrl = _make_vision_controller(
            monkeypatch,
            {
                "NAME": "anthropic.claude-haiku-4-5",
                "TYPE": "mantle",
                "REGION": "us-east-1",
                "API_PROTOCOL": "anthropic",
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["_class"] == "ChatAnthropic"
        assert patch_mantle["anthropic_api_url"] == (
            "https://bedrock-mantle.us-east-1.api.aws/anthropic"
        )


class TestMantleFactory:
    def test_invalid_protocol_raises(self, patch_mantle):
        from mnemoai.models.mantle_factory import build_mantle_model

        with pytest.raises(ValueError, match="Unknown Mantle API_PROTOCOL"):
            build_mantle_model(
                {"NAME": "x", "TYPE": "mantle", "API_PROTOCOL": "bogus"}
            )

    def test_explicit_endpoint_url_overrides_anthropic_default(self, patch_mantle):
        from mnemoai.models.mantle_factory import build_mantle_model

        build_mantle_model(
            {
                "NAME": "anthropic.claude-haiku-4-5",
                "API_PROTOCOL": "anthropic",
                "REGION": "us-east-1",
                "ENDPOINT_URL": "https://custom.example/anthropic",
            }
        )
        assert patch_mantle["anthropic_api_url"] == "https://custom.example/anthropic"

    def test_anthropic_defaults_max_tokens_when_unset(self, patch_mantle):
        from mnemoai.models.mantle_factory import build_mantle_model

        build_mantle_model(
            {
                "NAME": "anthropic.claude-haiku-4-5",
                "API_PROTOCOL": "anthropic",
                "REGION": "us-east-1",
            }
        )
        # Anthropic requires max_tokens; factory supplies a default.
        assert patch_mantle["max_tokens"] == 4096


class TestMantleApiKeyAuth:
    def test_config_api_key_used_without_minting(self, patch_mantle, monkeypatch):
        # An explicit API_KEY must be used directly, and provide_token must NOT
        # be called (would raise here, proving the mint path is skipped).
        import aws_bedrock_token_generator

        def _boom(*a, **k):
            raise AssertionError("provide_token should not be called when a key is set")

        monkeypatch.setattr(aws_bedrock_token_generator, "provide_token", _boom)
        monkeypatch.delenv("BEDROCK_API_KEY", raising=False)

        from mnemoai.models.mantle_factory import build_mantle_model

        build_mantle_model(
            {
                "NAME": "qwen.qwen3-32b",
                "TYPE": "mantle",
                "REGION": "us-east-1",
                "API_KEY": "bedrock-api-key-explicit",
            }
        )
        assert patch_mantle["api_key"] == "bedrock-api-key-explicit"

    def test_env_bedrock_api_key_used_without_minting(self, patch_mantle, monkeypatch):
        import aws_bedrock_token_generator

        def _boom(*a, **k):
            raise AssertionError("provide_token should not be called when a key is set")

        monkeypatch.setattr(aws_bedrock_token_generator, "provide_token", _boom)
        monkeypatch.setenv("BEDROCK_API_KEY", "bedrock-api-key-from-env")

        from mnemoai.models.mantle_factory import build_mantle_model

        build_mantle_model(
            {"NAME": "qwen.qwen3-32b", "TYPE": "mantle", "REGION": "us-east-1"}
        )
        assert patch_mantle["api_key"] == "bedrock-api-key-from-env"

    def test_config_api_key_takes_precedence_over_env(self, patch_mantle, monkeypatch):
        monkeypatch.setenv("BEDROCK_API_KEY", "from-env")
        from mnemoai.models.mantle_factory import build_mantle_model

        build_mantle_model(
            {
                "NAME": "qwen.qwen3-32b",
                "TYPE": "mantle",
                "REGION": "us-east-1",
                "API_KEY": "from-config",
            }
        )
        assert patch_mantle["api_key"] == "from-config"

    def test_falls_back_to_minting_when_no_key(self, patch_mantle, monkeypatch):
        # No key set anywhere -> mints via the (mocked) token generator.
        monkeypatch.delenv("BEDROCK_API_KEY", raising=False)
        from mnemoai.models.mantle_factory import build_mantle_model

        build_mantle_model(
            {"NAME": "qwen.qwen3-32b", "TYPE": "mantle", "REGION": "us-east-1"}
        )
        assert patch_mantle["api_key"] == "bedrock-api-fake-token"


class TestExtraParamsPassthrough:
    """EXTRA_PARAMS: a generic dict forwarded verbatim to the model.

    Verifies the passthrough reaches the right sink per provider/protocol:
    - Mantle responses / direct OpenAI: reasoning_effort lifts to a first-class
      arg; other keys go into model_kwargs.
    - Mantle anthropic / direct Anthropic: passed as top-level constructor args
      (e.g. thinking).
    """

    def test_mantle_responses_reasoning_effort_and_model_kwargs(
        self, patch_mantle, monkeypatch
    ):
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "openai.gpt-5.5",
                "TYPE": "mantle",
                "REGION": "us-west-2",
                "API_PROTOCOL": "responses",
                "EXTRA_PARAMS": {"reasoning_effort": "high", "verbosity": "low"},
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["_class"] == "ChatOpenAI"
        # reasoning_effort is a first-class ChatOpenAI arg.
        assert patch_mantle["reasoning_effort"] == "high"
        # Remaining keys go into the request body.
        assert patch_mantle["model_kwargs"] == {"verbosity": "low"}

    def test_mantle_anthropic_thinking_passthrough(self, patch_mantle, monkeypatch):
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "anthropic.claude-opus-4-8",
                "TYPE": "mantle",
                "REGION": "us-east-1",
                "API_PROTOCOL": "anthropic",
                "MAX_TOKENS": 8192,
                "EXTRA_PARAMS": {
                    "thinking": {"type": "enabled", "budget_tokens": 4096}
                },
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["_class"] == "ChatAnthropic"
        assert patch_mantle["thinking"] == {"type": "enabled", "budget_tokens": 4096}

    def test_direct_openai_extra_params(self, patch_mantle, monkeypatch):
        import langchain_openai

        captured = {}

        def rec(**kwargs):
            captured.clear()
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(langchain_openai, "ChatOpenAI", rec)
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "gpt-5.5",
                "TYPE": "openai",
                "EXTRA_PARAMS": {"reasoning_effort": "high", "verbosity": "low"},
            },
        )
        ctrl.initialize_model()
        assert captured["model_kwargs"]["verbosity"] == "low"
        # reasoning_effort flows via model_kwargs on the direct OpenAI path
        # (registry already maps REASONING_EFFORT there), or as a key in
        # model_kwargs from EXTRA_PARAMS — assert it reached the request body.
        assert "reasoning_effort" in captured["model_kwargs"]

    def test_direct_anthropic_extra_params(self, patch_mantle, monkeypatch):
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "claude-opus-4-8",
                "TYPE": "anthropic",
                "API_KEY": "k",
                "MAX_TOKENS": 4096,
                "EXTRA_PARAMS": {"thinking": {"type": "adaptive"}},
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["_class"] == "ChatAnthropic"
        assert patch_mantle["thinking"] == {"type": "adaptive"}

    def test_absent_extra_params_is_noop(self, patch_mantle, monkeypatch):
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "openai.gpt-5.4",
                "TYPE": "mantle",
                "REGION": "us-west-2",
                "API_PROTOCOL": "responses",
            },
        )
        ctrl.initialize_model()
        assert "reasoning_effort" not in patch_mantle
        assert "model_kwargs" not in patch_mantle

    def test_non_dict_extra_params_ignored(self, patch_mantle, monkeypatch):
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "openai.gpt-5.4",
                "TYPE": "mantle",
                "REGION": "us-west-2",
                "API_PROTOCOL": "responses",
                "EXTRA_PARAMS": "not-a-dict",
            },
        )
        ctrl.initialize_model()  # must not raise
        assert "model_kwargs" not in patch_mantle


class TestReasoningEffortFirstClass:
    """REASONING_EFFORT as a first-class, provider-translated knob.

    - Mantle responses: becomes a `reasoning` object that also requests a
      summary (`{"effort": …, "summary": "auto"}`) so the reasoning is visible.
    - Mantle chat_completions / direct OpenAI: forwarded as `reasoning_effort`.
    - Mantle anthropic / direct Bedrock / direct Anthropic: mapped to a
      `thinking` budget (token budget, not an effort enum).
    - LiteLLM: forwarded via model_kwargs (LiteLLM translates per backend).
    """

    def test_mantle_responses_reasoning_effort(self, patch_mantle, monkeypatch):
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "openai.gpt-5.5",
                "TYPE": "mantle",
                "REGION": "us-west-2",
                "API_PROTOCOL": "responses",
                "REASONING_EFFORT": "high",
                "MAX_TOKENS": 4096,
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["_class"] == "ChatOpenAI"
        # responses protocol: effort + auto summary so reasoning is visible.
        assert patch_mantle["reasoning"] == {"effort": "high", "summary": "auto"}

    def test_is_anthropic_model_predicate(self):
        # The is-Anthropic gate: every Claude id shape true, every non-Claude
        # Bedrock/Mantle family false. Not derived from _claude_version.
        from mnemoai.models.mantle_factory import is_anthropic_model

        for name in (
            "claude-opus-5",
            "claude-opus-4-8",
            "claude-sonnet-5",
            "claude-haiku-4-5",
            "claude-fable-5",
            "claude-mythos-5",
            "claude-opus-4-5-20250929",
            "claude-3-7-sonnet",
            "anthropic.claude-opus-4-8",
            "anthropic.claude-opus-4-5-20250929-v1:0",
            "us.anthropic.claude-opus-5",
            "eu.anthropic.claude-opus-5",
            "apac.anthropic.claude-opus-5",
            "global.anthropic.claude-sonnet-5",
        ):
            assert is_anthropic_model(name) is True, name

        for name in (
            "amazon.nova-pro-v1:0",
            "amazon.nova-premier-v1:0",
            "deepseek.r1-v1:0",
            "us.deepseek.r1-v1:0",
            "mistral.large-latest",
            "meta.llama3-1-405b-instruct-v1:0",
            "qwen.qwen3-32b",
            "zhipu.glm-4-9b-chat",
            "nvidia.nemotron-nano-12b-v2-vl-bf16",
            "amazon.titan-text-express-v1",
            "cohere.command-r-plus-v1:0",
            "ai21.j2-ultra",
            "",
        ):
            assert is_anthropic_model(name) is False, name

    def test_mantle_anthropic_non_claude_injects_no_thinking(
        self, patch_mantle, monkeypatch
    ):
        # A non-Claude on the Mantle anthropic protocol is a misconfig — guard
        # defensively and inject no thinking block.
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "qwen.qwen3-32b",
                "TYPE": "mantle",
                "REGION": "us-east-1",
                "API_PROTOCOL": "anthropic",
                "REASONING_EFFORT": "high",
                "MAX_TOKENS": 4096,
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["_class"] == "ChatAnthropic"
        assert "thinking" not in patch_mantle
        assert "output_config" not in patch_mantle

    def test_anthropic_thinking_form_by_version(self):
        # The helper picks the request form by model version.
        from mnemoai.models.mantle_factory import _anthropic_thinking_kwargs

        # 4.7+ -> adaptive + summarized display + effort
        out = _anthropic_thinking_kwargs("claude-opus-4-8", "high", 16384)
        assert out["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert out["output_config"] == {"effort": "high"}
        # 4.6 -> adaptive but no display key
        out = _anthropic_thinking_kwargs("claude-opus-4-6", "high", 16384)
        assert out["thinking"] == {"type": "adaptive"}
        # older -> enabled + budget, no output_config
        out = _anthropic_thinking_kwargs("claude-3-7-sonnet", "high", 16384)
        assert out["thinking"] == {"type": "enabled", "budget_tokens": 16384}
        assert "output_config" not in out
        # unknown/unparseable -> assume current (adaptive + summarized)
        out = _anthropic_thinking_kwargs("mystery-model", "high", 16384)
        assert out["thinking"]["type"] == "adaptive"

    def test_mantle_anthropic_newer_model_uses_adaptive(
        self, patch_mantle, monkeypatch
    ):
        # Opus 4.7+ on Mantle: the old enabled+budget form is rejected by the
        # API; we must send adaptive + output_config.effort.
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "anthropic.claude-opus-4-8",
                "TYPE": "mantle",
                "REGION": "us-west-2",
                "API_PROTOCOL": "anthropic",
                "REASONING_EFFORT": "high",
                "MAX_TOKENS": 4096,
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["_class"] == "ChatAnthropic"
        assert patch_mantle["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert patch_mantle["output_config"] == {"effort": "high"}
        assert patch_mantle["max_tokens"] > 16384
        assert "temperature" not in patch_mantle

    def test_mantle_anthropic_older_model_uses_enabled_budget(
        self, patch_mantle, monkeypatch
    ):
        # Older Claude (<=4.5, 3.x) reject `adaptive` and need enabled+budget.
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "anthropic.claude-3-7-sonnet",
                "TYPE": "mantle",
                "REGION": "us-west-2",
                "API_PROTOCOL": "anthropic",
                "REASONING_EFFORT": "high",
                "MAX_TOKENS": 4096,
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["thinking"] == {"type": "enabled", "budget_tokens": 16384}
        assert "output_config" not in patch_mantle

    def test_mantle_anthropic_no_optin_requests_no_thinking(
        self, patch_mantle, monkeypatch
    ):
        # A Claude model on Mantle WITHOUT any reasoning opt-in must NOT be sent
        # a thinking block — a non-reasoning model would reject it (400). Thinking
        # is opt-in (REASONING_EFFORT or REASONING: true), like direct anthropic.
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "anthropic.claude-3-5-sonnet",
                "TYPE": "mantle",
                "REGION": "us-east-1",
                "API_PROTOCOL": "anthropic",
                "MAX_TOKENS": 4096,
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["_class"] == "ChatAnthropic"
        assert "thinking" not in patch_mantle
        assert "output_config" not in patch_mantle

    def test_mantle_anthropic_reasoning_flag_enables_thinking(
        self, patch_mantle, monkeypatch
    ):
        # REASONING: true (no effort) also opts in on Mantle anthropic.
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "anthropic.claude-3-5-sonnet",
                "TYPE": "mantle",
                "REGION": "us-east-1",
                "API_PROTOCOL": "anthropic",
                "REASONING": True,
                "MAX_TOKENS": 4096,
            },
        )
        ctrl.initialize_model()
        assert "thinking" in patch_mantle

    def test_direct_anthropic_effort_alone_enables_thinking(
        self, patch_mantle, monkeypatch
    ):
        # Direct TYPE: anthropic — REASONING_EFFORT alone (no REASONING: true)
        # enables extended thinking, matching the OpenAI responses behavior.
        # claude-opus-4-8 is 4.7+ -> adaptive form.
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "claude-opus-4-8",
                "TYPE": "anthropic",
                "REASONING_EFFORT": "high",
                "MAX_TOKENS": 4096,
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["_class"] == "ChatAnthropic"
        assert patch_mantle["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert patch_mantle["output_config"] == {"effort": "high"}
        assert "temperature" not in patch_mantle

    def test_direct_anthropic_no_reasoning_no_thinking(
        self, patch_mantle, monkeypatch
    ):
        # Neither REASONING nor REASONING_EFFORT -> thinking stays off.
        ctrl = _make_llm_controller(
            monkeypatch,
            {"NAME": "claude-opus-4-8", "TYPE": "anthropic", "MAX_TOKENS": 4096},
        )
        ctrl.initialize_model()
        assert "thinking" not in patch_mantle

    def test_extra_params_overrides_reasoning_effort(self, patch_mantle, monkeypatch):
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "openai.gpt-5.5",
                "TYPE": "mantle",
                "REGION": "us-west-2",
                "API_PROTOCOL": "responses",
                "REASONING_EFFORT": "low",
                "EXTRA_PARAMS": {"reasoning_effort": "xhigh"},
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["reasoning_effort"] == "xhigh"

    def test_litellm_reasoning_effort_in_model_kwargs(self, patch_mantle, monkeypatch):
        # ChatLiteLLM is imported lazily (function-local) inside
        # _initialize_litellm_model, so patch it at the SOURCE (langchain_litellm)
        # where the deferred import looks it up — not on the controller module.
        import langchain_litellm

        cap = {}

        def rec(**kwargs):
            cap.clear()
            cap.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(langchain_litellm, "ChatLiteLLM", rec)
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "anthropic/claude-3-7-sonnet",
                "TYPE": "litellm",
                "REASONING_EFFORT": "medium",
                "MAX_TOKENS": 4096,
            },
        )
        ctrl.initialize_model()
        assert cap["model_kwargs"]["reasoning_effort"] == "medium"

    def test_direct_openai_responses_requests_summary(
        self, patch_mantle, monkeypatch
    ):
        # Direct TYPE: openai on the responses protocol must use the Responses
        # API and request a reasoning summary, mirroring Mantle responses.
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "gpt-5.5",
                "TYPE": "openai",
                "API_PROTOCOL": "responses",
                "REASONING_EFFORT": "high",
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["_class"] == "ChatOpenAI"
        assert patch_mantle["use_responses_api"] is True
        assert patch_mantle["reasoning"] == {"effort": "high", "summary": "auto"}
        # No bare reasoning_effort left in model_kwargs (would double-specify).
        assert "reasoning_effort" not in (patch_mantle.get("model_kwargs") or {})

    def test_direct_openai_chat_completions_stays_plain(
        self, patch_mantle, monkeypatch
    ):
        # Default (chat_completions): plain reasoning_effort enum, no Responses
        # API, no reasoning summary object.
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "o3",
                "TYPE": "openai",
                "REASONING_EFFORT": "high",
            },
        )
        ctrl.initialize_model()
        assert "use_responses_api" not in patch_mantle
        assert "reasoning" not in patch_mantle
        assert patch_mantle["model_kwargs"]["reasoning_effort"] == "high"


class TestOpenAICompatibleLocalEndpoint:
    """TYPE: openai pointed at a local OpenAI-compatible server.

    Lets a local llama-server (llama.cpp) / LM Studio / vLLM stand in for the
    OpenAI API without a separate provider — the article's recommended Ollama
    alternatives all expose this API.
    """

    def test_api_base_sets_base_url_and_placeholder_key(
        self, patch_mantle, monkeypatch
    ):
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "qwen2.5-7b-instruct",
                "TYPE": "openai",
                "API_BASE": "http://localhost:8080/v1",
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["_class"] == "ChatOpenAI"
        assert patch_mantle["base_url"] == "http://localhost:8080/v1"
        # Local servers ignore auth; a placeholder key lets the client build
        # without a real OPENAI_API_KEY.
        assert patch_mantle["api_key"] == "sk-local"

    def test_endpoint_url_alias_and_explicit_key(self, patch_mantle, monkeypatch):
        ctrl = _make_llm_controller(
            monkeypatch,
            {
                "NAME": "local-model",
                "TYPE": "openai",
                "ENDPOINT_URL": "http://localhost:1234/v1",
                "API_KEY": "lm-studio",
            },
        )
        ctrl.initialize_model()
        assert patch_mantle["base_url"] == "http://localhost:1234/v1"
        assert patch_mantle["api_key"] == "lm-studio"

    def test_plain_openai_unchanged(self, patch_mantle, monkeypatch):
        # No API_BASE/ENDPOINT_URL -> no base_url, no placeholder key (uses the
        # OPENAI_API_KEY env var as before).
        ctrl = _make_llm_controller(
            monkeypatch,
            {"NAME": "gpt-5.5", "TYPE": "openai"},
        )
        ctrl.initialize_model()
        assert "base_url" not in patch_mantle
        assert "api_key" not in patch_mantle
