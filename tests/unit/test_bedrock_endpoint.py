"""Unit tests for Bedrock endpoint wiring and the Bedrock Mantle model type.

No real AWS calls: langchain classes and the Mantle token generator are
replaced with capturing mocks. Live end-to-end verification of Mantle is done
separately (it requires AWS credentials + a reachable Mantle endpoint).
"""

import sys
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
    import langchain_openai
    import langchain_anthropic
    import aws_bedrock_token_generator

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
    import personal_ai_assistant.models.controllers.llm_controller as mod

    def fake_get(key, default=None):
        if key == "MODEL_ID":
            return model_id
        if key == "MAX_CONVERSATION_TOKENS":
            return 8192
        return default

    monkeypatch.setattr(mod.config, "get", fake_get)
    return mod.LangChainLLMController(verbose=False)


def _make_vision_controller(monkeypatch, model_id: dict):
    import personal_ai_assistant.models.controllers.vision_model_controller as mod

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
        from personal_ai_assistant.models.mantle_factory import build_mantle_model

        with pytest.raises(ValueError, match="Unknown Mantle API_PROTOCOL"):
            build_mantle_model(
                {"NAME": "x", "TYPE": "mantle", "API_PROTOCOL": "bogus"}
            )

    def test_explicit_endpoint_url_overrides_anthropic_default(self, patch_mantle):
        from personal_ai_assistant.models.mantle_factory import build_mantle_model

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
        from personal_ai_assistant.models.mantle_factory import build_mantle_model

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

        from personal_ai_assistant.models.mantle_factory import build_mantle_model

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

        from personal_ai_assistant.models.mantle_factory import build_mantle_model

        build_mantle_model(
            {"NAME": "qwen.qwen3-32b", "TYPE": "mantle", "REGION": "us-east-1"}
        )
        assert patch_mantle["api_key"] == "bedrock-api-key-from-env"

    def test_config_api_key_takes_precedence_over_env(self, patch_mantle, monkeypatch):
        monkeypatch.setenv("BEDROCK_API_KEY", "from-env")
        from personal_ai_assistant.models.mantle_factory import build_mantle_model

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
        from personal_ai_assistant.models.mantle_factory import build_mantle_model

        build_mantle_model(
            {"NAME": "qwen.qwen3-32b", "TYPE": "mantle", "REGION": "us-east-1"}
        )
        assert patch_mantle["api_key"] == "bedrock-api-fake-token"
