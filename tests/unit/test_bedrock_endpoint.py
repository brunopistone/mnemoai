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
    """Replace ChatOpenAI and the Mantle token generator with mocks.

    Returns the dict of kwargs ChatOpenAI was constructed with.
    """
    import langchain_openai
    import aws_bedrock_token_generator

    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return MagicMock(name="ChatOpenAI")

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", fake_chat_openai)
    monkeypatch.setattr(
        aws_bedrock_token_generator,
        "provide_token",
        lambda region=None, **kw: "bedrock-api-fake-token",
    )
    return captured


def _make_llm_controller(monkeypatch, model_id: dict):
    import models.llm_controller as mod

    def fake_get(key, default=None):
        if key == "MODEL_ID":
            return model_id
        if key == "MAX_CONVERSATION_TOKENS":
            return 8192
        return default

    monkeypatch.setattr(mod.config, "get", fake_get)
    return mod.LangChainLLMController(verbose=False)


def _make_vision_controller(monkeypatch, model_id: dict):
    import models.vision_model_controller as mod

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
