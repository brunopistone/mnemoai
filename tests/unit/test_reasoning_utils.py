"""Unit tests for reasoning model helpers (client/reasoning_utils.py)."""

from mnemoai.client.agent.reasoning_utils import (
    disable_reasoning,
    restore_reasoning,
    extract_visible_text,
)


class FakeOllamaModel:
    """Mimics ChatOllamaWrapper: has a `reasoning` attribute."""

    def __init__(self, reasoning=True):
        self.reasoning = reasoning


class FakeBedrockModel:
    """Mimics ChatBedrock (old API): thinking lives in model_kwargs."""

    def __init__(self):
        self.model_kwargs = {"thinking": {"type": "enabled"}, "temperature": 0.7}


class FakeBedrockConverseModel:
    """Mimics ChatBedrockConverse: thinking in additional_model_request_fields."""

    def __init__(self):
        self.additional_model_request_fields = {"thinking": {"type": "enabled"}}
        self.temperature = 0.8


class FakePlainModel:
    """A model with no reasoning knobs at all."""


class TestDisableRestoreOllama:
    def test_disables_then_restores_reasoning(self):
        model = FakeOllamaModel(reasoning=True)
        saved = disable_reasoning(model)
        assert model.reasoning is False
        restore_reasoning(model, saved)
        assert model.reasoning is True

    def test_reasoning_false_stays_consistent(self):
        model = FakeOllamaModel(reasoning=False)
        saved = disable_reasoning(model)
        assert model.reasoning is False
        restore_reasoning(model, saved)
        assert model.reasoning is False


class TestDisableRestoreBedrock:
    def test_pops_thinking_and_lowers_temp_then_restores(self):
        model = FakeBedrockModel()
        saved = disable_reasoning(model)
        assert "thinking" not in model.model_kwargs
        assert model.model_kwargs["temperature"] == 0.1
        restore_reasoning(model, saved)
        assert model.model_kwargs["thinking"] == {"type": "enabled"}
        assert model.model_kwargs["temperature"] == 0.7


class TestDisableRestoreBedrockConverse:
    def test_pops_thinking_and_restores_temperature(self):
        model = FakeBedrockConverseModel()
        saved = disable_reasoning(model)
        assert "thinking" not in model.additional_model_request_fields
        assert model.temperature == 0.1
        restore_reasoning(model, saved)
        assert model.additional_model_request_fields["thinking"] == {"type": "enabled"}
        assert model.temperature == 0.8

    def test_none_temperature_left_untouched(self):
        # Regression: newer Bedrock Claude models reject `temperature` as
        # deprecated, so these models run with temperature=None and
        # disable_reasoning must NOT set one.
        model = FakeBedrockConverseModel()
        model.temperature = None
        saved = disable_reasoning(model)
        assert model.temperature is None
        restore_reasoning(model, saved)
        assert model.temperature is None


class TestDisableRestorePlainModel:
    def test_noop_on_model_without_reasoning(self):
        model = FakePlainModel()
        saved = disable_reasoning(model)
        assert saved == {}
        # Should not raise.
        restore_reasoning(model, saved)


class TestExtractVisibleText:
    def test_strips_think_tags(self):
        text = "<think>internal reasoning</think>The answer is 42."
        assert extract_visible_text(text) == "The answer is 42."

    def test_strips_thinking_tags_case_insensitive(self):
        text = "<Thinking>hmm</Thinking>  Hello"
        assert extract_visible_text(text) == "Hello"

    def test_plain_text_unchanged(self):
        assert extract_visible_text("just a normal answer") == "just a normal answer"

    def test_none_content(self):
        assert extract_visible_text(None) == ""

    def test_bedrock_content_blocks(self):
        content = [
            {"type": "thinking", "thinking": "reasoning here"},
            {"type": "text", "text": "Visible answer"},
        ]
        assert extract_visible_text(content) == "Visible answer"

    def test_bedrock_blocks_with_no_text(self):
        content = [{"type": "thinking", "thinking": "only reasoning"}]
        assert extract_visible_text(content) == ""
