"""Unit tests for reasoning model helpers (client/reasoning_utils.py)."""

from pydantic import BaseModel

from mnemoai.client.agent.reasoning_utils import (
    disable_reasoning,
    extract_visible_text,
    restore_reasoning,
    without_reasoning,
)
from mnemoai.models.chat_models.chat_ollama_wrapper import ChatOllamaWrapper


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


class FakeResponsesModel:
    """Mimics ChatOpenAI on the Responses API (Mantle GPT-5 / Grok).

    Reasoning is controlled via `reasoning_effort`; `use_responses_api` is True.
    """

    def __init__(self, reasoning_effort=None):
        self.use_responses_api = True
        self.reasoning_effort = reasoning_effort


class FakeResponsesSummaryModel:
    """ChatOpenAI on the Responses API built with a `reasoning` OBJECT.

    This is the 0.10.3 shape: reasoning={"effort": …, "summary": "auto"} to get
    a visible summary. ChatOpenAI ALSO exposes a `reasoning_effort` field
    (defaulting None). disable_reasoning must set the object's effort to "none"
    and must NOT populate reasoning_effort (the Responses API rejects both).
    """

    def __init__(self, reasoning=None):
        self.use_responses_api = True
        self.reasoning = reasoning if reasoning is not None else {
            "effort": "high",
            "summary": "auto",
        }
        self.reasoning_effort = None


class FakeChatCompletionsModel:
    """ChatOpenAI on the classic chat_completions API (not a reasoning model).

    Has `reasoning_effort` (ChatOpenAI always does) but use_responses_api False,
    so disable_reasoning must NOT touch it.
    """

    def __init__(self):
        self.use_responses_api = False
        self.reasoning_effort = None


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


class TestDisableRestoreResponsesModel:
    def test_forces_reasoning_effort_none_then_restores(self):
        # Regression: Mantle Grok/GPT-5 on the Responses API spend their token
        # budget reasoning, leaving auxiliary calls (classify/decompose) with
        # empty content. disable_reasoning must set reasoning_effort="none".
        model = FakeResponsesModel(reasoning_effort="high")
        saved = disable_reasoning(model)
        assert model.reasoning_effort == "none"
        restore_reasoning(model, saved)
        assert model.reasoning_effort == "high"

    def test_restores_none_effort(self):
        model = FakeResponsesModel(reasoning_effort=None)
        saved = disable_reasoning(model)
        assert model.reasoning_effort == "none"
        restore_reasoning(model, saved)
        assert model.reasoning_effort is None

    def test_chat_completions_model_untouched(self):
        # A non-Responses ChatOpenAI is not forced to reason-none.
        model = FakeChatCompletionsModel()
        saved = disable_reasoning(model)
        assert "reasoning_effort" not in saved
        assert model.reasoning_effort is None
        restore_reasoning(model, saved)
        assert model.reasoning_effort is None


class TestDisableRestoreResponsesSummaryModel:
    """The 0.10.3 `reasoning` OBJECT shape (with a summary request).

    Regression for the live "Responses.create() got an unexpected keyword
    argument 'reasoning_effort'" crash: when the model carries a `reasoning`
    dict, disable_reasoning must set THAT object's effort to "none" and must NOT
    set the scalar reasoning_effort (which would then be sent alongside the
    object and rejected by the Responses API).
    """

    def test_sets_object_effort_none_not_scalar(self):
        model = FakeResponsesSummaryModel()
        saved = disable_reasoning(model)
        assert model.reasoning == {"effort": "none", "summary": "auto"}
        # The scalar must stay unset, or the API rejects both together.
        assert model.reasoning_effort is None
        assert "reasoning_effort" not in saved

    def test_restores_original_object(self):
        model = FakeResponsesSummaryModel()
        saved = disable_reasoning(model)
        restore_reasoning(model, saved)
        assert model.reasoning == {"effort": "high", "summary": "auto"}

    def test_preserves_extra_keys_in_object(self):
        model = FakeResponsesSummaryModel(
            reasoning={"effort": "max", "summary": "detailed"}
        )
        disable_reasoning(model)
        assert model.reasoning == {"effort": "none", "summary": "detailed"}


class TestDisableRestorePlainModel:
    def test_noop_on_model_without_reasoning(self):
        model = FakePlainModel()
        saved = disable_reasoning(model)
        assert saved == {}
        # Should not raise.
        restore_reasoning(model, saved)


class PydOllamaModel(BaseModel):
    """Scalar-attribute shape, copyable (pydantic, like the real chat models)."""

    reasoning: bool = True


class PydBedrockModel(BaseModel):
    """Dict-shaped provider params — the fields model_copy only SHALLOW-copies."""

    model_kwargs: dict = {}


class PydConverseModel(BaseModel):
    additional_model_request_fields: dict = {}
    temperature: float = 0.8


class TestWithoutReasoning:
    """The twin must be disabled and the parent untouched — on every shape.

    Both halves of the bug this replaces were real: the shared model was mutated
    for the duration of an auxiliary call (visible to a concurrent worker and to
    QueryRouter), and on the dict-shaped providers a pydantic shallow copy would
    still have aliased the very dicts `disable_reasoning` pops `thinking` out of.
    """

    def test_scalar_attribute_twin_leaves_the_parent_alone(self):
        parent = PydOllamaModel(reasoning=True)
        peer = without_reasoning(parent)
        assert peer.reasoning is False
        assert parent.reasoning is True

    def test_model_kwargs_dict_is_detached_not_aliased(self):
        parent = PydBedrockModel(model_kwargs={"thinking": {"type": "enabled"}})
        peer = without_reasoning(parent)
        assert "thinking" not in peer.model_kwargs
        assert parent.model_kwargs["thinking"] == {"type": "enabled"}
        assert peer.model_kwargs is not parent.model_kwargs

    def test_converse_fields_dict_is_detached_not_aliased(self):
        parent = PydConverseModel(
            additional_model_request_fields={"thinking": {"type": "enabled"}}
        )
        peer = without_reasoning(parent)
        assert "thinking" not in peer.additional_model_request_fields
        assert parent.additional_model_request_fields["thinking"] == {"type": "enabled"}
        assert parent.temperature == 0.8  # the parent's temperature is untouched too

    def test_a_tool_bound_model_keeps_its_tools(self):
        # The site that matters most (_call_model) invokes the tool-BOUND model, a
        # RunnableBinding with no reasoning knobs of its own. Copying the binding
        # naively would return a twin with reasoning still on; dropping down to
        # `.bound` without rebinding would lose the tools.
        parent = ChatOllamaWrapper(model="qwen3", reasoning=True)
        bound = parent.bind_tools([])
        peer = without_reasoning(bound)
        assert peer.bound.reasoning is False
        assert parent.reasoning is True
        assert bound.bound.reasoning is True
        assert peer.kwargs == bound.kwargs  # tools survived the copy

    def test_an_uncopyable_model_returns_none_for_the_fallback(self):
        # Callers fall back to the in-place disable/restore pair on None, so this
        # must not raise.
        assert without_reasoning(FakePlainModel()) is None


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
