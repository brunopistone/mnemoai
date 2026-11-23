"""Auxiliary JSON requests preserve native messages and provider capabilities."""

import json
from pathlib import Path

import yaml
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from mnemoai.client.work_review import parse_strategy
from mnemoai.models.chat_models.chat_ollama_wrapper import ChatOllamaWrapper
from mnemoai.models.structured_output import with_json_schema
from mnemoai.utils import config as config_module


def test_native_schema_reaches_sdk_request_without_mutating_model_or_binding():
    model = ChatOllamaWrapper(model="unused")
    bound = model.bind(stop=["END"])
    schema = {
        "type": "object",
        "required": ["value"],
        "properties": {"value": {"type": "string", "enum": ["verified"]}},
    }
    constrained = with_json_schema(bound, schema)
    schema["properties"]["value"]["enum"].append("changed later")
    params = constrained.bound._chat_params(
        [HumanMessage("Return the verified value.")],
        **constrained.kwargs,
    )
    assert params["format"]["properties"]["value"]["enum"] == ["verified"]
    assert params["options"]["stop"] == ["END"]
    assert model.format is None
    assert bound.kwargs == {"stop": ["END"]}


def test_provider_without_object_format_keeps_normal_request_interface():
    class Model(BaseModel):
        format: str | None = None

        def bind(self, **kwargs):
            raise AssertionError("Do not send an unsupported native-format option")

    model = Model()
    assert with_json_schema(model, {"type": "object"}) is model


def test_shipped_strategy_example_is_accepted_by_its_actual_validator():
    prompts = yaml.safe_load(
        Path(config_module.__file__).with_name("prompts.yaml").read_text()
    )
    decoder = json.JSONDecoder()

    def example(key):
        prompt = prompts[key]
        value, _ = decoder.raw_decode(prompt[prompt.index("{") :])
        return AIMessage(content=json.dumps(value))

    assert parse_strategy(example("WORK_STRATEGY_PROMPT"))["kind"] == "artifact"
