"""Opt-in Nebius tests using configured credentials and isolated MCP state."""

import io

import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from PIL import Image

from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.models import nebius
from mnemoai.models.controllers.embeddings_controller import EmbeddingsController
from mnemoai.models.controllers.llm_controller import LangChainLLMController
from mnemoai.models.controllers.vision_model_controller import VisionModelController
from mnemoai.utils.config import config

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        (config.get("MODEL_ID", {}) or {}).get("TYPE") != "nebius",
        reason="requires an explicitly configured Nebius test profile",
    ),
]

MODELS = [
    (nebius.DEFAULT_CHAT_MODEL, False, None),
    ("openai/gpt-oss-120b", True, "low"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", True, None),
    ("moonshotai/Kimi-K2.6", True, None),
]


@pytest.mark.parametrize("model_id,reasoning,effort", MODELS)
@pytest.mark.parametrize("visible", [True, False], ids=["verbose", "quiet"])
def test_nebius_tools_reasoning_and_output(
    live_client, tmp_path, monkeypatch, capsys, record_testsuite_property,
    model_id, reasoning, effort, visible,
):
    controller = LangChainLLMController(verbose=visible)
    model = controller.build_model_variant({
        "TYPE": "nebius", "NAME": model_id, "MAX_TOKENS": 2048,
        "REASONING_EFFORT": effort, "STREAM": True,
    })
    tools = [tool for tool in live_client.tools if tool.name in ("fs_write", "fs_read")]
    assert len(tools) == 2
    agent = LangGraphAgent(
        model, tools, verbose=visible,
        system_prompt="Complete the requested temporary-file task with the available tools.",
        auto_approve_provider=lambda: "all",
    )
    agent.styled_turn_view = True
    target = tmp_path / "nebius-tool-check.txt"
    result = agent.invoke(
        f"Use fs_write(command='create', path='{target}', file_text='NEBIUS_TOOL_OK'), "
        f"then fs_read(path='{target}', mode='Line'). After reading it, reply with "
        "NEBIUS_TOOL_OK. Perform both calls; do not merely describe them."
    )
    assert target.read_text() == "NEBIUS_TOOL_OK"
    assert "NEBIUS_TOOL_OK" in result
    messages = agent.messages
    assert sum(isinstance(message, ToolMessage) for message in messages) >= 2
    thoughts = [
        message.additional_kwargs.get("reasoning_content", "")
        for message in messages if isinstance(message, AIMessage)
    ]
    rendered = capsys.readouterr().out
    if reasoning:
        assert any(thoughts), f"{model_id} must preserve the reasoning it returns"
        assert ("Thought for" in rendered) is visible
    else:
        assert not any(thoughts)
    assert agent._last_input_tokens and agent._last_input_tokens > 0
    assert agent.usage.totals()["calls"] >= 2
    record_testsuite_property(f"nebius-{model_id}-{visible}", {
        "model": model_id, "reasoning_chars": sum(len(t) for t in thoughts),
        "reasoning_visible": visible and reasoning, "last_input_tokens": agent._last_input_tokens,
        "tool_results": sum(isinstance(message, ToolMessage) for message in messages),
    })


def test_nebius_vision_describes_generated_image(live_client, monkeypatch):
    monkeypatch.setitem(config._config_data, "VISION_MODEL_ID", {
        "TYPE": "nebius", "NAME": nebius.DEFAULT_VISION_MODEL, "MAX_TOKENS": 256,
    })
    image = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(image, format="PNG")
    result = VisionModelController().describe_image(
        "What is the dominant color? Reply with one color word.", image.getvalue(),
    )
    assert "red" in result.lower()


def test_nebius_embeddings_honor_dimensions(live_client):
    controller = EmbeddingsController({
        "TYPE": "nebius", "NAME": nebius.DEFAULT_EMBED_MODEL, "DIMENSION": 64,
    })
    embeddings = controller.embed(["A red apple is a fruit.", "A dog is an animal."])
    assert embeddings.shape == (2, 64)
    assert np.isfinite(embeddings).all()
    assert np.linalg.norm(embeddings[0]) > 0
    assert not np.array_equal(embeddings[0], embeddings[1])


def test_nebius_area_models_make_real_calls(live_client, monkeypatch):
    overrides = {
        name: {"TYPE": "nebius", "NAME": nebius.DEFAULT_CHAT_MODEL, "MAX_TOKENS": 128}
        for name in ("ROUTER", "ORCHESTRATOR", "SUMMARY", "REVIEWER")
    }
    monkeypatch.setitem(config._config_data, "AREA_MODELS", overrides)
    live_client._area_model_cache.clear()
    try:
        for area in overrides:
            model = live_client._area_model(area)
            response = model.invoke([HumanMessage(content="Reply exactly AREA_OK. Do not use tools.")])
            assert "AREA_OK" in response.content, area
            assert response.usage_metadata["input_tokens"] > 0
    finally:
        live_client._area_model_cache.clear()
