"""Nebius wire compatibility, reasoning, credentials and configuration."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import httpx
import numpy as np
import pytest
import yaml
from langchain_core.messages import HumanMessage, ToolMessage
from openai import OpenAI

from mnemoai.client import doctor
from mnemoai.models import nebius
from mnemoai.models.chat_models import chat_nebius
from mnemoai.models.controllers import embeddings_controller
from mnemoai.models.controllers.embeddings_controller import EmbeddingsController
from mnemoai.models.controllers.llm_controller import LangChainLLMController
from mnemoai.models.controllers.vision_model_controller import VisionModelController
from mnemoai.utils import configurator, paths
from mnemoai.utils.config import config


@pytest.fixture
def settings(monkeypatch):
    data = {
        "MODEL_ID": {"TYPE": "nebius", "NAME": nebius.DEFAULT_CHAT_MODEL},
        "VISION_MODEL_ID": {"TYPE": "nebius", "NAME": nebius.DEFAULT_VISION_MODEL},
        "RAG": {
            "EMBED_MODEL_ID": {"TYPE": "nebius", "NAME": nebius.DEFAULT_EMBED_MODEL, "DIMENSION": 2},
            "EMBEDDINGS": {"CACHE_ENABLED": False, "RETRIES": 0},
        },
        "LLM": {"REQUEST_TIMEOUT": 7},
        "MAX_CONVERSATION_TOKENS": 8192,
    }
    monkeypatch.setattr(config, "_config_data", data)
    monkeypatch.setenv("NEBIUS_API_KEY", "nebius-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    monkeypatch.setenv("OPENAI_ORG_ID", "unrelated-openai-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "unrelated-openai-project")
    return data


@pytest.fixture
def wire(monkeypatch):
    requests = []
    clients = []

    def respond(request):
        body = json.loads(request.content)
        requests.append((request, body))
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={
                "object": "list", "model": body["model"],
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
                ],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            })
        usage = {
            "prompt_tokens": 37, "completion_tokens": 11, "total_tokens": 48,
            "prompt_tokens_details": {"cached_tokens": 8},
            "completion_tokens_details": {"reasoning_tokens": 3},
        }
        tool_turn = bool(body.get("tools")) and body["messages"][-1]["role"] != "tool"
        answer = body["messages"][-1]["content"] if body["messages"][-1]["role"] == "tool" else "ANSWER"
        calls = [{
            "id": "call-add", "type": "function",
            "function": {"name": "add", "arguments": '{"a":2,"b":3}'},
        }] if tool_turn else []
        message = {"role": "assistant", "content": "" if calls else answer, "reasoning": "Check facts."}
        if calls:
            message["tool_calls"] = calls
        finish = "tool_calls" if calls else "stop"
        if not body.get("stream"):
            return httpx.Response(200, json={
                "id": "chat-test", "object": "chat.completion", "created": 1,
                "model": body["model"], "usage": usage,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            })
        deltas = [{"role": "assistant", "reasoning": "Check "}, {"reasoning": "facts."}]
        if calls:
            deltas += [
                {"tool_calls": [{"index": 0, "id": "call-add", "type": "function",
                                 "function": {"name": "add", "arguments": '{"a":2,'}}]},
                {"tool_calls": [{"index": 0, "function": {"arguments": '"b":3}'}}]},
            ]
        else:
            deltas.append({"content": answer})
        chunks = [{
            "id": "chat-test", "object": "chat.completion.chunk", "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        } for delta in deltas]
        chunks[-1]["choices"][0]["finish_reason"] = finish
        # Nebius reports cumulative counters on every chunk, then repeats the
        # final total in the empty-choices usage chunk.
        for index, chunk in enumerate(chunks):
            chunk["usage"] = {
                "prompt_tokens": 37, "completion_tokens": index,
                "total_tokens": 37 + index,
            }
        chunks[-1]["usage"] = {k: v for k, v in usage.items() if not isinstance(v, dict)}
        chunks.append({
            "id": "chat-test", "object": "chat.completion.chunk", "created": 1,
            "model": body["model"], "choices": [], "usage": usage,
        })
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content="".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n")

    transport = httpx.MockTransport(respond)
    original = chat_nebius.ChatNebius

    def model(**kwargs):
        sync = httpx.Client(transport=transport)
        asynchronous = httpx.AsyncClient(transport=transport)
        clients.extend((sync, asynchronous))
        return original(http_client=sync, http_async_client=asynchronous, **kwargs)

    def embed_client(**kwargs):
        client = httpx.Client(transport=transport)
        clients.append(client)
        return OpenAI(http_client=client, **kwargs)

    monkeypatch.setattr(chat_nebius, "ChatNebius", model)
    monkeypatch.setattr(embeddings_controller, "OpenAI", embed_client)
    yield requests
    for client in clients:
        if isinstance(client, httpx.AsyncClient):
            asyncio.run(client.aclose())
        else:
            client.close()


def _chat():
    controller = LangChainLLMController()
    controller.initialize_model()
    return controller.get_model()


@pytest.mark.parametrize("streaming", [True, False])
def test_chat_stream_switch_reasoning_usage_and_account_isolation(settings, wire, streaming):
    settings["MODEL_ID"]["STREAM"] = streaming
    model = _chat()
    chunks = list(model.stream([HumanMessage(content="Question")]))
    response = chunks[0]
    for chunk in chunks[1:]:
        response += chunk
    assert response.content == "ANSWER"
    assert response.additional_kwargs["reasoning_content"] == "Check facts."
    assert response.usage_metadata["input_tokens"] == 37
    assert response.usage_metadata["output_tokens"] == 11
    assert response.usage_metadata["total_tokens"] == 48
    assert response.usage_metadata["input_token_details"]["cache_read"] == 8
    assert response.usage_metadata["output_token_details"]["reasoning"] == 3
    request, body = wire[-1]
    assert str(request.url) == nebius.DEFAULT_BASE_URL + "/chat/completions"
    assert request.headers["Authorization"] == "Bearer nebius-test-key"
    assert "openai-organization" not in request.headers and "openai-project" not in request.headers
    assert body["stream"] is streaming
    if streaming:
        assert body["stream_options"]["include_usage"] is True
    assert model.max_retries == 0


def test_async_reasoning_and_usage(settings, wire):
    async def stream():
        chunks = [chunk async for chunk in _chat().astream("Question")]
        result = chunks[0]
        for chunk in chunks[1:]:
            result += chunk
        return result
    result = asyncio.run(stream())
    assert result.content == "ANSWER"
    assert result.additional_kwargs["reasoning_content"] == "Check facts."
    assert result.usage_metadata["input_tokens"] == 37
    assert result.usage_metadata["output_tokens"] == 11
    assert result.usage_metadata["total_tokens"] == 48


def test_cumulative_usage_is_local_to_each_concurrent_stream(settings, wire):
    model = _chat()

    def run(_):
        chunks = list(model.stream("Question"))
        result = chunks[0]
        for chunk in chunks[1:]:
            result += chunk
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(run, range(4)))
    assert all(response.usage_metadata["total_tokens"] == 48 for response in responses)
    assert all(response.content == "ANSWER" for response in responses)


def test_usage_delta_retains_late_details_and_counter_corrections():
    previous = {}
    first = chat_nebius._usage_delta({"input_tokens": 40, "output_tokens": 1}, previous)
    final = chat_nebius._usage_delta({
        "input_tokens": 39, "output_tokens": 4, "input_token_details": {"cache_read": 8},
    }, previous)
    assert first["input_tokens"] + final["input_tokens"] == 39
    assert first["output_tokens"] + final["output_tokens"] == 4
    assert final["input_token_details"]["cache_read"] == 8


def test_sampling_and_advanced_parameters_reach_nebius(settings, wire):
    settings["MODEL_ID"].update(
        MAX_TOKENS=123, TEMPERATURE=0.2, TOP_P=0.8, TOP_K=12, STOP=["END"],
        REPETITION_PENALTY=1.1, FREQUENCY_PENALTY=0.3, PRESENCE_PENALTY=0.4,
        REASONING_EFFORT="low", EXTRA_PARAMS={"chat_template_kwargs": {"enable_thinking": True}},
    )
    before = deepcopy(settings["MODEL_ID"])
    _chat().invoke("Question")
    body = wire[-1][1]
    assert body.get("max_completion_tokens", body.get("max_tokens")) == 123
    for key, value in {
        "temperature": 0.2, "top_p": 0.8, "top_k": 12, "stop": ["END"],
        "repetition_penalty": 1.1, "frequency_penalty": 0.3, "presence_penalty": 0.4,
        "reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": True},
    }.items():
        assert body[key] == value
    assert settings["MODEL_ID"] == before


def test_tool_round_trip_keeps_reasoning_and_tool_results(settings, wire):
    schema = {
        "type": "function",
        "function": {"name": "add", "description": "Add integers", "parameters": {
            "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }},
    }
    model = _chat().bind_tools([schema])
    prompt = HumanMessage(content="Add 2 and 3 with the add tool")
    response = model.invoke([prompt])
    call = response.tool_calls[0]
    assert call["name"] == "add" and call["args"] == {"a": 2, "b": 3}
    value = call["args"]["a"] + call["args"]["b"]
    final = model.invoke([prompt, response, ToolMessage(content=str(value), tool_call_id=call["id"])])
    assert final.content == "5"
    messages = wire[-1][1]["messages"]
    assert messages[-2]["reasoning_content"] == "Check facts."
    assert messages[-1]["tool_call_id"] == "call-add" and messages[-1]["content"] == "5"


def test_vision_parameters_and_image_content(settings, wire):
    settings["VISION_MODEL_ID"].update(MAX_TOKENS=100, TOP_P=0.9, TOP_K=7, REASONING_EFFORT="low")
    controller = VisionModelController()
    controller.initialize_model()
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    response = controller.get_model().invoke([HumanMessage(content=[{"type": "text", "text": "Describe"}, image])])
    assert response.content == "ANSWER"
    body = wire[-1][1]
    assert body["messages"][0]["content"][1] == image
    assert body["model"] == nebius.DEFAULT_VISION_MODEL
    assert body["top_k"] == 7 and body["reasoning_effort"] == "low"


def test_embeddings_dimensions_order_and_credentials(settings, wire):
    controller = EmbeddingsController()
    result = controller.embed(["first", "second"])
    np.testing.assert_array_equal(result, [[1, 0], [0, 1]])
    assert result.dtype == np.float32
    request, body = wire[-1]
    assert str(request.url) == nebius.DEFAULT_BASE_URL + "/embeddings"
    assert body["dimensions"] == 2 and body["encoding_format"] == "float"
    assert body["input"] == ["first", "second"]
    assert request.headers["Authorization"] == "Bearer nebius-test-key"
    assert "openai-organization" not in request.headers and "openai-project" not in request.headers


@pytest.mark.parametrize("model_limit,global_limit,expected", [
    (12, 8, 12), (0, 8, 100), (None, 8, 8), (None, 0, 100),
    (None, None, 100),
])
def test_embedding_character_limit_reaches_provider_with_legacy_fallback(
    settings, wire, model_limit, global_limit, expected,
):
    settings["RAG"]["EMBED_MODEL_ID"]["MAX_INPUT_CHARS"] = model_limit
    if global_limit is not None:
        settings["RAG"]["EMBEDDINGS"]["MAX_INPUT_CHARS"] = global_limit
    EmbeddingsController().embed(["0123456789" * 10, "abcdefghij" * 10])
    assert all(len(text) == expected for text in wire[-1][1]["input"])


def test_params_saved_character_limit_is_applied_after_reload(settings, wire, monkeypatch, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(configurator, "_config_to_edit", lambda: path)
    monkeypatch.setattr(configurator, "_is_tty", lambda: False)
    choice = next(
        key for key, (section, _) in configurator._PARAM_SECTIONS.items()
        if section == "EMBED_MODEL_ID"
    )
    monkeypatch.setattr(configurator, "_ask_choice", lambda *a, **kw: choice)
    monkeypatch.setattr(configurator, "_ask", lambda prompt, default=None, **kw: (
        "12" if prompt.split()[0] == "MAX_INPUT_CHARS" else default or ""
    ))
    assert configurator.run_params_override() is not None
    monkeypatch.setattr(config, "_config_data", yaml.safe_load(path.read_text()))
    EmbeddingsController().embed(["0123456789" * 10, "abcdefghij" * 10])
    assert wire[-1][1]["input"] == ["012345678901", "abcdefghijab"]


def test_disabled_character_cap_keeps_the_token_limit(settings, wire):
    settings["RAG"]["EMBED_MODEL_ID"].update(MAX_INPUT_CHARS=0, MAX_INPUT_TOKENS=16)
    original = "an embedding input " * 100
    controller = EmbeddingsController()
    controller.embed([original, original])
    assert controller.max_input_chars == 0
    assert all(
        0 < embeddings_controller.count_tokens(text) <= 16
        for text in wire[-1][1]["input"]
    )


def test_default_and_explicit_embedding_endpoints_have_same_identity(settings):
    implicit = EmbeddingsController()
    cfg = dict(settings["RAG"]["EMBED_MODEL_ID"], API_BASE=nebius.DEFAULT_BASE_URL + "/")
    explicit = EmbeddingsController(cfg)
    assert implicit.fingerprint() == explicit.fingerprint()
    cfg["API_BASE"] = "https://example.invalid/v1"
    assert EmbeddingsController(cfg).fingerprint() != implicit.fingerprint()


@pytest.mark.parametrize("kind", ["chat", "vision", "embeddings"])
def test_missing_nebius_key_never_falls_back_to_openai(settings, monkeypatch, kind):
    monkeypatch.delenv("NEBIUS_API_KEY")
    with pytest.raises(ValueError, match="NEBIUS_API_KEY"):
        if kind == "chat":
            _chat()
        elif kind == "vision":
            VisionModelController().initialize_model()
        else:
            EmbeddingsController()._embed_nebius(["test"])
    check = doctor._credentials_check("nebius", {})
    assert "NEBIUS_API_KEY" in check.fix


def test_explicit_credentials_and_endpoint_override(settings, wire):
    settings["MODEL_ID"].update(API_KEY="explicit-nebius-key", ENDPOINT_URL="https://example.invalid/v1/")
    _chat().invoke("Question")
    assert str(wire[-1][0].url) == "https://example.invalid/v1/chat/completions"
    assert wire[-1][0].headers["Authorization"] == "Bearer explicit-nebius-key"


@pytest.mark.parametrize("section", ["MODEL_ID", "VISION_MODEL_ID", "EMBED_MODEL_ID", "ROUTER", "ORCHESTRATOR", "SUMMARY", "REVIEWER"])
def test_model_ui_saves_nebius_for_every_role(settings, monkeypatch, tmp_path, section):
    data = deepcopy(settings)
    data.update(ENABLE_ROUTING=False, ENABLE_ORCHESTRATION=False, ENABLE_REVIEW=False)
    data["AREA_MODELS"] = {
        area: {"TYPE": "openai", "NAME": "old", "API_BASE": "https://old.invalid/v1", "API_KEY": "old-key"}
        for area in ("ROUTER", "ORCHESTRATOR", "SUMMARY", "REVIEWER")
    }
    for block in (data["MODEL_ID"], data["VISION_MODEL_ID"], data["RAG"]["EMBED_MODEL_ID"]):
        block.update(TYPE="openai", API_BASE="https://old.invalid/v1", API_KEY="old-key")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(configurator, "_config_to_edit", lambda: path)
    monkeypatch.setattr(configurator, "_is_tty", lambda: False)
    choice = next(k for k, (s, *_) in configurator._MODEL_SECTIONS.items() if s == section)

    def choose(prompt, valid, default=None, **kwargs):
        if prompt == "Which model to change?":
            return choice
        return next(k for k, label in kwargs["labels"].items() if label == "nebius")

    def answer(prompt, default=None, **kwargs):
        if prompt == "Model name":
            return "chosen-model"
        if "API key" in prompt:
            return ""
        return default or ""

    monkeypatch.setattr(configurator, "_ask_choice", choose)
    monkeypatch.setattr(configurator, "_ask", answer)
    monkeypatch.setattr(configurator, "_ask_bool", lambda prompt, *a, **kw: "same model as chat" not in prompt.lower())
    monkeypatch.setattr(configurator, "_ask_number", lambda *a, **kw: "256")
    result = configurator.run_model_override()
    assert result and result.section == section
    saved = yaml.safe_load(path.read_text())
    block = saved["RAG"]["EMBED_MODEL_ID"] if section == "EMBED_MODEL_ID" else (
        saved["AREA_MODELS"][section] if section in data["AREA_MODELS"] else saved[section]
    )
    assert block["TYPE"] == "nebius" and block["NAME"] == "chosen-model"
    assert block["API_BASE"] == nebius.DEFAULT_BASE_URL
    assert "API_KEY" not in block and "ENDPOINT_URL" not in block
    if section in configurator._AREA_FEATURE:
        assert saved[configurator._AREA_FEATURE[section][0]] is True


@pytest.mark.parametrize("section", ["MODEL_ID", "VISION_MODEL_ID", "EMBED_MODEL_ID", "ROUTER", "ORCHESTRATOR", "SUMMARY", "REVIEWER"])
def test_params_ui_persists_values_for_every_nebius_role(settings, monkeypatch, tmp_path, section):
    data = deepcopy(settings)
    data["AREA_MODELS"] = {name: {"TYPE": "nebius", "NAME": "reasoning-model"}
                           for name in ("ROUTER", "ORCHESTRATOR", "SUMMARY", "REVIEWER")}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(configurator, "_config_to_edit", lambda: path)
    monkeypatch.setattr(configurator, "_is_tty", lambda: False)
    choice = next(k for k, (s, _) in configurator._PARAM_SECTIONS.items() if s == section)
    monkeypatch.setattr(configurator, "_ask_choice", lambda *a, **kw: choice)
    values = {"TEMPERATURE": "0.2", "TOP_K": "12", "REASONING_EFFORT": "high", "STREAM": "false",
              "DIMENSION": "64", "MAX_INPUT_TOKENS": "2048", "MAX_INPUT_CHARS": "12000"}
    monkeypatch.setattr(configurator, "_ask", lambda prompt, default=None, **kw: values.get(prompt.split()[0], default or ""))
    result = configurator.run_params_override()
    assert result and result.section == section
    saved = yaml.safe_load(path.read_text())
    block = saved["RAG"]["EMBED_MODEL_ID"] if section == "EMBED_MODEL_ID" else (
        saved["AREA_MODELS"][section] if section in data["AREA_MODELS"] else saved[section]
    )
    if section == "EMBED_MODEL_ID":
        assert block["DIMENSION"] == 64 and block["MAX_INPUT_TOKENS"] == 2048 and block["MAX_INPUT_CHARS"] == 12000
    else:
        assert block["TEMPERATURE"] == 0.2 and block["TOP_K"] == 12 and block["REASONING_EFFORT"] == "high"
        if section != "VISION_MODEL_ID":
            assert block["STREAM"] is False


def test_first_run_setup_uses_complete_nebius_template(settings, monkeypatch, tmp_path):
    path = tmp_path / "config.yaml"
    monkeypatch.setattr(configurator, "_is_tty", lambda: False)
    monkeypatch.setattr(configurator, "_ask_choice", lambda prompt, valid, default=None, **kw: (
        "9" if prompt == "Choose your LLM provider"
        else next(k for k, label in kw["labels"].items() if label == "nebius")
    ))
    monkeypatch.setattr(configurator, "_ask", lambda prompt, default=None, **kw: (
        nebius.DEFAULT_CHAT_MODEL if prompt == "Chat model name" else default or ""
    ))
    monkeypatch.setattr(configurator, "_ask_bool", lambda prompt, default=True, **kw: (
        False if "same model as chat" in prompt.lower() or "web search" in prompt.lower()
        else default
    ))
    monkeypatch.setattr(configurator, "_ask_number", lambda prompt, default=None, **kw: (
        None if default == "none" else default
    ))
    assert configurator._run_configurator(path) == path
    parsed = yaml.safe_load(path.read_text())
    assert parsed["MODEL_ID"]["TYPE"] == parsed["VISION_MODEL_ID"]["TYPE"] == parsed["RAG"]["EMBED_MODEL_ID"]["TYPE"] == "nebius"
    assert parsed["VISION_MODEL_ID"]["NAME"] == nebius.DEFAULT_VISION_MODEL
    assert parsed["ENABLE_PLAYBOOK"] is True
    assert not parsed.get("ENV", {}).get("NEBIUS_API_KEY")


def test_features_toggle_preserves_nebius_models(settings, monkeypatch, tmp_path):
    data = deepcopy(settings)
    data.update(ENABLE_ROUTING=False, ENABLE_ORCHESTRATION=False, ENABLE_REVIEW=False,
                ENABLE_RAG=False, ENABLE_EPISODIC_MEMORY=False, ENABLE_PLAYBOOK=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(configurator, "_config_to_edit", lambda: path)
    monkeypatch.setattr(configurator, "_is_tty", lambda: True)
    enabled = ["ENABLE_ROUTING", "ENABLE_ORCHESTRATION", "ENABLE_RAG", "ENABLE_EPISODIC_MEMORY", "ENABLE_PLAYBOOK"]
    monkeypatch.setattr(configurator, "_dialog_checkbox", lambda *a, **kw: enabled)
    result = configurator.run_features_override()
    assert result is not None
    parsed = yaml.safe_load(path.read_text())
    assert all(parsed[key] for key in enabled)
    assert parsed["MODEL_ID"] == data["MODEL_ID"]
    assert parsed["RAG"]["EMBED_MODEL_ID"] == data["RAG"]["EMBED_MODEL_ID"]


def test_nebius_template_reaches_existing_installs(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))
    (tmp_path / "config").mkdir()
    active = tmp_path / "config/config.yaml"
    active.write_text("MODEL_ID: {TYPE: bedrock, NAME: keep-this-model}\n")
    before = active.read_bytes()
    paths.seed_example_files()
    example = tmp_path / "config/config.yaml.nebius.example"
    assert example.is_file()
    assert yaml.safe_load(example.read_text())["MODEL_ID"]["TYPE"] == "nebius"
    assert active.read_bytes() == before
