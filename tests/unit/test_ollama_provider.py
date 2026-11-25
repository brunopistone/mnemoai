"""Ollama runtime settings must reach real SDK request bodies, without network."""

import json

import pytest
from langchain_core.messages import HumanMessage

from mnemoai.models.chat_models.chat_ollama_wrapper import ChatOllamaWrapper
from mnemoai.models.controllers.embeddings_controller import EmbeddingsController
from mnemoai.models.controllers.llm_controller import LangChainLLMController
from mnemoai.models.controllers.vision_model_controller import VisionModelController
from mnemoai.utils.config import config

_UNSET = object()


@pytest.fixture
def ollama_requests(monkeypatch, sdk_http_modules):
    requests = []

    for module in sdk_http_modules:
        def respond(request, http=module):
            body = json.loads(request.content)
            requests.append((request.url.path, body))
            if request.url.path == "/api/embed":
                response = {
                    "model": body["model"],
                    "embeddings": [[1.0, 0.0] for _ in body["input"]],
                }
            else:
                assert request.url.path == "/api/chat"
                response = {
                    "model": body["model"],
                    "message": {"role": "assistant", "content": "OK"},
                    "done": True,
                }
            return http.Response(
                200, request=request, content=json.dumps(response) + "\n",
                headers={"content-type": "application/x-ndjson"},
            )

        def send(self, request, *args, _respond=respond, **kwargs):
            return _respond(request)

        async def asend(self, request, *args, _respond=respond, **kwargs):
            return _respond(request)

        monkeypatch.setattr(module.Client, "send", send)
        monkeypatch.setattr(module.AsyncClient, "send", asend)
    return requests


def _controller(monkeypatch, section, batch=_UNSET, extra=None):
    model = {"TYPE": "ollama", "NAME": "test-model"}
    if batch is not _UNSET:
        model["NUM_BATCH"] = batch
    if extra:
        model["EXTRA_PARAMS"] = extra
    data = {
        section: model,
        "MAX_CONVERSATION_TOKENS": 8192,
        "RAG": {"EMBEDDINGS": {"CACHE_ENABLED": False}},
    }
    monkeypatch.setattr(config, "get", lambda key, default=None: data.get(key, default))
    if section == "EMBED_MODEL_ID":
        return EmbeddingsController(model)
    cls = LangChainLLMController if section == "MODEL_ID" else VisionModelController
    return cls()


@pytest.mark.parametrize("section", ["MODEL_ID", "VISION_MODEL_ID", "EMBED_MODEL_ID"])
@pytest.mark.parametrize("batch", [_UNSET, None, 512, 2048], ids=["omitted", "null", "512", "2048"])
def test_num_batch_reaches_ollama_options(monkeypatch, ollama_requests, section, batch):
    controller = _controller(monkeypatch, section, batch)
    if section == "EMBED_MODEL_ID":
        assert controller._embed_raw(["hello"]).tolist() == [[1.0, 0.0]]
    else:
        assert controller.get_model().invoke([HumanMessage(content="hello")]).content == "OK"
    assert len(ollama_requests) == 1
    _, body = ollama_requests[0]
    assert "num_batch" not in body  # Ollama expects options, not a top-level arg.
    options = body.get("options") or {}
    assert options["num_batch"] == (2048 if batch is _UNSET or batch is None else batch)
    if section == "MODEL_ID":
        assert options["num_ctx"] == 8192


@pytest.mark.parametrize("section", ["MODEL_ID", "VISION_MODEL_ID", "EMBED_MODEL_ID"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "512"])
def test_invalid_batch_fails_before_a_request(
    monkeypatch, ollama_requests, section, value
):
    with pytest.raises(ValueError):
        controller = _controller(monkeypatch, section, value)
        if section != "EMBED_MODEL_ID":
            controller.get_model()
    assert ollama_requests == []


@pytest.mark.parametrize("section", ["MODEL_ID", "VISION_MODEL_ID"])
def test_extra_params_override_configured_batch(monkeypatch, ollama_requests, section):
    model = _controller(
        monkeypatch, section, 512, {"num_batch": 1024}
    ).get_model()
    model.invoke("hello")
    assert ollama_requests[0][1]["options"]["num_batch"] == 1024


def test_request_options_override_without_mutating_the_callers_dict(ollama_requests):
    model = ChatOllamaWrapper(model="test-model", num_batch=512)
    options = {"num_batch": 1024, "temperature": 0.2}
    model.invoke("hello", options=options)
    assert ollama_requests[0][1]["options"] == options
    assert options == {"num_batch": 1024, "temperature": 0.2}
    options = {"temperature": 0.3}
    model.invoke("hello", options=options)
    assert ollama_requests[1][1]["options"]["num_batch"] == 512
    assert options == {"temperature": 0.3}


@pytest.mark.parametrize("stream", [False, True])
def test_async_requests_forward_batch(ollama_requests, stream):
    import asyncio

    async def call():
        model = ChatOllamaWrapper(model="test-model", num_batch=512)
        if stream:
            return "".join([chunk.content async for chunk in model.astream("hello")])
        return (await model.ainvoke("hello")).content

    assert asyncio.run(call()) == "OK"
    assert ollama_requests[0][1]["options"]["num_batch"] == 512


def test_auxiliary_variant_uses_its_own_batch(monkeypatch, ollama_requests):
    controller = _controller(monkeypatch, "MODEL_ID", 512)
    main = controller.get_model()
    variant = controller.build_model_variant({"NAME": "small-model", "NUM_BATCH": 128})
    variant.invoke("hello")
    main.invoke("hello")
    assert [(body["model"], body["options"]["num_batch"])
            for _, body in ollama_requests] == [("small-model", 128), ("test-model", 512)]
