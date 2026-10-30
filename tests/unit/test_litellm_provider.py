"""Exercise the installed LiteLLM adapter without contacting a model provider."""

import base64
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path

from mnemoai.models.controllers.llm_controller import LangChainLLMController
from mnemoai.utils.config import config


def test_patched_litellm_owns_loaded_modules_and_serves_chat(monkeypatch, sdk_http_modules):
    # Crawl4AI also depends on unclecode-litellm, which shares the module name.
    # Verify the files actually loaded, not only the installed version metadata.
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(config, "_config_data", {
        "MODEL_ID": {
            "TYPE": "litellm", "NAME": "openai/gpt-4o-mini",
            "API_BASE": "https://sdk-test.invalid/v1",
            "API_KEY": "fixture-litellm-key", "MAX_TOKENS": 16, "STREAM": False,
        },
    })
    requests = []
    for backend in sdk_http_modules:
        def respond(self, request, backend=backend):
            requests.append(request)
            assert str(request.url) == "https://sdk-test.invalid/v1/chat/completions"
            assert request.headers["authorization"] == "Bearer fixture-litellm-key"
            assert json.loads(request.content)["model"] == "gpt-4o-mini"
            return backend.Response(200, request=request, json={
                "id": "fixture", "object": "chat.completion", "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0, "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            })
        monkeypatch.setattr(backend.HTTPTransport, "handle_request", respond)

    model = LangChainLLMController().get_model()
    assert model.invoke("Reply OK").content == "OK"
    assert len(requests) == 1

    distribution = importlib.metadata.distribution("litellm")
    for name in ("litellm", "litellm.main"):
        loaded = Path(importlib.import_module(name).__file__).resolve()
        record = next(
            entry for entry in distribution.files
            if Path(distribution.locate_file(entry)).resolve() == loaded
        )
        assert record.hash is not None and record.hash.mode == "sha256"
        digest = base64.urlsafe_b64encode(hashlib.sha256(loaded.read_bytes()).digest()).decode().rstrip("=")
        assert digest == record.hash.value, f"{name} was overwritten by another distribution"
