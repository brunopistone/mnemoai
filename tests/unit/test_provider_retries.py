"""Application retry limits must count actual SDK transport attempts."""

import pytest

from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.models.controllers import llm_controller


@pytest.mark.parametrize("provider,protocol", [
    ("openai", "chat_completions"), ("openai", "responses"),
    ("mlx", "chat_completions"), ("anthropic", "anthropic"),
    ("mantle", "chat_completions"), ("mantle", "responses"), ("mantle", "anthropic"),
])
def test_three_application_attempts_mean_three_http_attempts(monkeypatch, provider, protocol, sdk_http_modules):
    attempts = []

    model_id = {
        "TYPE": provider, "NAME": "claude-sonnet-4-5", "API_KEY": "test-key",
        "API_PROTOCOL": protocol, "MAX_TOKENS": 64,
        "API_BASE": "https://sdk-test.invalid", "ENDPOINT_URL": "https://sdk-test.invalid",
    }
    monkeypatch.setattr(llm_controller.config, "get", lambda key, default=None: (
        model_id if key == "MODEL_ID" else default
    ))
    for backend in sdk_http_modules:
        def unavailable(self, request, backend=backend):
            attempts.append(request)
            return backend.Response(503, request=request, json={
                "error": {"type": "server_error", "message": "Service unavailable"},
            })

        monkeypatch.setattr(backend.HTTPTransport, "handle_request", unavailable)
    controller = llm_controller.LangChainLLMController()
    controller.initialize_model()
    model = controller.get_model()
    assert model.max_retries == 0
    agent = LangGraphAgent.__new__(LangGraphAgent)
    agent._stream_idle_timeout = 0
    agent._empty_response_retries = 2
    agent._transient_retry_delay = lambda *args: 0
    with pytest.raises(Exception):
        agent._stream_response([], {}, model=model, quiet=True)
    assert len(attempts) == 3
