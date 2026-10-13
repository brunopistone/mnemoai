"""Real SDK request construction with in-memory HTTP transports, never AWS calls."""

import asyncio

import pytest

from mnemoai.models.chat_models.mantle_auth import RotatingTokenHeaders
from mnemoai.models.mantle_factory import build_mantle_model


@pytest.mark.parametrize("protocol", ["chat_completions", "responses", "anthropic"])
@pytest.mark.parametrize("explicit_key", [False, True])
def test_generated_tokens_rotate_for_sync_and_async_requests(
    monkeypatch, protocol, explicit_key, sdk_http_modules,
):
    import aws_bedrock_token_generator

    minted, headers = [], []

    def provide_token(**kwargs):
        assert not explicit_key
        minted.append(f"generated-{len(minted) + 1}")
        return minted[-1]

    def response(request, backend):
        header = "x-api-key" if protocol == "anthropic" else "authorization"
        headers.append(request.headers[header])
        return backend.Response(200, request=request, json={
            "id": "fixture", "model": "fixture", "object": "chat.completion",
            "created": 1, "created_at": 1, "status": "completed",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
            "type": "message", "role": "assistant", "content": [{"type": "text", "text": "OK"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "output": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    monkeypatch.delenv("BEDROCK_API_KEY", raising=False)
    monkeypatch.setattr(aws_bedrock_token_generator, "provide_token", provide_token)
    for backend in sdk_http_modules:
        def sync_transport(self, request, backend=backend):
            return response(request, backend)

        async def async_transport(self, request, backend=backend):
            await asyncio.sleep(0)
            return response(request, backend)

        monkeypatch.setattr(backend.HTTPTransport, "handle_request", sync_transport)
        monkeypatch.setattr(backend.AsyncHTTPTransport, "handle_async_request", async_transport)
    model = build_mantle_model({
        "TYPE": "mantle", "NAME": "fixture", "REGION": "us-east-1", "API_PROTOCOL": protocol,
        "ENDPOINT_URL": "https://sdk-test.invalid",
        **({"API_KEY": "explicit-key"} if explicit_key else {}),
    }, max_retries=0)
    messages = [{"role": "user", "content": "hello"}]
    if protocol == "anthropic":
        sync = lambda: model._client.messages.create(model="fixture", messages=messages, max_tokens=10)
        asynchronous = lambda: model._async_client.messages.create(model="fixture", messages=messages, max_tokens=10)
    elif protocol == "responses":
        sync = lambda: model.root_client.responses.create(model="fixture", input="hello")
        asynchronous = lambda: model.root_async_client.responses.create(model="fixture", input="hello")
    else:
        sync = lambda: model.client.create(model="fixture", messages=messages)
        asynchronous = lambda: model.async_client.create(model="fixture", messages=messages)
    sync()
    sync()
    asyncio.run(asynchronous())
    asyncio.run(asynchronous())
    prefix = "" if protocol == "anthropic" else "Bearer "
    expected = ["explicit-key"] * 4 if explicit_key else ["generated-2", "generated-3", "generated-4", "generated-5"]
    assert headers == [prefix + value for value in expected]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_generated_credentials_are_never_added_to_a_cross_origin_redirect(sdk_http_modules, asynchronous):
    for backend in sdk_http_modules:
        minted = []
        headers = RotatingTokenHeaders(lambda: minted.append(1) or "secret", "https://sdk-test.invalid")
        request = backend.Request("GET", "https://different-origin.invalid/collect")
        with pytest.raises(ValueError, match="different origin"):
            if asynchronous:
                asyncio.run(headers.async_request(request))
            else:
                headers.request(request)
        assert minted == [] and "authorization" not in request.headers
