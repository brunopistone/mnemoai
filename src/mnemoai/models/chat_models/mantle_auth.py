"""Refresh generated Mantle bearer credentials at the HTTP request boundary."""

import httpx
from anyio import to_thread


class RotatingTokenAuth(httpx.Auth):
    """Explicit user API keys never enter this path; generated tokens are re-minted."""

    def __init__(self, provider, anthropic=False):
        self.provider = provider
        self.header = "X-Api-Key" if anthropic else "Authorization"
        self.prefix = "" if anthropic else "Bearer "

    def sync_auth_flow(self, request):
        request.headers[self.header] = self.prefix + self.provider()
        yield request

    async def async_auth_flow(self, request):
        token = await to_thread.run_sync(self.provider)
        request.headers[self.header] = self.prefix + token
        yield request


def install_token_refresh(model, provider, protocol):
    """Keep independent SDK transports; never mutate LangChain's shared HTTP cache."""
    auth = RotatingTokenAuth(provider, anthropic=protocol == "anthropic")
    if protocol == "anthropic":
        sync, asynchronous = model._client, model._async_client
        proxy = getattr(model, "anthropic_proxy", None)
        model._client = sync.with_options(
            http_client=httpx.Client(auth=auth, timeout=sync.timeout, proxy=proxy),
        )
        model._async_client = asynchronous.with_options(
            http_client=httpx.AsyncClient(auth=auth, timeout=asynchronous.timeout, proxy=proxy),
        )
    else:
        sync, asynchronous = model.root_client, model.root_async_client
        proxy = getattr(model, "openai_proxy", None) or None
        model.root_client = sync.with_options(
            http_client=httpx.Client(auth=auth, timeout=sync.timeout, proxy=proxy),
        )
        model.root_async_client = asynchronous.with_options(
            http_client=httpx.AsyncClient(auth=auth, timeout=asynchronous.timeout, proxy=proxy),
        )
        model.client = model.root_client.chat.completions
        model.async_client = model.root_async_client.chat.completions
    return model
