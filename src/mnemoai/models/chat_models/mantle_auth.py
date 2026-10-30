"""Refresh generated Mantle bearer credentials at the HTTP request boundary."""

from urllib.parse import urlsplit

from anyio import to_thread


def _origin(url):
    parsed = urlsplit(str(url))
    return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


class RotatingTokenHeaders:
    """Explicit user API keys never enter this path; generated tokens are re-minted."""

    def __init__(self, provider, base_url, anthropic=False):
        self.provider = provider
        self.origin = _origin(base_url)
        self.header = "X-Api-Key" if anthropic else "Authorization"
        self.prefix = "" if anthropic else "Bearer "

    def _check_origin(self, request):
        if _origin(request.url) != self.origin:
            raise ValueError("Refusing to send generated Mantle credentials to a different origin")

    def _set_token(self, request, token):
        # Anthropic SDK 2.x uses Authorization for API keys. Refresh that header
        # when present, rather than leaving the initial token beside a fresh
        # X-Api-Key. Older SDKs keep their original X-Api-Key authentication.
        if self.header == "X-Api-Key" and "Authorization" in request.headers:
            request.headers.pop("X-Api-Key", None)
            request.headers["Authorization"] = "Bearer " + token
        else:
            request.headers[self.header] = self.prefix + token

    def request(self, request):
        self._check_origin(request)
        self._set_token(request, self.provider())

    async def async_request(self, request):
        self._check_origin(request)
        token = await to_thread.run_sync(self.provider)
        self._set_token(request, token)


def _transport(sdk, provider, *, anthropic, asynchronous, proxy):
    """Use the SDK's transport family (httpx or httpx2), not a hard-coded client."""
    headers = RotatingTokenHeaders(provider, sdk.base_url, anthropic=anthropic)
    hook = headers.async_request if asynchronous else headers.request
    return type(sdk._client)(
        timeout=sdk.timeout, proxy=proxy, follow_redirects=sdk._client.follow_redirects,
        event_hooks={"request": [hook]},
    )


def install_token_refresh(model, provider, protocol):
    """Keep independent SDK transports; never mutate LangChain's shared HTTP cache."""
    if protocol == "anthropic":
        sync, asynchronous = model._client, model._async_client
        proxy = getattr(model, "anthropic_proxy", None)
        model._client = sync.with_options(
            http_client=_transport(sync, provider, anthropic=True, asynchronous=False, proxy=proxy),
        )
        model._async_client = asynchronous.with_options(
            http_client=_transport(asynchronous, provider, anthropic=True, asynchronous=True, proxy=proxy),
        )
    else:
        sync, asynchronous = model.root_client, model.root_async_client
        proxy = getattr(model, "openai_proxy", None) or None
        model.root_client = sync.with_options(
            http_client=_transport(sync, provider, anthropic=False, asynchronous=False, proxy=proxy),
        )
        model.root_async_client = asynchronous.with_options(
            http_client=_transport(asynchronous, provider, anthropic=False, asynchronous=True, proxy=proxy),
        )
        model.client = model.root_client.chat.completions
        model.async_client = model.root_async_client.chat.completions
    return model
