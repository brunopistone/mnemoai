"""Select a web-search adapter independently of the chat-model provider."""

from importlib import import_module

from .web_search_providers.base import WebSearchProvider

_ADAPTERS = {
    "brave": (".web_search_providers.brave", "BraveSearchProvider"),
    "tavily": (".web_search_providers.tavily", "TavilySearchProvider"),
}


class WebSearchController:
    def __init__(self, provider: str = "brave", api_key: str | None = None):
        if provider not in _ADAPTERS:
            raise ValueError("WEB_SEARCH_PROVIDER must be brave or tavily")
        module, name = _ADAPTERS[provider]
        adapter_type = getattr(import_module(module, __package__), name)
        self.provider: WebSearchProvider = adapter_type(api_key)

    async def search(self, query: str, **kwargs) -> str:
        return await self.provider.search(query, **kwargs)

    def register(self, mcp, include_web_tools: bool = False) -> None:
        # Register the concrete bound signature: the LLM sees exactly the
        # selected provider's parameters, not a **kwargs schema or a union.
        mcp.tool(name="web_search")(self.provider.search)
        if include_web_tools:
            for name in self.provider.web_tools:
                mcp.tool(name=name)(getattr(self.provider, name))
