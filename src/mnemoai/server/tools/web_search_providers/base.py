"""Contract implemented by web-search adapters."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable


class WebSearchProvider(ABC):
    """A search adapter returns citation-ready JSON and may offer extra web tools.

    Each search method declares its concrete parameters so MCP exposes only
    capabilities supported by the selected provider. SDK imports belong inside
    calls, not constructors. Credentials belong to the instance, not global env.
    """

    web_tools: tuple[str, ...] = ()

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    @abstractmethod
    def search(self, query: str, **kwargs) -> Awaitable[str]:
        """Return an awaitable yielding JSON results or a structured error.

        Concrete adapters implement this with an async method. The abstract
        contract itself performs no work and creates no coroutine.
        """
