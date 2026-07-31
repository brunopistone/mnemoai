"""Client package initialization."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import only for type checkers, never at runtime
    from .client import LangGraphClient

__all__ = ["LangGraphClient"]


def __getattr__(name: str):
    """Resolve ``LangGraphClient`` lazily.

    Binding it here eagerly meant that importing ANY module in this package
    pulled the whole client — agent.py → langchain chat models →
    transformers/torch. The MCP server does exactly that: ``memory_tool`` and
    ``skill_tool`` import ``mnemoai.client.memory.*`` for the shared stores, so
    the server process loaded torch on every start, alongside the faiss that
    ``.rag`` brings — the two OpenMP runtimes whose coexistence aborts the
    interpreter (``OMP: Error #15``). Nothing imported this name from the
    package, so making it lazy costs nothing and keeps a submodule import cheap.
    """
    if name == "LangGraphClient":
        from .client import LangGraphClient

        return LangGraphClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
