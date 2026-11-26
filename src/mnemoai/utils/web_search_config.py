"""Shared, SDK-free web-search settings for setup, diagnostics and registration."""

import os

PROVIDERS = {"brave": "Brave Search", "tavily": "Tavily"}
KEY_NAMES = {"brave": "BRAVE_API_KEY", "tavily": "TAVILY_API_KEY"}


def provider_for(settings) -> str:
    """Old configurations continue to use Brave; never silently change providers."""
    provider = str(settings.get("WEB_SEARCH_PROVIDER") or "brave").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError("WEB_SEARCH_PROVIDER must be brave or tavily")
    return provider


def usable_key(value) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if value.lower() in {"", "none", "your_brave_api_key", "your_tavily_api_key"}:
        return ""
    return value


def api_key_for(provider: str, settings) -> str:
    """A real YAML key wins; unset/template keys fall back to the environment."""
    name = KEY_NAMES[provider]
    return usable_key(settings.get(name)) or usable_key(os.environ.get(name))
