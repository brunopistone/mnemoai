"""mnemoai — local agentic AI assistant (LangGraph + MCP)."""


def app_version() -> str:
    """Installed distribution version, or "" when running from a checkout.

    Resolved lazily (not at import time) so importing the package never touches
    the filesystem, and returned EMPTY rather than as a placeholder: the launch
    banner prints the line only when there is a real version to print, while
    ``/doctor`` supplies its own wording for the checkout case.
    """
    try:
        from importlib.metadata import version

        return version("mnemoai-assistant")
    except Exception:  # noqa: BLE001 — PackageNotFoundError and anything odder
        return ""
