"""Fixtures for integration tests.

These tests exercise the real agent against a live Ollama server and the MCP
subprocess. They are skipped automatically unless BOTH are available:
  1. A real utils/config.yaml exists (the runtime config, gitignored).
  2. The configured Ollama host is reachable.

Run only these with:   python -m pytest -m integration
Skip them with:        python -m pytest -m "not integration"
"""

import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _config_exists() -> bool:
    # The repo-relative runtime config now lives inside the package.
    return os.path.isfile(
        os.path.join(
            REPO_ROOT, "src", "mnemoai", "utils", "config.yaml"
        )
    )


def _ollama_reachable() -> bool:
    """Return True if the Ollama HTTP API answers on the configured host:port."""
    try:
        from mnemoai.utils.config import config

        model_id = config.get("MODEL_ID", {}) or {}
        host = model_id.get("HOST", "localhost")
        port = model_id.get("PORT", 11434)
    except Exception:
        host, port = "localhost", 11434

    import socket

    try:
        with socket.create_connection((host, int(port)), timeout=2):
            return True
    except OSError:
        return False


# A single module-scoped skip guard keeps the whole tier inert in CI / dev
# machines without Ollama, so the default `pytest` run stays fast and green.
_SKIP_REASON = None
if not _config_exists():
    _SKIP_REASON = "no utils/config.yaml (runtime config) present"
elif not _ollama_reachable():
    _SKIP_REASON = "Ollama server not reachable"


@pytest.fixture(scope="session")
def live_client():
    """Start a real LangGraphClient once for the whole integration session."""
    if _SKIP_REASON:
        pytest.skip(_SKIP_REASON)

    from mnemoai.client.client import LangGraphClient

    client = LangGraphClient(verbose=False)
    client.start()
    yield client
    # Best-effort teardown; the MCP wrapper also registers an atexit shutdown.
    try:
        if getattr(client, "mcp_client", None):
            client.mcp_client.shutdown()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _isolate_conversation(live_client):
    """Clear conversation history before each test so prior answers don't
    contaminate later queries (the client is session-scoped and shared)."""
    live_client.clear_context()
    yield
