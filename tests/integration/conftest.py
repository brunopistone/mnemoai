"""Fixtures for integration tests.

These tests exercise the real agent against the configured chat model and the
MCP subprocess. They are skipped automatically unless:
  1. A real utils/config.yaml exists (the runtime config, gitignored); and
  2. The configured model looks usable — for a local Ollama model the server
     must be reachable; for cloud providers (bedrock/mantle/openai/anthropic/
     sagemaker/litellm) a present config is treated as sufficient.

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


def _model_reason() -> str:
    """Return a skip reason if the configured chat model isn't usable, else "".

    Provider-aware: for a local Ollama model we probe the configured host:port
    (the server must be running). For cloud providers (bedrock, mantle, openai,
    anthropic, sagemaker, litellm) we can't cheaply verify reachability/creds
    here, so a present config is treated as sufficient — a genuinely
    unreachable backend then surfaces as a normal test failure rather than a
    misleading "Ollama not reachable" skip.
    """
    try:
        from mnemoai.utils.config import config

        model_id = config.get("MODEL_ID", {}) or {}
    except Exception:
        return "could not load config"

    model_type = str(model_id.get("TYPE", "ollama")).lower()
    if model_type != "ollama":
        return ""  # cloud provider: assume usable; failures surface as failures

    host = model_id.get("HOST", "localhost")
    port = model_id.get("PORT", 11434)
    import socket

    try:
        with socket.create_connection((host, int(port)), timeout=2):
            return ""
    except OSError:
        return f"Ollama server not reachable at {host}:{port}"


# A single module-scoped skip guard keeps the whole tier inert in CI / dev
# machines without a usable model, so the default `pytest` run stays fast/green.
_SKIP_REASON = None
if not _config_exists():
    _SKIP_REASON = "no utils/config.yaml (runtime config) present"
else:
    _SKIP_REASON = _model_reason() or None


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
