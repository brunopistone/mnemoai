"""Fixtures for integration tests.

These tests exercise the real agent against the configured chat model and the
MCP subprocess. They are skipped automatically unless:
  1. A runtime config resolves the same way the app loads it ($MNEMOAI_CONFIG →
     ~/.mnemoai/config/config.yaml → legacy → package-relative); and
  2. The configured model looks usable — for a local Ollama model the server
     must be reachable; for cloud providers (bedrock/mantle/openai/anthropic/
     sagemaker/litellm) a present config is treated as sufficient.

Run only these with:   python -m pytest -m integration
Skip them with:        python -m pytest -m "not integration"
"""

import os

import pytest


def _config_exists() -> bool:
    # Mirror the loader exactly: reuse its own resolution ($MNEMOAI_CONFIG →
    # ~/.mnemoai/config/config.yaml → legacy → package-relative), so the gate
    # can't drift from where the app actually reads its config.
    try:
        from mnemoai.utils.config import Config

        return Config._resolve_config_path() is not None
    except Exception:
        return False


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
    _SKIP_REASON = "no runtime config resolves (set MNEMOAI_CONFIG or ~/.mnemoai/config/config.yaml)"
else:
    _SKIP_REASON = _model_reason() or None


@pytest.fixture(scope="session")
def _isolated_app_home(tmp_path_factory):
    """Redirect ``$MNEMOAI_HOME`` at a throwaway dir for the whole tier.

    This tier drives a REAL client, so every side effect it has on the app home
    is a real one: session transcripts (which then show up in the user's
    ``--resume`` picker as fake "conversations" full of test prompts), episodic
    memory and playbook entries learned from test queries, RAG indexes, and the
    user profile. Without this, running the integration tier permanently
    pollutes the developer's own ``~/.mnemoai``.

    ``$MNEMOAI_CONFIG`` is pinned to the config that ALREADY resolved (the skip
    guard above resolved it against the real home) so the tier still runs
    against the user's configured provider — only writes are redirected.
    """
    from mnemoai.utils.config import Config

    real_config = Config._resolve_config_path()
    if real_config is not None:
        os.environ["MNEMOAI_CONFIG"] = str(real_config)
    real_prompts = Config._resolve_prompts_path()
    if real_prompts is not None:
        os.environ["MNEMOAI_PROMPTS"] = str(real_prompts)

    home = tmp_path_factory.mktemp("mnemoai_home")
    previous = os.environ.get("MNEMOAI_HOME")
    os.environ["MNEMOAI_HOME"] = str(home)
    yield home
    if previous is None:
        os.environ.pop("MNEMOAI_HOME", None)
    else:
        os.environ["MNEMOAI_HOME"] = previous


@pytest.fixture(scope="session")
def live_client(_isolated_app_home):
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
