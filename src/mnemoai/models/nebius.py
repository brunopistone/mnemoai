"""Nebius Token Factory connection and model construction."""

import os
from copy import deepcopy

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
DEFAULT_CHAT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_VISION_MODEL = "openbmb/MiniCPM-V-4_5"
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"


def base_url(model_config):
    """Canonical endpoint identity, including when the default is implicit."""
    value = model_config.get("API_BASE") or model_config.get("ENDPOINT_URL")
    return str(value or DEFAULT_BASE_URL).strip().rstrip("/") or DEFAULT_BASE_URL


def api_key(model_config):
    """Use this provider's credential, never an unrelated OpenAI key."""
    value = model_config.get("API_KEY") or os.environ.get("NEBIUS_API_KEY")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Set NEBIUS_API_KEY or API_KEY in the Nebius model section.")
    return value.strip()


def build_model(model_config, params, body, *, streaming=False, callbacks=None, timeout=600, max_retries=0):
    """Build the OpenAI-compatible client with reasoning and usage preserved."""
    # Defer the optional OpenAI/LangChain stack for other providers' startup.
    from mnemoai.models.chat_models.chat_nebius import ChatNebius

    extra = deepcopy(body)
    configured = model_config.get("EXTRA_PARAMS")
    if isinstance(configured, dict):
        extra.update(deepcopy(configured))
    model = ChatNebius(
        model=model_config["NAME"],
        base_url=base_url(model_config),
        api_key=api_key(model_config),
        streaming=streaming,
        disable_streaming=not streaming,
        stream_usage=True,
        use_responses_api=False,
        callbacks=callbacks,
        timeout=timeout,
        max_retries=max_retries,
        extra_body=extra or None,
        **params,
    )
    # These clients belong to Nebius; OpenAI account routing must not carry over.
    model.openai_organization = None
    for attr in ("root_client", "root_async_client"):
        client = getattr(model, attr, None)
        if client is not None:
            client.organization = None
            client.project = None
    return model
