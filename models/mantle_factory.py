"""Shared factory for AWS Bedrock Mantle models.

Mantle is reachable through standard AWS (SigV4) credentials, exchanged for a
short-lived bearer token via ``aws_bedrock_token_generator``. It serves models
under three OpenAI-/Anthropic-compatible protocols, selected per model with the
``API_PROTOCOL`` config key:

    chat_completions  (default)  base /v1            -> ChatOpenAI
    responses                    base /openai/v1     -> ChatOpenAI(use_responses_api=True)
    anthropic                    base /anthropic     -> ChatAnthropic (Messages API)

All three were verified live against bedrock-mantle.<region>.api.aws.

Both the chat LLM controller and the vision controller delegate here so the
provider behaves identically for text and image inputs.
"""

from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from utils.logger import logger

VALID_PROTOCOLS = ("chat_completions", "responses", "anthropic")


def _mantle_base_url(region: str, protocol: str, override: Optional[str]) -> str:
    """Resolve the Mantle base URL for a protocol (or use an explicit override)."""
    if override:
        return override
    root = f"https://bedrock-mantle.{region}.api.aws"
    if protocol == "responses":
        return f"{root}/openai/v1"
    if protocol == "anthropic":
        return f"{root}/anthropic"
    return f"{root}/v1"


def build_mantle_model(
    model_id: Dict[str, Any],
    *,
    callbacks: Optional[List[Any]] = None,
    streaming: Optional[bool] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
) -> BaseChatModel:
    """Build a LangChain chat model for a Bedrock Mantle endpoint.

    Args:
        model_id: The MODEL_ID / VISION_MODEL_ID config dict. Reads NAME,
            REGION, API_PROTOCOL, and optional ENDPOINT_URL.
        callbacks: Optional LangChain callbacks (chat controller passes these;
            vision passes None).
        streaming: Whether to stream (chat controller only; ignored for the
            Anthropic path which LangChain streams by default).
        temperature, max_tokens, top_p: Optional inference params; only sent
            when not None.

    Returns:
        An initialized ChatOpenAI or ChatAnthropic instance.
    """
    from aws_bedrock_token_generator import provide_token

    name = model_id["NAME"]
    region = model_id.get("REGION", "us-east-1")
    protocol = model_id.get("API_PROTOCOL", "chat_completions")
    if protocol not in VALID_PROTOCOLS:
        raise ValueError(
            f"Unknown Mantle API_PROTOCOL '{protocol}'. "
            f"Expected one of: {', '.join(VALID_PROTOCOLS)}"
        )

    base_url = _mantle_base_url(region, protocol, model_id.get("ENDPOINT_URL"))
    # Short-lived bearer token (default ~12h) minted from AWS credentials.
    token = provide_token(region=region)

    logger.info(f"Initializing Bedrock Mantle model '{name}' (protocol={protocol})")

    if protocol == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Mantle accepts the bearer token supplied as the Anthropic API key
        # (sent as the x-api-key header). Anthropic requires max_tokens.
        kwargs: Dict[str, Any] = {
            "model": name,
            "anthropic_api_url": base_url,
            "anthropic_api_key": token,
            "max_tokens": max_tokens if max_tokens is not None else 4096,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if callbacks is not None:
            kwargs["callbacks"] = callbacks
        return ChatAnthropic(**kwargs)

    # OpenAI-compatible protocols (chat_completions / responses)
    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": name,
        "base_url": base_url,
        "api_key": token,
    }
    if protocol == "responses":
        kwargs["use_responses_api"] = True
    if callbacks is not None:
        kwargs["callbacks"] = callbacks
    if streaming is not None:
        kwargs["streaming"] = streaming
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if top_p is not None:
        kwargs["top_p"] = top_p
    return ChatOpenAI(**kwargs)
