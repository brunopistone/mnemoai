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

import os
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from mnemoai.utils.logger import logger

VALID_PROTOCOLS = ("chat_completions", "responses", "anthropic")

# REASONING_EFFORT -> thinking budget_tokens, used on the anthropic protocol
# (which takes a token budget, not an effort enum). Mirrors the mapping in
# llm_controller for direct Bedrock/Anthropic. The responses/chat_completions
# protocols take the effort string directly.
_EFFORT_TO_TOKENS = {"low": 1024, "medium": 8192, "high": 16384, "max": 32768}


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
    reasoning_effort: Optional[str] = None,
    thinking_tokens: Optional[int] = None,
    extra_params: Optional[Dict[str, Any]] = None,
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
        reasoning_effort: First-class ``REASONING_EFFORT`` knob, translated per
            protocol — sent as ``reasoning_effort`` on responses/chat_completions
            and as a ``thinking`` budget (mapped via ``_EFFORT_TO_TOKENS``) on
            anthropic. ``EXTRA_PARAMS`` overrides it if it sets the same key.
        thinking_tokens: Fallback thinking budget for the anthropic protocol when
            ``reasoning_effort`` is unset but a budget is wanted (currently only
            used if explicitly provided).
        extra_params: Generic ``EXTRA_PARAMS`` passthrough forwarded verbatim.
            For chat_completions / responses it is merged into ``model_kwargs``
            (request body) — e.g. ``reasoning_effort``, ``reasoning``,
            ``verbosity``. For the anthropic protocol it is passed as top-level
            constructor kwargs — e.g. ``thinking``.

    Returns:
        An initialized ChatOpenAI or ChatAnthropic instance.
    """
    extra = dict(extra_params or {})
    name = model_id["NAME"]
    region = model_id.get("REGION", "us-east-1")
    protocol = model_id.get("API_PROTOCOL", "chat_completions")
    if protocol not in VALID_PROTOCOLS:
        raise ValueError(
            f"Unknown Mantle API_PROTOCOL '{protocol}'. "
            f"Expected one of: {', '.join(VALID_PROTOCOLS)}"
        )

    base_url = _mantle_base_url(region, protocol, model_id.get("ENDPOINT_URL"))

    # Bearer token used as the API key for all three protocols. Prefer an
    # explicit Bedrock API key (e.g. the short-term `bedrock-api-key-...`
    # exported as BEDROCK_API_KEY) so the app works without local AWS
    # credentials; fall back to minting a short-lived token (~12h) from the
    # standard AWS SigV4 credential chain.
    token = model_id.get("API_KEY") or os.environ.get("BEDROCK_API_KEY")
    if token:
        logger.info("Using Bedrock Mantle API key for authentication")
    else:
        from aws_bedrock_token_generator import provide_token

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
        # REASONING_EFFORT -> a thinking budget (the anthropic protocol takes a
        # token budget, not an effort enum). Ensure the budget fits under
        # max_tokens. EXTRA_PARAMS may override `thinking` below.
        if reasoning_effort or thinking_tokens:
            budget = (
                _EFFORT_TO_TOKENS.get(reasoning_effort, thinking_tokens or 2048)
                if reasoning_effort
                else thinking_tokens
            )
            if kwargs["max_tokens"] <= budget:
                kwargs["max_tokens"] = budget + 1024
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            # Anthropic rejects temperature/top_p/top_k when thinking is on.
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
        # Generic passthrough (e.g. thinking={...}); applied last so it wins.
        kwargs.update(extra)
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
    # REASONING_EFFORT is a first-class ChatOpenAI arg on the OpenAI-compatible
    # protocols (effort enum forwarded as-is). EXTRA_PARAMS overrides it below.
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    # Generic passthrough: merge EXTRA_PARAMS into the request body so knobs the
    # registry doesn't model (reasoning_effort, reasoning={...}, verbosity, …)
    # reach the API. reasoning_effort is a first-class ChatOpenAI arg, so lift it
    # out of model_kwargs to avoid a "specified in both" error.
    if extra:
        if "reasoning_effort" in extra:
            kwargs["reasoning_effort"] = extra.pop("reasoning_effort")
        if extra:
            kwargs["model_kwargs"] = extra
    return ChatOpenAI(**kwargs)
