"""Shared factory for AWS Bedrock Mantle models.

Mantle is reachable through standard AWS (SigV4) credentials, exchanged for a
short-lived bearer token via ``aws_bedrock_token_generator``. It serves models
under three OpenAI-/Anthropic-compatible protocols, selected per model with the
``API_PROTOCOL`` config key:

    chat_completions  (default)  base /v1            -> ChatOpenAIReasoning
    responses                    base /openai/v1     -> …(use_responses_api=True)
    anthropic                    base /anthropic     -> ChatAnthropic (Messages API)

All three were verified live against bedrock-mantle.<region>.api.aws.

Both the chat LLM controller and the vision controller delegate here so the
provider behaves identically for text and image inputs.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from mnemoai.utils.logger import logger

VALID_PROTOCOLS = ("chat_completions", "responses", "anthropic")

# REASONING_EFFORT -> thinking budget_tokens, used on OLDER Claude models on the
# anthropic protocol (which take a token budget, not an effort enum). Newer
# models (Opus 4.6+) use the `adaptive` form with `output_config.effort` — see
# _anthropic_thinking_kwargs — where this map only sizes the max_tokens headroom
# bump (the effort STRING is passed through verbatim). The responses/
# chat_completions protocols take the effort string directly. `xhigh` (added with
# Opus 4.7, between `high` and `max`) is the recommended coding/agentic effort on
# Opus 4.7+/Sonnet 5/Fable 5.
_EFFORT_TO_TOKENS = {
    "low": 1024, "medium": 8192, "high": 16384, "xhigh": 24576, "max": 32768,
}


def is_anthropic_model(name: str) -> bool:
    """True if the model id names a Claude / Anthropic model.

    Substring-based so it's robust to Bedrock inference-profile prefixes
    (``us.``/``eu.``/``apac.``/``global.``), the ``anthropic.`` provider prefix,
    and the bare ``claude-`` API id (e.g. ``anthropic.claude-opus-4-8``,
    ``us.anthropic.claude-opus-5``, ``claude-3-7-sonnet``). Deliberately NOT
    derived from ``_claude_version`` — a Claude id whose version doesn't parse
    still counts. This is the is-Anthropic gate for extended-thinking injection;
    non-Anthropic families (nova/mistral/llama/deepseek/qwen/glm/nemotron/titan/
    cohere/ai21) must never be sent Anthropic-only fields.
    """
    n = (name or "").lower()
    return "claude" in n or "anthropic." in n


def _claude_version(name: str) -> Optional[Tuple[int, int]]:
    """Best-effort (major, minor) parsed from a Claude model id, or None.

    Matches e.g. ``claude-opus-4-8`` / ``anthropic.claude-opus-4-8`` -> (4, 8),
    ``claude-3-7-sonnet`` -> (3, 7). Returns None when no version is present.
    NOTE: this is a VERSION probe only, NOT an is-Claude signal — callers must
    already have confirmed the model is Anthropic (via ``is_anthropic_model``);
    None here means "Claude of unknown version, assume current", never "not a
    Claude".
    """
    m = re.search(r"(\d+)[.\-](\d+)", name or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def _anthropic_thinking_kwargs(name: str, effort: Optional[str], budget: int) -> dict:
    """Pick the extended-thinking request form Claude's API expects per version.

    Assumes ``name`` is already known to be a Claude id (gate with
    ``is_anthropic_model`` first). Opus 4.7+:
    ``thinking={"type":"adaptive","display":"summarized"}`` +
    ``output_config.effort``. Opus 4.6: adaptive without ``display``. ≤4.5/3.x:
    ``thinking={"type":"enabled","budget_tokens": …}`` (reject adaptive). Unknown
    version (an unusual/newer Claude id) assumes adaptive. Returns kwargs to merge.
    """
    version = _claude_version(name)
    use_adaptive = version is None or version >= (4, 6)
    if not use_adaptive:
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}

    thinking: Dict[str, Any] = {"type": "adaptive"}
    if version is None or version >= (4, 7):
        thinking["display"] = "summarized"  # surfaces a reasoning summary
    out: Dict[str, Any] = {"thinking": thinking}
    if effort:
        out["output_config"] = {"effort": effort}
    return out


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
    reasoning_model: bool = False,
    extra_params: Optional[Dict[str, Any]] = None,
) -> BaseChatModel:
    """Build a LangChain chat model (ChatOpenAI or ChatAnthropic) for Mantle.

    ``model_id`` supplies NAME/REGION/API_PROTOCOL/ENDPOINT_URL. Inference params
    are sent only when not None. ``reasoning_effort`` is translated per protocol
    (``reasoning_effort`` on OpenAI protocols, a ``thinking`` budget on anthropic);
    ``reasoning_model`` opts anthropic into thinking. ``extra_params``
    (``EXTRA_PARAMS``) is forwarded verbatim and overrides the above.
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

    # Bearer token as the API key. Prefer an explicit Bedrock API key
    # (API_KEY / BEDROCK_API_KEY) so it works without local AWS credentials;
    # else mint a short-lived (~12h) token from the SigV4 chain.
    token = model_id.get("API_KEY") or os.environ.get("BEDROCK_API_KEY")
    if token:
        logger.info("Using Bedrock Mantle API key for authentication")
    else:
        from aws_bedrock_token_generator import provide_token

        token = provide_token(region=region)

    logger.info(f"Initializing Bedrock Mantle model '{name}' (protocol={protocol})")

    if protocol == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Bearer token supplied as the Anthropic API key; max_tokens required.
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
        # Enable thinking only when opted in (reasoning_effort / reasoning_model),
        # so a non-thinking Claude isn't sent a block it rejects.
        # _anthropic_thinking_kwargs picks the version-specific form. The
        # anthropic protocol is Claude-only, so a non-Claude name here is a
        # misconfig; guard defensively and never inject Anthropic fields for it
        # (EXTRA_PARAMS below stays available for a deliberate hand-injection).
        if reasoning_effort or reasoning_model:
            if not is_anthropic_model(name):
                logger.debug(
                    "Skipping Anthropic thinking fields for non-Claude model '%s' "
                    "on the Mantle anthropic protocol (misconfig); use EXTRA_PARAMS "
                    "to hand-inject provider-specific reasoning.",
                    name,
                )
            else:
                budget = (
                    _EFFORT_TO_TOKENS.get(reasoning_effort, thinking_tokens or 2048)
                    if reasoning_effort
                    else (thinking_tokens or 2048)
                )
                if kwargs["max_tokens"] <= budget:
                    kwargs["max_tokens"] = budget + 1024
                kwargs.update(
                    _anthropic_thinking_kwargs(name, reasoning_effort, budget)
                )
                # Anthropic rejects temperature/top_p/top_k when thinking is on.
                kwargs.pop("temperature", None)
                kwargs.pop("top_p", None)
        # EXTRA_PARAMS applied last so an explicit override wins.
        kwargs.update(extra)
        return ChatAnthropic(**kwargs)

    # OpenAI-compatible protocols (chat_completions / responses). The subclass
    # keeps a reasoning field the OpenAI schema has no room for (see
    # chat_openai_reasoning); on responses, which carries reasoning as typed
    # content blocks through its own converters, it is inert.
    from mnemoai.models.chat_models.chat_openai_reasoning import ChatOpenAIReasoning

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
    # Reasoning differs per protocol: responses needs reasoning={"effort":…,
    # "summary":"auto"} to get a readable summary (effort alone is hidden);
    # chat_completions takes the plain reasoning_effort enum. EXTRA_PARAMS wins.
    extra_sets_reasoning = "reasoning" in extra or "reasoning_effort" in extra
    if reasoning_effort and not extra_sets_reasoning:
        if protocol == "responses":
            kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
        else:
            kwargs["reasoning_effort"] = reasoning_effort
    # Merge EXTRA_PARAMS into the request body; lift the first-class `reasoning`/
    # `reasoning_effort` out of model_kwargs to avoid a "specified in both" error.
    if extra:
        if "reasoning" in extra:
            kwargs["reasoning"] = extra.pop("reasoning")
        if "reasoning_effort" in extra:
            kwargs["reasoning_effort"] = extra.pop("reasoning_effort")
        if extra:
            kwargs["model_kwargs"] = extra
    return ChatOpenAIReasoning(**kwargs)
