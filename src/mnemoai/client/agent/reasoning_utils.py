"""Shared helpers for handling reasoning/thinking models.

Auxiliary LLM calls (query classification, task decomposition) need a direct,
structured answer rather than chain-of-thought. Reasoning models (e.g. qwen3 via
Ollama, Claude extended thinking) place their output in a separate reasoning
field, leaving ``response.content`` empty. Disabling reasoning for these calls
keeps the visible content populated.
"""

import re
from typing import Any, Dict


def disable_reasoning(model) -> Dict[str, Any]:
    """Temporarily disable reasoning/thinking; returns saved state for restore."""
    saved: Dict[str, Any] = {}

    # Ollama wrapper / LiteLLM: a boolean `reasoning` toggle. Guard on bool so
    # this doesn't swallow ChatOpenAI's `reasoning` DICT (handled below as the
    # Responses-API object) — setting that to False would drop the summary
    # request and leave a stale flag.
    reasoning = getattr(model, "reasoning", None)
    if isinstance(reasoning, bool):
        saved["reasoning"] = reasoning
        model.reasoning = False

    # ChatBedrock (old API)
    if hasattr(model, "model_kwargs") and "thinking" in model.model_kwargs:
        saved["thinking"] = model.model_kwargs.pop("thinking")
        # Same deprecation caveat as below: only adjust temperature when the
        # model was already sending one.
        if model.model_kwargs.get("temperature") is not None:
            saved["temperature"] = model.model_kwargs["temperature"]
            model.model_kwargs["temperature"] = 0.1

    # ChatBedrockConverse (Converse API)
    additional = getattr(model, "additional_model_request_fields", None)
    if additional and "thinking" in additional:
        saved["additional_thinking"] = additional.pop("thinking")
        # Only touch temperature if the model already sends one. Newer Bedrock
        # Claude models reject `temperature` as deprecated, and these models
        # are initialized with temperature=None.
        if getattr(model, "temperature", None) is not None:
            saved["converse_temperature"] = model.temperature
            model.temperature = 0.1

    # ChatOpenAI on the Responses API (Mantle GPT-5 / Grok): with reasoning on, a
    # short auxiliary call spends its budget reasoning and returns empty content;
    # effort "none" makes it answer directly. Two mutually-exclusive shapes — a
    # `reasoning` dict or a scalar `reasoning_effort` (the API rejects both).
    if getattr(model, "use_responses_api", False):
        reasoning_obj = getattr(model, "reasoning", None)
        if isinstance(reasoning_obj, dict):
            saved["reasoning"] = dict(reasoning_obj)
            try:
                model.reasoning = {**reasoning_obj, "effort": "none"}
            except Exception:
                saved.pop("reasoning", None)
        elif hasattr(model, "reasoning_effort"):
            saved["reasoning_effort"] = getattr(model, "reasoning_effort", None)
            try:
                model.reasoning_effort = "none"
            except Exception:
                # Some providers reject "none"; leave reasoning as-is rather
                # than crash the auxiliary call.
                saved.pop("reasoning_effort", None)

    return saved


def restore_reasoning(model, saved: Dict[str, Any]) -> None:
    """Restore reasoning/thinking settings saved by disable_reasoning()."""
    if "reasoning" in saved:
        model.reasoning = saved["reasoning"]
    if "thinking" in saved:
        model.model_kwargs["thinking"] = saved["thinking"]
    if "additional_thinking" in saved:
        model.additional_model_request_fields["thinking"] = saved["additional_thinking"]
        if saved.get("converse_temperature") is not None:
            model.temperature = saved["converse_temperature"]
    if "temperature" in saved:
        model.model_kwargs["temperature"] = saved["temperature"]
    if "reasoning_effort" in saved:
        model.reasoning_effort = saved["reasoning_effort"]


def extract_visible_text(content) -> str:
    """Extract visible text from AIMessage.content, stripping reasoning.

    Handles plain strings (with optional <think>/<thinking> tags) and
    Bedrock-style content blocks (list of typed dicts).
    """
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()

    text = content or ""
    text = re.sub(
        r"<think(?:ing)?>.*?</think(?:ing)?>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return text.strip()
