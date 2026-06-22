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
    """Temporarily disable reasoning/thinking on a model.

    Args:
        model: LangChain chat model

    Returns:
        Saved state to pass to restore_reasoning()
    """
    saved: Dict[str, Any] = {}

    # Ollama wrapper / LiteLLM
    reasoning = getattr(model, "reasoning", None)
    if reasoning is not None:
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

    # ChatOpenAI on the Responses API (e.g. Mantle GPT-5 / Grok). These are
    # reasoning models: with reasoning on, a short auxiliary call (classify /
    # decompose) spends its whole token budget reasoning and returns empty
    # `content`. `reasoning_effort="none"` makes the model answer directly.
    # We only force it when the model exposes the knob AND speaks the Responses
    # API, leaving non-reasoning chat_completions models untouched.
    if getattr(model, "use_responses_api", False) and hasattr(
        model, "reasoning_effort"
    ):
        saved["reasoning_effort"] = getattr(model, "reasoning_effort", None)
        try:
            model.reasoning_effort = "none"
        except Exception:
            # Some providers reject "none"; leave reasoning as-is rather than
            # crash the auxiliary call.
            saved.pop("reasoning_effort", None)

    return saved


def restore_reasoning(model, saved: Dict[str, Any]) -> None:
    """Restore reasoning/thinking settings on a model.

    Args:
        model: LangChain chat model
        saved: State from disable_reasoning()
    """
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
    """Extract visible text from a response content, stripping reasoning.

    Handles plain strings (with optional <think>/<thinking> tags) and
    Bedrock-style content blocks (list of dicts with 'type').

    Args:
        content: AIMessage.content (str or list of blocks)

    Returns:
        Visible text with thinking removed
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
