"""Orchestrator for decomposing complex tasks into worker subtasks."""

import json
import re
from typing import Any, Dict, List, Optional

from mnemoai.utils.config import config
from mnemoai.utils.logger import logger


def get_orchestrator_prompt() -> str:
    """Get the orchestrator (task-decomposition) prompt from prompts.yaml.

    Returns:
        The ORCHESTRATOR_PROMPT string.

    Raises:
        PromptError: if missing — only called when orchestration is enabled, so
        the prompt is required (no in-code fallback).
    """
    return config.require_prompt("ORCHESTRATOR_PROMPT")


def get_aggregator_prompt() -> str:
    """Get the aggregator (result-synthesis) prompt from prompts.yaml.

    Returns:
        The AGGREGATOR_PROMPT string.

    Raises:
        PromptError: if missing (required when orchestration is enabled).
    """
    return config.require_prompt("AGGREGATOR_PROMPT")


def parse_subtasks(
    content: str,
    fallback_query: str,
    valid_categories: set,
) -> List[Dict[str, Any]]:
    """Parse the orchestrator's response into a list of subtasks.

    Handles thinking tags, markdown fences, and malformed JSON gracefully.

    Args:
        content: Raw model response
        fallback_query: Original query to use if parsing fails
        valid_categories: Set of valid category names

    Returns:
        List of subtask dicts with 'description' and 'category' keys
    """
    # Handle Bedrock-style list content blocks (thinking enabled)
    if isinstance(content, list):
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        text = content or ""

    # Strip thinking tags
    text = re.sub(
        r"<think(?:ing)?>.*?</think(?:ing)?>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    try:
        # Try to find a JSON array in the response
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if json_match:
            subtasks = json.loads(json_match.group())
        else:
            subtasks = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        # Expected for models that don't emit clean JSON; we degrade gracefully
        # by treating the whole query as a single subtask, so this is debug-level.
        logger.debug(
            f"Orchestrator returned no parseable JSON ({e}); "
            "falling back to a single subtask"
        )
        return [{"description": fallback_query, "category": "full"}]

    if not isinstance(subtasks, list):
        return [{"description": fallback_query, "category": "full"}]

    # Validate and normalize
    validated = []
    for st in subtasks:
        if not isinstance(st, dict) or "description" not in st:
            continue
        category = st.get("category", "full")
        if category not in valid_categories:
            category = "full"
        validated.append(
            {
                "description": st["description"],
                "category": category,
            }
        )

    if not validated:
        return [{"description": fallback_query, "category": "full"}]

    return validated
