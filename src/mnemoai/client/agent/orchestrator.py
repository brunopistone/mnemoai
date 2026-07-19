"""Orchestrator for decomposing complex tasks into worker subtasks."""

import json
import re
from typing import Any, Dict, List, Optional

from mnemoai.utils.config import config
from mnemoai.utils.logger import logger


def get_orchestrator_prompt() -> str:
    """Get the ORCHESTRATOR_PROMPT (task-decomposition) from prompts.yaml."""
    return config.require_prompt("ORCHESTRATOR_PROMPT")


def get_aggregator_prompt() -> str:
    """Get the AGGREGATOR_PROMPT (result-synthesis) from prompts.yaml."""
    return config.require_prompt("AGGREGATOR_PROMPT")


def parse_subtasks(
    content: str,
    fallback_query: str,
    valid_categories: set,
) -> List[Dict[str, Any]]:
    """Parse the orchestrator response into ``[{description, category, depends_on}]``.

    Tolerant of thinking tags, markdown fences, and malformed JSON — falls back
    to a single ``full`` subtask from ``fallback_query`` when parsing fails.

    Each subtask may carry an optional ``depends_on``: a list of 0-based indices
    of EARLIER subtasks whose results it needs. Subtasks with no (unmet)
    dependency can run concurrently; dependents wait. Absent/empty ``depends_on``
    means "no dependency". Malformed/out-of-range/forward/self references are
    dropped (treated as no dependency) so a bad plan degrades to safe behavior.
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
        return [{"description": fallback_query, "category": "full", "depends_on": []}]

    if not isinstance(subtasks, list):
        return [{"description": fallback_query, "category": "full", "depends_on": []}]

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
                # Raw deps kept for a second pass (indices refer to the ORIGINAL
                # positions; we validate after we know the final length).
                "_raw_depends_on": st.get("depends_on"),
            }
        )

    if not validated:
        return [{"description": fallback_query, "category": "full", "depends_on": []}]

    # Second pass: sanitize depends_on against the validated list. An index is
    # kept only if it's an int in range and refers to an EARLIER subtask (no
    # self/forward refs → the dependency graph is a DAG, safe to schedule).
    for i, st in enumerate(validated):
        raw = st.pop("_raw_depends_on", None)
        deps = []
        if isinstance(raw, list):
            for d in raw:
                if isinstance(d, bool):
                    continue  # bool is an int subclass — reject explicitly
                if isinstance(d, int) and 0 <= d < i:
                    deps.append(d)
        st["depends_on"] = sorted(set(deps))

    return validated
