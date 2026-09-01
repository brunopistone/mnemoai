"""Per-area model overrides: a different model for a specific internal call.

A turn is not one model call. Before the answer is streamed, the router
classifies the query and — when orchestration is on — a decomposer splits it into
subtasks; afterwards a summarizer may compact the history. Those calls are short,
structured and invisible, and they don't need the model that writes the answer:
classification is a one-word label, and paying a large reasoning model for it
adds latency to *every* turn. Equally, a decomposition that goes wrong wastes the
whole task, so it may deserve a BIGGER model than the chat one.

So each area can name its own model, and an area with nothing configured uses the
main ``MODEL_ID`` — which is what every existing install gets with no config edit
(the section's absence is the default, not a new key someone has to add).

Config shape — a partial ``MODEL_ID`` per area, merged OVER the main one, so only
what differs is written::

    AREA_MODELS:
      ROUTER: "qwen2.5:3b"                 # shorthand: just the model name
      ORCHESTRATOR:
        NAME: "us.anthropic.claude-opus-4-5-v1:0"
        REASONING_EFFORT: high
      SUMMARY:
        TYPE: ollama                       # a different PROVIDER is fine too
        NAME: "qwen2.5:7b"

Because the override is merged over ``MODEL_ID`` and rebuilt through the ordinary
controller, an area may change anything a model block holds — including ``TYPE``,
so a local model can serve the router while the answer comes from a hosted one.
Any key the target provider doesn't support is dropped by ``build_kwargs`` as
usual.

Pure: reads the config singleton and returns dicts. No model is built here — the
client owns that (and the caching), because building one needs a live controller.
"""

from typing import Any, Dict, Tuple

from mnemoai.utils.config import config

CONFIG_SECTION = "AREA_MODELS"

# The internal calls that may run on their own model.
#
# ROUTER        query classification (`QueryRouter.classify`) — one label per turn.
# ORCHESTRATOR  task decomposition (`agent._decompose_task`) — the subtask JSON.
# SUMMARY       conversation compaction (`client._summary_model`).
#
# Deliberately NOT an area: the aggregator. Its output is the user-visible answer,
# streamed through the same path as an ordinary reply — a different model there
# would change the voice of the answer, not just the cost of an internal step.
AREAS: Tuple[str, ...] = ("ROUTER", "ORCHESTRATOR", "SUMMARY")

# One line each, for `/doctor` and the startup log.
DESCRIPTIONS = {
    "ROUTER": "query classification",
    "ORCHESTRATOR": "task decomposition",
    "SUMMARY": "conversation compaction",
}


def overrides_for(area: str) -> Dict[str, Any]:
    """The partial ``MODEL_ID`` configured for ``area``, or ``{}`` if none.

    Tolerant by design — this is optional config that most installs never write,
    so anything unusable degrades to "no override" (the main model) rather than
    raising at startup. A bare string is accepted as shorthand for ``NAME``,
    since naming a smaller model is the whole point of the common case.
    """
    section = config.get(CONFIG_SECTION, {})
    if not isinstance(section, dict):
        return {}
    raw = section.get(_canonical(area))
    if raw is None:
        # Accept a lower/mixed-case key too: this is hand-written YAML.
        for key, value in section.items():
            if isinstance(key, str) and _canonical(key) == _canonical(area):
                raw = value
                break
    if isinstance(raw, str):
        name = raw.strip()
        return {"NAME": name} if name else {}
    if not isinstance(raw, dict):
        return {}
    # Drop empty values so a commented-out key can't blank a real MODEL_ID entry.
    return {k: v for k, v in raw.items() if v is not None and v != ""}


def configured() -> Dict[str, Dict[str, Any]]:
    """Every area with a usable override, keyed by canonical area name."""
    found = {}
    for area in AREAS:
        ov = overrides_for(area)
        if ov:
            found[area] = ov
    return found


def unknown_keys() -> Tuple[str, ...]:
    """Area names present in the config section that aren't real areas.

    A typo here is silent — the area simply keeps the main model — so `/doctor`
    reports these rather than leaving the user to wonder why nothing changed.
    """
    section = config.get(CONFIG_SECTION, {})
    if not isinstance(section, dict):
        return ()
    known = {_canonical(a) for a in AREAS}
    return tuple(
        str(k) for k in section if isinstance(k, str) and _canonical(k) not in known
    )


def label(overrides: Dict[str, Any], fallback_name: str = "", fallback_type: str = "") -> str:
    """``name (type)`` for an override, filling either half from the main model."""
    name = str(overrides.get("NAME") or fallback_name or "?")
    kind = str(overrides.get("TYPE") or fallback_type or "")
    return f"{name} ({kind})" if kind else name


def _canonical(area: str) -> str:
    """Area names are written by hand, so match case- and space-insensitively."""
    return str(area).strip().upper()
