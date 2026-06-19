"""Single source of truth for the config keys each provider consumes.

This module owns, per modality (``MODEL_ID`` / ``VISION_MODEL_ID`` /
``EMBED_MODEL_ID``) and per provider ``TYPE``:

- the **passthrough inference params** — config key -> the client kwarg it maps
  to, and whether it goes into the main kwargs or a nested ``model_kwargs`` —
  consumed by the controllers via :func:`build_kwargs`;
- the **connection/auth** keys and **special** keys the controller handles
  inline (region, host/port, endpoint, reasoning specials, …);

so :func:`supported_keys` can report exactly what a provider accepts. The
controllers build their client kwargs from this table (no inline ``if
self.x is not None`` ladders), and the configurator prunes any key a provider
doesn't accept when ``/model`` switches providers. One definition feeds both —
change it here when a provider starts/stops reading a key.

Derived strictly from ``models/llm_controller.py`` and
``models/vision_model_controller.py`` ``_initialize_*_model`` methods.
"""

from collections import namedtuple
from typing import Any, Dict, Optional, Tuple

# config_key: the YAML key under the model section
# attr:       the controller attribute holding the parsed value
# kwarg:      the client kwarg name to emit
# dest:       "main" (top-level kwargs) or "model_kwargs" (nested dict)
ParamSpec = namedtuple("ParamSpec", ["config_key", "attr", "kwarg", "dest"])

# Keys included when *truthy* (e.g. an empty STOP list is dropped); every other
# param is included when its value ``is not None``.
_TRUTHY_KEYS = {"STOP"}


def _p(config_key, attr, kwarg, dest="main"):
    return ParamSpec(config_key, attr, kwarg, dest)


# --- MODEL_ID: mirrors llm_controller._initialize_*_model -------------------
_LLM = {
    "ollama": {
        "params": [
            _p("TEMPERATURE", "temperature", "temperature"),
            _p("TOP_P", "top_p", "top_p"),
            _p("TOP_K", "top_k", "top_k"),
            _p("MAX_TOKENS", "max_tokens", "num_predict"),
            _p("STOP", "stop", "stop"),
            _p("REPETITION_PENALTY", "repetition_penalty", "repeat_penalty"),
            _p("PRESENCE_PENALTY", "presence_penalty", "presence_penalty"),
            _p("FREQUENCY_PENALTY", "frequency_penalty", "frequency_penalty"),
        ],
        "connection": {"HOST", "PORT"},
        "special": set(),
    },
    "bedrock": {
        "params": [
            _p("TEMPERATURE", "temperature", "temperature"),
            _p("TOP_P", "top_p", "top_p"),
            _p("MAX_TOKENS", "max_tokens", "max_tokens"),
            _p("STOP", "stop", "stop"),
        ],
        "connection": {"REGION", "ENDPOINT_URL"},
        "special": {"REASONING", "REASONING_EFFORT", "THINKING_TOKENS"},
    },
    "mantle": {
        # Delegates to mantle_factory (temperature/max_tokens/top_p/streaming),
        # so no passthrough specs here; the factory reads these from the dict.
        "params": [],
        "connection": {"REGION", "API_PROTOCOL", "ENDPOINT_URL", "API_KEY"},
        "special": {"TEMPERATURE", "MAX_TOKENS", "TOP_P", "STREAM"},
    },
    "openai": {
        "params": [
            _p("TEMPERATURE", "temperature", "temperature"),
            _p("MAX_TOKENS", "max_tokens", "max_tokens"),
            _p("TOP_P", "top_p", "top_p"),
            _p("PRESENCE_PENALTY", "presence_penalty", "presence_penalty"),
            _p("REASONING_EFFORT", "reasoning_effort", "reasoning_effort", "model_kwargs"),
        ],
        "connection": set(),
        "special": {"STREAM"},
    },
    "sagemaker": {
        "params": [
            _p("TEMPERATURE", "temperature", "temperature"),
            _p("MAX_TOKENS", "max_tokens", "max_tokens"),
            _p("TOP_P", "top_p", "top_p"),
            _p("TOP_K", "top_k", "top_k"),
            _p("STOP", "stop", "stop"),
        ],
        "connection": {"REGION", "INPUT_FORMAT"},
        "special": set(),
    },
    "litellm": {
        "params": [
            _p("TEMPERATURE", "temperature", "temperature"),
            _p("MAX_TOKENS", "max_tokens", "max_tokens"),
            _p("TOP_P", "top_p", "top_p"),
            _p("STOP", "stop", "stop", "model_kwargs"),
            _p("REPETITION_PENALTY", "repetition_penalty", "repeat_penalty", "model_kwargs"),
        ],
        "connection": {"API_BASE", "API_KEY"},
        "special": {"STREAM"},
    },
}

# --- VISION_MODEL_ID: mirrors vision_model_controller._initialize_*_model ----
_VISION = {
    "bedrock": {
        # Bedrock vision passes these inside model_kwargs (nested).
        "params": [
            _p("TEMPERATURE", "temperature", "temperature", "model_kwargs"),
            _p("TOP_P", "top_p", "top_p", "model_kwargs"),
            _p("MAX_TOKENS", "max_tokens", "max_tokens", "model_kwargs"),
        ],
        "connection": {"REGION", "ENDPOINT_URL"},
        "special": set(),
    },
    "ollama": {
        "params": [
            _p("TEMPERATURE", "temperature", "temperature"),
            _p("TOP_P", "top_p", "top_p"),
            _p("TOP_K", "top_k", "top_k"),
            _p("MAX_TOKENS", "max_tokens", "num_predict"),
            _p("STOP", "stop", "stop"),
        ],
        "connection": {"HOST", "PORT"},
        "special": set(),
    },
    "openai": {
        "params": [
            _p("TEMPERATURE", "temperature", "temperature"),
            _p("MAX_TOKENS", "max_tokens", "max_tokens"),
            _p("TOP_P", "top_p", "top_p"),
        ],
        "connection": set(),
        "special": set(),
    },
    "mantle": {
        "params": [],  # delegates to mantle_factory
        "connection": {"REGION", "API_PROTOCOL", "ENDPOINT_URL", "API_KEY"},
        "special": {"TEMPERATURE", "MAX_TOKENS", "TOP_P"},
    },
    "sagemaker": {
        "params": [
            _p("TEMPERATURE", "temperature", "temperature"),
            _p("MAX_TOKENS", "max_tokens", "max_tokens"),
            _p("TOP_P", "top_p", "top_p"),
            _p("TOP_K", "top_k", "top_k"),
            _p("STOP", "stop", "stop"),
        ],
        "connection": {"REGION", "INPUT_FORMAT"},
        "special": set(),
    },
    "litellm": {
        "params": [
            _p("TEMPERATURE", "temperature", "temperature"),
            _p("MAX_TOKENS", "max_tokens", "max_tokens"),
            _p("TOP_P", "top_p", "top_p"),
        ],
        "connection": {"API_BASE", "API_KEY"},
        "special": set(),
    },
}

# --- RAG.EMBED_MODEL_ID: mirrors embeddings_controller -----------------------
# Embeddings use no inference params — only connection/identity. The Ollama
# embed path uses the default host, so only REGION matters for bedrock/sagemaker.
_EMBED = {
    "ollama": {"params": [], "connection": {"HOST", "PORT"}, "special": set()},
    "bedrock": {"params": [], "connection": {"REGION"}, "special": set()},
    "openai": {"params": [], "connection": set(), "special": set()},
    "sagemaker": {"params": [], "connection": {"REGION"}, "special": set()},
    "litellm": {"params": [], "connection": {"API_BASE", "API_KEY"}, "special": set()},
}

_TABLES = {
    "MODEL_ID": _LLM,
    "VISION_MODEL_ID": _VISION,
    "EMBED_MODEL_ID": _EMBED,
}


def providers(section: str) -> Tuple[str, ...]:
    """Provider TYPEs supported for a config section."""
    return tuple(_TABLES.get(section, {}).keys())


def supported_keys(section: str, provider: str) -> Optional[set]:
    """All config keys ``provider`` accepts for ``section`` (excluding NAME/TYPE).

    Returns None for an unknown section/provider so callers can choose not to
    prune anything in that case.
    """
    entry = _TABLES.get(section, {}).get(provider)
    if entry is None:
        return None
    return {s.config_key for s in entry["params"]} | entry["connection"] | entry["special"]


def build_kwargs(section: str, provider: str, controller: Any) -> Tuple[Dict, Dict]:
    """Build (main_kwargs, model_kwargs) for a provider from a controller.

    Reads each spec's value off ``controller`` (e.g. ``controller.temperature``)
    and emits it under the spec's client kwarg name, into the main dict or the
    nested ``model_kwargs`` dict per ``dest``. STOP is included when truthy;
    every other param when ``is not None`` — matching the original controller
    init logic exactly.
    """
    entry = _TABLES.get(section, {}).get(provider, {})
    main: Dict[str, Any] = {}
    model_kwargs: Dict[str, Any] = {}
    for spec in entry.get("params", []):
        val = getattr(controller, spec.attr, None)
        include = bool(val) if spec.config_key in _TRUTHY_KEYS else val is not None
        if not include:
            continue
        (main if spec.dest == "main" else model_kwargs)[spec.kwarg] = val
    return main, model_kwargs
