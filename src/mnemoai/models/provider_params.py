"""Single source of truth for the config keys each provider consumes.

Per modality (``MODEL_ID`` / ``VISION_MODEL_ID`` / ``EMBED_MODEL_ID``) and
provider ``TYPE``, records the passthrough inference params (config key → client
kwarg + destination, consumed via :func:`build_kwargs`) plus the connection/auth
and controller-handled "special" keys. Feeds both the controllers (which build
their kwargs from this table) and the configurator (which prunes unsupported
keys on a ``/model`` provider switch). Mirrors the controllers'
``_initialize_*_model`` methods — keep in sync.
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

# A generic passthrough every provider accepts: an arbitrary dict merged into
# the model's request body (model_kwargs). Universally supported so `/model`
# pruning never strips it on a provider switch. See :func:`extra_params`.
EXTRA_PARAMS_KEY = "EXTRA_PARAMS"


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
        # Delegates to mantle_factory (temperature/max_tokens/top_p/streaming +
        # reasoning), so no passthrough specs here; the factory reads these from
        # the dict and translates REASONING_EFFORT per protocol (effort enum on
        # responses, a thinking budget on anthropic).
        "params": [],
        "connection": {"REGION", "API_PROTOCOL", "ENDPOINT_URL", "API_KEY"},
        "special": {"TEMPERATURE", "MAX_TOKENS", "TOP_P", "STREAM", "REASONING_EFFORT"},
    },
    "openai": {
        "params": [
            _p("TEMPERATURE", "temperature", "temperature"),
            _p("MAX_TOKENS", "max_tokens", "max_tokens"),
            _p("TOP_P", "top_p", "top_p"),
            _p("PRESENCE_PENALTY", "presence_penalty", "presence_penalty"),
            _p("REASONING_EFFORT", "reasoning_effort", "reasoning_effort", "model_kwargs"),
        ],
        # API_PROTOCOL selects chat_completions (default) vs the Responses API;
        # on `responses`, REASONING_EFFORT is sent as reasoning={effort, summary}
        # so the reasoning summary is returned and shown (handled inline).
        # API_BASE/ENDPOINT_URL + API_KEY point at an OpenAI-compatible server
        # (local llama-server / LM Studio / vLLM, …) instead of the OpenAI API.
        "connection": {"API_PROTOCOL", "API_BASE", "ENDPOINT_URL", "API_KEY"},
        "special": {"STREAM"},
    },
    "anthropic": {
        # Direct Anthropic API (api.anthropic.com) via langchain-anthropic's
        # ChatAnthropic. STOP maps to Anthropic's `stop_sequences`. Extended
        # thinking is handled inline (REASONING/REASONING_EFFORT/THINKING_TOKENS).
        "params": [
            _p("TEMPERATURE", "temperature", "temperature"),
            _p("MAX_TOKENS", "max_tokens", "max_tokens"),
            _p("TOP_P", "top_p", "top_p"),
            _p("TOP_K", "top_k", "top_k"),
            _p("STOP", "stop", "stop_sequences"),
        ],
        "connection": {"API_KEY", "ENDPOINT_URL"},
        "special": {"REASONING", "REASONING_EFFORT", "THINKING_TOKENS", "STREAM"},
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
            # LiteLLM's unified reasoning knob; it translates per backend
            # (effort enum for OpenAI, thinking budget for Anthropic). Passed via
            # model_kwargs since ChatLiteLLM has no top-level field for it.
            _p("REASONING_EFFORT", "reasoning_effort", "reasoning_effort", "model_kwargs"),
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
        # API_BASE/ENDPOINT_URL + API_KEY point a vision model at an
        # OpenAI-compatible server (local llama-server / LM Studio / vLLM).
        "connection": {"API_BASE", "ENDPOINT_URL", "API_KEY"},
        "special": set(),
    },
    "anthropic": {
        # Direct Anthropic API vision via ChatAnthropic (Claude is multimodal).
        "params": [
            _p("TEMPERATURE", "temperature", "temperature"),
            _p("MAX_TOKENS", "max_tokens", "max_tokens"),
            _p("TOP_P", "top_p", "top_p"),
            _p("TOP_K", "top_k", "top_k"),
        ],
        "connection": {"API_KEY", "ENDPOINT_URL"},
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
# Embeddings take no inference params, only connection/identity. DIMENSION is an
# optional vector-size override (fallback shape only); it's in `special` so it's
# both kept by /model and tunable via /params.
_EMBED = {
    "ollama": {"params": [], "connection": {"HOST", "PORT"}, "special": {"DIMENSION"}},
    "bedrock": {"params": [], "connection": {"REGION"}, "special": {"DIMENSION"}},
    # API_BASE/ENDPOINT_URL + API_KEY point embeddings at an OpenAI-compatible
    # server (local llama-server / LM Studio / vLLM).
    "openai": {
        "params": [],
        "connection": {"API_BASE", "ENDPOINT_URL", "API_KEY"},
        "special": {"DIMENSION"},
    },
    "sagemaker": {"params": [], "connection": {"REGION"}, "special": {"DIMENSION"}},
    "litellm": {
        "params": [],
        "connection": {"API_BASE", "API_KEY"},
        "special": {"DIMENSION"},
    },
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
    """All config keys ``provider`` accepts for ``section`` (excl. NAME/TYPE);
    None for an unknown section/provider (so callers can skip pruning)."""
    entry = _TABLES.get(section, {}).get(provider)
    if entry is None:
        return None
    return (
        {s.config_key for s in entry["params"]}
        | entry["connection"]
        | entry["special"]
        | {EXTRA_PARAMS_KEY}
    )


def tunable_params(section: str, provider: str) -> Optional[set]:
    """Generation knobs ``provider`` accepts for ``section`` — :func:`supported_keys`
    minus connection/auth keys, i.e. what ``/params`` may tune; None if unknown."""
    entry = _TABLES.get(section, {}).get(provider)
    if entry is None:
        return None
    return {s.config_key for s in entry["params"]} | entry["special"]


def extra_params(model_id: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of the ``EXTRA_PARAMS`` passthrough dict, or ``{}``.

    A generic escape hatch merged verbatim into the model's request body (no
    interpretation) for provider knobs the registry doesn't model (e.g.
    ``reasoning``, ``thinking``, ``verbosity``). A non-dict value → ``{}``.
    """
    raw = (model_id or {}).get("EXTRA_PARAMS")
    return dict(raw) if isinstance(raw, dict) and raw else {}


def build_kwargs(section: str, provider: str, controller: Any) -> Tuple[Dict, Dict]:
    """Build ``(main_kwargs, model_kwargs)`` for a provider from a controller.

    Reads each spec's value off ``controller`` and emits it under the spec's
    client kwarg, into main or nested ``model_kwargs`` per ``dest``. STOP is
    included when truthy; every other param when ``is not None``.
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
