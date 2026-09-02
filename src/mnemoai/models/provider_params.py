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
# dest:       "main" (top-level client kwargs) or a nested-dict name. The nested
#             name is the client field the dict is handed to — "model_kwargs" for
#             params the client flattens into the request body, "extra_body" for
#             ones that must stay nested (see the mlx entry for why the
#             difference is load-bearing). build_kwargs returns both buckets.
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
        "special": {
            "REASONING", "REASONING_EFFORT", "THINKING_TOKENS", "STREAM",
            # Prompt-cache breakpoints (models/prompt_cache.py): on by default
            # where supported, so these only ever opt out or lengthen the TTL.
            # Registered so a /model provider switch doesn't prune them.
            "PROMPT_CACHE", "PROMPT_CACHE_TTL",
        },
    },
    "mantle": {
        # Delegates to mantle_factory (temperature/max_tokens/top_p/streaming +
        # reasoning), so no passthrough specs here; the factory reads these from
        # the dict and translates REASONING_EFFORT per protocol (effort enum on
        # responses, a thinking budget on anthropic).
        "params": [],
        "connection": {"REGION", "API_PROTOCOL", "ENDPOINT_URL", "API_KEY"},
        "special": {
            "TEMPERATURE", "MAX_TOKENS", "TOP_P", "STREAM", "REASONING_EFFORT",
            # Only honored on API_PROTOCOL: anthropic (see prompt_cache.policy).
            "PROMPT_CACHE", "PROMPT_CACHE_TTL",
        },
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
        "special": {
            "REASONING", "REASONING_EFFORT", "THINKING_TOKENS", "STREAM",
            "PROMPT_CACHE", "PROMPT_CACHE_TTL",
        },
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
    "mlx": {
        # Local MLX server on Apple Silicon (OpenAI-compatible surface), reached
        # over HOST/PORT like the other local runner rather than a full API_BASE.
        # TOP_K / MIN_P / REPETITION_PENALTY are real knobs on its `lm` path but
        # are not part of the OpenAI API, so their dest is "extra_body", NOT
        # "model_kwargs": the openai SDK's create() is a typed method, so a
        # non-OpenAI key flattened into the top-level payload (which is exactly
        # what model_kwargs does) raises TypeError before any request is sent.
        # extra_body is the documented passthrough for OpenAI-compatible servers.
        "params": [
            _p("TEMPERATURE", "temperature", "temperature"),
            _p("MAX_TOKENS", "max_tokens", "max_tokens"),
            _p("TOP_P", "top_p", "top_p"),
            _p("STOP", "stop", "stop"),
            _p("PRESENCE_PENALTY", "presence_penalty", "presence_penalty"),
            _p("FREQUENCY_PENALTY", "frequency_penalty", "frequency_penalty"),
            _p("TOP_K", "top_k", "top_k", "extra_body"),
            _p("MIN_P", "min_p", "min_p", "extra_body"),
            _p("REPETITION_PENALTY", "repetition_penalty", "repetition_penalty", "extra_body"),
        ],
        # API_BASE/API_KEY stay available for a non-default mount point or a
        # server behind auth; HOST/PORT is the ordinary path. ENDPOINT_URL is an
        # accepted alias for API_BASE (as on openai) — all three controllers read
        # it, so it must be declared or a /model switch would prune it away.
        "connection": {"HOST", "PORT", "API_BASE", "ENDPOINT_URL", "API_KEY"},
        # KEEP_ALIVE is MLX-specific: how long the server keeps the model
        # resident after the request (e.g. `30m`, `0` to unload immediately,
        # `-1` to pin). Handled inline by the controller, hence "special".
        "special": {"STREAM", "KEEP_ALIVE"},
    },
}

# --- VISION_MODEL_ID: mirrors vision_model_controller._initialize_*_model ----
_VISION = {
    "bedrock": {
        # Top-level, like the chat entry: vision goes through Converse, which
        # takes these as client fields it maps to `inferenceConfig`. They used to
        # be nested in `model_kwargs` for the legacy InvokeModel client, where
        # they landed in the raw request body — and a body field is per-family, so
        # `max_tokens` there is rejected outright by a GPT model on Bedrock.
        "params": [
            _p("TEMPERATURE", "temperature", "temperature"),
            _p("TOP_P", "top_p", "top_p"),
            _p("MAX_TOKENS", "max_tokens", "max_tokens"),
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
    "mlx": {
        # A vision model on the MLX server is `model_type: multimodal`, whose
        # handler forwards only temperature / top_p / max_tokens / stop (and
        # repetition params the vision controller does not read). TOP_K is
        # deliberately absent: that path never passes it to the sampler, so
        # offering it here would write config that silently does nothing.
        "params": [
            _p("TEMPERATURE", "temperature", "temperature"),
            _p("MAX_TOKENS", "max_tokens", "max_tokens"),
            _p("TOP_P", "top_p", "top_p"),
            _p("STOP", "stop", "stop"),
        ],
        "connection": {"HOST", "PORT", "API_BASE", "ENDPOINT_URL", "API_KEY"},
        "special": {"KEEP_ALIVE"},
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
    # The same MLX server serves `model_type: embeddings` entries over
    # /v1/embeddings, so HOST/PORT reaches them; KEEP_ALIVE controls residency
    # for the embedding worker exactly as it does for chat.
    "mlx": {
        "params": [],
        "connection": {"HOST", "PORT", "API_BASE", "ENDPOINT_URL", "API_KEY"},
        "special": {"DIMENSION", "KEEP_ALIVE"},
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
    """Build ``(main_kwargs, nested_kwargs)`` for a provider from a controller.

    Reads each spec's value off ``controller`` and emits it under the spec's
    client kwarg, into main or the nested bucket per ``dest``. STOP is included
    when truthy; every other param when ``is not None``.

    A provider has at most one nested bucket, so the second return value is just
    "the nested dict" — which client field it becomes (``model_kwargs`` or
    ``extra_body``) is the caller's business, and each ``_initialize_*_model``
    hands it over under the name its provider's ``dest`` names.
    """
    entry = _TABLES.get(section, {}).get(provider, {})
    main: Dict[str, Any] = {}
    nested: Dict[str, Any] = {}
    for spec in entry.get("params", []):
        val = getattr(controller, spec.attr, None)
        include = bool(val) if spec.config_key in _TRUTHY_KEYS else val is not None
        if not include:
            continue
        (main if spec.dest == "main" else nested)[spec.kwarg] = val
    return main, nested
