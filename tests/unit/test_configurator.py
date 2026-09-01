"""Unit tests for the first-run configurator's template patching.

These cover the pure line-editing helpers — no TTY/interaction needed. They
verify that edits target the right key (first match within a section, or the
top level) and that the rich prompt blocks survive as valid YAML.
"""

import textwrap

import pytest

yaml = pytest.importorskip("yaml")

from mnemoai.utils.configurator import (
    _get_field,
    _get_in_section,
    _get_top_level,
    _remove_field,
    _remove_top_section,
    _section_summary,
    _set_bool,
    _set_field,
    _set_in_section,
    _set_or_add_in_section,
    _set_top_level,
    _set_top_level_or_add,
    _truthy,
)

SAMPLE = textwrap.dedent(
    """\
    MODEL_ID:
      NAME: qwen3.5:4b
      TYPE: ollama
      HOST: localhost
      PORT: 11434
      MAX_TOKENS: 8192
    VISION_MODEL_ID:
      NAME: qwen2.5vl:3b
      TYPE: ollama
    RAG:
      EMBED_MODEL_ID:
        NAME: embed-model
    ENABLE_WEB_SEARCH: true
    ENABLE_RAG: true
    BRAVE_API_KEY: your_brave_api_key
    PROFILE:
      NAME: bpistone
      USE_PROFILING: true
    """
)


def test_set_in_section_targets_first_key_in_named_section():
    out = _set_in_section(SAMPLE, "MODEL_ID", "NAME", "llama3.1:8b")
    d = yaml.safe_load(out)
    assert d["MODEL_ID"]["NAME"] == "llama3.1:8b"


def test_set_in_section_does_not_touch_same_key_in_other_sections():
    out = _set_in_section(SAMPLE, "MODEL_ID", "NAME", "llama3.1:8b")
    d = yaml.safe_load(out)
    # VISION_MODEL_ID.NAME and the nested RAG embed NAME must be untouched.
    assert d["VISION_MODEL_ID"]["NAME"] == "qwen2.5vl:3b"
    assert d["RAG"]["EMBED_MODEL_ID"]["NAME"] == "embed-model"


def test_set_in_section_replaces_host_and_port():
    out = _set_in_section(SAMPLE, "MODEL_ID", "HOST", "10.0.0.2")
    out = _set_in_section(out, "MODEL_ID", "PORT", "9999")
    d = yaml.safe_load(out)
    assert d["MODEL_ID"]["HOST"] == "10.0.0.2"
    assert d["MODEL_ID"]["PORT"] == 9999


def test_set_top_level_replaces_root_key():
    out = _set_top_level(SAMPLE, "BRAVE_API_KEY", "secret123")
    out = _set_top_level(out, "ENABLE_WEB_SEARCH", "false")
    d = yaml.safe_load(out)
    assert d["BRAVE_API_KEY"] == "secret123"
    assert d["ENABLE_WEB_SEARCH"] is False


def test_set_top_level_does_not_match_indented_key():
    # PROFILE.NAME is indented; a top-level set for "NAME" must not touch it.
    out = _set_top_level(SAMPLE, "NAME", "should-not-apply")
    d = yaml.safe_load(out)
    assert d["PROFILE"]["NAME"] == "bpistone"


def test_output_stays_valid_yaml_and_preserves_unrelated_keys():
    out = _set_in_section(SAMPLE, "PROFILE", "NAME", "alice")
    d = yaml.safe_load(out)
    assert d["PROFILE"]["NAME"] == "alice"
    assert d["MODEL_ID"]["TYPE"] == "ollama"
    assert d["ENABLE_WEB_SEARCH"] is True


def test_get_in_section_reads_first_key_value():
    assert _get_in_section(SAMPLE, "MODEL_ID", "NAME") == "qwen3.5:4b"
    assert _get_in_section(SAMPLE, "MODEL_ID", "MAX_TOKENS") == "8192"
    assert _get_in_section(SAMPLE, "PROFILE", "USE_PROFILING") == "true"


def test_get_in_section_missing_key_returns_none():
    assert _get_in_section(SAMPLE, "MODEL_ID", "REGION") is None


def test_get_top_level_reads_value_and_ignores_indented():
    assert _get_top_level(SAMPLE, "ENABLE_RAG") == "true"
    # NAME only exists indented; a top-level read must not find it.
    assert _get_top_level(SAMPLE, "NAME") is None


def test_set_bool_top_level_and_section():
    out = _set_bool(SAMPLE, "ENABLE_RAG", False)
    out = _set_bool(out, "USE_PROFILING", False, section="PROFILE")
    d = yaml.safe_load(out)
    assert d["ENABLE_RAG"] is False
    assert d["PROFILE"]["USE_PROFILING"] is False
    # Other section booleans untouched.
    assert d["ENABLE_WEB_SEARCH"] is True


def test_truthy_interprets_template_scalars():
    assert _truthy("true") is True
    assert _truthy("false") is False
    assert _truthy(None) is True  # missing -> default on
    assert _truthy("yes") is True


def test_set_or_add_inserts_missing_key_after_header():
    # MODEL_ID has no API_PROTOCOL in SAMPLE; it should be inserted.
    out = _set_or_add_in_section(SAMPLE, "MODEL_ID", "API_PROTOCOL", "anthropic")
    d = yaml.safe_load(out)
    assert d["MODEL_ID"]["API_PROTOCOL"] == "anthropic"
    # Inserted line is indented to match the section's children (2 spaces).
    assert "\n  API_PROTOCOL: anthropic" in out
    # Other sections untouched.
    assert "API_PROTOCOL" not in d["VISION_MODEL_ID"]


def test_set_or_add_replaces_existing_key():
    base = _set_or_add_in_section(SAMPLE, "MODEL_ID", "API_PROTOCOL", "responses")
    out = _set_or_add_in_section(base, "MODEL_ID", "API_PROTOCOL", "anthropic")
    d = yaml.safe_load(out)
    assert d["MODEL_ID"]["API_PROTOCOL"] == "anthropic"
    # No duplicate line was added on the second call.
    assert out.count("API_PROTOCOL:") == 1


def test_set_or_add_only_touches_named_section():
    out = _set_or_add_in_section(SAMPLE, "VISION_MODEL_ID", "API_PROTOCOL", "responses")
    d = yaml.safe_load(out)
    assert d["VISION_MODEL_ID"]["API_PROTOCOL"] == "responses"
    assert "API_PROTOCOL" not in d["MODEL_ID"]


# --- Depth-agnostic field helpers (used by /model), incl. nested embeddings ---

NESTED = textwrap.dedent(
    """\
    MODEL_ID:
      NAME: qwen3.5:4b
      TYPE: ollama
      HOST: localhost
      PORT: 11434
    RAG:
      MAX_TOKENS: 8192
      EMBED_MODEL_ID:
        NAME: embed-model
        TYPE: ollama
        HOST: localhost
        PORT: 11434
      CHUNK_TOKENS: 1024
    VISION_MODEL_ID:
      NAME: vlm
      TYPE: ollama
    """
)


def test_get_field_reads_top_level_section():
    assert _get_field(NESTED, "MODEL_ID", "NAME") == "qwen3.5:4b"
    assert _get_field(NESTED, "MODEL_ID", "TYPE") == "ollama"


def test_get_field_reads_nested_section():
    assert _get_field(NESTED, "EMBED_MODEL_ID", "NAME") == "embed-model"
    assert _get_field(NESTED, "EMBED_MODEL_ID", "PORT") == "11434"


def test_get_field_missing_returns_none():
    assert _get_field(NESTED, "EMBED_MODEL_ID", "REGION") is None


def test_set_field_replaces_in_nested_section():
    out = _set_field(NESTED, "EMBED_MODEL_ID", "NAME", "amazon.titan-embed-text-v2:0")
    d = yaml.safe_load(out)
    assert d["RAG"]["EMBED_MODEL_ID"]["NAME"] == "amazon.titan-embed-text-v2:0"
    # Sibling RAG keys and other sections untouched.
    assert d["RAG"]["MAX_TOKENS"] == 8192
    assert d["RAG"]["CHUNK_TOKENS"] == 1024
    assert d["MODEL_ID"]["NAME"] == "qwen3.5:4b"


def test_set_field_inserts_into_nested_section_at_right_indent():
    out = _set_field(NESTED, "EMBED_MODEL_ID", "REGION", "us-west-2")
    d = yaml.safe_load(out)
    assert d["RAG"]["EMBED_MODEL_ID"]["REGION"] == "us-west-2"
    # Inserted at the nested body's 4-space indent.
    assert "\n    REGION: us-west-2" in out


def test_set_field_switches_provider_type_in_nested_section():
    out = _set_field(NESTED, "EMBED_MODEL_ID", "TYPE", "bedrock")
    d = yaml.safe_load(out)
    assert d["RAG"]["EMBED_MODEL_ID"]["TYPE"] == "bedrock"
    # The top-level MODEL_ID TYPE must not change.
    assert d["MODEL_ID"]["TYPE"] == "ollama"


def test_set_field_noop_when_section_absent():
    out = _set_field(NESTED, "NONEXISTENT_SECTION", "NAME", "x")
    assert out == NESTED


def test_set_field_inserts_at_body_indent_with_nested_list_present():
    # Regression: a nested list (deeper indent) inside the section must not
    # shift where a new key is inserted. MODEL_ID's body is 2-space; STOP's
    # items are 4-space.
    text = textwrap.dedent(
        """\
        MODEL_ID:
          NAME: m
          TYPE: ollama
          STOP:
            - "<|im_end|>"
            - "<|endoftext|>"
        """
    )
    out = _set_field(text, "MODEL_ID", "MAX_TOKENS", "4096")
    d = yaml.safe_load(out)  # must stay valid YAML
    assert d["MODEL_ID"]["MAX_TOKENS"] == 4096
    assert "\n  MAX_TOKENS: 4096" in out  # 2-space, not 4
    assert d["MODEL_ID"]["STOP"] == ["<|im_end|>", "<|endoftext|>"]


def test_remove_field_drops_key_in_nested_section():
    out = _remove_field(NESTED, "EMBED_MODEL_ID", "PORT")
    d = yaml.safe_load(out)
    assert "PORT" not in d["RAG"]["EMBED_MODEL_ID"]
    assert d["RAG"]["EMBED_MODEL_ID"]["NAME"] == "embed-model"


def test_remove_field_absent_is_noop():
    assert _remove_field(NESTED, "MODEL_ID", "MAX_TOKENS") == NESTED
    assert _remove_field(NESTED, "NOPE", "NAME") == NESTED


def test_remove_field_drops_multiline_list_block():
    # A list value (e.g. STOP) and its items must be removed together, and a
    # preceding comment describing it absorbed — leaving valid YAML.
    text = textwrap.dedent(
        """\
        MODEL_ID:
          NAME: m
          TYPE: ollama
          # stop sequences for this chat template
          STOP:
            - "<|im_start|>"
            - "<|im_end|>"
          TEMPERATURE: 0.6
        """
    )
    out = _remove_field(text, "MODEL_ID", "STOP")
    d = yaml.safe_load(out)
    assert "STOP" not in d["MODEL_ID"]
    assert "<|im_start|>" not in out and "stop sequences" not in out
    # Surrounding keys survive.
    assert d["MODEL_ID"]["NAME"] == "m"
    assert d["MODEL_ID"]["TEMPERATURE"] == 0.6


def test_set_top_level_or_add_appends_when_missing():
    text = "MODEL_ID:\n  NAME: m\n"
    out = _set_top_level_or_add(text, "MAX_CONVERSATION_TOKENS", "65536")
    d = yaml.safe_load(out)
    assert d["MAX_CONVERSATION_TOKENS"] == 65536


def test_set_top_level_or_add_replaces_when_present():
    text = "MAX_CONVERSATION_TOKENS: 1000\nMODEL_ID:\n  NAME: m\n"
    out = _set_top_level_or_add(text, "MAX_CONVERSATION_TOKENS", "65536")
    d = yaml.safe_load(out)
    assert d["MAX_CONVERSATION_TOKENS"] == 65536
    assert out.count("MAX_CONVERSATION_TOKENS:") == 1


def test_prune_unsupported_params_mantle_to_ollama():
    # The switch that motivated this: a mantle section carries REGION +
    # API_PROTOCOL; switching to ollama must drop both (ollama doesn't consume
    # them) while NAME/TYPE survive.
    from mnemoai.utils.configurator import _prune_unsupported_params

    text = textwrap.dedent(
        """\
        MODEL_ID:
          NAME: xai.grok-4.3
          TYPE: ollama
          REGION: us-west-2
          API_PROTOCOL: responses
          HOST: localhost
          PORT: 11434
        """
    )
    out = _prune_unsupported_params(text, "MODEL_ID", "ollama")
    d = yaml.safe_load(out)
    assert "REGION" not in d["MODEL_ID"]
    assert "API_PROTOCOL" not in d["MODEL_ID"]
    assert d["MODEL_ID"]["HOST"] == "localhost" and d["MODEL_ID"]["PORT"] == 11434
    assert d["MODEL_ID"]["NAME"] == "xai.grok-4.3"


def test_prune_unsupported_params_strips_inference_keys_too():
    # ollama -> bedrock must drop HOST/PORT *and* ollama-only inference params
    # (FREQUENCY_PENALTY), keeping bedrock-valid keys (REGION, TEMPERATURE).
    from mnemoai.utils.configurator import _prune_unsupported_params

    text = textwrap.dedent(
        """\
        MODEL_ID:
          NAME: m
          TYPE: bedrock
          HOST: localhost
          PORT: 11434
          FREQUENCY_PENALTY: 0.0
          TEMPERATURE: 0.5
          REGION: us-east-1
        """
    )
    out = _prune_unsupported_params(text, "MODEL_ID", "bedrock")
    d = yaml.safe_load(out)
    assert "HOST" not in d["MODEL_ID"] and "PORT" not in d["MODEL_ID"]
    assert "FREQUENCY_PENALTY" not in d["MODEL_ID"]
    assert d["MODEL_ID"]["REGION"] == "us-east-1"
    assert d["MODEL_ID"]["TEMPERATURE"] == 0.5


def test_prune_unknown_provider_is_noop():
    from mnemoai.utils.configurator import _prune_unsupported_params

    text = "MODEL_ID:\n  NAME: m\n  TYPE: weird\n  FOO: bar\n"
    assert _prune_unsupported_params(text, "MODEL_ID", "weird") == text


def test_clear_inference_params_drops_tunables_keeps_identity():
    # /model clears model-specific inference params on a model change (so a
    # leftover TEMPERATURE isn't carried into a model that rejects it), while
    # keeping identity/connection and any `keep` set (MAX_TOKENS).
    from mnemoai.utils.configurator import _clear_inference_params

    text = textwrap.dedent(
        """\
        MODEL_ID:
          NAME: qwen3.5:4b
          TYPE: ollama
          HOST: localhost
          PORT: 11434
          MAX_TOKENS: 8192
          TEMPERATURE: 0.6
          TOP_P: 0.9
          FREQUENCY_PENALTY: 0.0
          STOP:
            - "<|im_end|>"
        """
    )
    out = _clear_inference_params(text, "MODEL_ID", keep={"MAX_TOKENS"})
    d = yaml.safe_load(out)["MODEL_ID"]
    # Inference params gone.
    for k in ("TEMPERATURE", "TOP_P", "FREQUENCY_PENALTY", "STOP"):
        assert k not in d, f"{k} should be cleared"
    # Identity / connection / kept key survive.
    assert d["NAME"] == "qwen3.5:4b" and d["TYPE"] == "ollama"
    assert d["HOST"] == "localhost" and d["PORT"] == 11434
    assert d["MAX_TOKENS"] == 8192


def test_clear_inference_params_noop_when_none_present():
    from mnemoai.utils.configurator import _clear_inference_params

    text = "MODEL_ID:\n  NAME: m\n  TYPE: ollama\n  HOST: localhost\n"
    assert _clear_inference_params(text, "MODEL_ID") == text


def test_provider_params_registry_shape():
    # Guard against drift: each section must advertise the provider set the
    # configurator/controllers expect, and supported_keys must report sane sets.
    from mnemoai.models.provider_params import providers, supported_keys

    assert set(providers("MODEL_ID")) == {
        "ollama", "bedrock", "mantle", "openai", "anthropic", "sagemaker",
        "litellm", "mlx",
    }
    assert set(providers("VISION_MODEL_ID")) == {
        "ollama", "bedrock", "mantle", "openai", "anthropic", "sagemaker",
        "litellm", "mlx",
    }
    # Anthropic (direct Claude API): STOP-capable, with extended-thinking
    # specials. EXTRA_PARAMS (the generic passthrough) is supported everywhere.
    assert supported_keys("MODEL_ID", "anthropic") == {
        "TEMPERATURE", "MAX_TOKENS", "TOP_P", "TOP_K", "STOP",
        "API_KEY", "ENDPOINT_URL",
        "REASONING", "REASONING_EFFORT", "THINKING_TOKENS", "STREAM",
        "PROMPT_CACHE", "PROMPT_CACHE_TTL",
        "EXTRA_PARAMS",
    }
    # Bedrock must offer STREAM like the other streaming providers, so /params can
    # tune it — the controller translates it to ChatBedrockConverse's inverted
    # `disable_streaming` (it has no passthrough spec, hence "special").
    assert supported_keys("MODEL_ID", "bedrock") == {
        "TEMPERATURE", "TOP_P", "MAX_TOKENS", "STOP",
        "REGION", "ENDPOINT_URL",
        "REASONING", "REASONING_EFFORT", "THINKING_TOKENS", "STREAM",
        # Prompt-cache breakpoints: registered so /model pruning keeps an opt-out.
        "PROMPT_CACHE", "PROMPT_CACHE_TTL",
        "EXTRA_PARAMS",
    }
    assert supported_keys("VISION_MODEL_ID", "litellm") == {
        "API_BASE", "API_KEY", "TEMPERATURE", "MAX_TOKENS", "TOP_P", "EXTRA_PARAMS"
    }
    assert set(providers("EMBED_MODEL_ID")) == {
        "ollama", "bedrock", "openai", "sagemaker", "litellm", "mlx"
    }
    # Embeddings take no inference params — only connection keys, the optional
    # DIMENSION override, and EXTRA_PARAMS.
    assert supported_keys("EMBED_MODEL_ID", "ollama") == {
        "HOST", "PORT", "DIMENSION", "EXTRA_PARAMS"
    }
    # openai embeddings can target a local OpenAI-compatible server.
    assert supported_keys("EMBED_MODEL_ID", "openai") == {
        "API_BASE", "ENDPOINT_URL", "API_KEY", "DIMENSION", "EXTRA_PARAMS"
    }
    assert supported_keys("EMBED_MODEL_ID", "litellm") == {
        "API_BASE", "API_KEY", "DIMENSION", "EXTRA_PARAMS"
    }
    # A local MLX server: HOST/PORT is the ordinary path (API_BASE/API_KEY cover a
    # proxied or auth'd one, ENDPOINT_URL being API_BASE's alias), plus
    # KEEP_ALIVE — how long it keeps the model resident after a request.
    assert supported_keys("MODEL_ID", "mlx") == {
        "TEMPERATURE", "MAX_TOKENS", "TOP_P", "STOP",
        "PRESENCE_PENALTY", "FREQUENCY_PENALTY",
        "TOP_K", "MIN_P", "REPETITION_PENALTY",
        "HOST", "PORT", "API_BASE", "ENDPOINT_URL", "API_KEY",
        "STREAM", "KEEP_ALIVE",
        "EXTRA_PARAMS",
    }
    # Vision on MLX is a `multimodal` model, whose handler forwards only these —
    # TOP_K is deliberately absent (it never reaches that sampler).
    assert supported_keys("VISION_MODEL_ID", "mlx") == {
        "TEMPERATURE", "MAX_TOKENS", "TOP_P", "STOP",
        "HOST", "PORT", "API_BASE", "ENDPOINT_URL", "API_KEY",
        "KEEP_ALIVE", "EXTRA_PARAMS",
    }
    assert supported_keys("EMBED_MODEL_ID", "mlx") == {
        "HOST", "PORT", "API_BASE", "ENDPOINT_URL", "API_KEY", "DIMENSION",
        "KEEP_ALIVE", "EXTRA_PARAMS",
    }
    # Unknown provider -> None (configurator then prunes nothing).
    assert supported_keys("MODEL_ID", "bogus") is None


# --- /params: inference-parameter tuning ------------------------------------


def test_tunable_params_excludes_connection_keys():
    # tunable_params == supported_keys minus connection/auth (HOST/PORT/REGION/…).
    from mnemoai.models.provider_params import supported_keys, tunable_params

    # ollama chat: keeps the inference knobs, drops HOST/PORT.
    t = tunable_params("MODEL_ID", "ollama")
    assert "TEMPERATURE" in t and "FREQUENCY_PENALTY" in t and "STOP" in t
    assert "HOST" not in t and "PORT" not in t
    # bedrock chat: reasoning specials are tunable; REGION (connection) is not.
    tb = tunable_params("MODEL_ID", "bedrock")
    assert {"REASONING", "REASONING_EFFORT", "THINKING_TOKENS"} <= tb
    assert "REGION" not in tb
    # Capability filtering: /params only offers params a provider supports.
    # Anthropic has no OpenAI-style penalties — they must NOT be offered/written.
    ta = tunable_params("MODEL_ID", "anthropic")
    assert "FREQUENCY_PENALTY" not in ta and "PRESENCE_PENALTY" not in ta
    assert "REASONING_EFFORT" in ta  # but reasoning effort IS supported
    # litellm gains a first-class REASONING_EFFORT (LiteLLM translates per backend).
    assert "REASONING_EFFORT" in tunable_params("MODEL_ID", "litellm")
    # mantle: connection (REGION/API_PROTOCOL) excluded, specials kept —
    # including REASONING_EFFORT (translated per protocol by the factory).
    tm = tunable_params("MODEL_ID", "mantle")
    assert "REGION" not in tm and "API_PROTOCOL" not in tm
    assert {
        "TEMPERATURE", "MAX_TOKENS", "TOP_P", "STREAM", "REASONING_EFFORT",
        "PROMPT_CACHE", "PROMPT_CACHE_TTL",
    } == tm
    # It's exactly supported minus the connection set and the generic
    # EXTRA_PARAMS passthrough (which is not a /params-tunable scalar).
    from mnemoai.models.provider_params import _TABLES  # type: ignore

    for prov in ("ollama", "bedrock", "openai", "sagemaker", "litellm", "mlx"):
        conn = _TABLES["MODEL_ID"][prov]["connection"]
        expected = supported_keys("MODEL_ID", prov) - conn - {"EXTRA_PARAMS"}
        assert tunable_params("MODEL_ID", prov) == expected
    # EXTRA_PARAMS is supported but never tunable via /params.
    assert "EXTRA_PARAMS" not in tunable_params("MODEL_ID", "openai")
    # Embeddings expose only DIMENSION as a tunable (the vector-size override);
    # connection keys (HOST/PORT/REGION/API_BASE/…) are still excluded.
    assert tunable_params("EMBED_MODEL_ID", "ollama") == {"DIMENSION"}
    assert tunable_params("EMBED_MODEL_ID", "openai") == {"DIMENSION"}
    assert "HOST" not in tunable_params("EMBED_MODEL_ID", "ollama")
    assert "API_BASE" not in tunable_params("EMBED_MODEL_ID", "openai")
    # mlx embeddings add KEEP_ALIVE (model residency) to the DIMENSION override.
    assert tunable_params("EMBED_MODEL_ID", "mlx") == {"DIMENSION", "KEEP_ALIVE"}
    # mlx chat: MIN_P is tunable here (its first consumer) and HOST/PORT are not.
    tx = tunable_params("MODEL_ID", "mlx")
    assert {"MIN_P", "TOP_K", "REPETITION_PENALTY", "KEEP_ALIVE"} <= tx
    assert "HOST" not in tx and "PORT" not in tx and "API_BASE" not in tx
    # Unknown provider -> None.
    assert tunable_params("MODEL_ID", "bogus") is None


def test_validate_param_coerces_by_kind():
    from mnemoai.utils.configurator import _validate_param as v

    assert v("TEMPERATURE", "float", "0.7") == "0.7"
    assert v("TEMPERATURE", "float", "abc") is None
    assert v("TOP_K", "int", "40") == "40"
    assert v("TOP_K", "int", "4.5") is None
    assert v("STREAM", "bool", "yes") == "true"
    assert v("STREAM", "bool", "off") == "false"
    assert v("STREAM", "bool", "maybe") is None
    assert v("REASONING_EFFORT", "enum:minimal,low,medium,high", "high") == "high"
    assert v("REASONING_EFFORT", "enum:minimal,low,medium,high", "ultra") is None
    # A list becomes a YAML flow sequence of quoted items.
    assert v("STOP", "list", "</s>, ###") == '["</s>", "###"]'
    assert v("STOP", "list", "   ") is None
    # KEEP_ALIVE: bare seconds (negative pins the model) or unit-suffixed parts,
    # exactly what the MLX server's parser accepts.
    assert v("KEEP_ALIVE", "duration", "30m") == "30m"
    assert v("KEEP_ALIVE", "duration", "1h30m") == "1h30m"
    assert v("KEEP_ALIVE", "duration", "500ms") == "500ms"
    assert v("KEEP_ALIVE", "duration", "0") == "0"
    assert v("KEEP_ALIVE", "duration", "-1") == "-1"
    assert v("KEEP_ALIVE", "duration", "30M") == "30m"  # normalized
    assert v("KEEP_ALIVE", "duration", "30 minutes") is None
    assert v("KEEP_ALIVE", "duration", "forever") is None
    assert v("KEEP_ALIVE", "duration", "-30m") is None  # sign only on bare numbers


def test_every_tunable_key_has_prompt_metadata():
    # Guard against drift: any key tunable_params can report must have an entry
    # in _PARAM_META and a slot in _PARAM_ORDER, else /params would skip/crash.
    from mnemoai.models.provider_params import providers, tunable_params
    from mnemoai.utils.configurator import _PARAM_META, _PARAM_ORDER

    keys = set()
    for section in ("MODEL_ID", "VISION_MODEL_ID", "EMBED_MODEL_ID"):
        for prov in providers(section):
            keys |= tunable_params(section, prov) or set()
    assert keys <= set(_PARAM_META)
    assert keys <= set(_PARAM_ORDER)


def test_set_field_replaces_multiline_list_with_inline_scalar():
    # The /params fix: replacing a block-list value (STOP) with an inline value
    # must drop the orphaned list items, leaving valid YAML.
    text = textwrap.dedent(
        """\
        MODEL_ID:
          NAME: m
          TYPE: ollama
          STOP:
            - "<|im_start|>"
            - "<|im_end|>"
          TEMPERATURE: 0.6
        """
    )
    out = _set_field(text, "MODEL_ID", "STOP", '["</s>"]')
    d = yaml.safe_load(out)
    assert d["MODEL_ID"]["STOP"] == ["</s>"]
    assert "<|im_start|>" not in out
    # Keys after the replaced block survive untouched.
    assert d["MODEL_ID"]["TEMPERATURE"] == 0.6
    assert d["MODEL_ID"]["NAME"] == "m"


def test_build_kwargs_matches_controller_logic():
    # build_kwargs must reproduce the controller init behavior: STOP included
    # only when truthy, others when not-None, mapped to the right client kwarg,
    # routed to main vs model_kwargs.
    from mnemoai.models.provider_params import build_kwargs

    class FakeController:
        temperature = 0.0      # not-None -> included (the bedrock truthy bug is gone)
        top_p = None           # None -> dropped
        top_k = 40
        max_tokens = 8192
        stop = []              # falsy list -> dropped (truthy rule)
        repetition_penalty = 1.1
        presence_penalty = None
        frequency_penalty = 0.0
        reasoning_effort = None

    main, model_kwargs = build_kwargs("MODEL_ID", "ollama", FakeController())
    assert main["temperature"] == 0.0          # not-None kept
    assert "top_p" not in main                 # None dropped
    assert main["top_k"] == 40
    assert main["num_predict"] == 8192         # MAX_TOKENS -> num_predict
    assert "stop" not in main                  # empty list dropped
    assert main["repeat_penalty"] == 1.1       # REPETITION_PENALTY -> repeat_penalty
    assert main["frequency_penalty"] == 0.0
    assert model_kwargs == {}                  # ollama has no nested kwargs


def test_build_kwargs_routes_to_model_kwargs():
    from mnemoai.models.provider_params import build_kwargs

    class FakeController:
        temperature = 0.5
        max_tokens = None
        top_p = None
        presence_penalty = None
        reasoning_effort = "high"

    main, model_kwargs = build_kwargs("MODEL_ID", "openai", FakeController())
    assert main["temperature"] == 0.5
    assert model_kwargs == {"reasoning_effort": "high"}  # nested, not main


# --- Optional-section removal and current-setup summary (used by /config, /model) ---


def test_remove_top_section_drops_block_and_keeps_others():
    out = _remove_top_section(NESTED, "VISION_MODEL_ID")
    d = yaml.safe_load(out)
    assert "VISION_MODEL_ID" not in d
    assert d["MODEL_ID"]["NAME"] == "qwen3.5:4b"
    assert d["RAG"]["EMBED_MODEL_ID"]["NAME"] == "embed-model"


def test_remove_top_section_absent_is_noop():
    out = _remove_top_section(NESTED, "NOPE")
    assert yaml.safe_load(out) == yaml.safe_load(NESTED)


def test_remove_top_section_drops_leading_comment():
    text = textwrap.dedent(
        """\
        MODEL_ID:
          NAME: m
          TYPE: ollama
        # vision is optional
        VISION_MODEL_ID:
          NAME: v
          TYPE: ollama
        PROFILE:
          NAME: bob
        """
    )
    out = _remove_top_section(text, "VISION_MODEL_ID")
    assert "vision is optional" not in out
    d = yaml.safe_load(out)
    assert "VISION_MODEL_ID" not in d and d["PROFILE"]["NAME"] == "bob"


def test_section_summary_formats_present_section():
    summary = _section_summary(NESTED, "MODEL_ID")
    assert summary == "ollama / qwen3.5:4b (localhost:11434)"


def test_section_summary_none_when_absent():
    out = _remove_top_section(NESTED, "VISION_MODEL_ID")
    assert _section_summary(out, "VISION_MODEL_ID") is None


def test_section_summary_includes_region_and_protocol():
    text = textwrap.dedent(
        """\
        MODEL_ID:
          NAME: anthropic.claude-haiku-4-5
          TYPE: mantle
          REGION: us-east-1
          API_PROTOCOL: anthropic
        """
    )
    summary = _section_summary(text, "MODEL_ID")
    # The TYPE is shown with its human label: mantle -> "bedrock-mantle".
    assert summary == "bedrock-mantle / anthropic.claude-haiku-4-5 (us-east-1, anthropic)"


def test_prompt_provider_type_numbered_menu_returns_canonical_value(monkeypatch):
    # The menu shows labels (bedrock-mantle) but returns the canonical config
    # value (mantle). Picking option 3 for the chat model -> "mantle".
    import builtins

    from mnemoai.utils import configurator as C

    monkeypatch.setattr(builtins, "input", lambda prompt="": "3")
    assert C._prompt_provider_type("MODEL_ID", "ollama") == "mantle"


def test_prompt_provider_type_invalid_reasks_then_accepts(monkeypatch):
    # New behavior: an invalid choice RE-ASKS instead of silently keeping the
    # current value. Feed "99" (invalid) then "2" (bedrock) -> bedrock.
    import builtins

    from mnemoai.utils import configurator as C

    answers = iter(["99", "2"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    assert C._prompt_provider_type("MODEL_ID", "ollama") == "bedrock"


def test_prompt_provider_type_eof_cancels(monkeypatch):
    # Ctrl+D / EOF aborts the flow with _Cancelled rather than defaulting.
    import builtins

    import pytest as _pytest

    from mnemoai.utils import configurator as C

    def _raise_eof(prompt=""):
        raise EOFError()

    monkeypatch.setattr(builtins, "input", _raise_eof)
    with _pytest.raises(C._Cancelled):
        C._prompt_provider_type("MODEL_ID", "ollama")


class TestValidatingPrompts:
    """The re-asking input helpers used across the configurator."""

    def test_ask_choice_reasks_until_valid(self, monkeypatch):
        import builtins

        from mnemoai.utils import configurator as C

        answers = iter(["x", "9", "2"])
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(answers))
        assert C._ask_choice("Pick", {"1", "2", "3"}, "1") == "2"

    def test_ask_choice_enter_uses_default(self, monkeypatch):
        import builtins

        from mnemoai.utils import configurator as C

        monkeypatch.setattr(builtins, "input", lambda *a, **k: "")
        assert C._ask_choice("Pick", {"1", "2"}, "2") == "2"

    def test_ask_choice_eof_cancels(self, monkeypatch):
        import builtins

        import pytest as _pytest

        from mnemoai.utils import configurator as C

        monkeypatch.setattr(
            builtins, "input", lambda *a, **k: (_ for _ in ()).throw(EOFError())
        )
        with _pytest.raises(C._Cancelled):
            C._ask_choice("Pick", {"1"}, "1")

    def test_ask_number_reasks_on_non_numeric(self, monkeypatch):
        import builtins

        from mnemoai.utils import configurator as C

        answers = iter(["abc", "1.5x", "4096"])
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(answers))
        assert C._ask_number("Tokens", default=None, kind="int") == "4096"

    def test_ask_number_float_accepts_float(self, monkeypatch):
        import builtins

        from mnemoai.utils import configurator as C

        answers = iter(["nope", "0.7"])
        monkeypatch.setattr(builtins, "input", lambda *a, **k: next(answers))
        assert C._ask_number("Temp", default=None, kind="float") == "0.7"

    def test_ask_number_none_returns_none(self, monkeypatch):
        import builtins

        from mnemoai.utils import configurator as C

        monkeypatch.setattr(builtins, "input", lambda *a, **k: "none")
        assert C._ask_number("Tokens", default="none", allow_none=True) is None

    def test_ask_number_enter_uses_default(self, monkeypatch):
        import builtins

        from mnemoai.utils import configurator as C

        monkeypatch.setattr(builtins, "input", lambda *a, **k: "")
        assert C._ask_number("Ctx", default="65536", kind="int") == "65536"


def test_prompt_provider_type_embeddings_has_no_mantle(monkeypatch):
    # Embeddings provider set excludes mantle; option count reflects that.
    import builtins

    from mnemoai.models.provider_params import providers
    from mnemoai.utils import configurator as C

    opts = list(providers("EMBED_MODEL_ID"))
    assert "mantle" not in opts
    # Enter keeps the current (default) selection -> ollama.
    monkeypatch.setattr(builtins, "input", lambda prompt="": "")
    assert C._prompt_provider_type("EMBED_MODEL_ID", "ollama") == "ollama"


# --- /config can create OpenAI / SageMaker / LiteLLM (base-template transform) ---


def _run_build(provider, default_model, answers):
    """Drive _build_config against the base template with scripted answers.

    Note the two ``AREA_MODELS`` prompts that follow the ROUTING/ORCH toggles when
    those are enabled ("use the same model as Chat?"); answering yes writes
    nothing, which is why these sequences still produce no AREA_MODELS section.
    """
    import builtins

    from mnemoai.utils import configurator as C

    text = (C._templates_dir() / "config.yaml.example").read_text()
    it = iter(answers)
    builtins.input = lambda *a, **k: next(it)
    return yaml.safe_load(
        C._build_config(provider, default_model, text, "config.yaml.example")
    )


def test_config_openai_transforms_base_template():
    d = _run_build(
        "openai", "gpt-5-mini",
        # chat name, [blank base URL, blank key], MAX_TOKENS none, context,
        # vision? y, "same as chat?" y (copies chat), embeddings? n, profile, brave,
        # then toggles: RAG, EPISODIC, PLAYBOOK, MEMORY, AUTO_EXTRACT, SKILLS,
        # WEB_CRAWL, ROUTING, ORCH, [router + orchestrator "same as chat?"],
        # PROFILING, BASH_CONFIRM, WRITE_CONFIRM, MEM_CONFIRM, GIT_CONFIRM.
        ["gpt-5-mini", "", "", "none", "65536", "y", "y", "n", "alice", "",
         "y", "y", "y", "y", "y", "y", "y", "y", "y",
         "y", "y",
         "y", "y", "y", "y", "y"],
    )
    m = d["MODEL_ID"]
    assert m["TYPE"] == "openai" and m["NAME"] == "gpt-5-mini"
    # Both areas answered "same as chat" -> nothing written, so the section stays
    # the commented example the template ships.
    assert "AREA_MODELS" not in d
    # Ollama-only keys pruned; OpenAI-valid TEMPERATURE/PRESENCE_PENALTY kept.
    for bad in ("HOST", "PORT", "TOP_K", "FREQUENCY_PENALTY"):
        assert bad not in m
    # Vision copied from chat via the "same as chat?" shortcut.
    assert d["VISION_MODEL_ID"]["TYPE"] == "openai"
    assert d["VISION_MODEL_ID"]["NAME"] == "gpt-5-mini"


def test_config_sagemaker_sets_region_and_input_format():
    d = _run_build(
        "sagemaker", "my-endpoint",
        # chat name, region, input_format, MAX_TOKENS none, ctx, vision? n,
        # embeddings? n, profile, brave, then 14 toggles + the 2 area prompts
        # (see the openai test).
        ["my-endpoint", "eu-west-1", "huggingface", "none", "65536", "n", "n",
         "bob", "", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y",
         "y", "y", "y", "y"],
    )
    m = d["MODEL_ID"]
    assert m["TYPE"] == "sagemaker"
    assert m["REGION"] == "eu-west-1" and m["INPUT_FORMAT"] == "huggingface"
    assert "HOST" not in m and "PORT" not in m


def test_config_litellm_sets_api_base_and_key():
    d = _run_build(
        "litellm", "openai/gpt-4o",
        # chat name, api_base, api_key, MAX_TOKENS none, ctx, vision? n,
        # embeddings? n, profile, brave, then 14 toggles + the 2 area prompts
        # (see the openai test).
        ["openai/gpt-4o", "http://localhost:8000/v1", "sk-xyz", "none", "65536",
         "n", "n", "carol", "",
         "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y",
         "y", "y"],
    )
    m = d["MODEL_ID"]
    assert m["TYPE"] == "litellm"
    assert m["API_BASE"] == "http://localhost:8000/v1" and m["API_KEY"] == "sk-xyz"
    assert "HOST" not in m


def test_config_mlx_sets_host_port_and_mirrors_vision():
    d = _run_build(
        "mlx", "mlx-community/Qwen3-4B-4bit",
        # chat name, host, port, [blank base URL, blank key], MAX_TOKENS none, ctx,
        # vision? y, "same as chat?" y (copies chat), embeddings? n, profile,
        # brave, then 14 toggles + the 2 area prompts (see the openai test).
        ["qwen-agentcoder", "127.0.0.1", "8000", "", "", "none", "65536",
         "y", "y", "n", "erin", "",
         "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y",
         "y", "y"],
    )
    m = d["MODEL_ID"]
    assert m["TYPE"] == "mlx" and m["NAME"] == "qwen-agentcoder"
    assert m["HOST"] == "127.0.0.1" and m["PORT"] == 8000
    # Blank optional answers are not written (HOST/PORT stays the live path).
    assert "API_BASE" not in m and "API_KEY" not in m
    # No AWS/protocol keys leaked in from another provider's prompts.
    for bad in ("REGION", "INPUT_FORMAT", "API_PROTOCOL"):
        assert bad not in m
    # An MLX server can host a multimodal model, so "same as chat" applies here
    # too and carries the connection over.
    v = d["VISION_MODEL_ID"]
    assert v["TYPE"] == "mlx" and v["NAME"] == "qwen-agentcoder"
    assert v["HOST"] == "127.0.0.1" and v["PORT"] == 8000


def test_config_anthropic_transforms_base_template():
    # answers: chat name, API_KEY, base URL (blank), MAX_TOKENS, ctx, configure
    # vision? (y), "same as chat?" (y → copies chat), embeddings? (n), profile,
    # brave (blank), then 14 toggles: RAG, EPISODIC, PLAYBOOK, MEMORY,
    # AUTO_EXTRACT, SKILLS, WEB_CRAWL, ROUTING, ORCH, [the 2 area prompts],
    # PROFILING, BASH_CONFIRM, WRITE_CONFIRM, MEM_CONFIRM, GIT_CONFIRM.
    d = _run_build(
        "anthropic", "claude-opus-4-8",
        ["claude-opus-4-8", "fake-anthropic-key", "", "none", "65536",
         "y", "y", "n", "dave", "",
         "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y",
         "y", "y"],
    )
    m = d["MODEL_ID"]
    assert m["TYPE"] == "anthropic" and m["NAME"] == "claude-opus-4-8"
    assert m["API_KEY"] == "fake-anthropic-key"
    # Ollama-only keys pruned; no AWS region leaked in.
    for bad in ("HOST", "PORT", "FREQUENCY_PENALTY", "REGION"):
        assert bad not in m
    # Claude is multimodal -> vision section switched to anthropic too.
    assert d["VISION_MODEL_ID"]["TYPE"] == "anthropic"
    # Both confirmation toggles are prompted and written ('y' -> true).
    assert d["REQUIRE_BASH_CONFIRMATION"] is True
    assert d["REQUIRE_WRITE_CONFIRMATION"] is True
    # The persistent-memory toggle is prompted and written.
    assert d["ENABLE_MEMORY"] is True
    # The newer toggles are now in the /config flow too (all answered 'y').
    assert d["ENABLE_MEMORY_AUTO_EXTRACTION"] is True  # asked because memory=y
    assert d["ENABLE_SKILLS"] is True
    assert d["REQUIRE_MEMORY_CONFIRMATION"] is True
    assert d["REQUIRE_GIT_CONFIRMATION"] is True


def test_config_skips_auto_extract_when_memory_off():
    # When persistent memory is declined, the auto-extraction sub-prompt is not
    # asked (one fewer answer). Memory=n here; sequence stays aligned.
    d = _run_build(
        "anthropic", "claude-opus-4-8",
        ["claude-opus-4-8", "fake-key", "", "none", "65536",
         "n", "n", "dave", "",   # vision? n, embeddings? n
         # RAG, EPISODIC, PLAYBOOK, MEMORY(n → no auto-extract prompt), SKILLS,
         # WEB_CRAWL, ROUTING, ORCH, [the 2 area prompts], PROFILING, BASH,
         # WRITE, MEM_CONFIRM, GIT_CONFIRM.
         "y", "y", "y", "n", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y",
         "y"],
    )
    assert d["ENABLE_MEMORY"] is False
    # Not prompted → key stays at its template value (not forced by this run).


def test_config_providers_menu_covers_every_llm_provider():
    # The first-run menu must offer every provider the LLM registry supports —
    # a provider only reachable by hand-editing the config is a half-wired one.
    from mnemoai.models.provider_params import providers
    from mnemoai.utils.configurator import _PROVIDER_LABELS, _PROVIDERS

    types = {v[0] for v in _PROVIDERS.values()}
    assert types == {
        "ollama", "bedrock", "mantle", "openai", "anthropic", "sagemaker",
        "litellm", "mlx",
    }
    assert types == set(providers("MODEL_ID"))
    # Every menu entry also needs a /model label (else it shows its raw key).
    assert types <= set(_PROVIDER_LABELS)


# --- Shared connection-prompt helper: /config and /model ask the same params ---


def test_prompt_provider_connection_sagemaker_asks_region_and_format(monkeypatch):
    from mnemoai.utils import configurator as C

    answers = iter(["eu-west-1", "huggingface"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    text = "MODEL_ID:\n  NAME: ep\n  TYPE: sagemaker\n"
    out, conn = C._prompt_provider_connection(text, "MODEL_ID", "sagemaker")
    d = yaml.safe_load(out)
    assert d["MODEL_ID"]["REGION"] == "eu-west-1"
    assert d["MODEL_ID"]["INPUT_FORMAT"] == "huggingface"
    assert conn["REGION"] == "eu-west-1"


def test_prompt_provider_connection_litellm_asks_base_and_key(monkeypatch):
    from mnemoai.utils import configurator as C

    answers = iter(["http://localhost:8000/v1", "sk-abc"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    text = "MODEL_ID:\n  NAME: openai/gpt-4o\n  TYPE: litellm\n"
    out, _ = C._prompt_provider_connection(text, "MODEL_ID", "litellm")
    d = yaml.safe_load(out)
    assert d["MODEL_ID"]["API_BASE"] == "http://localhost:8000/v1"
    assert d["MODEL_ID"]["API_KEY"] == "sk-abc"


def test_prompt_provider_connection_mlx_asks_host_port_with_its_own_defaults(
    monkeypatch,
):
    # The HOST/PORT prompts are shared with the other local runner, so they must
    # be provider-aware: Enter-through has to land on the MLX server's own
    # 127.0.0.1:8000, not the other runner's localhost:11434.
    from mnemoai.utils import configurator as C

    prompts = []

    def _fake_input(prompt=""):
        prompts.append(prompt)
        return ""  # Enter through every step -> the offered defaults are kept

    monkeypatch.setattr("builtins.input", _fake_input)
    text = "MODEL_ID:\n  NAME: qwen-agentcoder\n  TYPE: mlx\n"
    out, conn = C._prompt_provider_connection(text, "MODEL_ID", "mlx")
    d = yaml.safe_load(out)["MODEL_ID"]
    assert d["HOST"] == "127.0.0.1" and d["PORT"] == 8000
    assert conn == {"HOST": "127.0.0.1", "PORT": "8000"}
    # Optional keys: blank -> not written, so HOST/PORT stays the live path.
    assert "API_BASE" not in d and "API_KEY" not in d
    assert "11434" not in "".join(prompts)  # no other runner's default offered
    assert any("MLX server host" in p for p in prompts)


def test_prompt_provider_connection_mlx_base_url_overrides(monkeypatch):
    from mnemoai.utils import configurator as C

    answers = iter(["127.0.0.1", "8000", "https://mac.internal/mlx/v1", "tok"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    text = "MODEL_ID:\n  NAME: qwen-agentcoder\n  TYPE: mlx\n"
    out, _ = C._prompt_provider_connection(text, "MODEL_ID", "mlx")
    d = yaml.safe_load(out)["MODEL_ID"]
    assert d["API_BASE"] == "https://mac.internal/mlx/v1"
    assert d["API_KEY"] == "tok"


def test_prompt_provider_connection_ollama_defaults_unchanged(monkeypatch):
    # The provider-aware prompts must not have moved the other runner's defaults.
    from mnemoai.utils import configurator as C

    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    text = "MODEL_ID:\n  NAME: qwen3.5:4b\n  TYPE: ollama\n"
    out, _ = C._prompt_provider_connection(text, "MODEL_ID", "ollama")
    d = yaml.safe_load(out)["MODEL_ID"]
    assert d["HOST"] == "localhost" and d["PORT"] == 11434


def test_every_host_port_provider_has_its_own_prompt_wording():
    # A provider that takes HOST/PORT must have its own _HOST_PORT_PROMPTS row:
    # borrowing another's would offer that runner's name and port for a server
    # that is neither, and a wrong default is accepted by pressing Enter.
    from mnemoai.models.provider_params import providers, supported_keys
    from mnemoai.utils.configurator import _HOST_PORT_PROMPTS

    for section in ("MODEL_ID", "VISION_MODEL_ID", "EMBED_MODEL_ID"):
        for provider in providers(section):
            keys = supported_keys(section, provider) or set()
            if keys & {"HOST", "PORT"}:
                assert provider in _HOST_PORT_PROMPTS, (
                    f"{section}/{provider} takes HOST/PORT but has no prompt row"
                )


def test_host_port_fallback_borrows_no_other_providers_defaults(monkeypatch):
    # The safety net behind the test above: with the row missing, the wording and
    # the default must be neutral, and Enter must leave the port unwritten rather
    # than silently configuring some other runner's.
    from mnemoai.utils import configurator as C

    monkeypatch.setattr(C, "_HOST_PORT_PROMPTS", {})
    prompts = []

    def _fake_input(prompt=""):
        prompts.append(prompt)
        return ""

    monkeypatch.setattr("builtins.input", _fake_input)
    text = "MODEL_ID:\n  NAME: m\n  TYPE: mlx\n"
    out, _ = C._prompt_provider_connection(text, "MODEL_ID", "mlx")
    d = yaml.safe_load(out)["MODEL_ID"]
    joined = "".join(prompts)
    assert "Ollama" not in joined and "11434" not in joined
    assert "PORT" not in d  # no default to offer -> nothing written


def test_prompt_provider_connection_openai_optional_base_url(monkeypatch):
    # OpenAI prompts for an OPTIONAL base URL + key (to target a local
    # OpenAI-compatible server); blank answers leave the config untouched
    # (defaults to the OpenAI API via OPENAI_API_KEY).
    from mnemoai.utils import configurator as C

    answers = iter(["", ""])  # blank base URL, blank key
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    text = "MODEL_ID:\n  NAME: gpt-5-mini\n  TYPE: openai\n"
    out, conn = C._prompt_provider_connection(text, "MODEL_ID", "openai")
    assert conn == {}
    d = yaml.safe_load(out)["MODEL_ID"]
    assert "HOST" not in d
    assert "API_BASE" not in d and "API_KEY" not in d  # blank -> not written


def test_prompt_provider_connection_openai_sets_base_url(monkeypatch):
    # A non-blank base URL points OpenAI at a local server.
    from mnemoai.utils import configurator as C

    answers = iter(["http://localhost:8080/v1", ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    text = "MODEL_ID:\n  NAME: qwen\n  TYPE: openai\n"
    out, _ = C._prompt_provider_connection(text, "MODEL_ID", "openai")
    assert yaml.safe_load(out)["MODEL_ID"]["API_BASE"] == "http://localhost:8080/v1"


def test_prompt_provider_connection_embeddings_skips_input_format(monkeypatch):
    # INPUT_FORMAT is a SageMaker *chat* key; embeddings sagemaker only needs REGION.
    from mnemoai.utils import configurator as C

    # REGION, then the optional embeddings DIMENSION prompt (left blank).
    answers = iter(["us-west-2", ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    text = "RAG:\n  EMBED_MODEL_ID:\n    NAME: e\n    TYPE: sagemaker\n"
    out, _ = C._prompt_provider_connection(text, "EMBED_MODEL_ID", "sagemaker")
    d = yaml.safe_load(out)
    assert d["RAG"]["EMBED_MODEL_ID"]["REGION"] == "us-west-2"
    assert "INPUT_FORMAT" not in d["RAG"]["EMBED_MODEL_ID"]
    assert "DIMENSION" not in d["RAG"]["EMBED_MODEL_ID"]  # blank -> not written


def test_prompt_one_param_keeps_current_on_empty(monkeypatch):
    # A set param, Enter (empty) -> unchanged.
    from mnemoai.utils import configurator as C

    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    text = "MODEL_ID:\n  TYPE: openai\n  TEMPERATURE: 0.7\n"
    out = C._prompt_one_param(text, "MODEL_ID", "TEMPERATURE", "float", "temp")
    assert C._get_field(out, "MODEL_ID", "TEMPERATURE") == "0.7"


def test_prompt_one_param_default_stays_absent(monkeypatch):
    # Provider-default param (absent), Enter -> still not written.
    from mnemoai.utils import configurator as C

    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    text = "MODEL_ID:\n  TYPE: openai\n"
    out = C._prompt_one_param(text, "MODEL_ID", "TEMPERATURE", "float", "temp")
    assert C._get_field(out, "MODEL_ID", "TEMPERATURE") is None


def test_prompt_one_param_sets_from_default(monkeypatch):
    from mnemoai.utils import configurator as C

    monkeypatch.setattr("builtins.input", lambda *a, **k: "0.3")
    text = "MODEL_ID:\n  TYPE: openai\n"
    out = C._prompt_one_param(text, "MODEL_ID", "TEMPERATURE", "float", "temp")
    assert C._get_field(out, "MODEL_ID", "TEMPERATURE") == "0.3"


def test_prompt_one_param_none_clears(monkeypatch):
    from mnemoai.utils import configurator as C

    monkeypatch.setattr("builtins.input", lambda *a, **k: "none")
    text = "MODEL_ID:\n  TYPE: openai\n  TEMPERATURE: 0.7\n"
    out = C._prompt_one_param(text, "MODEL_ID", "TEMPERATURE", "float", "temp")
    assert C._get_field(out, "MODEL_ID", "TEMPERATURE") is None


def test_prompt_one_param_reasks_on_invalid(monkeypatch):
    from mnemoai.utils import configurator as C

    answers = iter(["abc", "0.9"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    text = "MODEL_ID:\n  TYPE: openai\n"
    out = C._prompt_one_param(text, "MODEL_ID", "TEMPERATURE", "float", "temp")
    assert C._get_field(out, "MODEL_ID", "TEMPERATURE") == "0.9"


class TestRunStepsBackNavigation:
    """The wizard step-runner (_run_steps) with Back support. Steps are
    fn(text, allow_back)->text; a _GoBack rewinds to the previous step and
    restores the text as it was before that step ran."""

    def test_runs_all_steps_in_order_first_has_no_back(self):
        from mnemoai.utils import configurator as C

        seen = []

        def step(tag):
            def _s(text, allow_back):
                seen.append((tag, allow_back))
                return text + tag
            return _s

        out = C._run_steps("", [step("a"), step("b"), step("c")])
        assert out == "abc"
        # First step gets allow_back=False; the rest True.
        assert seen == [("a", False), ("b", True), ("c", True)]

    def test_back_rewinds_and_reruns_previous_step(self):
        from mnemoai.utils import configurator as C

        calls = []

        def step_a(text, allow_back):
            calls.append("a")
            return text + "A"

        # step_b goes Back the first time it runs, then succeeds.
        state = {"backed": False}

        def step_b(text, allow_back):
            calls.append("b")
            if not state["backed"]:
                state["backed"] = True
                raise C._GoBack()
            return text + "B"

        out = C._run_steps("", [step_a, step_b])
        # a ran, b raised Back, a re-ran, b succeeded.
        assert calls == ["a", "b", "a", "b"]
        assert out == "AB"  # a's edit applied once (pre-run text restored on rewind)

    def test_back_restores_pre_step_text_no_stacking(self):
        from mnemoai.utils import configurator as C

        # step_a appends " x" to a key each run; if Back didn't restore the
        # pre-run text, re-running would stack " x x".
        def step_a(text, allow_back):
            return text + " x"

        state = {"n": 0}

        def step_b(text, allow_back):
            state["n"] += 1
            if state["n"] == 1:
                raise C._GoBack()
            return text + " y"

        out = C._run_steps("base", [step_a, step_b])
        assert out == "base x y"  # not "base x x y"

    def test_back_from_first_step_stays_on_first(self):
        from mnemoai.utils import configurator as C

        state = {"n": 0}

        def only(text, allow_back):
            state["n"] += 1
            if state["n"] == 1:
                raise C._GoBack()  # Back on the first step is a no-op rerun
            return text + "Z"

        out = C._run_steps("", [only])
        assert out == "Z"
        assert state["n"] == 2

    def test_cancelled_propagates(self):
        from mnemoai.utils import configurator as C

        def boom(text, allow_back):
            raise C._Cancelled()

        with pytest.raises(C._Cancelled):
            C._run_steps("", [boom])


class TestCopyChatToVision:
    """The /model vision shortcut: point VISION_MODEL_ID at the same model as the
    chat LLM (same provider, name, connection keys, max_tokens)."""

    MANTLE = textwrap.dedent(
        """\
        MODEL_ID:
          NAME: anthropic.claude-opus-4-8
          TYPE: mantle
          REGION: us-east-1
          API_PROTOCOL: anthropic
          MAX_TOKENS: 128000
          REASONING_EFFORT: max
        VISION_MODEL_ID:
          NAME: qwen2.5vl:3b
          TYPE: ollama
          HOST: localhost
          PORT: 11434
          TEMPERATURE: 0.3
        """
    )

    def test_copies_provider_name_and_connection(self):
        from mnemoai.utils.configurator import _copy_chat_to_vision

        out = _copy_chat_to_vision(self.MANTLE)
        d = yaml.safe_load(out)
        v = d["VISION_MODEL_ID"]
        assert v["TYPE"] == "mantle"
        assert v["NAME"] == "anthropic.claude-opus-4-8"
        assert v["REGION"] == "us-east-1"
        assert v["API_PROTOCOL"] == "anthropic"
        assert str(v["MAX_TOKENS"]) == "128000"

    def test_prunes_old_provider_keys_on_switch(self):
        # Vision was ollama (HOST/PORT); after copying a mantle chat model those
        # ollama-only keys must be gone, and the ollama TEMPERATURE cleared.
        from mnemoai.utils.configurator import _copy_chat_to_vision

        out = _copy_chat_to_vision(self.MANTLE)
        d = yaml.safe_load(out)
        v = d["VISION_MODEL_ID"]
        assert "HOST" not in v
        assert "PORT" not in v

    def test_does_not_copy_chat_only_inference_params(self):
        # REASONING_EFFORT is a chat knob; vision shouldn't inherit chat's
        # generation params wholesale (only NAME/TYPE/connection + max_tokens).
        from mnemoai.utils.configurator import _copy_chat_to_vision

        out = _copy_chat_to_vision(self.MANTLE)
        d = yaml.safe_load(out)
        assert "REASONING_EFFORT" not in d["VISION_MODEL_ID"]

    def test_chat_section_untouched(self):
        from mnemoai.utils.configurator import _copy_chat_to_vision

        out = _copy_chat_to_vision(self.MANTLE)
        d = yaml.safe_load(out)
        assert d["MODEL_ID"]["NAME"] == "anthropic.claude-opus-4-8"
        assert d["MODEL_ID"]["REASONING_EFFORT"] == "max"

    def test_ollama_to_ollama_copies_host_port(self):
        base = textwrap.dedent(
            """\
            MODEL_ID:
              NAME: llama3.1:8b
              TYPE: ollama
              HOST: localhost
              PORT: 11434
              MAX_TOKENS: 8192
            VISION_MODEL_ID:
              NAME: old-vision
              TYPE: ollama
            """
        )
        from mnemoai.utils.configurator import _copy_chat_to_vision

        out = _copy_chat_to_vision(base)
        d = yaml.safe_load(out)
        v = d["VISION_MODEL_ID"]
        assert v["NAME"] == "llama3.1:8b"
        assert v["HOST"] == "localhost"
        assert str(v["PORT"]) == "11434"


class TestEmbeddingsSetupWithRagOff:
    """/model must let you set embeddings even when RAG is off (its block may be
    absent), then offer to enable the features that consume embeddings."""

    RAG_OFF = textwrap.dedent(
        """\
        MODEL_ID:
          NAME: anthropic.claude-opus-4-8
          TYPE: mantle
        ENABLE_RAG: false
        RAG:
          MAX_TOKENS: 8192
          VECTOR_STORE:
            TYPE: chromadb
        ENABLE_EPISODIC_MEMORY: false
        """
    )

    def test_ensure_embed_section_inserts_block_under_rag(self):
        from mnemoai.utils.configurator import _ensure_embed_section, _get_field

        out = _ensure_embed_section(self.RAG_OFF)
        d = yaml.safe_load(out)
        assert "EMBED_MODEL_ID" in d["RAG"]
        # And it's addressable by the section helpers (nested lookup).
        assert _get_field(out, "EMBED_MODEL_ID", "TYPE") == "ollama"

    def test_ensure_embed_section_noop_when_present(self):
        from mnemoai.utils.configurator import _ensure_embed_section

        base = self.RAG_OFF.replace(
            "  VECTOR_STORE:",
            "  EMBED_MODEL_ID:\n    NAME: existing\n    TYPE: ollama\n  VECTOR_STORE:",
        )
        assert _ensure_embed_section(base) == base

    def test_ensure_embed_section_creates_rag_when_absent(self):
        from mnemoai.utils.configurator import _ensure_embed_section

        base = "MODEL_ID:\n  NAME: x\n  TYPE: ollama\n"
        out = _ensure_embed_section(base)
        d = yaml.safe_load(out)
        assert "EMBED_MODEL_ID" in d["RAG"]

    def test_embeddings_can_be_set_after_scaffold(self):
        # The whole point: after scaffolding, _set_field writes into the new block.
        from mnemoai.utils.configurator import (
            _ensure_embed_section,
            _get_field,
            _set_field,
        )

        out = _ensure_embed_section(self.RAG_OFF)
        out = _set_field(out, "EMBED_MODEL_ID", "NAME", "qwen3-embedding:0.6b")
        assert _get_field(out, "EMBED_MODEL_ID", "NAME") == "qwen3-embedding:0.6b"

    def test_enable_features_prompts_when_off(self, monkeypatch):
        # Both toggles off → both prompts asked; answering yes flips both on.
        from mnemoai.utils import configurator as C

        asked = []
        monkeypatch.setattr(C, "_ask_bool", lambda p, default=True: asked.append(p) or True)
        out = C._prompt_enable_embedding_features(self.RAG_OFF)
        d = yaml.safe_load(out)
        assert d["ENABLE_RAG"] is True
        assert d["ENABLE_EPISODIC_MEMORY"] is True
        assert len(asked) == 2

    def test_enable_features_respects_no(self, monkeypatch):
        from mnemoai.utils import configurator as C

        monkeypatch.setattr(C, "_ask_bool", lambda p, default=True: False)
        out = C._prompt_enable_embedding_features(self.RAG_OFF)
        d = yaml.safe_load(out)
        assert d["ENABLE_RAG"] is False
        assert d["ENABLE_EPISODIC_MEMORY"] is False

    def test_enable_features_skips_already_on(self, monkeypatch):
        # RAG already on → only episodic is asked.
        from mnemoai.utils import configurator as C

        base = self.RAG_OFF.replace("ENABLE_RAG: false", "ENABLE_RAG: true")
        asked = []
        monkeypatch.setattr(C, "_ask_bool", lambda p, default=True: asked.append(p) or True)
        C._prompt_enable_embedding_features(base)
        assert len(asked) == 1
        assert "episodic" in asked[0].lower()

    def test_enable_features_asks_when_key_absent(self, monkeypatch):
        # An absent toggle is treated as OFF (still asked), not defaulted to on.
        from mnemoai.utils import configurator as C

        base = "MODEL_ID:\n  NAME: x\n  TYPE: ollama\n"  # no ENABLE_* keys
        asked = []
        monkeypatch.setattr(C, "_ask_bool", lambda p, default=True: asked.append(p) or True)
        out = C._prompt_enable_embedding_features(base)
        d = yaml.safe_load(out)
        assert len(asked) == 2
        assert d["ENABLE_RAG"] is True
        assert d["ENABLE_EPISODIC_MEMORY"] is True


class TestFeaturesToggles:
    """/features: flip ENABLE_* toggles and gather info a newly-on feature needs."""

    CFG = textwrap.dedent(
        """\
        MODEL_ID:
          NAME: x
          TYPE: ollama
        ENABLE_RAG: false
        RAG:
          MAX_TOKENS: 8192
        ENABLE_EPISODIC_MEMORY: false
        ENABLE_WEB_SEARCH: false
        BRAVE_API_KEY: your_brave_api_key
        """
    )

    def test_placeholder_brave_key_detection(self):
        from mnemoai.utils.configurator import _is_placeholder_brave_key
        assert _is_placeholder_brave_key(None)
        assert _is_placeholder_brave_key("")
        assert _is_placeholder_brave_key("your_brave_api_key")
        assert not _is_placeholder_brave_key("BSA-realkey123")

    def test_web_search_prompts_for_brave_key(self, monkeypatch):
        from mnemoai.utils import configurator as C
        monkeypatch.setattr(C, "_ask", lambda *a, **k: "BSA-realkey")
        out = C._prompt_feature_dependencies(self.CFG, {"ENABLE_WEB_SEARCH"})
        d = yaml.safe_load(out)
        assert d["BRAVE_API_KEY"] == "BSA-realkey"

    def test_web_search_skips_key_when_already_set(self, monkeypatch):
        from mnemoai.utils import configurator as C
        base = self.CFG.replace("BRAVE_API_KEY: your_brave_api_key",
                                "BRAVE_API_KEY: BSA-existing")
        called = {"n": 0}
        monkeypatch.setattr(C, "_ask", lambda *a, **k: called.__setitem__("n", 1))
        C._prompt_feature_dependencies(base, {"ENABLE_WEB_SEARCH"})
        assert called["n"] == 0  # already set — not asked

    def test_rag_prompts_for_embeddings_when_missing(self, monkeypatch):
        from mnemoai.utils import configurator as C
        seen = {"section": None}

        def _fake_section(text, section, is_llm):
            seen["section"] = section
            return C._set_field(C._ensure_embed_section(text), section, "NAME", "embed-x")

        monkeypatch.setattr(C, "_prompt_model_section", _fake_section)
        out = C._prompt_feature_dependencies(self.CFG, {"ENABLE_RAG"})
        assert seen["section"] == "EMBED_MODEL_ID"
        assert C._get_field(out, "EMBED_MODEL_ID", "NAME") == "embed-x"

    def test_no_embeddings_prompt_when_already_configured(self, monkeypatch):
        from mnemoai.utils import configurator as C
        base = self.CFG.replace(
            "  MAX_TOKENS: 8192",
            "  MAX_TOKENS: 8192\n  EMBED_MODEL_ID:\n    NAME: existing\n    TYPE: ollama",
        )
        called = {"n": 0}
        monkeypatch.setattr(
            C, "_prompt_model_section",
            lambda *a, **k: called.__setitem__("n", 1) or a[0],
        )
        C._prompt_feature_dependencies(base, {"ENABLE_RAG"})
        assert called["n"] == 0  # embeddings already configured — not asked

    def test_dependencies_noop_for_features_without_extra_info(self, monkeypatch):
        from mnemoai.utils import configurator as C
        # Turning on the playbook needs nothing extra.
        monkeypatch.setattr(C, "_ask", lambda *a, **k: pytest.fail("should not ask"))
        out = C._prompt_feature_dependencies(self.CFG, {"ENABLE_PLAYBOOK"})
        assert out == self.CFG


class TestPerAreaModels:
    """AREA_MODELS in the configurator: /config offers a model for each enabled
    area, /model and /params can change one afterwards.

    The distinctive rule is what "same as chat" means here. Vision copies the chat
    block; an area must have NO block at all — absence is what makes it follow the
    chat model, so a copy would freeze today's model into a duplicate.
    """

    CFG = textwrap.dedent(
        """\
        MODEL_ID:
          NAME: llama3.1:8b
          TYPE: ollama
          HOST: localhost
          PORT: 11434
          TEMPERATURE: 0.7
        ENABLE_ROUTING: false
        ENABLE_ORCHESTRATION: true
        """
    )

    def _template(self):
        from mnemoai.utils import configurator as C

        return (C._templates_dir() / "config.yaml.example").read_text()

    # --- the two meanings of a section name ---------------------------------

    def test_an_area_is_described_by_the_chat_model_table(self):
        # An area lives under its own block but IS a partial MODEL_ID, so every
        # registry lookup must resolve to the chat model's providers and keys.
        from mnemoai.models.provider_params import providers, supported_keys
        from mnemoai.utils.configurator import _registry_section

        for area in ("ROUTER", "ORCHESTRATOR", "SUMMARY"):
            assert _registry_section(area) == "MODEL_ID"
            assert supported_keys(_registry_section(area), "bedrock") == (
                supported_keys("MODEL_ID", "bedrock")
            )
        # Not an area -> unchanged (vision has its own, narrower table).
        assert _registry_section("VISION_MODEL_ID") == "VISION_MODEL_ID"
        assert set(providers(_registry_section("ROUTER"))) == set(providers("MODEL_ID"))

    def test_the_provider_menu_for_an_area_is_the_llm_one(self, monkeypatch):
        # Proof through the real prompt: an area may switch provider entirely, so
        # it must be offered every LLM provider, defaulting to the chat one.
        from mnemoai.models.provider_params import providers
        from mnemoai.utils import configurator as C

        seen = {}

        def _spy(prompt, valid, default, labels=None, **kw):
            seen.update(valid=valid, labels=labels, default=default)
            return default

        monkeypatch.setattr(C, "_ask_choice", _spy)
        assert C._prompt_provider_type("ROUTER", "ollama") == "ollama"
        assert len(seen["valid"]) == len(list(providers("MODEL_ID")))

    def test_area_provider_falls_back_to_the_chat_one(self):
        from mnemoai.utils.configurator import _effective_type

        text = self.CFG + "AREA_MODELS:\n  ROUTER:\n    NAME: tiny:1b\n"
        # No TYPE of its own -> the merged block runs on the chat provider.
        assert _effective_type(text, "ROUTER") == "ollama"
        text = text.replace("    NAME: tiny:1b", "    NAME: tiny:1b\n    TYPE: bedrock")
        assert _effective_type(text, "ROUTER") == "bedrock"

    # --- scaffolding and clearing the block ---------------------------------

    def test_scaffold_lands_under_the_commented_example(self):
        from mnemoai.utils import configurator as C

        out = C._ensure_area_section(self._template(), "ROUTER")
        d = yaml.safe_load(out)
        assert list(d["AREA_MODELS"]) == ["ROUTER"]
        lines = out.splitlines()
        at = lines.index("AREA_MODELS:")
        # Written directly below the commented example it documents, not at EOF.
        assert lines[at - 1].lstrip().startswith("#")
        assert "# AREA_MODELS:" in out  # the example itself survives

    def test_scaffold_is_a_noop_when_the_block_exists(self):
        from mnemoai.utils import configurator as C

        text = self.CFG + "AREA_MODELS:\n  SUMMARY:\n    NAME: tiny:1b\n"
        assert C._ensure_area_section(text, "SUMMARY") == text

    def test_scaffold_matches_an_existing_bodys_indent(self):
        # Mixed sibling indents are a YAML error, so the new key can't assume the
        # two spaces we would write ourselves.
        from mnemoai.utils import configurator as C

        text = self.CFG + "AREA_MODELS:\n    SUMMARY: tiny:1b\n"
        out = C._ensure_area_section(text, "ROUTER")
        d = yaml.safe_load(out)  # would raise on mixed indents
        assert set(d["AREA_MODELS"]) == {"ROUTER", "SUMMARY"}

    def test_bare_name_shorthand_expands_to_a_block(self):
        # `ROUTER: tiny:1b` is valid config, so a prompt that needs to write TYPE
        # has to expand it in place — keeping the name, without duplicating the key.
        from mnemoai.utils import configurator as C

        text = self.CFG + "AREA_MODELS:\n  ROUTER: tiny:1b\n"
        out = C._ensure_area_section(text, "ROUTER")
        d = yaml.safe_load(out)
        assert d["AREA_MODELS"]["ROUTER"] == {"NAME": "tiny:1b"}
        assert out.count("ROUTER:") == 1
        # And it's now writable through the ordinary section helpers.
        out = C._set_field(out, "ROUTER", "TYPE", "bedrock")
        assert yaml.safe_load(out)["AREA_MODELS"]["ROUTER"]["TYPE"] == "bedrock"

    def test_clearing_removes_the_emptied_header_and_keeps_the_comments(self):
        from mnemoai.utils import configurator as C

        template = self._template()
        out = C._clear_area_override(C._ensure_area_section(template, "ROUTER"), "ROUTER")
        # Back to byte-for-byte what shipped: the commented example is what
        # documents the section, so removing the block must not eat it.
        assert out == template
        assert "AREA_MODELS" not in (yaml.safe_load(out) or {})

    def test_clearing_keeps_a_sibling_area(self):
        from mnemoai.utils import configurator as C

        text = (
            self.CFG
            + "AREA_MODELS:\n  ROUTER: tiny:1b\n  SUMMARY:\n    NAME: small:3b\n"
        )
        d = yaml.safe_load(C._clear_area_override(text, "ROUTER"))
        assert d["AREA_MODELS"] == {"SUMMARY": {"NAME": "small:3b"}}

    def test_clearing_a_bare_name_shorthand(self):
        from mnemoai.utils import configurator as C

        text = self.CFG + "AREA_MODELS:\n  ORCHESTRATOR: big:70b\n"
        out = C._clear_area_override(text, "ORCHESTRATOR")
        assert "AREA_MODELS" not in yaml.safe_load(out)
        assert C._clear_area_override(out, "ORCHESTRATOR") == out  # absent -> no-op

    def test_configured_sees_both_config_shapes(self):
        from mnemoai.utils import configurator as C

        assert not C._area_configured(self.CFG, "ROUTER")
        inline = self.CFG + "AREA_MODELS:\n  ROUTER: tiny:1b\n"
        block = self.CFG + "AREA_MODELS:\n  ROUTER:\n    NAME: tiny:1b\n"
        assert C._area_configured(inline, "ROUTER")
        assert C._area_configured(block, "ROUTER")
        assert not C._area_configured(block, "SUMMARY")

    # --- the feature gate ---------------------------------------------------

    def test_the_gate_names_the_feature_that_is_off(self):
        from mnemoai.utils import configurator as C

        assert C._area_gate(self.CFG, "ROUTER") == ("ENABLE_ROUTING", "Query routing")
        assert C._area_gate(self.CFG, "ORCHESTRATOR") is None  # explicitly on
        on = self.CFG.replace("ENABLE_ROUTING: false", "ENABLE_ROUTING: true")
        assert C._area_gate(on, "ROUTER") is None

    def test_compaction_always_runs_so_summary_is_never_gated(self):
        from mnemoai.utils import configurator as C

        assert C._area_gate(self.CFG, "SUMMARY") is None
        assert C._area_gate(self.CFG, "MODEL_ID") is None

    # --- what the pickers offer --------------------------------------------

    def test_every_area_is_reachable_from_model_and_params(self):
        # A model that can only be set by hand-editing the config is half-wired.
        from mnemoai.models.area_models import AREAS, DESCRIPTIONS
        from mnemoai.utils.configurator import _MODEL_SECTIONS, _PARAM_SECTIONS

        model_labels = {s: lbl for s, lbl, _ in _MODEL_SECTIONS.values()}
        assert set(AREAS) <= set(model_labels)
        assert set(AREAS) <= {s for s, _ in _PARAM_SECTIONS.values()}
        # The job wording comes from the area registry, so /model and /doctor
        # describe the same call the same way.
        for area in AREAS:
            assert DESCRIPTIONS[area] in model_labels[area]

    def test_the_overview_lists_every_area(self):
        from mnemoai.utils import configurator as C

        out = C._current_setup_text(self.CFG)
        rows = dict(
            line.strip().split(":", 1) for line in out.splitlines()[1:]
        )
        # Absence is an answer to "which model does the router use", not a gap.
        assert rows["Router"].strip().startswith("same as chat")
        assert "query routing is off" in rows["Router"]
        assert rows["Orchestrator"].strip() == "same as chat"
        assert rows["Summary"].strip() == "same as chat"

    def test_the_overview_shows_a_configured_area(self):
        from mnemoai.utils import configurator as C

        inline = self.CFG + "AREA_MODELS:\n  SUMMARY: tiny:1b\n"
        rows = C._current_setup_text(inline).splitlines()
        # The shorthand names no provider, so the chat model's fills it in.
        assert any("Summary:" in r and "ollama / tiny:1b" in r for r in rows)

    # --- the prompt itself --------------------------------------------------

    def _script(self, monkeypatch, *, same_as_chat, provider="ollama", name="tiny:1b"):
        """Answer the area prompts: same-as-chat y/n, then provider, name, conn."""
        from mnemoai.utils import configurator as C

        monkeypatch.setattr(C, "_ask_bool", lambda p, default=True, **k: same_as_chat)
        monkeypatch.setattr(C, "_prompt_provider_type", lambda s, cur: provider)
        monkeypatch.setattr(
            C, "_ask",
            lambda p, default=None, **k: name if "Model name" in p else (default or ""),
        )
        monkeypatch.setattr(C, "_ask_number", lambda *a, **k: None)
        return C

    def test_same_as_chat_writes_nothing(self, monkeypatch):
        C = self._script(monkeypatch, same_as_chat=True)
        template = self._template()
        assert C._prompt_model_section(template, "ROUTER", is_llm=False) == template

    def test_same_as_chat_removes_an_existing_override(self, monkeypatch):
        C = self._script(monkeypatch, same_as_chat=True)
        text = self.CFG + "AREA_MODELS:\n  ROUTER:\n    NAME: tiny:1b\n    TYPE: mlx\n"
        out = C._prompt_model_section(text, "ROUTER", is_llm=False)
        assert "AREA_MODELS" not in yaml.safe_load(out)

    def test_declining_configures_the_area_on_its_own_provider(self, monkeypatch):
        C = self._script(monkeypatch, same_as_chat=False, provider="bedrock",
                         name="global.anthropic.claude-haiku-4-5")
        out = C._prompt_model_section(self.CFG, "ROUTER", is_llm=False)
        d = yaml.safe_load(out)
        r = d["AREA_MODELS"]["ROUTER"]
        assert r["TYPE"] == "bedrock"
        assert r["NAME"] == "global.anthropic.claude-haiku-4-5"
        assert r["REGION"] == "us-east-1"
        # An area is a partial MODEL_ID: the chat model is left alone.
        assert d["MODEL_ID"]["NAME"] == "llama3.1:8b"
        assert d["MODEL_ID"]["TYPE"] == "ollama"
        # The context-window prompt belongs to the chat model only.
        assert "MAX_CONVERSATION_TOKENS" not in d

    def test_a_blank_name_keeps_the_chat_model(self, monkeypatch):
        # Nothing named -> no block, rather than one that says nothing.
        C = self._script(monkeypatch, same_as_chat=False, name="")
        out = C._prompt_model_section(self.CFG, "SUMMARY", is_llm=False)
        assert "AREA_MODELS" not in yaml.safe_load(out)

    def test_params_scaffold_a_shorthand_before_tuning(self, monkeypatch):
        # /params writes into the block, so the bare-name form has to grow one —
        # and the params offered are the chat provider's.
        from mnemoai.utils import configurator as C

        monkeypatch.setattr(C, "_ask", lambda p, default=None, **k: default or "")
        monkeypatch.setattr(
            C, "_prompt_one_param",
            lambda t, s, key, kind, hint, back: C._set_field(t, s, key, "0.3")
            if key == "TEMPERATURE" else t,
        )
        text = self.CFG + "AREA_MODELS:\n  ROUTER: tiny:1b\n"
        d = yaml.safe_load(C._prompt_inference_params(text, "ROUTER"))
        assert d["AREA_MODELS"]["ROUTER"] == {"NAME": "tiny:1b", "TEMPERATURE": 0.3}
        assert d["MODEL_ID"]["TEMPERATURE"] == 0.7  # chat model untouched


class TestChoiceDialogTracksTheArrowKeys:
    """`/model` + the `/config` wizard route every single-choice prompt through
    `_dialog_radio`, which built its `RadioList` without `select_on_focus`.

    A `RadioList` keeps the highlighted row separate from its committed
    `current_value`, and the dialog's own enter binding (needed so Enter confirms
    without a Tab-to-OK step) shadows the one binding that reconciles them. So the
    `(*)` marker sat on the opening row while the arrows moved only the highlight:
    the pick had to be committed with Space — undocumented — and Enter otherwise
    confirmed a row the user had moved off. `--resume` never had this, which is
    how the two pickers came to answer the same key differently.
    """

    def _spy_radio(self, monkeypatch):
        """Run _dialog_radio with a no-op Application, capturing the RadioList kwargs."""
        import prompt_toolkit.widgets as widgets

        from mnemoai.utils import configurator as C

        seen = {}
        real = widgets.RadioList

        def _spy(values, **kwargs):
            seen.update(kwargs)
            return real(values, **kwargs)

        monkeypatch.setattr(C, "RadioList", _spy)
        monkeypatch.setattr(
            C, "Application",
            lambda **k: type("_A", (), {"run": lambda self: C._DIALOG_CANCEL})(),
        )
        return C, seen

    def test_the_dialog_asks_for_select_on_focus(self, monkeypatch):
        C, seen = self._spy_radio(monkeypatch)
        C._dialog_radio("Which model?", [("a", "a"), ("b", "b")])
        assert seen.get("select_on_focus") is True

    def test_a_default_still_opens_on_that_row(self, monkeypatch):
        # select_on_focus and default are independent: the pre-selected provider
        # must still be the row the dialog opens (and commits) on.
        C, seen = self._spy_radio(monkeypatch)
        C._dialog_radio("Provider", [("a", "a"), ("b", "b")], default="b")
        assert seen.get("default") == "b" and seen.get("select_on_focus") is True

    def test_arrows_move_the_committed_value_on_a_real_radiolist(self):
        # The behavior itself, on the real widget: what _ok() reads must follow
        # the highlight, with no Space in between.
        from prompt_toolkit.keys import Keys
        from prompt_toolkit.widgets import RadioList

        radio = RadioList(
            values=[("a", "a"), ("b", "b"), ("c", "c")], select_on_focus=True
        )
        down = next(
            b.handler for b in radio.control.key_bindings.bindings
            if b.keys == (Keys.Down,)
        )
        down(None)
        assert radio.current_value == "b"
        down(None)
        assert radio.current_value == "c"

    def test_multi_select_keeps_space_to_toggle(self, monkeypatch):
        # /features is a CheckboxList: there the marker MUST NOT follow the
        # highlight (moving past a row would tick it), so it takes no such flag —
        # and prompt_toolkit does not even offer one for multiple selection.
        import inspect

        from prompt_toolkit.widgets import CheckboxList

        assert "select_on_focus" not in inspect.signature(CheckboxList).parameters
