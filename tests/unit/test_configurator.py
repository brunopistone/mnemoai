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
        "ollama", "bedrock", "mantle", "openai", "anthropic", "sagemaker", "litellm"
    }
    assert set(providers("VISION_MODEL_ID")) == {
        "ollama", "bedrock", "mantle", "openai", "anthropic", "sagemaker", "litellm"
    }
    # Anthropic (direct Claude API): STOP-capable, with extended-thinking
    # specials. EXTRA_PARAMS (the generic passthrough) is supported everywhere.
    assert supported_keys("MODEL_ID", "anthropic") == {
        "TEMPERATURE", "MAX_TOKENS", "TOP_P", "TOP_K", "STOP",
        "API_KEY", "ENDPOINT_URL",
        "REASONING", "REASONING_EFFORT", "THINKING_TOKENS", "STREAM",
        "EXTRA_PARAMS",
    }
    # Bedrock must offer STREAM like the other streaming providers, so /params can
    # tune it — the controller translates it to ChatBedrockConverse's inverted
    # `disable_streaming` (it has no passthrough spec, hence "special").
    assert supported_keys("MODEL_ID", "bedrock") == {
        "TEMPERATURE", "TOP_P", "MAX_TOKENS", "STOP",
        "REGION", "ENDPOINT_URL",
        "REASONING", "REASONING_EFFORT", "THINKING_TOKENS", "STREAM",
        "EXTRA_PARAMS",
    }
    assert supported_keys("VISION_MODEL_ID", "litellm") == {
        "API_BASE", "API_KEY", "TEMPERATURE", "MAX_TOKENS", "TOP_P", "EXTRA_PARAMS"
    }
    assert set(providers("EMBED_MODEL_ID")) == {
        "ollama", "bedrock", "openai", "sagemaker", "litellm"
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
    assert {"TEMPERATURE", "MAX_TOKENS", "TOP_P", "STREAM", "REASONING_EFFORT"} == tm
    # It's exactly supported minus the connection set and the generic
    # EXTRA_PARAMS passthrough (which is not a /params-tunable scalar).
    from mnemoai.models.provider_params import _TABLES  # type: ignore

    for prov in ("ollama", "bedrock", "openai", "sagemaker", "litellm"):
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


def test_every_tunable_key_has_prompt_metadata():
    # Guard against drift: any key tunable_params can report must have an entry
    # in _PARAM_META and a slot in _PARAM_ORDER, else /params would skip/crash.
    from mnemoai.models.provider_params import tunable_params
    from mnemoai.utils.configurator import _PARAM_META, _PARAM_ORDER

    keys = set()
    for section in ("MODEL_ID", "VISION_MODEL_ID"):
        for prov in ("ollama", "bedrock", "mantle", "openai", "sagemaker", "litellm"):
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
    """Drive _build_config against the base template with scripted answers."""
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
        # WEB_CRAWL, ROUTING, ORCH, PROFILING, BASH_CONFIRM, WRITE_CONFIRM, MEM_CONFIRM.
        ["gpt-5-mini", "", "", "none", "65536", "y", "y", "n", "alice", "",
         "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y"],
    )
    m = d["MODEL_ID"]
    assert m["TYPE"] == "openai" and m["NAME"] == "gpt-5-mini"
    # Ollama-only keys pruned; OpenAI-valid TEMPERATURE/PRESENCE_PENALTY kept.
    for bad in ("HOST", "PORT", "TOP_K", "FREQUENCY_PENALTY"):
        assert bad not in m
    # Vision copied from chat via the "same as chat?" shortcut.
    assert d["VISION_MODEL_ID"]["TYPE"] == "openai"
    assert d["VISION_MODEL_ID"]["NAME"] == "gpt-5-mini"


def test_config_sagemaker_sets_region_and_input_format():
    d = _run_build(
        "sagemaker", "my-endpoint",
        # chat name, region, input_format, MAX_TOKENS none, ctx,
        # vision? n, embeddings? n, profile, brave, then 13 toggles (see openai test).
        ["my-endpoint", "eu-west-1", "huggingface", "none", "65536", "n", "n",
         "bob", "", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y"],
    )
    m = d["MODEL_ID"]
    assert m["TYPE"] == "sagemaker"
    assert m["REGION"] == "eu-west-1" and m["INPUT_FORMAT"] == "huggingface"
    assert "HOST" not in m and "PORT" not in m


def test_config_litellm_sets_api_base_and_key():
    d = _run_build(
        "litellm", "openai/gpt-4o",
        # chat name, api_base, api_key, MAX_TOKENS none, ctx,
        # vision? n, embeddings? n, profile, brave, then 13 toggles (see openai test).
        ["openai/gpt-4o", "http://localhost:8000/v1", "sk-xyz", "none", "65536",
         "n", "n", "carol", "",
         "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y"],
    )
    m = d["MODEL_ID"]
    assert m["TYPE"] == "litellm"
    assert m["API_BASE"] == "http://localhost:8000/v1" and m["API_KEY"] == "sk-xyz"
    assert "HOST" not in m


def test_config_anthropic_transforms_base_template():
    # answers: chat name, API_KEY, base URL (blank), MAX_TOKENS, ctx, configure
    # vision? (y), "same as chat?" (y → copies chat), embeddings? (n), profile,
    # brave (blank), then 13 toggles: RAG, EPISODIC, PLAYBOOK, MEMORY,
    # AUTO_EXTRACT, SKILLS, WEB_CRAWL, ROUTING, ORCH, PROFILING, BASH_CONFIRM,
    # WRITE_CONFIRM, MEM_CONFIRM.
    d = _run_build(
        "anthropic", "claude-opus-4-8",
        ["claude-opus-4-8", "fake-anthropic-key", "", "none", "65536",
         "y", "y", "n", "dave", "",
         "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y", "y"],
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


def test_config_skips_auto_extract_when_memory_off():
    # When persistent memory is declined, the auto-extraction sub-prompt is not
    # asked (one fewer answer). Memory=n here; sequence stays aligned.
    d = _run_build(
        "anthropic", "claude-opus-4-8",
        ["claude-opus-4-8", "fake-key", "", "none", "65536",
         "n", "n", "dave", "",   # vision? n, embeddings? n
         # RAG, EPISODIC, PLAYBOOK, MEMORY(n → no auto-extract prompt), SKILLS,
         # WEB_CRAWL, ROUTING, ORCH, PROFILING, BASH, WRITE, MEM_CONFIRM.
         "y", "y", "y", "n", "y", "y", "y", "y", "y", "y", "y", "y"],
    )
    assert d["ENABLE_MEMORY"] is False
    # Not prompted → key stays at its template value (not forced by this run).


def test_config_providers_menu_has_all_seven():
    from mnemoai.utils.configurator import _PROVIDERS

    types = {v[0] for v in _PROVIDERS.values()}
    assert types == {
        "ollama", "bedrock", "mantle", "openai", "anthropic", "sagemaker", "litellm"
    }


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
