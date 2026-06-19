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


def test_provider_params_registry_shape():
    # Guard against drift: each section must advertise the provider set the
    # configurator/controllers expect, and supported_keys must report sane sets.
    from mnemoai.models.provider_params import providers, supported_keys

    assert set(providers("MODEL_ID")) == {
        "ollama", "bedrock", "mantle", "openai", "sagemaker", "litellm"
    }
    assert set(providers("VISION_MODEL_ID")) == {
        "ollama", "bedrock", "mantle", "openai", "sagemaker", "litellm"
    }
    assert supported_keys("VISION_MODEL_ID", "litellm") == {
        "API_BASE", "API_KEY", "TEMPERATURE", "MAX_TOKENS", "TOP_P"
    }
    assert set(providers("EMBED_MODEL_ID")) == {
        "ollama", "bedrock", "openai", "sagemaker", "litellm"
    }
    # Embeddings take no inference params — only connection keys.
    assert supported_keys("EMBED_MODEL_ID", "ollama") == {"HOST", "PORT"}
    assert supported_keys("EMBED_MODEL_ID", "openai") == set()
    assert supported_keys("EMBED_MODEL_ID", "litellm") == {"API_BASE", "API_KEY"}
    # Unknown provider -> None (configurator then prunes nothing).
    assert supported_keys("MODEL_ID", "bogus") is None


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
    assert summary == "mantle / anthropic.claude-haiku-4-5 (us-east-1, anthropic)"


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
        ["gpt-5-mini", "none", "65536", "y", "gpt-5-mini", "none", "alice", "",
         "y", "y", "y", "y", "y", "y", "y"],
    )
    m = d["MODEL_ID"]
    assert m["TYPE"] == "openai" and m["NAME"] == "gpt-5-mini"
    # Ollama-only keys pruned; OpenAI-valid TEMPERATURE/PRESENCE_PENALTY kept.
    for bad in ("HOST", "PORT", "TOP_K", "FREQUENCY_PENALTY"):
        assert bad not in m
    assert d["VISION_MODEL_ID"]["TYPE"] == "openai"


def test_config_sagemaker_sets_region_and_input_format():
    d = _run_build(
        "sagemaker", "my-endpoint",
        ["my-endpoint", "eu-west-1", "huggingface", "none", "65536", "n", "bob", "",
         "y", "y", "y", "y", "y", "y", "y"],
    )
    m = d["MODEL_ID"]
    assert m["TYPE"] == "sagemaker"
    assert m["REGION"] == "eu-west-1" and m["INPUT_FORMAT"] == "huggingface"
    assert "HOST" not in m and "PORT" not in m


def test_config_litellm_sets_api_base_and_key():
    d = _run_build(
        "litellm", "openai/gpt-4o",
        ["openai/gpt-4o", "http://localhost:8000/v1", "sk-xyz", "none", "65536", "n",
         "carol", "", "y", "y", "y", "y", "y", "y", "y"],
    )
    m = d["MODEL_ID"]
    assert m["TYPE"] == "litellm"
    assert m["API_BASE"] == "http://localhost:8000/v1" and m["API_KEY"] == "sk-xyz"
    assert "HOST" not in m


def test_config_providers_menu_has_all_six():
    from mnemoai.utils.configurator import _PROVIDERS

    types = {v[0] for v in _PROVIDERS.values()}
    assert types == {"ollama", "bedrock", "mantle", "openai", "sagemaker", "litellm"}


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


def test_prompt_provider_connection_openai_asks_nothing(monkeypatch):
    # OpenAI is env-based (OPENAI_API_KEY); no connection keys to prompt.
    from mnemoai.utils import configurator as C

    def _no_input(*a, **k):
        raise AssertionError("OpenAI should not prompt for connection keys")

    monkeypatch.setattr("builtins.input", _no_input)
    text = "MODEL_ID:\n  NAME: gpt-5-mini\n  TYPE: openai\n"
    out, conn = C._prompt_provider_connection(text, "MODEL_ID", "openai")
    assert conn == {}
    assert "HOST" not in yaml.safe_load(out)["MODEL_ID"]


def test_prompt_provider_connection_embeddings_skips_input_format(monkeypatch):
    # INPUT_FORMAT is a SageMaker *chat* key; embeddings sagemaker only needs REGION.
    from mnemoai.utils import configurator as C

    answers = iter(["us-west-2"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    text = "RAG:\n  EMBED_MODEL_ID:\n    NAME: e\n    TYPE: sagemaker\n"
    out, _ = C._prompt_provider_connection(text, "EMBED_MODEL_ID", "sagemaker")
    d = yaml.safe_load(out)
    assert d["RAG"]["EMBED_MODEL_ID"]["REGION"] == "us-west-2"
    assert "INPUT_FORMAT" not in d["RAG"]["EMBED_MODEL_ID"]
