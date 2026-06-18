"""Unit tests for the first-run configurator's template patching.

These cover the pure line-editing helpers — no TTY/interaction needed. They
verify that edits target the right key (first match within a section, or the
top level) and that the rich prompt blocks survive as valid YAML.
"""

import textwrap

import pytest

yaml = pytest.importorskip("yaml")

from utils.configurator import (
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
