"""Unit tests for the first-run configurator's template patching.

These cover the pure line-editing helpers — no TTY/interaction needed. They
verify that edits target the right key (first match within a section, or the
top level) and that the rich prompt blocks survive as valid YAML.
"""

import textwrap

import pytest

yaml = pytest.importorskip("yaml")

from utils.configurator import (
    _get_in_section,
    _get_top_level,
    _set_bool,
    _set_in_section,
    _set_or_add_in_section,
    _set_top_level,
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
