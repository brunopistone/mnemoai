"""Unit tests for the prompts fallback-merge in Config._load_prompts.

A NEW prompt shipped in a release must resolve on an EXISTING install whose
prompts.yaml was seeded once and is never overwritten. The loader layers the
bundled package prompts.yaml underneath the user's file (user keys win), so a
missing key falls back to the bundled default while customizations are preserved.
"""

import mnemoai.utils.config as config_mod
from mnemoai.utils.config import Config


def _bare_config():
    c = Config.__new__(Config)
    c._config_data = {}
    c._prompts_data = {}
    return c


def test_bundled_prompt_used_when_user_file_missing_key(monkeypatch, tmp_path):
    # User prompts.yaml has only SYSTEM_PROMPT; a bundled key must still resolve.
    user_file = tmp_path / "prompts.yaml"
    user_file.write_text("SYSTEM_PROMPT: user override\n")
    monkeypatch.setattr(
        Config, "_resolve_prompts_path", staticmethod(lambda: user_file)
    )
    c = _bare_config()
    c._load_prompts()
    # User key wins…
    assert c._prompts_data["SYSTEM_PROMPT"] == "user override"
    # …and a key only in the bundle (never in the user's old file) resolves.
    assert c._prompts_data.get("MEMORY_EXTRACTION_PROMPT")
    assert c._prompts_data.get("SUMMARY_TASK_PROMPT")


def test_user_prompt_overrides_bundled(monkeypatch, tmp_path):
    user_file = tmp_path / "prompts.yaml"
    user_file.write_text("SUMMARY_SYSTEM_PROMPT: my custom summary framing\n")
    monkeypatch.setattr(
        Config, "_resolve_prompts_path", staticmethod(lambda: user_file)
    )
    c = _bare_config()
    c._load_prompts()
    assert c._prompts_data["SUMMARY_SYSTEM_PROMPT"] == "my custom summary framing"


def test_no_user_file_still_loads_bundle(monkeypatch):
    # No resolvable user prompts → the bundled defaults still populate everything.
    monkeypatch.setattr(
        Config, "_resolve_prompts_path", staticmethod(lambda: None)
    )
    c = _bare_config()
    c._load_prompts()
    assert c._prompts_data.get("SYSTEM_PROMPT")
    assert c._prompts_data.get("MEMORY_EXTRACTION_PROMPT")


def test_memory_extraction_prompt_has_format_placeholders():
    # The shipped prompt must expose the two fields the extractor fills.
    c = _bare_config()

    # Force the bundled-only path.
    import types

    def _none():
        return None

    orig = Config._resolve_prompts_path
    Config._resolve_prompts_path = staticmethod(_none)
    try:
        c._load_prompts()
    finally:
        Config._resolve_prompts_path = orig
    prompt = c._prompts_data["MEMORY_EXTRACTION_PROMPT"]
    assert "{existing_memory}" in prompt
    assert "{exchange}" in prompt
    # And it must .format() cleanly with just those two named fields.
    out = prompt.format(existing_memory="X", exchange="Y")
    assert "X" in out and "Y" in out
