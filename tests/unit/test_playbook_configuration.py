"""Playbook configuration is reachable through every relevant interactive path."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from mnemoai.client.client import LangGraphClient
from mnemoai.client.ui.chat_interface import ChatInterface
from mnemoai.utils import configurator as C

CFG = """\
# retain the user's comments
MODEL_ID:
  NAME: chat-fixture
  TYPE: ollama
ENABLE_PLAYBOOK: false
PLAYBOOK:
  REFLECTION_TIMEOUT: 30 # a user comment
  MAX_ENTRIES: 500
  MAX_INJECT: 10
  SIMILARITY_THRESHOLD: 0.85
  CUSTOM:
    MAX_ENTRIES: 999
    retain: true
"""


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(CFG)
    monkeypatch.setattr(C, "_config_to_edit", lambda: path)
    return path


def answers(monkeypatch, **values):
    def answer(label, default=None, **kwargs):
        for key, spec in C._PLAYBOOK_SETTINGS.items():
            if label.startswith(spec[0]):
                return str(values.get(key, default))
        return default
    monkeypatch.setattr(C, "_ask_number", answer)


def test_scoped_config_changes_only_playbook_and_preserves_comments(config_file, monkeypatch):
    answers(monkeypatch, REFLECTION_TIMEOUT=45, MAX_ENTRIES=80, MAX_INJECT=6)
    assert C.run_playbook_settings() == config_file
    text = config_file.read_text()
    data, old = yaml.safe_load(text), yaml.safe_load(CFG)
    assert data["PLAYBOOK"]["REFLECTION_TIMEOUT"] == 45
    assert data["PLAYBOOK"]["MAX_ENTRIES"] == 80
    assert data["PLAYBOOK"]["MAX_INJECT"] == 6
    assert data["PLAYBOOK"]["CUSTOM"] == old["PLAYBOOK"]["CUSTOM"]
    assert data["MODEL_ID"] == old["MODEL_ID"] and not data["ENABLE_PLAYBOOK"]
    assert "# retain the user's comments" in text and "# a user comment" in text


def test_cancel_and_no_change_preserve_file(config_file, monkeypatch):
    answers(monkeypatch)
    assert C.run_playbook_settings() is None
    assert config_file.read_text() == CFG
    monkeypatch.setattr(C, "_ask_number", Mock(side_effect=C._Cancelled()))
    assert C.run_playbook_settings() is None
    assert config_file.read_text() == CFG


def test_settings_reject_nonfinite_and_out_of_range_input(monkeypatch):
    attempts = iter(["nan", "0", "121", "25"])
    monkeypatch.setattr(C, "_ask_number", lambda *a, **k: next(attempts))
    data = yaml.safe_load(C._prompt_playbook_settings(CFG, timeout_only=True))
    assert data["PLAYBOOK"]["REFLECTION_TIMEOUT"] == 25
    assert list(attempts) == []


@pytest.mark.parametrize("section", [
    "",
    "PLAYBOOK: {}\n",
    "PLAYBOOK: {CUSTOM: {keep: yes}, REFLECTION_TIMEOUT: 30} # keep header\n",
    "PLAYBOOK:\n# keep unindented comment\n    CUSTOM:\n      keep: true\n",
])
def test_settings_create_or_expand_mapping_without_losing_unknown_fields(section, monkeypatch):
    answers(monkeypatch, REFLECTION_TIMEOUT=45)
    text = "MODEL_ID:\n  NAME: fixture\n" + section
    updated = C._prompt_playbook_settings(text, timeout_only=True)
    data = yaml.safe_load(updated)
    assert data["PLAYBOOK"]["REFLECTION_TIMEOUT"] == 45
    if "CUSTOM" in section:
        assert data["PLAYBOOK"]["CUSTOM"]["keep"] is True
    if "# keep" in section:
        assert "# keep" in updated


def test_adding_settings_preserves_trailing_newlines_in_unknown_block_scalar(monkeypatch):
    original = "MODEL_ID:\n  NAME: fixture\nCUSTOM_TEXT: |+\n  retain\n\n\n"
    answers(monkeypatch, REFLECTION_TIMEOUT=45)
    updated = C._prompt_playbook_settings(original, timeout_only=True)
    assert yaml.safe_load(updated)["CUSTOM_TEXT"] == yaml.safe_load(original)["CUSTOM_TEXT"]


def test_scoped_config_detects_concurrent_edits(config_file, monkeypatch):
    def answer(label, default=None, **kwargs):
        config_file.write_text(CFG + "\nNEW_SETTING: true\n")
        return "45" if label.startswith("Reflection wait") else default
    monkeypatch.setattr(C, "_ask_number", answer)
    assert C.run_playbook_settings() is None
    assert "NEW_SETTING: true" in config_file.read_text()


def test_failed_save_preserves_existing_config(config_file, monkeypatch):
    answers(monkeypatch, REFLECTION_TIMEOUT=45)
    monkeypatch.setattr(C, "atomic_write_text", Mock(side_effect=OSError("disk full")))
    assert C.run_playbook_settings() is None
    assert config_file.read_text() == CFG


def test_settings_follow_symlink_without_replacing_it(config_file, tmp_path, monkeypatch):
    link = tmp_path / "config-link.yaml"
    link.symlink_to(config_file)
    monkeypatch.setattr(C, "_config_to_edit", lambda: link)
    answers(monkeypatch, REFLECTION_TIMEOUT=45)
    assert C.run_playbook_settings() == link
    assert link.is_symlink()
    assert yaml.safe_load(config_file.read_text())["PLAYBOOK"]["REFLECTION_TIMEOUT"] == 45


def test_features_enabling_learning_offers_reflector_and_wait_limit(config_file, monkeypatch):
    monkeypatch.setattr(C, "_is_tty", lambda: True)
    monkeypatch.setattr(C, "_dialog_checkbox",
                        lambda title, options, checked, **kw: [*checked, "ENABLE_PLAYBOOK"])
    model_prompt = Mock(side_effect=lambda text, *a, **kw: text)
    monkeypatch.setattr(C, "_prompt_model_section", model_prompt)
    answers(monkeypatch, REFLECTION_TIMEOUT=45)
    assert C.run_features_override() == config_file
    data = yaml.safe_load(config_file.read_text())
    assert data["ENABLE_PLAYBOOK"] is True
    assert data["PLAYBOOK"]["REFLECTION_TIMEOUT"] == 45
    assert model_prompt.call_args.args[1] == "REFLECTOR"


def test_full_config_offers_reflector_when_learning_enabled(monkeypatch):
    seen = []
    monkeypatch.setattr(C, "_ask", lambda prompt, default=None, **kw: default or "fixture")
    monkeypatch.setattr(C, "_ask_number",
                        lambda prompt, default=None, **kw: None if kw.get("allow_none") else default)
    monkeypatch.setattr(C, "_ask_bool", lambda prompt, default=True, **kw: "playbook" in prompt.lower())
    monkeypatch.setattr(C, "_prompt_model_section",
                        lambda text, section, **kw: seen.append(section) or text)
    text = C._build_config("ollama", "fixture", CFG)
    assert "REFLECTOR" in seen
    assert yaml.safe_load(text)["ENABLE_PLAYBOOK"] is True


def test_model_dialog_writes_reflector_override_and_enables_learning(config_file, monkeypatch):
    monkeypatch.setattr(C, "_is_tty", lambda: False)
    monkeypatch.setattr(C, "_ask_choice", lambda *a, **kw: "7")
    monkeypatch.setattr(C, "_ask_bool", lambda prompt, **kw: "same model" not in prompt)
    monkeypatch.setattr(C, "_prompt_provider_type", lambda *a: "ollama")
    monkeypatch.setattr(C, "_ask", lambda prompt, default=None, **kw:
                        "reflector-fixture" if prompt == "Model name" else default or "")
    monkeypatch.setattr(C, "_ask_number", lambda *a, **kw: None)
    result = C.run_model_override()
    data = yaml.safe_load(config_file.read_text())
    assert result.section == "REFLECTOR" and result.enabled_feature
    assert data["AREA_MODELS"]["REFLECTOR"]["NAME"] == "reflector-fixture"
    assert data["ENABLE_PLAYBOOK"] is True
    assert data["MODEL_ID"]["NAME"] == "chat-fixture"


def test_params_dialog_tunes_reflector_not_chat(config_file, monkeypatch):
    config_file.write_text(CFG + "AREA_MODELS:\n  REFLECTOR: reflector-fixture\n")
    monkeypatch.setattr(C, "_is_tty", lambda: False)
    monkeypatch.setattr(C, "_ask_choice", lambda *a, **kw: "7")
    monkeypatch.setattr(C, "_prompt_one_param", lambda text, section, key, *args:
                        C._set_field(text, section, key, "0.25") if key == "TEMPERATURE" else text)
    assert C.run_params_override() == config_file
    data = yaml.safe_load(config_file.read_text())
    assert data["AREA_MODELS"]["REFLECTOR"]["TEMPERATURE"] == 0.25
    assert "TEMPERATURE" not in data["MODEL_ID"]


def test_scoped_config_dispatch_applies_without_restart(monkeypatch):
    ui = ChatInterface.__new__(ChatInterface)
    ui.client = SimpleNamespace(reload_playbook_settings=Mock(return_value=True))
    ui._restart_in_place = Mock(side_effect=AssertionError("must not restart"))
    monkeypatch.setattr("mnemoai.client.ui.chat_interface.run_playbook_settings",
                        lambda: Path("/temporary/config.yaml"))
    ui._dispatch("/config playbook")
    ui.client.reload_playbook_settings.assert_called_once()
    ui._restart_in_place.assert_not_called()


def test_reload_preserves_model_conversation_and_session_injection_choice(monkeypatch):
    client = LangGraphClient.__new__(LangGraphClient)
    client.model = object()
    client.agent = SimpleNamespace(messages=["history"])
    client.playbook = SimpleNamespace(enabled=False, max_entries=500, similarity_threshold=0.85)
    client.refresh_playbook_context = Mock()
    original_model = client.model
    monkeypatch.setattr("mnemoai.client.client.config.reload", lambda: None)
    monkeypatch.setattr("mnemoai.client.client.config.get",
                        lambda key, default=None: {"MAX_ENTRIES": 80, "SIMILARITY_THRESHOLD": 0.9})
    assert client.reload_playbook_settings()
    assert client.model is original_model and client.agent.messages == ["history"]
    assert not client.playbook.enabled and client.playbook.max_entries == 80
    assert client.playbook.similarity_threshold == 0.9
