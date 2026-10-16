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
    monkeypatch.setattr(C, "_ask_bool", lambda prompt, default=True, **kw: default)
    def answer(label, default=None, **kwargs):
        for key, spec in C._PLAYBOOK_SETTINGS.items():
            if label.startswith(spec[0]):
                return str(values.get(key, default))
        return default
    monkeypatch.setattr(C, "_ask_number", answer)


def test_scoped_config_changes_only_playbook_and_preserves_comments(config_file, monkeypatch):
    answers(monkeypatch, MAX_ENTRIES=80, MAX_INJECT=6)
    assert C.run_playbook_settings() == config_file
    text = config_file.read_text()
    data, old = yaml.safe_load(text), yaml.safe_load(CFG)
    assert data["PLAYBOOK"]["REFLECTION_TIMEOUT"] == 30  # ignored legacy field, not rewritten
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
    attempts = iter(["nan", "0", "-1", "25"])
    answers(monkeypatch)
    monkeypatch.setattr(C, "_ask_number", lambda label, **kw:
                        next(attempts) if label.startswith("Active notes") else kw["default"])
    data = yaml.safe_load(C._prompt_playbook_settings(CFG))
    assert data["PLAYBOOK"]["MAX_ENTRIES"] == 25
    assert list(attempts) == []


@pytest.mark.parametrize("section", [
    "",
    "PLAYBOOK: {}\n",
    "PLAYBOOK: {CUSTOM: {keep: yes}, REFLECTION_TIMEOUT: 30} # keep header\n",
    "PLAYBOOK:\n# keep unindented comment\n    CUSTOM:\n      keep: true\n",
])
def test_settings_create_or_expand_mapping_without_losing_unknown_fields(section, monkeypatch):
    answers(monkeypatch, MAX_ENTRIES=80)
    text = "MODEL_ID:\n  NAME: fixture\n" + section
    updated = C._prompt_playbook_settings(text)
    data = yaml.safe_load(updated)
    assert data["PLAYBOOK"]["MAX_ENTRIES"] == 80
    if "CUSTOM" in section:
        assert data["PLAYBOOK"]["CUSTOM"]["keep"] is True
    if "# keep" in section:
        assert "# keep" in updated


def test_adding_settings_preserves_trailing_newlines_in_unknown_block_scalar(monkeypatch):
    original = "MODEL_ID:\n  NAME: fixture\nCUSTOM_TEXT: |+\n  retain\n\n\n"
    answers(monkeypatch, MAX_ENTRIES=80)
    updated = C._prompt_playbook_settings(original)
    assert yaml.safe_load(updated)["CUSTOM_TEXT"] == yaml.safe_load(original)["CUSTOM_TEXT"]


def test_scoped_config_detects_concurrent_edits(config_file, monkeypatch):
    answers(monkeypatch)
    def answer(label, default=None, **kwargs):
        config_file.write_text(CFG + "\nNEW_SETTING: true\n")
        return "80" if label.startswith("Active notes") else default
    monkeypatch.setattr(C, "_ask_number", answer)
    assert C.run_playbook_settings() is None
    assert "NEW_SETTING: true" in config_file.read_text()


def test_failed_save_preserves_existing_config(config_file, monkeypatch):
    answers(monkeypatch, MAX_ENTRIES=80)
    monkeypatch.setattr(C, "atomic_write_text", Mock(side_effect=OSError("disk full")))
    assert C.run_playbook_settings() is None
    assert config_file.read_text() == CFG


def test_settings_follow_symlink_without_replacing_it(config_file, tmp_path, monkeypatch):
    link = tmp_path / "config-link.yaml"
    link.symlink_to(config_file)
    monkeypatch.setattr(C, "_config_to_edit", lambda: link)
    answers(monkeypatch, MAX_ENTRIES=80)
    assert C.run_playbook_settings() == link
    assert link.is_symlink()
    assert yaml.safe_load(config_file.read_text())["PLAYBOOK"]["MAX_ENTRIES"] == 80


def test_features_enabling_local_learning_needs_no_model_or_wait_limit(config_file, monkeypatch):
    monkeypatch.setattr(C, "_is_tty", lambda: True)
    monkeypatch.setattr(C, "_dialog_checkbox",
                        lambda title, options, checked, **kw: [*checked, "ENABLE_PLAYBOOK"])
    model_prompt = Mock(side_effect=lambda text, *a, **kw: text)
    monkeypatch.setattr(C, "_prompt_model_section", model_prompt)
    written = C.run_features_override()
    assert written.path == config_file
    assert written.changed_features == frozenset({"ENABLE_PLAYBOOK"})
    data = yaml.safe_load(config_file.read_text())
    assert data["ENABLE_PLAYBOOK"] is True
    assert data["PLAYBOOK"]["REFLECTION_TIMEOUT"] == 30
    model_prompt.assert_not_called()


def test_full_config_enables_local_learning_without_offering_a_model(monkeypatch):
    seen = []
    monkeypatch.setattr(C, "_ask", lambda prompt, default=None, **kw: default or "fixture")
    monkeypatch.setattr(C, "_ask_number",
                        lambda prompt, default=None, **kw: None if kw.get("allow_none") else default)
    monkeypatch.setattr(C, "_ask_bool", lambda prompt, default=True, **kw: "playbook" in prompt.lower())
    monkeypatch.setattr(C, "_prompt_model_section",
                        lambda text, section, **kw: seen.append(section) or text)
    text = C._build_config("ollama", "fixture", CFG)
    assert "REFLECTOR" not in seen
    assert yaml.safe_load(text)["ENABLE_PLAYBOOK"] is True


def test_reflector_is_not_available_in_model_or_parameter_dialogs():
    assert all(section[0] != "REFLECTOR" for section in C._MODEL_SECTIONS.values())
    assert all(section[0] != "REFLECTOR" for section in C._PARAM_SECTIONS.values())
    assert "REFLECTION_TIMEOUT" not in C._PLAYBOOK_SETTINGS
    assert "SIMILARITY_THRESHOLD" not in C._PLAYBOOK_SETTINGS


def test_features_shows_local_learning_on_when_key_is_missing(config_file, monkeypatch):
    config_file.write_text("MODEL_ID:\n  NAME: fixture\n  TYPE: ollama\n")
    monkeypatch.setattr(C, "_is_tty", lambda: True)
    def choose(title, options, checked, **kwargs):
        assert "ENABLE_PLAYBOOK" in checked
        return [key for key in checked if key != "ENABLE_PLAYBOOK"]
    monkeypatch.setattr(C, "_dialog_checkbox", choose)
    C.run_features_override()
    assert yaml.safe_load(config_file.read_text())["ENABLE_PLAYBOOK"] is False


def test_scoped_config_can_disable_learning_without_editing_other_settings(config_file, monkeypatch):
    config_file.write_text(CFG.replace("ENABLE_PLAYBOOK: false", "ENABLE_PLAYBOOK: true"))
    answers(monkeypatch)
    monkeypatch.setattr(C, "_ask_bool", lambda *a, **kw: False)
    assert C.run_playbook_settings() == config_file
    assert yaml.safe_load(config_file.read_text())["ENABLE_PLAYBOOK"] is False


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
    client.playbook = SimpleNamespace(enabled=False, max_entries=500)
    client.refresh_playbook_context = Mock()
    original_model = client.model
    monkeypatch.setattr("mnemoai.client.client.config.reload", lambda: None)
    monkeypatch.setattr("mnemoai.client.client.config.get",
                        lambda key, default=None: {"MAX_ENTRIES": 80} if key == "PLAYBOOK" else default)
    assert client.reload_playbook_settings()
    assert client.model is original_model and client.agent.messages == ["history"]
    assert not client.playbook.enabled and client.playbook.max_entries == 80
    assert not hasattr(client.playbook, "similarity_threshold")


def test_disabling_does_not_require_valid_learning_settings(monkeypatch):
    client = LangGraphClient.__new__(LangGraphClient)
    client.playbook = SimpleNamespace(error="previous failure")
    client.reflector = object()
    client._playbook_error = "previous failure"
    client.refresh_playbook_context = Mock()
    monkeypatch.setattr("mnemoai.client.client.config.reload", lambda: None)
    values = {"ENABLE_PLAYBOOK": False, "PLAYBOOK": {"MAX_ENTRIES": "invalid"}}
    monkeypatch.setattr("mnemoai.client.client.config.get",
                        lambda key, default=None: values.get(key, default))
    assert client.reload_playbook_settings()
    assert client.playbook is None and client.reflector is None
    assert client._playbook_error is None
    client.refresh_playbook_context.assert_called_once()


def test_features_learning_only_change_applies_without_restart(monkeypatch):
    ui = ChatInterface.__new__(ChatInterface)
    ui.client = SimpleNamespace(reload_playbook_settings=Mock(return_value=True))
    ui._restart_in_place = Mock(side_effect=AssertionError("must not restart"))
    monkeypatch.setattr("mnemoai.client.ui.chat_interface.run_features_override",
                        lambda: C.FeatureOverride(Path("/cfg"), frozenset({"ENABLE_PLAYBOOK"})))
    ui._dispatch("/features")
    ui.client.reload_playbook_settings.assert_called_once()
    ui._restart_in_place.assert_not_called()
