"""Every reviewer setting has a working interactive path and safe defaults."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from mnemoai.client import review
from mnemoai.client.client import LangGraphClient
from mnemoai.client.ui.chat_interface import ChatInterface
from mnemoai.client.user_commands import UserCommandStore
from mnemoai.models import area_models
from mnemoai.utils import configurator as C

CFG = """\
MODEL_ID:
  NAME: main-fixture
  TYPE: ollama
ENABLE_REVIEW: false
REVIEW:
  TIMEOUT: 45 # preserve comment
  MAX_INPUT_TOKENS: 6000
  MAX_ROUNDS: 2
  TOTAL_TIMEOUT: 180
  CUSTOM:
    TIMEOUT: 900
"""


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(CFG)
    monkeypatch.setattr(C, "_config_to_edit", lambda: path)
    return path


def answers(monkeypatch, **values):
    def answer(label, default=None, **kwargs):
        for key, spec in C._REVIEW_SETTINGS.items():
            if label.startswith(spec[0]):
                return str(values.get(key, default))
        return default
    monkeypatch.setattr(C, "_ask_number", answer)


def test_scoped_settings_preserve_everything_else(config_file, monkeypatch):
    answers(monkeypatch, TIMEOUT=60, MAX_INPUT_TOKENS=8000)
    assert C.run_review_settings() == config_file
    data = yaml.safe_load(config_file.read_text())
    assert data["REVIEW"] == {"TIMEOUT": 60, "MAX_INPUT_TOKENS": 8000,
                              "MAX_ROUNDS": 2, "TOTAL_TIMEOUT": 180, "CUSTOM": {"TIMEOUT": 900}}
    assert data["MODEL_ID"] == yaml.safe_load(CFG)["MODEL_ID"]
    assert data["ENABLE_REVIEW"] is False
    assert "# preserve comment" in config_file.read_text()


def test_cancel_unchanged_and_concurrent_edits_do_not_overwrite(config_file, monkeypatch):
    answers(monkeypatch)
    assert C.run_review_settings() is None
    monkeypatch.setattr(C, "_ask_number", Mock(side_effect=C._Cancelled()))
    assert C.run_review_settings() is None and config_file.read_text() == CFG
    def changed(*args, **kwargs):
        config_file.write_text(CFG + "USER_EDIT: true\n")
        return CFG.replace("TIMEOUT: 45", "TIMEOUT: 60")
    monkeypatch.setattr(C, "_prompt_review_settings", changed)
    assert C.run_review_settings() is None
    assert "USER_EDIT: true" in config_file.read_text()


@pytest.mark.parametrize("section", ["", "REVIEW: {}", "REVIEW: {TIMEOUT: 45, CUSTOM: yes}"])
def test_missing_and_inline_sections_are_editable(section, monkeypatch):
    answers(monkeypatch, TIMEOUT=60)
    result = C._prompt_review_settings("MODEL_ID:\n  NAME: fixture\n" + section + "\n")
    assert yaml.safe_load(result)["REVIEW"]["TIMEOUT"] == 60
    if "CUSTOM" in section:
        assert yaml.safe_load(result)["REVIEW"]["CUSTOM"] is True


def test_features_enabling_review_offers_model_and_limits(config_file, monkeypatch):
    monkeypatch.setattr(C, "_is_tty", lambda: True)
    monkeypatch.setattr(C, "_dialog_checkbox", lambda *a, **kw: ["ENABLE_REVIEW"])
    offered = Mock(side_effect=lambda text, *a, **kw: text)
    monkeypatch.setattr(C, "_prompt_model_section", offered)
    answers(monkeypatch, TIMEOUT=60)
    assert C.run_features_override() == config_file
    assert offered.call_args.args[1] == "REVIEWER"
    assert yaml.safe_load(config_file.read_text())["ENABLE_REVIEW"] is True


def test_full_setup_offers_review_when_selected(monkeypatch):
    seen = []
    monkeypatch.setattr(C, "_ask", lambda prompt, default=None, **kw: default or "fixture")
    monkeypatch.setattr(C, "_ask_number",
                        lambda *a, **kw: None if kw.get("allow_none") else kw.get("default"))
    monkeypatch.setattr(C, "_ask_bool", lambda prompt, *args, **kw: "peer review" in prompt)
    monkeypatch.setattr(C, "_prompt_model_section",
                        lambda text, section, **kw: seen.append(section) or text)
    data = yaml.safe_load(C._build_config("ollama", "fixture", CFG))
    assert "REVIEWER" in seen and data["ENABLE_REVIEW"] is True


def test_model_picker_wires_reviewer_and_feature(config_file, monkeypatch):
    monkeypatch.setattr(C, "_is_tty", lambda: False)
    monkeypatch.setattr(C, "_ask_choice", lambda *a, **kw: "8")
    monkeypatch.setattr(C, "_ask_bool", lambda prompt, **kw: "same model" not in prompt)
    monkeypatch.setattr(C, "_prompt_provider_type", lambda *a: "ollama")
    monkeypatch.setattr(C, "_ask", lambda prompt, default=None, **kw:
                        "review-fixture" if prompt == "Model name" else default or "")
    monkeypatch.setattr(C, "_ask_number", lambda *a, **kw: None)
    result = C.run_model_override()
    data = yaml.safe_load(config_file.read_text())
    assert result.section == "REVIEWER" and result.enabled_feature
    assert data["AREA_MODELS"]["REVIEWER"]["NAME"] == "review-fixture"
    assert data["ENABLE_REVIEW"] is True


def test_params_changes_only_the_reviewer(config_file, monkeypatch):
    config_file.write_text(CFG + "AREA_MODELS:\n  REVIEWER: review-fixture\n")
    monkeypatch.setattr(C, "_is_tty", lambda: False)
    monkeypatch.setattr(C, "_ask_choice", lambda *a, **kw: "8")
    monkeypatch.setattr(C, "_prompt_one_param", lambda text, section, key, *args:
                        C._set_field(text, section, key, "0.2") if key == "TEMPERATURE" else text)
    assert C.run_params_override() == config_file
    data = yaml.safe_load(config_file.read_text())
    assert data["AREA_MODELS"]["REVIEWER"]["TEMPERATURE"] == 0.2
    assert "TEMPERATURE" not in data["MODEL_ID"]


def test_dispatch_controls_and_scoped_reload_do_not_restart(monkeypatch):
    client = SimpleNamespace(reviewer=review.Reviewer(), reload_review_settings=Mock(return_value=True))
    ui = ChatInterface.__new__(ChatInterface)
    ui.client = client
    ui._restart_in_place = Mock(side_effect=AssertionError("must not restart"))
    monkeypatch.setattr("mnemoai.client.ui.chat_interface.run_review_settings",
                        lambda: Path("/temporary/config.yaml"))
    ui._dispatch("/review on")
    assert client.reviewer.enabled
    ui._dispatch("/config review")
    client.reload_review_settings.assert_called_once()
    ui._dispatch("/review off")
    assert not client.reviewer.enabled


@pytest.mark.parametrize("filename", ["review.md", "Review.md"])
def test_legacy_review_macro_keeps_priority_and_controls_have_an_alias(tmp_path, monkeypatch, filename):
    (tmp_path / filename).write_text("Legacy review: $ARGUMENTS")
    ui = ChatInterface.__new__(ChatInterface)
    ui._user_commands = UserCommandStore(root=tmp_path)
    ui.client = SimpleNamespace(
        reviewer=review.Reviewer(), query=Mock(return_value="answer"),
        episodic_memory=None, reflector=None, agent=SimpleNamespace(files=None),
    )
    ui._wrap_up_spinner = lambda on: None
    ui._files_mark = lambda: 0
    ui._turn_end_line = lambda *a, **kw: "done"
    ui._expand_mentions = lambda value: value
    monkeypatch.setattr("mnemoai.client.ui.chat_interface.notify.notify_turn_end", lambda *a: None)
    ui._dispatch("/review on")
    ui.client.query.assert_called_once_with("Legacy review: on")
    assert not ui.client.reviewer.enabled
    ui._dispatch("/config review on")
    assert ui.client.reviewer.enabled
    assert sum(name.lower() == "/review" for name, _ in ui._completion_commands()) == 1


def test_reload_keeps_actor_model_history_and_session_toggle(monkeypatch):
    client = LangGraphClient.__new__(LangGraphClient)
    client.model, client.agent = object(), SimpleNamespace(messages=["history"])
    client.reviewer = review.Reviewer(enabled=True)
    monkeypatch.setattr(review.config, "reload", lambda: None)
    monkeypatch.setitem(review.config._config_data, "REVIEW", {"TIMEOUT": 60})
    assert client.reload_review_settings()
    assert client.reviewer.enabled and client.agent.messages == ["history"]


def test_enabling_reviewer_in_model_picker_applies_without_restart(monkeypatch):
    ui = ChatInterface.__new__(ChatInterface)
    ui.client = SimpleNamespace(reviewer=review.Reviewer(), reload_area_models=Mock(return_value=True))
    ui._restart_in_place = Mock(side_effect=AssertionError("review activation needs no restart"))
    monkeypatch.setattr("mnemoai.client.ui.chat_interface.run_model_override",
                        lambda: SimpleNamespace(section="REVIEWER", enabled_feature=True))
    monkeypatch.setattr("mnemoai.client.ui.chat_interface.render_model_update",
                        lambda *a, **kw: "applied in place")
    ui._dispatch("/model")
    assert ui.client.reviewer.enabled
    ui.client.reload_area_models.assert_called_once()


def test_unconfigured_reviewer_gets_an_isolated_bounded_model(monkeypatch):
    client = LangGraphClient.__new__(LangGraphClient)
    client._area_model_cache = {}
    client.llm_controller = SimpleNamespace(build_model_variant=Mock(return_value="REVIEW"),
                                            model_name="main")
    monkeypatch.setitem(area_models.config._config_data, "AREA_MODELS", {})
    assert client._area_model("REVIEWER") == "REVIEW"
    client.llm_controller.build_model_variant.assert_called_once_with(
        {"MAX_TOKENS": 2048}, callbacks=[], non_reasoning=True,
    )


def test_explicit_reviewer_failure_cannot_fall_back_to_actor(monkeypatch):
    client = LangGraphClient.__new__(LangGraphClient)
    client._area_model_cache, client.model = {}, "ACTOR"
    client.llm_controller = SimpleNamespace(build_model_variant=Mock(side_effect=RuntimeError("offline")))
    monkeypatch.setitem(area_models.config._config_data, "AREA_MODELS", {"REVIEWER": "side-model"})
    assert client._area_model("REVIEWER") is None
    assert client._area_model("REVIEWER") is None
    client.llm_controller.build_model_variant.assert_called_once()


@pytest.mark.parametrize("value", [False, [], 3, {"NAME": False}, {"TYPE": []}])
def test_invalid_explicit_override_is_not_treated_as_no_override(value, monkeypatch):
    monkeypatch.setitem(area_models.config._config_data, "AREA_MODELS", {"REVIEWER": value})
    with pytest.raises(ValueError):
        area_models.reviewer_overrides()


def test_every_template_exposes_the_same_off_by_default_limits():
    root = Path(__file__).resolve().parents[2] / "src/mnemoai/utils"
    templates = list(root.glob("config.yaml*.example"))
    assert len(templates) == 4
    for path in templates:
        data = yaml.safe_load(path.read_text())
        assert data["ENABLE_REVIEW"] is False
        assert data["REVIEW"] == {k: spec[1] for k, spec in review.SETTINGS.items()}
        assert "#   REVIEWER:" in path.read_text()
