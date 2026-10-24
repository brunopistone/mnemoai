"""Configuration dialogs must not discard concurrent edits or tear live YAML."""

import threading

import pytest

from mnemoai.utils import configurator as C

BASE = "MODEL_ID:\n  NAME: original\n  TYPE: ollama\nENABLE_PLAYBOOK: true\n"


def test_features_preserves_a_model_change_made_while_its_dialog_is_open(tmp_path, monkeypatch):
    target = tmp_path / "config.yaml"
    target.write_text(BASE)
    monkeypatch.setattr(C, "_config_to_edit", lambda: target)
    monkeypatch.setattr(C, "_is_tty", lambda: True)
    concurrent = BASE.replace("original", "new-model")
    def choose(title, options, checked, **kwargs):
        target.write_text(concurrent)
        return [key for key in checked if key != "ENABLE_PLAYBOOK"]
    monkeypatch.setattr(C, "_dialog_checkbox", choose)
    assert C.run_features_override() is None
    assert target.read_text() == concurrent


@pytest.mark.parametrize("command,prompt", [
    ("run_params_override", "_prompt_inference_params"),
    ("run_model_override", "_prompt_model_section"),
])
def test_model_dialogs_reject_concurrent_changes(tmp_path, monkeypatch, command, prompt):
    target = tmp_path / "config.yaml"
    target.write_text(BASE)
    monkeypatch.setattr(C, "_config_to_edit", lambda: target)
    monkeypatch.setattr(C, "_is_tty", lambda: True)
    monkeypatch.setattr(C, "_ask_choice", lambda *a, **kw: "1")
    concurrent = BASE + "KEEP_EXTERNAL_CHANGE: true\n"
    def change(text, *args, **kwargs):
        target.write_text(concurrent)
        return text.replace("original", "dialog-model")
    monkeypatch.setattr(C, prompt, change)
    assert getattr(C, command)() is None
    assert target.read_text() == concurrent


def test_two_dialog_commits_cannot_both_publish_the_same_old_snapshot(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text(BASE)
    start = threading.Barrier(2)
    results = []
    def save(name):
        start.wait()
        results.append(C._save_config_update(target, BASE, BASE.replace("original", name)))
    threads = [threading.Thread(target=save, args=(name,)) for name in ("one", "two")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert sorted(results) == [False, True]


def test_invalid_yaml_is_not_published(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text(BASE)
    assert not C._save_config_update(target, BASE, "MODEL_ID: [")
    assert target.read_text() == BASE
