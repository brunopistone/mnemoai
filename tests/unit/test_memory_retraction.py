"""Explicit withdrawal, durable quarantine, and conservative handling of shared evidence."""

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from mnemoai.client import learned
from mnemoai.client.memory import retraction
from mnemoai.client.memory.playbook_records import PlaybookEntry
from mnemoai.client.memory.playbook_store import PlaybookStore
from mnemoai.client.memory.reflector import Reflector


def ref(call="a", session="session", turn=1):
    return {"session_id": session, "turn": turn, "tool_call_id": call, "tool": "file_edit"}


def note(strategy="Use a scoped retry", refs=None, scope=""):
    return PlaybookEntry(
        "editing", strategy, "fixture", source_refs=refs or [ref()],
        scope=scope, provenance="model",
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = PlaybookStore(str(tmp_path / "playbook"))
    result.append(note(scope=str(tmp_path)))
    return result


def apply(store, entry, action="retract", reason="The premise was incorrect"):
    plan = store.preview_retraction(entry["id"], action, reason)
    return store.apply_retraction(entry["id"], action=action, reason=reason, token=plan["token"])


def test_observation_identity_is_stable_without_optional_metadata():
    assert retraction.source_id(ref()) == retraction.source_id({**ref(), "result_sha256": "f" * 64})
    assert retraction.source_id(ref("b")) != retraction.source_id(ref())
    assert retraction.source_id(ref(session="another")) != retraction.source_id(ref())
    assert retraction.source_id({"tool_call_id": "a"}) is None
    assert retraction.source_id(ref(turn=True)) is None
    fallback = {**ref(), "tool_call_id": "", "args_sha256": "a" * 64, "result_sha256": "b" * 64}
    assert retraction.source_id(fallback)
    assert retraction.source_id({**fallback, "args_sha256": "invalid"}) is None


def test_preview_does_not_mutate_and_does_not_invent_dependencies(store):
    target = store.snapshot()[0]
    store.append(note("Another interpretation", refs=[ref(), ref("independent")], scope=target["scope"]))
    store.append(note("Unrelated lesson", refs=[ref("other")], scope=target["scope"]))
    before = Path(store.playbook_file).read_bytes()
    plan = store.preview_retraction(target["id"])
    assert plan["newly_quarantined"] == 1 and len(plan["related"]) == 1
    assert plan["other_entries_unchanged"] == 2
    assert Path(store.playbook_file).read_bytes() == before
    message = learned.render_impact(plan)
    assert "NOT a proven dependency" in message
    assert "episodic recall" in message and "permissions" in message


def test_retraction_survives_reload_and_preserves_other_records(store):
    target = store.snapshot()[0]
    store.append(note("Independent support", refs=[ref("independent")], scope=target["scope"]))
    independent = store.snapshot()[1]
    result = apply(store, target)
    reopened = PlaybookStore(store.persist_path)
    assert reopened.snapshot()[0] == result
    assert result["status"] == "retracted" and result["id"] == target["id"]
    assert result["history"][-1]["reason"] == "The premise was incorrect"
    assert result["source_refs"] == target["source_refs"]
    assert reopened.snapshot()[1] == independent
    assert [e["id"] for e in reopened.get_relevant_entries("editing")] == [independent["id"]]
    # A caller holding the old snapshot must not re-inject it.
    assert reopened.format_for_prompt([target]) == ""
    backups = list(Path(store.persist_path).glob("playbook.json.backup-*"))
    assert any(json.loads(p.read_text())[0]["status"] == "active" for p in backups)


def test_reextraction_cannot_resurrect_a_paraphrase_from_quarantined_evidence(store):
    target = store.snapshot()[0]
    apply(store, target)
    store.append(note("A paraphrase", scope=target["scope"]))
    store.append(note("Mixed evidence is not proof of independence",
                      refs=[ref(), ref("new")], scope=target["scope"]))
    store.append(note("Independent new lesson", refs=[ref("new")], scope=target["scope"]))
    records = store.snapshot()
    assert [e["strategy"] for e in records] == [target["strategy"], "Independent new lesson"]
    assert store.filter_learning_evidence([{"ref": ref()}, {"ref": ref("new")}], target["scope"]) == [
        {"ref": ref("new")},
    ]


def test_new_evidence_cannot_silently_restore_the_exact_retracted_note(store):
    target = store.snapshot()[0]
    withdrawn = apply(store, target)
    store.append(note(target["strategy"], refs=[ref("new")], scope=target["scope"]))
    assert store.snapshot() == [withdrawn]


def test_scope_boundaries_and_unknown_legacy_lineage_are_explicit(store):
    target = store.snapshot()[0]
    apply(store, target)
    store.append(note("Other project", scope="/unrelated-project"))
    assert len(store.snapshot()) == 2
    legacy = PlaybookEntry("legacy", "Unknown old note", "legacy", source_refs=[])
    store.append(legacy)
    old = store.snapshot()[-1]
    plan = store.preview_retraction(old["id"])
    assert plan["lineage_unknown"] and plan["source_ids"] == []
    apply(store, old)
    assert store.filter_learning_evidence([{"ref": ref("new")}], target["scope"])


def test_restore_returns_previous_state_and_keeps_the_audit_record(store):
    target = store.snapshot()[0]
    disabled = store.update(target["id"], target["revision"], action="disable")
    withdrawn = apply(store, disabled)
    restored = apply(store, withdrawn, "restore", "The withdrawal was mistaken")
    assert restored["status"] == "disabled"
    assert restored["retraction"]["restore_reason"] == "The withdrawal was mistaken"
    assert restored["history"][-1]["action"] == "restore_retraction"
    assert restored["history"][-1]["retraction"]["reason"] == "The premise was incorrect"
    assert store.filter_learning_evidence([{"ref": ref()}], target["scope"])
    active = store.update(restored["id"], restored["revision"], action="restore")
    assert active["status"] == "active"


def test_another_retraction_can_keep_evidence_quarantined_after_restore(store):
    first = store.snapshot()[0]
    store.append(note("Another note", scope=first["scope"]))
    second = store.snapshot()[1]
    first = apply(store, first)
    apply(store, second)
    plan = store.preview_retraction(first["id"], "restore", "Reconsidered")
    assert plan["released_sources"] == 0 and plan["sources_still_quarantined"] == 1
    apply(store, first, "restore", "Reconsidered")
    assert store.filter_learning_evidence([{"ref": ref()}], first["scope"]) == []


def test_stale_plan_is_refused_but_exposure_counters_do_not_invalidate_it(store):
    target = store.snapshot()[0]
    plan = store.preview_retraction(target["id"], "retract", "Incorrect")
    store.record_exposure([target["id"]])
    store.apply_retraction(target["id"], action="retract", reason="Incorrect", token=plan["token"])
    plan = store.preview_retraction(target["id"], "restore", "Reconsidered")
    another = PlaybookStore(store.persist_path)
    another.append(note("New independent note", refs=[ref("new")], scope=target["scope"]))
    with pytest.raises(ValueError, match="changed"):
        store.apply_retraction(target["id"], action="restore", reason="Reconsidered", token=plan["token"])
    assert store.snapshot()[0]["status"] == "retracted"


@pytest.mark.parametrize("action", ["edit", "disable", "restore", "helpful", "unhelpful"])
def test_old_update_api_cannot_bypass_a_retraction(store, action):
    target = apply(store, store.snapshot()[0])
    with pytest.raises(ValueError, match="preview"):
        store.update(target["id"], target["revision"], action=action, context="c", strategy="s")
    assert store.snapshot()[0] == target


@pytest.mark.parametrize("reason", ["", " " * 3, "x" * 501, "\x1b[2J"])
def test_invalid_reason_never_mutates(store, reason):
    target = store.snapshot()[0]
    with pytest.raises(ValueError):
        store.apply_retraction(target["id"], action="retract", reason=reason, token="invalid")
    assert store.snapshot() == [target]


def test_failed_write_leaves_original_state_and_quarantine_unchanged(store, monkeypatch):
    target = store.snapshot()[0]
    plan = store.preview_retraction(target["id"], "retract", "Incorrect")
    def fail(*args):
        raise OSError("disk full")
    with monkeypatch.context() as m:
        m.setattr("mnemoai.client.memory.playbook_store.atomic_write_json", fail)
        with pytest.raises(OSError):
            store.apply_retraction(target["id"], action="retract", reason="Incorrect", token=plan["token"])
        assert store.entries == [target]
    assert store.snapshot() == [target]
    assert store.filter_learning_evidence([{"ref": ref()}], target["scope"])


@pytest.mark.parametrize("field,value", [
    ("reason", ""), ("source_ids", []), ("scope", "/wrong"),
    ("previous_status", "retracted"), ("actor", "model"),
])
def test_corrupt_retraction_metadata_fails_closed_without_rewriting(store, field, value):
    entry = apply(store, store.snapshot()[0])
    entry["retraction"][field] = value
    payload = json.dumps([entry])
    path = Path(store.playbook_file)
    path.write_text(payload)
    reopened = PlaybookStore(store.persist_path)
    assert reopened.error and reopened.get_relevant_entries("") == []
    assert path.read_text() == payload


def test_cancelled_confirmation_and_clear_warning(store):
    client = SimpleNamespace(playbook=store, refresh_playbook_context=lambda: None)
    target = store.snapshot()[0]
    prompts = []
    def decline(prompt):
        prompts.append(prompt)
        return False
    result = learned.run(client, f"retract {target['id']} Incorrect premise",
                         confirm=decline, edit=lambda x: x)
    assert "Cancelled" in result and store.snapshot()[0] == target
    assert "Quarantine 1" in prompts[0]
    apply(store, target)
    learned.run(client, "clear", confirm=decline, edit=lambda x: x)
    assert "learning quarantines" in prompts[-1]
    store.clear()
    assert store.snapshot() == []
    assert store.filter_learning_evidence([{"ref": ref()}], target["scope"])


def test_cli_retraction_and_reasoned_restore(store):
    client = SimpleNamespace(playbook=store, refresh_playbook_context=lambda: None)
    target = store.snapshot()[0]
    assert "Quarantine 1" in learned.run(
        client, f"preview {target['id']}", confirm=lambda _: pytest.fail("read-only preview"),
        edit=lambda x: x,
    )
    assert "retracted" in learned.run(
        client, f"retract {target['id']} Incorrect premise", confirm=lambda _: True, edit=lambda x: x,
    )
    assert target["id"] in learned.run(client, "retracted", confirm=lambda _: False, edit=lambda x: x)
    assert "required" in learned.run(
        client, f"restore {target['id']}", confirm=lambda _: pytest.fail("missing reason"), edit=lambda x: x,
    )
    assert "active" in learned.run(
        client, f"restore {target['id']} Withdrawal was mistaken", confirm=lambda _: True, edit=lambda x: x,
    )
    assert store.snapshot()[0]["retraction"]["restore_reason"] == "Withdrawal was mistaken"


@pytest.mark.parametrize("reason", [
    "The provider's failure was temporary.",
    "The provider's failure isn't permanent.",
    '"The provider\'s failure isn\'t permanent."',
    "'The failure was temporary.'",
])
def test_cli_reasons_preserve_prose_including_apostrophes(store, reason):
    client = SimpleNamespace(playbook=store, refresh_playbook_context=lambda: None)
    target = store.snapshot()[0]
    expected = reason[1:-1] if reason[0] == reason[-1] and reason[0] in {"'", '"'} else reason
    for action, key in [("retract", "reason"), ("restore", "restore_reason")]:
        result = learned.run(
            client, f"{action} {target['id']} {reason}",
            confirm=lambda _: True, edit=lambda x: x,
        )
        assert "Updated" in result
        assert store.snapshot()[0]["retraction"][key] == expected


@pytest.mark.parametrize("phase,expected_status,code", [
    ("before", "active", 71), ("after", "retracted", 72),
])
def test_abrupt_exit_never_splits_retraction_from_its_quarantine(store, phase, expected_status, code):
    script = """
import os, sys
from mnemoai.client.memory.playbook_store import PlaybookStore
from mnemoai.utils import atomic_write
store = PlaybookStore(sys.argv[1])
entry = store.snapshot()[0]
plan = store.preview_retraction(entry["id"], "retract", "Crash fixture")
replace = atomic_write.os.replace
def crash(src, dst):
    if str(dst) == store.playbook_file:
        if sys.argv[2] == "before":
            os._exit(71)
        replace(src, dst)
        os._exit(72)
    return replace(src, dst)
atomic_write.os.replace = crash
store.apply_retraction(entry["id"], action="retract", reason="Crash fixture", token=plan["token"])
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    result = subprocess.run(
        [sys.executable, "-c", script, store.persist_path, phase],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == code, result.stderr
    reopened = PlaybookStore(store.persist_path)
    entry = reopened.snapshot()[0]
    assert entry["status"] == expected_status
    remaining = reopened.filter_learning_evidence([{"ref": ref()}], entry["scope"])
    assert bool(remaining) == (phase == "before")


def test_late_model_result_is_rechecked_when_committing(store):
    target = store.snapshot()[0]
    entered, released = threading.Event(), threading.Event()
    class Model:
        def invoke(self, *args, **kwargs):
            entered.set()
            released.wait(3)
            return SimpleNamespace(content=json.dumps({"lessons": [{
                "context": "editing", "strategy": "A late paraphrase", "evidence_ids": [0],
            }]}))
    trace = [
        SimpleNamespace(type="human", content="task"),
        SimpleNamespace(type="ai", tool_calls=[{"name": "file_edit", "args": {}, "id": "a"}]),
        SimpleNamespace(type="tool", tool_call_id="a", content="Error: target changed"),
    ]
    reflector = Reflector()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            reflector.reflect_on_trajectory, trace, "task", model=Model(),
            source={"session_id": "session", "turn": 1}, scope=target["scope"],
            evidence_filter=lambda evidence: store.filter_learning_evidence(evidence, target["scope"]),
        )
        try:
            assert entered.wait(2)
            apply(PlaybookStore(store.persist_path), target)
        finally:
            released.set()
        entries = future.result(timeout=5)
    assert entries
    store.append_batch(entries)
    assert len(store.snapshot()) == 1 and store.snapshot()[0]["status"] == "retracted"
    def forbidden(*args, **kwargs):
        pytest.fail("quarantined evidence must not trigger a model call")
    assert reflector.reflect_on_trajectory(
        trace, "task", model=SimpleNamespace(invoke=forbidden),
        source={"session_id": "session", "turn": 1}, scope=target["scope"],
        evidence_filter=lambda evidence: store.filter_learning_evidence(evidence, target["scope"]),
    ) == []
