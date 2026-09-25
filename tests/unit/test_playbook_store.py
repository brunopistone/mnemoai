"""Evidence identity, preservation, user edits, and concurrent playbook updates."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from mnemoai.client.memory.playbook_records import PlaybookEntry
from mnemoai.client.memory.playbook_store import PlaybookStore


def lesson(strategy="read before editing", **kwargs):
    return PlaybookEntry("editing Python", strategy, "test", **kwargs)


def test_legacy_migration_preserves_original_and_stable_ids(tmp_path):
    path = tmp_path / "playbook.json"
    original = '[{"context":"c","strategy":"s","custom":{"keep":1},"confidence":0.9}]'
    path.write_text(original)
    first = PlaybookStore(str(tmp_path)).snapshot()
    second = PlaybookStore(str(tmp_path)).snapshot()
    assert first == second
    assert first[0]["custom"] == {"keep": 1}
    assert first[0]["provenance"] == "legacy"
    assert first[0]["source_refs"] == []
    backups = list(tmp_path.glob("playbook.json.backup-*"))
    assert len(backups) == 1 and backups[0].read_text() == original


def test_legacy_confidence_is_not_mistaken_for_negative_user_feedback(tmp_path):
    (tmp_path / "playbook.json").write_text(json.dumps([
        {"context": "editing", "strategy": "legacy note", "confidence": 0.01},
    ]))
    store = PlaybookStore(str(tmp_path))
    assert store.get_relevant_entries("editing")


@pytest.mark.parametrize("payload", ['{"not":"a list"}', '[{"strategy":"missing context"}]', "{"])
def test_corrupt_file_cannot_be_overwritten_by_learning(tmp_path, payload):
    path = tmp_path / "playbook.json"
    path.write_text(payload)
    store = PlaybookStore(str(tmp_path))
    assert store.get_relevant_entries("") == []
    with pytest.raises(ValueError):
        store.append(lesson())
    assert path.read_text() == payload


def test_lock_permission_failure_disables_recall_without_losing_data(tmp_path, monkeypatch):
    store = PlaybookStore(str(tmp_path))
    store.append(lesson())
    before = (tmp_path / "playbook.json").read_bytes()
    def denied(*args):
        raise PermissionError("lock inaccessible")
    with monkeypatch.context() as m:
        m.setattr("mnemoai.client.memory.playbook_store.file_lock", denied)
        unavailable = PlaybookStore(str(tmp_path))
        assert unavailable.error
        assert store.get_relevant_entries("editing") == []
        assert store.error and store.entries == []
        with pytest.raises(PermissionError):
            store.append(lesson("another"))
    assert (tmp_path / "playbook.json").read_bytes() == before
    assert store.get_relevant_entries("editing")  # recovers after permissions return
    assert store.error is None


def test_evidence_merges_without_inventing_usefulness_or_reenabling(tmp_path):
    store = PlaybookStore(str(tmp_path))
    store.append(lesson(source_refs=[{"tool_call_id": "1"}], provenance="model"))
    first = store.snapshot()[0]
    store.update(first["id"], first["revision"], action="disable")
    store.append(lesson(source_refs=[{"tool_call_id": "2"}], provenance="model"))
    store.append(lesson(source_refs=[{"tool_call_id": "2"}], provenance="model"))
    current = store.snapshot()[0]
    assert current["id"] == first["id"]
    assert current["source_refs"] == [{"tool_call_id": "1"}, {"tool_call_id": "2"}]
    assert current["status"] == "disabled"
    assert current["confidence"] == first["confidence"]
    assert current["helpful_count"] == current["injection_count"] == 0
    assert store.get_relevant_entries("editing") == []
    assert store.format_for_prompt([current]) == ""


def test_scope_and_applicability_are_part_of_identity(tmp_path, monkeypatch):
    store = PlaybookStore(str(tmp_path))
    store.append(lesson(scope=str(tmp_path)))
    store.append(lesson(scope="/different/project"))
    assert len(store.snapshot()) == 2
    monkeypatch.chdir(tmp_path)
    assert len(store.get_relevant_entries("editing")) == 1


def test_edit_is_revision_checked_and_preserves_old_text(tmp_path):
    store = PlaybookStore(str(tmp_path))
    store.append(lesson())
    old = store.snapshot()[0]
    new = store.update(old["id"], old["revision"], action="edit",
                       context="editing", strategy="read the exact target first")
    assert new["id"] == old["id"] and new["revision"] == old["revision"] + 1
    assert new["history"][0]["strategy"] == old["strategy"]
    with pytest.raises(ValueError, match="changed"):
        store.update(old["id"], old["revision"], action="disable")
    store.append(lesson())  # must not recreate wording the user replaced
    assert len(store.snapshot()) == 1


def test_exposure_outcomes_and_feedback_are_separate(tmp_path):
    store = PlaybookStore(str(tmp_path))
    store.append(lesson())
    entry = store.snapshot()[0]
    store.record_exposure([entry["id"]])
    store.record_exposure([entry["id"]], success=True)
    same = store.snapshot()[0]
    assert same["injection_count"] == same["observed_success_count"] == 1
    assert same["confidence"] == entry["confidence"] and same["helpful_count"] == 0
    for _ in range(4):
        entry = store.snapshot()[0]
        store.update(entry["id"], entry["revision"], action="unhelpful")
    assert store.get_relevant_entries("editing") == []
    entry = store.snapshot()[0]
    assert entry["status"] == "active"  # dormant, not deleted
    store.update(entry["id"], entry["revision"], action="helpful")
    assert store.get_relevant_entries("editing")


def test_refinement_archives_without_losing_records(tmp_path):
    store = PlaybookStore(str(tmp_path), max_entries=1)
    store.append(lesson("first"))
    store.append(lesson("second"))
    entries = store.snapshot()
    assert len(entries) == 2
    assert {e["status"] for e in entries} == {"active", "archived"}
    archived = next(e for e in entries if e["status"] == "archived")
    store.update(archived["id"], archived["revision"], action="restore")
    assert any(e["id"] == archived["id"] for e in store.get_relevant_entries(""))


def test_two_instances_do_not_lose_each_others_changes(tmp_path):
    stores = [PlaybookStore(str(tmp_path)), PlaybookStore(str(tmp_path))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: stores[i % 2].append(lesson(str(i))), range(20)))
    assert len(stores[0].snapshot()) == 20


def test_failed_atomic_write_keeps_disk_and_memory_intact(tmp_path, monkeypatch):
    store = PlaybookStore(str(tmp_path))
    store.append(lesson())
    entry = store.snapshot()[0]
    before = (tmp_path / "playbook.json").read_bytes()
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr("mnemoai.client.memory.playbook_store.atomic_write_json", fail)
    with pytest.raises(OSError):
        store.update(entry["id"], entry["revision"], action="disable")
    assert store.entries[0]["status"] == "active"
    assert (tmp_path / "playbook.json").read_bytes() == before


def test_clear_keeps_backup_and_rejects_stale_preview(tmp_path):
    store = PlaybookStore(str(tmp_path))
    store.append(lesson())
    preview = [(e["id"], e["revision"]) for e in store.snapshot()]
    store.append(lesson("new"))
    with pytest.raises(ValueError):
        store.clear(expected=preview)
    store.clear()
    assert store.snapshot() == []
    backup = next(tmp_path.glob("playbook.json.backup-*"))
    assert len(json.loads(backup.read_text())) == 2
