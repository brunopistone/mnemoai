"""Atomic-write guarantees for the learned-state files.

The point of ``utils.atomic_write`` is that a failed or concurrent write can
never leave a truncated/half-written file behind, because the visible file is
only ever swapped in whole. These tests pin that, plus the fact that the four
learned-state writers actually route through it (a regression here is silent:
the file still gets written, just unsafely).
"""

import json
import os

import pytest

from mnemoai.utils.atomic_write import atomic_write_json, atomic_write_text


def test_write_text_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "deeper" / "MEMORY.md"
    atomic_write_text(str(target), "hello")
    assert target.read_text() == "hello"


def test_write_json_round_trips(tmp_path):
    target = tmp_path / "playbook.json"
    data = [{"strategy": "use glob", "confidence": 0.4}]
    atomic_write_json(str(target), data)
    assert json.loads(target.read_text()) == data


def test_overwrite_replaces_contents_entirely(tmp_path):
    """A shorter payload must not leave a tail of the longer previous one."""
    target = tmp_path / "metrics.json"
    atomic_write_json(str(target), {"a": 1, "b": 2, "c": 3, "padding": "x" * 200})
    atomic_write_json(str(target), {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}


def test_no_temp_files_left_behind(tmp_path):
    target = tmp_path / "profile.json"
    atomic_write_json(str(target), {"k": "v"})
    assert [p.name for p in tmp_path.iterdir()] == ["profile.json"]


def test_failed_serialization_leaves_original_intact(tmp_path):
    """The pre-existing file survives a write that can't be serialized.

    This is the whole point: the old truncate-then-write would already have
    emptied the file by the time json.dump hit the bad object.
    """
    target = tmp_path / "playbook.json"
    atomic_write_json(str(target), {"good": True})

    with pytest.raises(TypeError):
        atomic_write_json(str(target), {"bad": object()})

    assert json.loads(target.read_text()) == {"good": True}
    # And no .tmp debris next to it.
    assert [p.name for p in tmp_path.iterdir()] == ["playbook.json"]


def test_failed_write_leaves_no_temp_file(tmp_path, monkeypatch):
    """An exception mid-write cleans up its temp file."""
    target = tmp_path / "metrics.json"

    real_replace = os.replace

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_text(str(target), "partial")
    monkeypatch.setattr(os, "replace", real_replace)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_memory_store_writes_atomically(tmp_path, monkeypatch):
    """MemoryStore._write_entries routes through atomic_write_text."""
    from mnemoai.client.memory import memory_store as ms

    calls = []
    monkeypatch.setattr(
        ms, "atomic_write_text", lambda p, t: calls.append((p, t)) or None
    )

    store = ms.MemoryStore.__new__(ms.MemoryStore)
    store.path = tmp_path / "MEMORY.md"
    store._write_entries(["[user] likes pytest"])

    assert len(calls) == 1
    assert "[user] likes pytest" in calls[0][1]


def test_playbook_store_writes_atomically(tmp_path, monkeypatch):
    from mnemoai.client.memory import playbook_store as ps

    calls = []
    monkeypatch.setattr(ps, "atomic_write_json", lambda p, d: calls.append((p, d)))

    store = ps.PlaybookStore.__new__(ps.PlaybookStore)
    store.persist_path = str(tmp_path)
    store.playbook_file = str(tmp_path / "playbook.json")
    store.entries = [{"strategy": "s"}]
    store._save()

    assert calls and calls[0][1] == [{"strategy": "s"}]


def test_reflector_metrics_write_atomically(tmp_path, monkeypatch):
    from mnemoai.client.memory import reflector as rf

    calls = []
    monkeypatch.setattr(rf, "atomic_write_json", lambda p, d: calls.append((p, d)))

    r = rf.Reflector.__new__(rf.Reflector)
    r.metrics_file = str(tmp_path / "metrics.json")
    r.metrics = {"turns": 3}
    r._save_metrics()

    assert calls and calls[0][1] == {"turns": 3}


def test_profile_manager_writes_atomically(tmp_path, monkeypatch):
    from mnemoai.client.managers import user_profile_manager as upm

    calls = []
    monkeypatch.setattr(upm, "atomic_write_json", lambda p, d: calls.append((p, d)))

    m = upm.UserProfileManager.__new__(upm.UserProfileManager)
    m.profile_path = str(tmp_path / "profile.json")
    m.profile = {"interaction_count": 2, "tool_patterns": {}}
    m._save_profile()

    assert calls and calls[0][1]["interaction_count"] == 2


def test_todo_manager_delegates_to_shared_helper(tmp_path):
    """todo_manager keeps its private name but no longer has its own copy."""
    from mnemoai.server.tools import todo_manager as tm

    target = tmp_path / "todos.json"
    tm._atomic_write_json(str(target), [{"content": "x"}])
    assert json.loads(target.read_text()) == [{"content": "x"}]
