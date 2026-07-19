"""Unit tests for background-task artifact sweeping (server/tools/background_tasks).

The in-memory registries are cleared on each restart, so nothing removes prior
sessions' artifacts at runtime. `sweep_old_task_logs` prunes stale ones at
startup — both a bash task's ``*.log`` and a sub-agent's ``subagent_*.json``
record — so the task dir doesn't grow without bound.
"""

import os
import time

import mnemoai.server.tools.background_tasks as bt


def _touch(path, age_days):
    with open(path, "w") as f:
        f.write("x")
    old = time.time() - age_days * 86400
    os.utime(path, (old, old))


def test_sweeps_only_old_log_files(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "TASK_OUTPUT_DIR", str(tmp_path))
    _touch(tmp_path / "old.log", age_days=10)     # older than 7d -> removed
    _touch(tmp_path / "recent.log", age_days=1)   # within 7d -> kept

    removed = bt.sweep_old_task_logs(max_age_days=7)

    assert removed == 1
    assert not (tmp_path / "old.log").exists()
    assert (tmp_path / "recent.log").exists()


def test_ignores_non_log_files(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "TASK_OUTPUT_DIR", str(tmp_path))
    _touch(tmp_path / "old.log", age_days=30)
    _touch(tmp_path / "keepme.txt", age_days=30)  # not swept -> never touched
    _touch(tmp_path / ".DS_Store", age_days=30)

    bt.sweep_old_task_logs(max_age_days=7)

    assert not (tmp_path / "old.log").exists()
    assert (tmp_path / "keepme.txt").exists()
    assert (tmp_path / ".DS_Store").exists()


def test_sweeps_old_subagent_json_records(tmp_path, monkeypatch):
    # A background/resumed sub-agent's record must be pruned on the same policy
    # as .log files — else these accumulate forever (the bug this fixes).
    monkeypatch.setattr(bt, "TASK_OUTPUT_DIR", str(tmp_path))
    _touch(tmp_path / "subagent_explore-1.json", age_days=10)   # old -> removed
    _touch(tmp_path / "subagent_plan-2.json", age_days=1)       # recent -> kept

    removed = bt.sweep_old_task_logs(max_age_days=7)

    assert removed == 1
    assert not (tmp_path / "subagent_explore-1.json").exists()
    assert (tmp_path / "subagent_plan-2.json").exists()


def test_unrelated_json_is_not_swept(tmp_path, monkeypatch):
    # Only subagent_*.json is swept; another app's JSON in the dir is left alone.
    monkeypatch.setattr(bt, "TASK_OUTPUT_DIR", str(tmp_path))
    _touch(tmp_path / "config.json", age_days=30)
    _touch(tmp_path / "subagent_x-1.json", age_days=30)

    bt.sweep_old_task_logs(max_age_days=7)

    assert (tmp_path / "config.json").exists()
    assert not (tmp_path / "subagent_x-1.json").exists()


def test_zero_age_disables_sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "TASK_OUTPUT_DIR", str(tmp_path))
    _touch(tmp_path / "old.log", age_days=999)
    assert bt.sweep_old_task_logs(max_age_days=0) == 0
    assert (tmp_path / "old.log").exists()


def test_missing_dir_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "TASK_OUTPUT_DIR", str(tmp_path / "nope"))
    # No dir, no crash, nothing removed.
    assert bt.sweep_old_task_logs(max_age_days=7) == 0
