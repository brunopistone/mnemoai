"""Shared steering snapshots must stay quiet, fresh and faithful to the existing rules."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mnemoai.client import context_injection, context_report, doctor
from mnemoai.client.client import LangGraphClient
from mnemoai.client.memory.steering_store import (
    SteeringSnapshot,
    SteeringState,
    SteeringStore,
)
from mnemoai.client.ui.command_feedback import render_steering_notices
from mnemoai.utils import paths


def client_with(files):
    client = LangGraphClient.__new__(LangGraphClient)
    client._steering_state = SteeringState(files=files)
    client.agent = SimpleNamespace(system_prompt="system", messages=[], _last_input_tokens=None)
    client.tools = []
    client.plan_mode_active = False
    client.refresh_steering()
    return client


@pytest.fixture
def capped(monkeypatch):
    monkeypatch.setattr(SteeringStore, "_max_chars", staticmethod(lambda: 100))


@pytest.mark.parametrize("name", ["CLAUDE.md", "STEERING.md", "AGENTS.md"])
def test_all_supplied_filenames_keep_their_content_and_name(tmp_path, name):
    path = tmp_path / name
    path.write_text("Keep the user's conventions.")
    snapshot = SteeringState(files=[path]).current
    assert snapshot.files[0].path == path
    assert "Keep the user's conventions." in snapshot.prompt
    assert snapshot.roots == (path,)


def test_reader_sizing_and_estimates_never_log_warnings(tmp_path, capped, caplog):
    path = tmp_path / "CLAUDE.md"
    path.write_text("x" * 300)
    client = client_with([path])
    expected = client._steering_state.current
    for _ in range(4):
        assert client._steering_reminder() == expected.prompt
        context_report.estimate(client)
        SteeringStore(files=[path]).read()
        SteeringStore(files=[path]).sizes()
    assert not caplog.records
    assert "truncated" in expected.prompt
    assert expected.files[0].source_chars == 300
    assert expected.files[0].loaded_chars == 100


def test_one_snapshot_serves_request_report_and_footer_without_more_reads(tmp_path, monkeypatch):
    root = tmp_path / "CLAUDE.md"
    extra = tmp_path / "AGENTS.md"
    root.write_text("Apply @AGENTS.md.")
    extra.write_text("Included instructions.")
    reads = Mock(wraps=SteeringStore._read_text)
    monkeypatch.setattr(SteeringStore, "_read_text", staticmethod(reads))
    client = client_with([root])
    assert reads.call_count == 2
    snapshot = client._steering_state.current
    for _ in range(3):
        client.refresh_steering()
        context_report.collect(client)
        context_report.estimate_cache_key(client)
        assert client._steering_reminder() == snapshot.prompt
        assert client._steering_state.current is snapshot
    assert reads.call_count == 2
    with pytest.raises(FrozenInstanceError):
        snapshot.text = "changed"


def test_foreground_notice_is_gray_once_and_model_marker_is_preserved(tmp_path, capped, capsys):
    path = tmp_path / "CLAUDE.md"
    path.write_text("a" * 300)
    client = client_with([path])
    for _ in range(4):
        client.refresh_steering(notify=True)
        context_report.estimate(client)
    out = capsys.readouterr().out
    assert out.count("partially loaded") == 1
    assert "\033[90m" in out and "\033[93m" not in out
    assert "100/300 chars" in out and "/context" in out
    assert str(tmp_path) not in out
    assert "truncated" in client._steering_reminder()


def test_unchanged_condition_stays_quiet_but_cap_change_reannounces(tmp_path, monkeypatch, capsys):
    path = tmp_path / "STEERING.md"
    path.write_text("x" * 300)
    cap = [100]
    monkeypatch.setattr(SteeringStore, "_max_chars", staticmethod(lambda: cap[0]))
    client = client_with([path])
    client.refresh_steering(notify=True)
    capsys.readouterr()
    path.write_text("y" * 300)  # changed content, same warning condition
    client.refresh_steering(notify=True)
    assert capsys.readouterr().out == ""
    cap[0] = 200
    client.refresh_steering(notify=True)
    assert "200/300" in capsys.readouterr().out
    cap[0] = 0
    client.refresh_steering()  # a passive recovery also rearms the notice
    assert client._steering_state.current.notices == ()
    cap[0] = 100
    client.refresh_steering(notify=True)
    assert "100/300" in capsys.readouterr().out


def test_new_snapshot_invalidates_estimates_and_stale_provider_count(tmp_path):
    path = tmp_path / "CLAUDE.md"
    path.write_text("first")
    client = client_with([path])
    old_snapshot = client._steering_state.current
    key = context_report.estimate_cache_key(client)
    client.agent._last_input_tokens = 2000
    path.write_text("other")  # same length
    client.refresh_steering()
    assert "first" in old_snapshot.text
    assert "other" in client._steering_reminder()
    assert client.agent._last_input_tokens is None
    assert context_report.estimate_cache_key(client) != key


def test_include_creation_edit_and_removal_are_observed(tmp_path):
    path = tmp_path / "STEERING.md"
    path.write_text("Read @AGENTS.md.")
    extra = tmp_path / "AGENTS.md"
    state = SteeringState(files=[path])
    assert "new rules" not in state.current.text
    extra.write_text("new rules")
    assert "new rules" in state.refresh().text
    extra.write_text("replacement rules")
    assert "replacement rules" in state.refresh().text
    extra.unlink()
    assert "replacement rules" not in state.refresh().text


def test_transitive_include_and_symlink_retarget_refresh(tmp_path):
    path = tmp_path / "CLAUDE.md"
    path.write_text("@first.md")
    first = tmp_path / "first.md"
    first.write_text("@selected.md")
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_text("aaaa")
    b.write_text("bbbb")
    link = tmp_path / "selected.md"
    link.symlink_to(a)
    state = SteeringState(files=[path])
    assert "aaaa" in state.current.text
    link.unlink()
    link.symlink_to(b)
    assert "bbbb" in state.refresh().text
    b.write_text("cccc")
    assert "cccc" in state.refresh().text


def test_restored_mtime_cannot_hide_same_size_edits(tmp_path):
    path = tmp_path / "STEERING.md"
    path.write_text("aaaa")
    state = SteeringState(files=[path])
    old = path.stat()
    assert "aaaa" in state.current.text
    replacement = tmp_path / "replacement"
    replacement.write_text("bbbb")
    os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
    replacement.replace(path)
    assert "bbbb" in state.refresh().text


def test_new_roots_and_existing_precedence_are_refreshed(tmp_path, monkeypatch):
    home = tmp_path / "app"
    repo = tmp_path / "repo"
    home.mkdir()
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setattr(paths, "app_home", lambda: home)
    state = SteeringState(cwd=repo)
    assert state.current.text == ""
    (home / "CLAUDE.md").write_text("global")
    (repo / "CLAUDE.md").write_text("project")
    assert "global" in state.refresh().text and "project" in state.current.text
    (repo / "STEERING.md").write_text("preferred")
    assert "preferred" in state.refresh().text
    assert "project" not in state.current.text
    assert "global" in state.current.text


def test_omitted_includes_still_share_budget_and_are_diagnosable(tmp_path, capped):
    path = tmp_path / "STEERING.md"
    path.write_text("@large.md")
    include = tmp_path / "large.md"
    include.write_text("z" * 200)
    state = SteeringState(files=[path])
    assert "z" * 200 not in state.current.text
    assert "not included" in state.current.prompt
    assert state.current.notices[0].kind == "includes"
    include.write_text("now fits")
    assert "now fits" in state.refresh().text
    assert state.current.notices == ()


def test_transient_read_error_retries_without_spamming(tmp_path, monkeypatch):
    path = tmp_path / "CLAUDE.md"
    path.write_text("rules")
    original = SteeringStore._read_text
    unavailable = [True]

    def read(target):
        if unavailable[0]:
            raise OSError("temporary outage")
        return original(target)

    monkeypatch.setattr(SteeringStore, "_read_text", staticmethod(read))
    state = SteeringState(files=[path])
    assert state.current.text == ""
    assert len(state.take_notices()) == 1
    state.refresh()
    assert state.take_notices() == ()
    unavailable[0] = False
    assert "rules" in state.refresh().text
    unavailable[0] = True
    path.touch()
    state.refresh()
    assert len(state.take_notices()) == 1


def test_concurrent_reads_and_notice_consumers_share_one_result(tmp_path, capped, monkeypatch):
    path = tmp_path / "STEERING.md"
    path.write_text("x" * 300)
    reads = Mock(wraps=SteeringStore._read_text)
    monkeypatch.setattr(SteeringStore, "_read_text", staticmethod(reads))
    state = SteeringState(files=[path])
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: (state.current, state.take_notices()), range(24)))
    assert reads.call_count == 1
    assert all(snapshot is results[0][0] for snapshot, _ in results)
    assert sum(len(notices) for _, notices in results) == 1


def test_cached_reads_do_not_block_while_another_thread_refreshes(tmp_path, monkeypatch):
    path = tmp_path / "STEERING.md"
    path.write_text("first")
    state = SteeringState(files=[path])
    previous = state.current
    path.write_text("other")
    started, release, read_finished = threading.Event(), threading.Event(), threading.Event()
    original = SteeringStore._read_text

    def slow_read(target):
        started.set()
        release.wait(5)
        return original(target)

    monkeypatch.setattr(SteeringStore, "_read_text", staticmethod(slow_read))
    refreshed = threading.Thread(target=state.refresh, daemon=True)
    refreshed.start()
    assert started.wait(5)
    seen = []

    def read_cached():
        seen.append(state.current)
        read_finished.set()

    reader = threading.Thread(target=read_cached, daemon=True)
    reader.start()
    try:
        assert read_finished.wait(1), "the painter waited on a disk refresh"
        assert seen == [previous]
    finally:
        release.set()
        refreshed.join(5)
        reader.join(5)
    assert "other" in state.current.text


def test_repaint_path_performs_no_discovery_or_file_access(tmp_path, monkeypatch, capped):
    path = tmp_path / "CLAUDE.md"
    path.write_text("x" * 300)
    client = client_with([path])

    def forbidden(*args, **kwargs):
        pytest.fail("rendering must use the startup snapshot")

    monkeypatch.setattr(paths, "steering_files", forbidden)
    monkeypatch.setattr(SteeringStore, "_read_text", staticmethod(forbidden))
    monkeypatch.setattr(Path, "stat", forbidden)
    for _ in range(3):
        context_report.estimate_cache_key(client)
        context_report.estimate(client)
        assert "truncated" in client._steering_reminder()


def test_context_and_doctor_expose_truncation_without_reprinting_notice(tmp_path, capped, capsys):
    path = tmp_path / "CLAUDE.md"
    path.write_text("x" * 300)
    client = client_with([path])
    client.refresh_steering(notify=True)
    capsys.readouterr()
    context = context_report.report(client)
    assert "100/300 chars loaded" in context
    checks = doctor._size_checks(client)
    check = next(check for check in checks if check.name == "steering")
    assert check.status == doctor.WARN and "100/300 source chars loaded" in check.detail
    assert capsys.readouterr().out == ""


def test_notice_presentation_strips_control_characters():
    from mnemoai.client.memory.steering_store import SteeringNotice

    text = render_steering_notices([SteeringNotice(Path("bad\x1b[2J"), "unreadable", "bad\x1b[2J")])
    assert "\x1b[2J" not in text


def test_empty_snapshot_is_empty_in_context():
    assert SteeringSnapshot().prompt == ""
