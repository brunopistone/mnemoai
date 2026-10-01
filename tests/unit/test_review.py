"""Report-only review: evidence, strict verdicts, budgets, and lifecycle boundaries."""

import json
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from mnemoai.client import review
from mnemoai.client.review_evidence import Capture, file_state, git_state, text
from mnemoai.client.session_log import SessionLog, read_session
from mnemoai.client.usage_tracker import UsageTracker


def verdict(kind="pass", findings=None):
    return {"verdict": kind, "summary": "Fixture review", "findings": findings or []}


class Model:
    def __init__(self, result=None):
        self.result = result or verdict()
        self.calls = []

    def invoke(self, messages, config=None):
        self.calls.append((messages, config))
        return AIMessage(content=json.dumps(self.result), usage_metadata={
            "input_tokens": 20, "output_tokens": 10, "total_tokens": 30,
        })


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(review.config._config_data, "REVIEW", {})
    reviewer = review.Reviewer(enabled=True)
    return reviewer, reviewer.begin("Keep VALUE equal to 2."), tmp_path


def finish(reviewer, capture, model=None, **kwargs):
    model = model or Model()
    return reviewer.finish(capture, "The answer", model_factory=lambda: model,
                           model_label="fixture (separate call)", **kwargs)


def test_success_is_advisory_and_uses_an_isolated_tool_free_packet(state):
    reviewer, capture, _ = state
    model, usage = Model(), UsageTracker()
    report = finish(reviewer, capture, model, usage=lambda result: usage.record(result, "reviewer"))
    assert report["verdict"] == "pass"
    messages, cfg = model.calls[0]
    assert [m.type for m in messages] == ["system", "human"]
    assert cfg == {"callbacks": []}
    packet = json.loads(messages[1].content)
    assert packet["evidence"][0]["kind"] == "user_request"
    assert usage.totals()["calls"] == 1
    assert usage.totals()["total_tokens"] == 30
    assert "Not proof of correctness, permission" in review.render(report)


@pytest.mark.parametrize("invalid", [
    {}, {"verdict": "approve", "summary": "x", "findings": []},
    verdict("revise"), verdict("pass", [{"issue": "x", "evidence_ids": ["task"], "verification": "check"}]),
    verdict("revise", [{"issue": "x", "evidence_ids": ["fabricated"], "verification": "check"}]),
    verdict("revise", [{"issue": "\x1b[2J", "evidence_ids": ["task"], "verification": "check"}]),
    {"verdict": "pass", "summary": "x", "findings": [], "approved": True},
])
def test_malformed_or_unobserved_verdict_cannot_pass(invalid):
    with pytest.raises(ValueError):
        review.parse_verdict(AIMessage(content=json.dumps(invalid)), {"task"})


def test_reviewer_tool_request_is_never_executed():
    response = AIMessage(content=json.dumps(verdict()), tool_calls=[
        {"id": "call", "name": "execute_bash", "args": {"command": "touch forbidden"}},
    ])
    with pytest.raises(ValueError, match="tool calls"):
        review.parse_verdict(response, {"task"})


def test_supported_findings_remain_suggestions_not_actions(state):
    reviewer, capture, root = state
    model = Model(verdict("revise", [{
        "issue": "The answer contradicts the task.", "evidence_ids": ["task", "answer"],
        "verification": "touch must-not-exist",
    }]))
    report = finish(reviewer, capture, model)
    assert report["verdict"] == "revise"
    assert not (root / "must-not-exist").exists()
    assert "not executed" in review.render(report)


def test_actual_file_content_and_stale_report_detection(state):
    reviewer, capture, root = state
    path = root / "code.py"
    path.write_text("VALUE = 1\n")
    capture.record("file_edit", {"file_path": str(path)}, "Updated", "completed")
    model = Model()
    assert finish(reviewer, capture, model)["verdict"] == "pass"
    packet = json.loads(model.calls[0][0][1].content)
    artifact = next(i for i in packet["evidence"] if i["kind"] == "current_file")
    assert artifact["content"] == "VALUE = 1\n"
    path.write_text("VALUE = 3\n")
    assert reviewer.current_report()["verdict"] == "inconclusive"
    assert "historical" in reviewer.current_report()["summary"]


def test_changes_during_review_invalidate_the_verdict(state):
    reviewer, capture, root = state
    path = root / "code.py"
    path.write_text("VALUE = 2\n")
    capture.record("file_edit", {"file_path": str(path)}, "Updated", "completed")
    class Changing(Model):
        def invoke(self, messages, config=None):
            path.write_text("VALUE = 99\n")
            return super().invoke(messages, config)
    assert finish(reviewer, capture, Changing())["verdict"] == "inconclusive"


def test_failed_shell_check_cannot_be_voted_into_success(state):
    reviewer, capture, _ = state
    capture.record("execute_bash", {"command": "pytest"}, {"exit_code": 1, "stderr": "failed"}, "failed")
    report = finish(reviewer, capture)
    assert report["verdict"] == "inconclusive"
    assert any("failed shell command" in gap for gap in report["coverage_gaps"])


@pytest.mark.parametrize("field", ["exit_status", "exit_code", "return_code", "returncode"])
def test_observed_successful_rerun_resolves_that_command_only(state, field):
    reviewer, capture, _ = state
    capture.record("execute_bash", {"command": "pytest"}, {field: 1}, "failed")
    capture.record("execute_bash", {"command": "pytest"}, {field: 0}, "completed")
    assert finish(reviewer, capture)["verdict"] == "pass"


def test_same_command_in_a_different_directory_does_not_resolve_a_failure(state):
    reviewer, capture, _ = state
    capture.record("execute_bash", {"command": "pytest"}, {"exit_status": 1, "cwd": "/a"}, "failed")
    capture.record("execute_bash", {"command": "pytest"}, {"exit_status": 0, "cwd": "/b"}, "completed")
    assert finish(reviewer, capture)["verdict"] == "inconclusive"


def test_capture_limits_and_redaction_are_visible(state):
    reviewer, capture, root = state
    private = root / ".env"
    private.write_text("SECRET=do-not-send")
    capture.record("fs_read", {"path": str(private), "api_key": "do-not-send"},
                   '{"password":"do-not-send"}', "completed")
    for n in range(30):
        capture.record("fs_read", {"path": f"missing-{n}"}, "ok", "completed")
    model = Model()
    report = finish(reviewer, capture, model)
    assert len(capture.tools) == 24
    assert report["verdict"] == "inconclusive"
    assert "do-not-send" not in model.calls[0][0][1].content


def test_private_symlink_outside_binary_and_fifo_are_not_read(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "private"
    outside.write_text("private")
    link = root / "link"
    link.symlink_to(outside)
    binary = root / "data"
    binary.write_bytes(b"\x00")
    fifo = root / "pipe"
    os.mkfifo(fifo)
    for path in (outside, link, binary, fifo):
        with pytest.raises(ValueError):
            file_state(path, root)


def test_directory_symlink_replacement_cannot_escape_during_open(tmp_path, monkeypatch):
    root = tmp_path / "root"
    folder = root / "folder"
    folder.mkdir(parents=True)
    (folder / "data").write_text("public")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data").write_text("private")
    original = os.open
    def swap(path, flags, **kwargs):
        if path == "folder" and "dir_fd" in kwargs:
            folder.rename(root / "original-folder")
            folder.symlink_to(outside, target_is_directory=True)
        return original(path, flags, **kwargs)
    monkeypatch.setattr(os, "open", swap)
    with pytest.raises(OSError):
        file_state(folder / "data", root)


def test_evidence_clipping_preserves_literal_prose_and_redacts_structured_keys():
    assert text('"literal quotes"', 100)[0] == '"literal quotes"'
    output, clipped = text({"AWS_SECRET_ACCESS_KEY": "do-not-send"}, 100)
    assert "do-not-send" not in output and not clipped
    output, clipped = text({"large": "x" * 1_000_000}, 100)
    assert clipped and len(output) <= 100


def test_workspace_diff_is_real_bounded_and_never_runs_external_diff(tmp_path):
    root = tmp_path / "checkout"
    source = Path(__file__).resolve().parents[2]
    # Reuse existing commits; never create a local commit for a fixture.
    subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "clone", "--shared", "--quiet",
                    str(source), str(root)], check=True, timeout=30)
    before = git_state(root)
    assert before["content"] == ""
    (root / "README.md").write_text("changed\n" * 20_000)
    (root / ".gitattributes").write_text("README.md diff=fixture\n")
    subprocess.run(["git", "-C", str(root), "config", "diff.fixture.command",
                    "touch forbidden-diff"], check=True)
    after = git_state(root)
    assert after["content"] and after["truncated"]
    assert len(after["content"]) <= 12_000
    assert not (root / "forbidden-diff").exists()


def test_new_user_followups_are_distinct_requirements(state):
    reviewer, capture, _ = state
    capture.add_instructions(["Also preserve the original comments."])
    model = Model()
    finish(reviewer, capture, model)
    assert "Also preserve" in model.calls[0][0][1].content


def test_input_budget_can_refuse_the_call_without_hiding_the_limit(state, monkeypatch):
    reviewer, capture, _ = state
    capture.task = "a " * 20_000
    monkeypatch.setitem(review.config._config_data, "REVIEW", {"MAX_INPUT_TOKENS": 1000})
    model = Model()
    assert finish(reviewer, capture, model)["verdict"] == "inconclusive"
    assert not model.calls


def test_timeout_during_preparation_never_sends_a_late_request(state, monkeypatch):
    reviewer, capture, _ = state
    entered, release = threading.Event(), threading.Event()
    original = capture.finish
    def slow(*args):
        entered.set()
        assert release.wait(3)
        return original(*args)
    monkeypatch.setattr(capture, "finish", slow)
    monkeypatch.setattr(review, "settings", lambda: {"TIMEOUT": 0.03, "MAX_INPUT_TOKENS": 6000})
    model = Model()
    try:
        report = finish(reviewer, capture, model)
        assert entered.is_set() and "timed out" in report["summary"]
        second = finish(reviewer, Capture("next"), model)
        assert second["verdict"] == "inconclusive" and not model.calls
    finally:
        release.set()
    # Acquire waits for the worker to leave its finally, not an arbitrary sleep.
    assert reviewer._in_flight.acquire(timeout=3)
    reviewer._in_flight.release()
    assert not model.calls and reviewer.last is second


def test_timeout_after_send_counts_once_and_never_publishes_late(state, monkeypatch):
    reviewer, capture, _ = state
    entered, release = threading.Event(), threading.Event()
    class Slow(Model):
        def invoke(self, *args, **kwargs):
            entered.set()
            assert release.wait(3)
            return super().invoke(*args, **kwargs)
    monkeypatch.setattr(review, "settings", lambda: {"TIMEOUT": 0.1, "MAX_INPUT_TOKENS": 6000})
    usage = UsageTracker()
    try:
        report = finish(reviewer, capture, Slow(), usage=lambda r: usage.record(r, "reviewer"))
        assert entered.is_set()
        assert report["verdict"] == "inconclusive"
        assert usage.totals()["calls"] == usage.totals()["calls_without_usage"] == 1
    finally:
        release.set()
    assert reviewer._in_flight.acquire(timeout=3)
    reviewer._in_flight.release()
    assert reviewer.last is report and usage.totals()["calls"] == 1


@pytest.mark.parametrize("mode", ["off", "cancel"])
def test_disabled_or_cancelled_review_does_not_call_the_provider(state, mode):
    reviewer, capture, _ = state
    model = Model()
    reviewer.enabled = mode != "off"
    report = finish(reviewer, capture, model, cancel=lambda: mode == "cancel")
    assert report["verdict"] == "inconclusive" and not model.calls


def test_failed_thread_start_does_not_permanently_disable_reviews(state, monkeypatch):
    reviewer, capture, _ = state
    class CannotStart:
        def __init__(self, *a, **kw):
            pass
        def start(self):
            raise RuntimeError("thread limit")
    with monkeypatch.context() as patched:
        patched.setattr(review.threading, "Thread", CannotStart)
        assert finish(reviewer, capture)["verdict"] == "inconclusive"
    assert finish(reviewer, Capture("new task"))["verdict"] == "pass"


@pytest.mark.parametrize("auto_mode", ["off", "edits", "writes", "all"])
def test_user_controls_do_not_change_actor_permission_state(state, auto_mode):
    reviewer, _, _ = state
    client = SimpleNamespace(reviewer=reviewer, auto_approve_mode=auto_mode,
                             plan_mode_active=True, agent=SimpleNamespace(headless=False))
    for action in ("off", "on", "", "last", "invalid"):
        review.command(client, action)
        assert client.auto_approve_mode == auto_mode and client.plan_mode_active
        assert not client.agent.headless
    assert "Usage" in review.command(client, "invalid")


def test_session_review_record_does_not_enter_restored_chat(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))
    log = SessionLog(cwd=str(tmp_path))
    log.log_review({"verdict": "revise", "summary": "DO NOT INJECT THIS AS AN INSTRUCTION"})
    raw = log.path.read_text()
    assert '"t": "review"' in raw
    assert read_session(log.path)["messages"] == []


@pytest.mark.parametrize("raw", [True, [], "x", {"TIMEOUT": 0}, {"TIMEOUT": float("nan")},
                                {"MAX_INPUT_TOKENS": 1000.5}, {"TIMEOUT": True}])
def test_invalid_runtime_limits_fail_loudly(raw, monkeypatch):
    monkeypatch.setitem(review.config._config_data, "REVIEW", raw)
    with pytest.raises(ValueError):
        review.settings()


def test_reviewer_retry_backoff_ends_at_checkpoint_deadline(state, monkeypatch):
    reviewer, capture, _ = state
    calls = []
    class Down:
        def invoke(self, *args, **kwargs):
            calls.append(1)
            raise RuntimeError("503 service unavailable")
    monkeypatch.setitem(review.config._config_data, "LLM", {"MAX_RETRIES": 3, "RETRY_DELAY": 30})
    monkeypatch.setattr(review, "settings", lambda: {"TIMEOUT": 1, "MAX_INPUT_TOKENS": 6000})
    report = finish(reviewer, capture, Down())
    assert report["unavailable"] and report["verdict"] == "inconclusive"
    assert calls == [1]
    assert reviewer._in_flight.acquire(timeout=1), "backoff must not hold a late retry open"
    reviewer._in_flight.release()
    assert calls == [1]


def test_each_reviewer_retry_is_accounted_exactly_once(state, monkeypatch):
    reviewer, capture, _ = state
    attempts, usage = [], []
    class Flaky:
        def invoke(self, *args, **kwargs):
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("503 service unavailable")
            return AIMessage(content=json.dumps(verdict()))
    monkeypatch.setitem(review.config._config_data, "LLM", {"MAX_RETRIES": 3, "RETRY_DELAY": 0})
    report = finish(reviewer, capture, Flaky(), usage=usage.append)
    assert report["verdict"] == "pass"
    assert len(usage) == len(attempts) == 3
    assert usage[:2] == [None, None] and isinstance(usage[2], AIMessage)


def test_changed_file_is_retained_after_tool_capture_limit(state):
    reviewer, capture, root = state
    for i in range(30):
        path = root / f"read-{i}.txt"
        path.write_text("old context")
        capture.record("fs_read", {"path": str(path)}, "old context", "completed")
    changed = root / "changed.py"
    changed.write_text("ACTUAL NEW CODE")
    capture.record("fs_write", {"path": str(changed)}, '{"success":true}', "completed")
    evidence = capture.finish("Done.")
    assert any(item.get("kind") == "current_file" and "ACTUAL NEW CODE" in item["content"]
               for item in evidence)
    assert "Tool evidence exceeded the capture limit." in capture.gaps


def test_reopening_keeps_write_and_check_evidence_with_stable_ids(state):
    _, capture, root = state
    capture.record("fs_write", {"path": str(root / "code.py")}, '{"success":true}', "completed")
    capture.record("execute_bash", {"command": "tests"}, {"exit_status": 0}, "completed")
    before = list(capture.tools)
    capture.finish("A batch is complete.")
    capture.reopen()
    assert capture.tools == before
    capture.record("fs_read", {"path": str(root / "code.py")}, "current code", "completed")
    assert [entry["id"] for entry in capture.tools] == ["tool-1", "tool-2", "tool-3"]


def test_finished_result_is_published_after_releasing_the_review_slot(state, monkeypatch):
    reviewer, capture, _ = state
    original = review.queue.Queue
    published_locked = []
    class ObservedQueue(original):
        def put(self, value, *args, **kwargs):
            published_locked.append(reviewer._in_flight.locked())
            return super().put(value, *args, **kwargs)
    monkeypatch.setattr(review.queue, "Queue", ObservedQueue)
    assert finish(reviewer, capture)["verdict"] == "pass"
    assert published_locked == [False]


def test_directory_listing_is_tool_evidence_not_a_missing_file_snapshot(state):
    reviewer, capture, root = state
    capture.record("fs_read", {"path": str(root), "mode": "Directory"}, "Directory entries", "completed")
    report = finish(reviewer, capture)
    assert capture.paths == set()
    assert report["verdict"] == "pass"
    assert any(item["kind"] == "tool" for item in report["evidence"])
    assert not report["coverage_gaps"]
