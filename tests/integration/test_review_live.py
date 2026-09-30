"""Real reviewer/provider + MCP evidence; routing is controlled where noted."""

import json
import shlex
import sys
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from mnemoai.client.agent.supervised_turn import budget_for
from mnemoai.client.review import Reviewer
from mnemoai.client.session_log import read_session
from mnemoai.utils.config import config

pytestmark = pytest.mark.integration


def enable_review(client, monkeypatch):
    monkeypatch.setattr(client, "reviewer", Reviewer(enabled=True))
    monkeypatch.setitem(config._config_data, "REVIEW", {"TIMEOUT": 120, "MAX_INPUT_TOKENS": 6000})
    # The real provider/controller builds the separate, tool-free reviewer.
    client._area_model_cache.pop("REVIEWER", None)


def test_real_reviewer_checks_seeded_defect_and_verified_correction(
    live_client, _neutral_root, monkeypatch, record_testsuite_property
):
    enable_review(live_client, monkeypatch)
    record_testsuite_property("reviewer_model", live_client._area_usage_name("REVIEWER"))
    monkeypatch.setattr(live_client, "auto_approve_mode", "all")
    target = _neutral_root / "review_double.py"
    target.write_text("# KEEP THIS COMMENT\ndef double(n):\n    return n * 2\n")
    task = f"Implement double(n) in {target.name} as n * 2; preserve the existing comment."
    verify = (
        f"cd {shlex.quote(str(_neutral_root))} && {shlex.quote(sys.executable)} -B -c "
        + shlex.quote(
            "from review_double import double; "
            "assert [double(n) for n in (-3, 0, 5)] == [-6, 0, 10]"
        )
    )
    agent = live_client.agent
    measurements = []

    def call(name, args, messages):
        agent._run_tool_calls([{"name": name, "id": f"check-{len(messages)}", "args": args}],
                              agent.tools, messages)
        return messages[-1].content

    for bad in (True, False):
        capture = live_client.reviewer.begin(task)
        monkeypatch.setattr(agent, "_review_capture", capture)
        messages = []
        call("fs_read", {"path": str(target)}, messages)
        old, new = ("n * 2", "n + 2") if bad else ("n + 2", "n * 2")
        call("file_edit", {"file_path": str(target), "old_string": old, "new_string": new}, messages)
        checked = json.loads(call("execute_bash", {"command": verify}, messages))
        assert (checked["exit_status"] != 0) is bad
        contents = target.read_text()
        usage_before = agent.usage.totals()
        live_client._finish_review(capture, "Implementation is complete.", [])
        report = live_client.reviewer.last
        assert report["verdict"] == ("revise" if bad else "pass"), report
        if bad:
            assert report["findings"]
        assert target.read_text() == contents  # review never fixes the fixture
        assert any(e["kind"] == "current_file" for e in report["evidence"])
        usage_after = agent.usage.totals()
        measurements.append({
            "fixture": "seeded defect" if bad else "verified correction",
            "verdict": report["verdict"], "elapsed_seconds": report["elapsed_seconds"],
            "reported_tokens": usage_after["total_tokens"] - usage_before["total_tokens"],
            "calls_without_usage": usage_after["calls_without_usage"] - usage_before["calls_without_usage"],
        })
    record_testsuite_property("review_fixture_measurements", json.dumps(measurements))


@pytest.mark.parametrize("route", ["off", "direct", "atomic", "orchestrated"])
def test_real_completion_review_runs_once_after_each_parent_route(
    live_client, _neutral_root, monkeypatch, route, record_testsuite_property
):
    enable_review(live_client, monkeypatch)
    live_client.reviewer.enabled = route != "off"
    agent = live_client.agent
    monkeypatch.setattr(live_client, "auto_approve_mode", "edits")
    monkeypatch.setattr(live_client, "plan_mode_active", False)
    monkeypatch.setattr(agent, "_trusted_confirm_categories", set())
    targets = [_neutral_root / f"review-{route}-{i}.txt" for i in range(2 if route == "orchestrated" else 1)]
    tasks = [{
        "description": f"Use fs_write to create {p} containing exactly 'review-fixture'. Touch only that file.",
        "category": "full", "depends_on": [],
    } for p in targets]
    # Deterministic route/decomposition, but real graph, workers, actor, tools,
    # aggregator and reviewer. Existing agent-live tests cover classifier output.
    monkeypatch.setattr(agent, "router", None if route in {"direct", "off"} else SimpleNamespace(classify=lambda q, context: "full"))
    monkeypatch.setattr(agent, "orchestrator_enabled", route not in {"direct", "off"})
    monkeypatch.setattr(agent, "_decompose_task", lambda *a, **kw: tasks)
    monkeypatch.setattr(agent, "graph", agent._build_graph())
    record_testsuite_property("routing", "controlled; graph, workers, provider and MCP are real")
    finish = live_client.reviewer.finish
    calls = []
    def observe(*args, **kwargs):
        calls.append(1)
        return finish(*args, **kwargs)
    monkeypatch.setattr(live_client.reviewer, "finish", observe)
    response = live_client.query("Create the requested files using fs_write, with no other changes: "
                                 + "; ".join(t["description"] for t in tasks))
    assert response.strip()
    assert calls == ([] if route == "off" else [1])
    for target in targets:
        assert target.read_text().strip() == "review-fixture"
    report = live_client.reviewer.last
    if route == "off":
        assert report is None
    else:
        assert report and report["verdict"] == "pass", report
        assert sum(e["kind"] == "current_file" for e in report["evidence"]) == len(targets)
    assert not agent._is_headless()
    assert live_client.auto_approve_mode == "edits"
    assert agent._trusted_confirm_categories == set()


def test_live_supervisor_returns_findings_to_chat_which_fixes_and_verifies(
    live_client, _neutral_root, monkeypatch, record_testsuite_property
):
    """Seed only the initial bad claim; reviewer, corrections and MCP are live."""
    enable_review(live_client, monkeypatch)
    monkeypatch.setattr(live_client, "auto_approve_mode", "all")
    monkeypatch.setattr(live_client, "plan_mode_active", False)
    agent = live_client.agent
    monkeypatch.setattr(agent, "router", None)
    monkeypatch.setattr(agent, "orchestrator_enabled", False)
    monkeypatch.setattr(agent, "graph", agent._build_graph())
    target = _neutral_root / "supervised_double.py"
    target.write_text("# KEEP THIS COMMENT\ndef double(n):\n    return n + 2\n")
    verify = (
        f"cd {shlex.quote(str(_neutral_root))} && {shlex.quote(sys.executable)} -B -c "
        + shlex.quote("from supervised_double import double; "
                      "assert [double(n) for n in (-3, 0, 5)] == [-6, 0, 10]")
    )
    prompt = (
        f"Make double(n) in {target} return n * 2, preserving '# KEEP THIS COMMENT'. "
        f"Verify using this command: {verify}. Change only that file."
    )
    stream = agent._stream_response
    calls = 0
    def seeded_first_answer(messages, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(content="", tool_calls=[{
                "id": "supervisor-seed-read", "name": "fs_read", "args": {"path": str(target)},
            }]), False
        if calls == 2:
            return AIMessage(content="Implementation complete; double returns n * 2."), False
        return stream(messages, *args, **kwargs)
    monkeypatch.setattr(agent, "_stream_response", seeded_first_answer)
    invoke = agent._invoke_tool
    writes, checks = [], []
    def observed(tool, name, args, quiet=False):
        result = invoke(tool, name, args, quiet=quiet)
        if name in {"file_edit", "fs_write"}:
            writes.append((name, budget_for(agent) is not None, agent._turn_prompt))
        if name == "execute_bash":
            checks.append(json.loads(result))
        return result
    monkeypatch.setattr(agent, "_invoke_tool", observed)
    previous_turns = read_session(agent.session_log.path)["turns"]
    response = live_client.query(prompt)
    report = live_client.reviewer.last
    assert response.strip()
    assert report["verdict"] == "pass", report
    assert 1 <= report["revisions"] <= 2
    assert report["rounds"][0]["findings"]
    assert writes and all(supervised and cause == agent._strip_ephemeral(prompt) for _, supervised, cause in writes)
    assert any(item.get("exit_status") == 0 for item in checks)
    namespace = {}
    exec(target.read_text(), namespace)
    assert [namespace["double"](n) for n in (-3, 0, 5)] == [-6, 0, 10]
    assert "# KEEP THIS COMMENT" in target.read_text()
    assert read_session(agent.session_log.path)["turns"] == previous_turns + 1
    assert not agent._is_headless() and live_client.auto_approve_mode == "all"
    record_testsuite_property("supervisor_live", json.dumps({
        "initial_answer": "seeded bad claim; real fs_read",
        "reviewer_and_corrections": "live configured models and MCP",
        "revisions": report["revisions"], "verdict": report["verdict"],
        "elapsed_seconds": report["elapsed_seconds"], "write_calls": len(writes),
    }))
