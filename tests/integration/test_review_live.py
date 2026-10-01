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
from mnemoai.utils.review_protocol import is_feedback

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
def test_real_work_review_reaches_strategy_and_completion_after_each_parent_route(
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
    completion_hooks = []
    supervise = live_client._supervise_turn
    def final_checkpoint(*args, **kwargs):
        completion_hooks.append(1)
        return supervise(*args, **kwargs)
    monkeypatch.setattr(live_client, "_supervise_turn", final_checkpoint)
    def observe(*args, **kwargs):
        checkpoint = kwargs.get("checkpoint", "completion")
        if checkpoint == "strategy":
            assert not any(path.exists() for path in targets), "strategy must precede writes"
        calls.append(checkpoint)
        return finish(*args, **kwargs)
    monkeypatch.setattr(live_client.reviewer, "finish", observe)
    response = live_client.query("Create the requested files using fs_write, with no other changes: "
                                 + "; ".join(t["description"] for t in tasks))
    assert response.strip()
    if route == "off":
        assert calls == [] and completion_hooks == []
    else:
        assert calls[0] == "strategy" and calls[-1] == "completion"
        assert completion_hooks == [1]
        assert calls.count("completion") == 1 + live_client.reviewer.last["revisions"]
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
        if messages[0].content == config.require_prompt("WORK_STRATEGY_PROMPT"):
            return stream(messages, *args, **kwargs)
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


def test_live_general_question_does_not_call_reviewer(live_client, monkeypatch):
    enable_review(live_client, monkeypatch)
    monkeypatch.setattr(
        live_client.reviewer, "finish",
        lambda *a, **kw: pytest.fail("General Q&A must not call the reviewer"),
    )
    answer = live_client.query("What is the capital of France? Answer briefly.")
    assert "paris" in answer.lower()
    assert live_client.reviewer.last is None


def test_live_reviewer_outage_continues_edits_with_one_notice(
    live_client, _neutral_root, monkeypatch, capsys, record_testsuite_property
):
    enable_review(live_client, monkeypatch)
    monkeypatch.setattr(live_client, "auto_approve_mode", "edits")
    monkeypatch.setattr(live_client, "plan_mode_active", False)
    original = live_client._area_model
    monkeypatch.setitem(config._config_data, "LLM",
                        {**(config.get("LLM", {}) or {}), "MAX_RETRIES": 3, "RETRY_DELAY": 0})
    attempts = []
    class Unavailable:
        def invoke(self, *args, **kwargs):
            attempts.append(1)
            raise RuntimeError("503 service unavailable (injected reviewer outage)")
    monkeypatch.setattr(live_client, "_area_model",
                        lambda area: Unavailable() if area == "REVIEWER" else original(area))
    for i in range(2):
        path = _neutral_root / f"review-outage-{i}.txt"
        answer = live_client.query(
            f"Create {path} using fs_write with exactly the text 'outage-ok'. "
            "Only create that file; do not ask questions or do other work."
        )
        assert answer.strip()
        assert path.read_text() == "outage-ok"
        assert live_client.reviewer.last["unavailable"]
    assert len(attempts) == 6
    assert capsys.readouterr().out.count("Peer review unavailable; continuing without it.") == 1
    record_testsuite_property("review_outage", "injected reviewer failure; real actor and MCP; writes continue")


_DOCUMENT_FACTS = (
    "Previously downloaded documents remain readable offline.",
    "Downloading new documents requires an internet connection.",
)


def document_request(target=None):
    request = (
        "Write a concise Markdown release-note document titled '# Offline mode'. "
        "Include these exact factual sentences: " + " ".join(_DOCUMENT_FACTS)
        + " Do not add other product capabilities, guarantees, dates or claims."
    )
    if target is None:
        return request + " Return the document directly in your answer. This is self-contained; do not use tools or change files."
    return request + f" Save the document to {target} using fs_write, then read it back. Change only that file."


def direct_review(client, monkeypatch, *, shell=False):
    """Pin only the execution route; strategy, reviewer, actor and MCP stay live."""
    enable_review(client, monkeypatch)
    monkeypatch.setattr(client, "auto_approve_mode", "all" if shell else "edits")
    monkeypatch.setattr(client, "plan_mode_active", False)
    monkeypatch.setattr(client.agent, "router", None)
    monkeypatch.setattr(client.agent, "orchestrator_enabled", False)
    monkeypatch.setattr(client.agent, "graph", client.agent._build_graph())


@pytest.mark.parametrize("kind,destination", [
    ("code", "file"), ("document", "file"), ("document", "inline"),
])
def test_live_artifact_authoring_reviews_strategy_changes_and_final_result(
    live_client, _neutral_root, monkeypatch, kind, destination, record_testsuite_property,
):
    """No seeded model output: author a real artifact and inspect every phase."""
    direct_review(live_client, monkeypatch, shell=kind == "code")
    agent = live_client.agent
    target = _neutral_root / ("live_clamp.py" if kind == "code" else "live_release_note.md")
    if destination == "inline":
        target = None
    if kind == "code":
        verification = (
            "from live_clamp import clamp; "
            "assert [clamp(x, 0, 10) for x in (-3, 0, 4.5, 10, 15)] == [0, 0, 4.5, 10, 10]; "
            "assert clamp(-4, -3, -1) == -3; assert clamp(7, 2, 2) == 2; "
            "\ntry: clamp(1, 3, 2)\nexcept ValueError: pass\n"
            "else: raise AssertionError('inverted bounds must raise ValueError')"
        )
        command = (
            f"cd {shlex.quote(str(_neutral_root))} && {shlex.quote(sys.executable)} -B -c "
            + shlex.quote(verification)
        )
        prompt = (
            f"Create {target} defining clamp(value, low, high). Raise ValueError if low > high; "
            "otherwise return the nearest inclusive bound or the original value without casting it. "
            "Use only the standard library, write via fs_write, and change no other files. "
            f"Verify with exactly this command: {command}"
        )
    else:
        prompt = document_request(target)
    finish = live_client.reviewer.finish
    phases, tools, checks = [], [], []
    def observe_review(*args, **kwargs):
        phase = kwargs.get("checkpoint", "completion")
        if phase == "strategy":
            assert not tools, "no implementation tool may run before strategy review"
            assert target is None or not target.exists()
        phases.append(phase)
        return finish(*args, **kwargs)
    monkeypatch.setattr(live_client.reviewer, "finish", observe_review)
    invoke = agent._invoke_tool
    def observe_tool(tool, name, args, quiet=False):
        assert "strategy" in phases
        tools.append(name)
        result = invoke(tool, name, args, quiet=quiet)
        if name == "execute_bash":
            checks.append(json.loads(result))
        return result
    monkeypatch.setattr(agent, "_invoke_tool", observe_tool)
    answer = live_client.query(prompt)
    report = live_client.reviewer.last
    assert answer.strip()
    assert report and report["verdict"] == "pass", report
    assert report["task_kind"] == kind and report["strategy"].strip()
    assert phases[0] == "strategy" and phases[-1] == "completion"
    if target is not None:
        assert "changes" in phases and "fs_write" in tools
        assert any(item["kind"] == "current_file" for item in report["evidence"])
        artifact = target.read_text()
    else:
        assert "changes" not in phases and tools == []
        artifact = answer
    if kind == "code":
        namespace = {}
        exec(artifact, namespace)
        clamp = namespace["clamp"]
        assert [clamp(x, 0, 10) for x in (-3, 0, 4.5, 10, 15)] == [0, 0, 4.5, 10, 10]
        assert clamp(-4, -3, -1) == -3 and clamp(7, 2, 2) == 2
        with pytest.raises(ValueError):
            clamp(1, 3, 2)
        assert any(item.get("exit_status") == 0 for item in checks)
    else:
        assert "# Offline mode" in artifact
        assert all(fact in artifact for fact in _DOCUMENT_FACTS)
    record_testsuite_property(f"live_authoring_{kind}_{destination}", json.dumps({
        "actor": live_client.llm_controller.model_name,
        "reviewer": live_client._area_usage_name("REVIEWER"),
        "model_outputs": "all live; no seeded strategy, artifact or verdict",
        "phases": phases, "tools": tools, "verdict": report["verdict"],
        "revisions": report["revisions"], "artifact": artifact,
    }))


@pytest.mark.parametrize("destination", ["file", "inline"])
def test_live_document_feedback_corrects_a_seeded_factual_error(
    live_client, _neutral_root, monkeypatch, destination, record_testsuite_property,
):
    """Only the first incorrect draft is seeded; feedback and revisions are live."""
    direct_review(live_client, monkeypatch)
    agent = live_client.agent
    target = _neutral_root / "corrected_release_note.md" if destination == "file" else None
    wrong_fact = "Downloading new documents is available offline."
    bad_draft = "# Offline mode\n\n" + _DOCUMENT_FACTS[0] + "\n" + wrong_fact + "\n"
    stream = agent._stream_response
    seeded = False
    feedback_seen = []
    def first_bad_draft(messages, *args, **kwargs):
        nonlocal seeded
        if messages[0].content == config.require_prompt("WORK_STRATEGY_PROMPT"):
            return stream(messages, *args, **kwargs)
        if not seeded:
            seeded = True
            if target is None:
                return AIMessage(content=bad_draft), False
            return AIMessage(content="", tool_calls=[{
                "id": "seed-document-draft", "name": "fs_write",
                "args": {"path": str(target), "command": "create", "file_text": bad_draft},
            }]), False
        feedback_seen.extend(
            message.content for message in messages
            if getattr(message, "name", None) == "reviewer" and is_feedback(message.content)
        )
        return stream(messages, *args, **kwargs)
    monkeypatch.setattr(agent, "_stream_response", first_bad_draft)
    finish = live_client.reviewer.finish
    observed = []
    def observe_review(*args, **kwargs):
        phase = kwargs.get("checkpoint", "completion")
        if target is not None and phase == "changes" and not any(
            item["checkpoint"] == "changes" for item in observed
        ):
            assert target.read_text() == bad_draft, "the seeded defect must really exist before review"
        report = finish(*args, **kwargs)
        observed.append({
            "checkpoint": phase,
            "verdict": report["verdict"], "findings": report["findings"],
        })
        return report
    monkeypatch.setattr(live_client.reviewer, "finish", observe_review)
    answer = live_client.query(document_request(target))
    report = live_client.reviewer.last
    assert report and report["task_kind"] == "document" and report["verdict"] == "pass", report
    checkpoint = "changes" if target is not None else "completion"
    assert any(item["checkpoint"] == checkpoint and item["verdict"] == "revise"
               and item["findings"] for item in observed), observed
    assert any(wrong_fact in feedback for feedback in feedback_seen), feedback_seen
    artifact = target.read_text() if target is not None else answer
    assert all(fact in artifact for fact in _DOCUMENT_FACTS)
    assert wrong_fact not in artifact
    record_testsuite_property(f"live_document_correction_{destination}", json.dumps({
        "initial_draft": "seeded incorrect fact; file writes use real MCP",
        "strategy_reviewer_and_corrections": "live configured models",
        "checkpoints": observed, "verdict": report["verdict"], "artifact": artifact,
    }))
