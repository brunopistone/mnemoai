"""Local-only learning: factual observations, no model calls, and existing-store reuse."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from mcp import StdioServerParameters

from mnemoai.client.client import LangGraphClient
from mnemoai.client.mcp_tool_wrapper import MultiMCPClient
from mnemoai.client.memory import reflection
from mnemoai.client.memory.playbook_records import PlaybookEntry
from mnemoai.client.memory.playbook_store import PlaybookStore
from mnemoai.client.memory.reflector import Reflector
from mnemoai.client.usage_tracker import UsageTracker
from mnemoai.models.controllers.embeddings_controller import EmbeddingsController
from mnemoai.utils.config import config


def trace(calls):
    messages = [HumanMessage(content="Fix tensor parallel weight storage")]
    for i, (name, args, result) in enumerate(calls):
        messages.extend([
            AIMessage(content="", tool_calls=[{"id": str(i), "name": name, "args": args}]),
            ToolMessage(content=result, tool_call_id=str(i), name=name),
        ])
    return messages


def recovery():
    return trace([
        ("file_edit", {"file_path": "weights.py", "old_string": "ambiguous"},
         '{"error":true,"message":"not unique"}'),
        ("fs_read", {"path": "weights.py"}, "contents must not become a rule"),
        ("file_edit", {"file_path": "weights.py", "old_string": "unique context"},
         '{"success":true}'),
    ])


def test_failure_recovery_records_only_observed_facts(tmp_path):
    learner = Reflector(str(tmp_path))
    entries = learner.reflect_on_trajectory(
        recovery(), "Fix tensor parallel weight storage", scope="/project",
        source={"session_id": "s1", "turn": 4},
    )
    assert len(entries) == 1
    entry = entries[0].to_dict()
    assert entry["provenance"] == "observed"
    assert entry["scope"] == "/project" and entry["confidence"] == 0.5
    assert [ref["tool_call_id"] for ref in entry["source_refs"]] == ["0", "2"]
    assert "failed" in entry["strategy"] and "succeeded" in entry["strategy"]
    assert "not proof of cause" in entry["strategy"]
    assert "contents must not become a rule" not in entry["strategy"]
    assert learner.metrics["failed_calls"] == 1
    assert learner.metrics["successful_calls"] == 2


def test_different_target_or_command_does_not_claim_recovery():
    learner = Reflector()
    entries = learner.reflect_on_trajectory(trace([
        ("execute_bash", {"command": "pytest first.py"}, '{"exit_status":1}'),
        ("execute_bash", {"command": "pytest other.py"}, '{"exit_status":0}'),
    ]), "test", scope="/project")
    assert len(entries) == 2
    assert "Later" not in entries[0].strategy


def test_successful_routine_reads_do_not_manufacture_advice():
    learner = Reflector()
    assert learner.reflect_on_trajectory(
        trace([("fs_read", {"path": "file"}, '{"content":"ordinary"}')]), "read",
    ) == []
    assert learner.last_error is None
    assert learner.metrics["successful_calls"] == 1


def test_completed_check_is_observation_not_proof_of_task_success():
    entries = Reflector().reflect_on_trajectory(
        trace([("execute_bash", {"command": "python -m pytest"}, '{"exit_status":0}')]), "test",
    )
    assert len(entries) == 1
    assert "returned success" in entries[0].strategy
    assert "general correctness" in entries[0].strategy


def test_observations_are_bounded_and_failure_is_not_lost_to_later_reads():
    calls = [("file_edit", {"file_path": "x"}, '{"error":true}')]
    calls.extend(("fs_read", {"path": f"{i}"}, "text") for i in range(20))
    entries = Reflector().reflect_on_trajectory(trace(calls), "x" * 10_000)
    assert len(entries) == 1 and len(entries[0].context) <= 200
    assert len(entries[0].strategy) <= 800
    assert entries[0].source_refs[0]["tool_call_id"] == "0"


def test_long_file_bodies_do_not_hide_the_target_or_leak_into_observations():
    entries = Reflector().reflect_on_trajectory(trace([
        ("file_edit", {"file_path": "weights.py", "old_string": "x" * 10000},
         '{"error":true,"message":"Found 2 occurrences of old_string"}'),
        ("file_edit", {"file_path": "weights.py", "old_string": "y" * 15000},
         '{"success":true}'),
    ]), "edit")
    assert len(entries) == 1
    assert "weights.py" in entries[0].strategy
    assert "10000" in entries[0].strategy and "15000" in entries[0].strategy
    assert "ambiguous_match" in entries[0].strategy
    assert "x" * 20 not in entries[0].strategy and "y" * 20 not in entries[0].strategy


@pytest.mark.parametrize(("message", "kind"), [
    ("No occurrences of old_string found.", "string_not_found"),
    ("Found 2 occurrences of old_string.", "ambiguous_match"),
    ("The match is not unique.", "ambiguous_match"),
])
def test_local_observation_distinguishes_missing_and_ambiguous_matches(message, kind):
    entries = Reflector().reflect_on_trajectory(trace([
        ("file_edit", {"file_path": "weights.py"}, json.dumps({"error": True, "message": message})),
    ]), "edit")
    assert len(entries) == 1 and f"failed ({kind})" in entries[0].strategy


def test_no_tools_or_missing_results_create_nothing():
    learner = Reflector()
    assert learner.reflect_on_trajectory([HumanMessage(content="hello")], "hello") == []
    assert learner.reflect_on_trajectory(recovery()[:2], "edit") == []


@pytest.mark.parametrize("result", [
    '{"password": "private-value", "content": "ordinary text"}',
    'Authorization: Bearer private-value',
    'content includes api_key="private-value"',
])
def test_evidence_redacts_named_credentials(result):
    item = reflection.evidence_item("fs_read", {}, result, "call", "success", {})
    assert "private-value" not in item["result"]


def test_evidence_is_bounded_and_preserves_hash_only_provenance():
    item = reflection.evidence_item(
        "execute_bash", {"api_key": "private-value"},
        "password=private-value " + "x" * 10000, "call", "failure", {},
    )
    assert "private-value" not in item["args"] + item["result"]
    assert len(item["result"]) <= 1200 and item["truncated"]
    assert "result" not in item["ref"]


def test_client_never_builds_a_learning_model_and_preserves_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "get", lambda key, default=None: default)
    client = LangGraphClient.__new__(LangGraphClient)
    client.reflector = Reflector(str(tmp_path))
    client.playbook = PlaybookStore(str(tmp_path))
    client.agent = SimpleNamespace(messages=[], usage=UsageTracker())
    client._area_model = Mock(side_effect=AssertionError("local learning must not build a model"))
    client._reflection_messages = recovery()
    client._reflection_source = {"session_id": "s1", "turn": 1}
    before = client.agent.usage.totals()
    client.reflect_and_learn("Fix tensor parallel weight storage")
    client.reflect_and_learn("Fix tensor parallel weight storage")
    assert len(client.playbook.snapshot()) == 1
    assert client.agent.usage.totals() == before
    client._area_model.assert_not_called()
    assert client.reflector.metrics["total_tool_calls"] == 3


def test_learning_is_default_on_and_explicit_false_prevents_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))
    values = {"MODEL_ID": {"NAME": "fixture", "TYPE": "ollama"}}
    monkeypatch.setattr(config, "get", lambda key, default=None: values.get(key, default))
    enabled = LangGraphClient()
    assert enabled.playbook is not None and enabled.reflector is not None
    assert enabled.llm_controller.model is None
    values["ENABLE_PLAYBOOK"] = False
    disabled = LangGraphClient()
    assert disabled.playbook is disabled.reflector is None
    enabled._reflection_messages = recovery()
    enabled.agent = SimpleNamespace(messages=[])
    enabled.reflect_and_learn("test")
    assert enabled.playbook.snapshot() == []


def test_local_failure_is_file_only_and_visible_in_diagnostics(tmp_path, monkeypatch, capsys):
    client = LangGraphClient.__new__(LangGraphClient)
    client.reflector = Reflector()
    client.playbook = PlaybookStore(str(tmp_path))
    client.agent = SimpleNamespace(messages=[])
    client._reflection_messages = recovery()
    client.playbook.append_batch = Mock(side_effect=OSError("disk unavailable"))
    client.reflect_and_learn("test")
    assert client.reflector.last_error
    assert not capsys.readouterr().err


def test_task_relevance_beats_generic_legacy_confidence_and_updates_exposure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = PlaybookStore(str(tmp_path))
    for i in range(8):
        store.append(PlaybookEntry(
            "generic advice", f"unrelated advice {i}", "legacy", confidence=1.0,
        ))
    note = PlaybookEntry(
        "tensor parallel weight storage", "Observed sharding check returned success.",
        "local", scope=str(tmp_path), provenance="observed", confidence=0.5,
    )
    store.append(note)
    block, ids = store.prepare_prompt(task="Fix tensor parallel weight storage")
    assert note.id in ids and note.strategy in block
    store.record_exposure(ids)
    assert next(e for e in store.snapshot() if e["id"] == note.id)["injection_count"] == 1
    assert len(PlaybookStore(str(tmp_path)).snapshot()) == 9  # old notes preserved


def test_local_refinement_never_uses_an_embedding_provider(tmp_path):
    store = PlaybookStore(str(tmp_path), max_entries=1)
    # Even a legacy embedded caller carrying this obsolete attribute cannot invoke it.
    store.embeddings = SimpleNamespace(embed=Mock(side_effect=AssertionError("no embeddings")))
    store.append(PlaybookEntry("first", "first observation", "local"))
    store.append(PlaybookEntry("second", "second observation", "local"))
    assert len([e for e in store.snapshot() if e["status"] == "active"]) == 1
    store.embeddings.embed.assert_not_called()


def test_real_context_injection_uses_the_current_task(tmp_path, monkeypatch):
    from mnemoai.client import context_injection

    monkeypatch.chdir(tmp_path)
    store = PlaybookStore(str(tmp_path))
    for i in range(8):
        store.append(PlaybookEntry("generic", f"unrelated {i}", "legacy", confidence=1))
    observation = PlaybookEntry("tensor parallel weights", "Observed tensor storage check.", "local",
                                provenance="observed", scope=str(tmp_path))
    store.append(observation)
    client = SimpleNamespace(playbook=store, _playbook_task="tensor parallel weights")
    assert observation.strategy in context_injection.get_playbook_context(client)
    assert observation.id in client._playbook_selected_ids


def test_unavailable_local_notes_do_not_break_context_assembly():
    from mnemoai.client import context_injection

    client = SimpleNamespace(playbook=SimpleNamespace(
        prepare_prompt=Mock(side_effect=OSError("unreadable store")),
    ))
    assert context_injection.get_playbook_context(client) == ""
    assert client._playbook_selected_ids == []
    assert client._playbook_error


def test_recovered_local_notes_clear_the_old_diagnostic():
    from mnemoai.client import context_injection

    client = SimpleNamespace(
        playbook=SimpleNamespace(prepare_prompt=Mock(return_value=("recovered", ["note"]))),
        _playbook_error="previous storage failure",
    )
    assert context_injection.get_playbook_context(client) == "recovered"
    assert client._playbook_error is None
    assert client._playbook_selected_ids == ["note"]


def test_old_model_and_timeout_keys_cannot_enable_model_reflection(monkeypatch):
    from mnemoai.models import area_models

    monkeypatch.setattr(config, "get", lambda key, default=None: (
        {"REFLECTOR": {"NAME": "unused", "TYPE": "bedrock"}}
        if key == "AREA_MODELS" else default
    ))
    assert area_models.overrides_for("REFLECTOR") == {}
    assert "REFLECTOR" not in area_models.AREAS
    assert area_models.unknown_keys() == ()
    assert not hasattr(reflection, "extract")


def test_disable_and_reenable_preserve_notes_model_and_conversation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    enabled = [True]
    monkeypatch.setattr(config, "reload", lambda: None)
    monkeypatch.setattr(config, "get", lambda key, default=None: (
        enabled[0] if key == "ENABLE_PLAYBOOK" else default
    ))
    client = LangGraphClient.__new__(LangGraphClient)
    client._model_scoped_dir = lambda: str(tmp_path)
    client._open_playbook()
    client.system_prompt = "base"
    client.model = object()
    original_model = client.model
    client.agent = SimpleNamespace(system_prompt="base", messages=[HumanMessage(content="work")])
    client.playbook.append(PlaybookEntry("work", "keep existing note", "old"))
    path = Path(client.playbook.playbook_file)
    original = path.read_bytes()
    enabled[0] = False
    assert client.reload_playbook_settings()
    assert client.playbook is None and client.reflector is None
    assert "keep existing note" not in client.agent.system_prompt
    enabled[0] = True
    assert client.reload_playbook_settings()
    assert "keep existing note" in client.agent.system_prompt
    assert path.read_bytes() == original
    assert client.model is original_model and client.agent.messages[0].content == "work"


def test_real_mcp_fail_recover_learn_reuse_without_model_or_embedding_calls(tmp_path, monkeypatch):
    from mnemoai.client import context_injection

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(BaseChatModel, "invoke", Mock(side_effect=AssertionError("no LLM invocation")))
    monkeypatch.setattr(EmbeddingsController, "embed", Mock(side_effect=AssertionError("no embeddings")))
    source = tmp_path / "weights.py"
    source.write_text("# first\nVALUE = 1\n# second\nVALUE = 1\n")
    env = {**os.environ, "MNEMOAI_CONFIG": "/dev/null",
           "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    mcp = MultiMCPClient(StdioServerParameters(
        command=sys.executable, args=["-m", "mnemoai.server.server"], env=env,
    ))
    events = []
    try:
        with mcp:
            endpoint = mcp._members[0][1]
            for name, args in [
                ("fs_read", {"path": str(source)}),
                ("file_edit", {"file_path": str(source), "old_string": "VALUE = 1", "new_string": "VALUE = 2"}),
                ("file_edit", {"file_path": str(source), "old_string": "# first\nVALUE = 1", "new_string": "# first\nVALUE = 2"}),
            ]:
                events.append((name, args, endpoint.call_tool_sync(name, args)))
    finally:
        mcp.shutdown()
    assert json.loads(events[1][2])["error"]
    assert json.loads(events[2][2])["success"]
    assert source.read_text() == "# first\nVALUE = 2\n# second\nVALUE = 1\n"
    client = LangGraphClient.__new__(LangGraphClient)
    client._model_scoped_dir = lambda: str(tmp_path / "memory")
    client._open_playbook()
    client.agent = SimpleNamespace(messages=[], usage=UsageTracker())
    client._area_model = Mock(side_effect=AssertionError("no model construction"))
    client._reflection_messages = trace(events)
    client._reflection_source = {"session_id": "real-tools", "turn": 1}
    client.reflect_and_learn("Update the first weights section")
    stored = PlaybookStore(client.playbook.persist_path).snapshot()
    assert len(stored) == 1 and len(stored[0]["source_refs"]) == 2
    client._playbook_task = "Update the first weights section"
    assert stored[0]["strategy"] in context_injection.get_playbook_context(client)
    assert client.agent.usage.totals()["calls"] == 0
    client._area_model.assert_not_called()
    BaseChatModel.invoke.assert_not_called()
    EmbeddingsController.embed.assert_not_called()
