"""Real MCP failure/recovery evidence, real model extraction, isolated persistence."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException

from mnemoai.client import learned
from mnemoai.client.memory.playbook_records import PlaybookEntry
from mnemoai.client.memory.playbook_store import PlaybookStore
from mnemoai.client.memory.reflector import Reflector
from mnemoai.utils.config import config

pytestmark = pytest.mark.integration


def test_live_reflection_and_disable_restore(live_client, tmp_path, monkeypatch,
                                           record_testsuite_property):
    target = tmp_path / "settings.py"
    target.write_text("# first section\nVALUE = 1\n# second section\nVALUE = 1\n")
    tools = {tool.name: tool for tool in live_client.tools}
    trace = [HumanMessage(content="Update only the first section's VALUE.")]

    def call(name, args):
        call_id = f"f2-live-{len(trace)}"
        trace.append(AIMessage(content="", tool_calls=[
            {"name": name, "args": args, "id": call_id},
        ]))
        try:
            result = tools[name].invoke(args)
        except ToolException as exc:
            result = "Error: " + str(exc)
        trace.append(ToolMessage(content=result, tool_call_id=call_id))
        return result

    call("fs_read", {"path": str(target)})
    failed = call("file_edit", {
        "file_path": str(target), "old_string": "VALUE = 1", "new_string": "VALUE = 2",
    })
    assert target.read_text().count("VALUE = 1") == 2
    assert Reflector.__new__(Reflector)._is_actual_error(failed.lower(), failed)
    call("file_edit", {
        "file_path": str(target), "old_string": "# first section\nVALUE = 1",
        "new_string": "# first section\nVALUE = 2",
    })
    assert target.read_text() == "# first section\nVALUE = 2\n# second section\nVALUE = 1\n"

    store = PlaybookStore(str(tmp_path / "playbook"))
    monkeypatch.setattr(live_client, "playbook", store)
    monkeypatch.setattr(live_client, "reflector", Reflector(str(tmp_path / "playbook")))
    monkeypatch.setattr(live_client, "_reflection_messages", trace, raising=False)
    monkeypatch.setattr(live_client, "_reflection_source",
                        {"session_id": "live-reflection", "turn": 1}, raising=False)
    monkeypatch.setattr(live_client, "_last_reflected_source", None, raising=False)
    monkeypatch.setattr(live_client, "system_prompt", "Live reflection verification.")
    monkeypatch.setattr(live_client.agent, "system_prompt", "Live reflection verification.")
    monkeypatch.setitem(config._config_data, "PLAYBOOK",
                        {**config.get("PLAYBOOK", {}), "REFLECTION_TIMEOUT": 120})
    record_testsuite_property("reflector_model", live_client._area_usage_name("REFLECTOR"))
    live_client.reflect_and_learn(
        "Update only the first section's VALUE. The ambiguous edit failed; retrying "
        "with the section header succeeded. Extract the supported editing lesson."
    )
    entries = store.snapshot()
    assert entries, live_client.reflector.last_error or "Model extracted no lesson"
    assert all(e["provenance"] == "model" and e["source_refs"] for e in entries)
    chosen = entries[0]
    live_client.refresh_playbook_context()
    assert chosen["strategy"] in live_client.agent.system_prompt
    assert "Updated" in learned.run(
        live_client, "disable " + chosen["id"], confirm=lambda _: True, edit=lambda x: x,
    )
    assert chosen["strategy"] not in live_client.agent.system_prompt
    reloaded = PlaybookStore(str(tmp_path / "playbook"))
    assert next(e for e in reloaded.snapshot() if e["id"] == chosen["id"])["status"] == "disabled"
    assert "Updated" in learned.run(
        live_client, "restore " + chosen["id"], confirm=lambda _: True, edit=lambda x: x,
    )
    assert chosen["strategy"] in live_client.agent.system_prompt


def test_live_retraction_filters_next_request_and_replayed_evidence(
    live_client, tmp_path, monkeypatch
):
    store = PlaybookStore(str(tmp_path / "retraction"))
    source = {"session_id": "f4-live", "turn": 1, "tool_call_id": "withdrawn-call"}
    store.append(PlaybookEntry(
        "F4 live fixture", "F4_WITHDRAWN_FIXTURE_49391", "fixture",
        scope=str(tmp_path), source_refs=[source],
    ))
    # Scope belongs to the neutral test workspace, not the developer's project.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(live_client, "playbook", store)
    monkeypatch.setattr(live_client, "system_prompt", "You are a test assistant.")
    monkeypatch.setattr(live_client.agent, "system_prompt", "You are a test assistant.")
    live_client.refresh_playbook_context()
    stale = live_client.agent._system_message()
    target = store.snapshot()[0]
    result = learned.run(
        live_client, f"retract {target['id']} This test premise is withdrawn",
        confirm=lambda _: True, edit=lambda text: text,
    )
    assert "retracted" in result
    assert "F4_WITHDRAWN_FIXTURE_49391" not in live_client.agent.system_prompt
    assert PlaybookStore(store.persist_path).snapshot()[0]["status"] == "retracted"

    observed = []
    class RecordingModel:
        def stream(self, messages, config=None):
            observed.extend(messages)
            yield from live_client.model.stream(messages, config={"callbacks": []})

    response, _ = live_client.agent._stream_once(
        RecordingModel(), [stale, HumanMessage(content="Reply with OK only.")],
        {}, quiet=True,
    )
    assert response is not None and response.content
    assert "F4_WITHDRAWN_FIXTURE_49391" not in json.dumps([m.content for m in observed])
    assert "F4_WITHDRAWN_FIXTURE_49391" in str(stale.content)  # history wasn't rewritten

    def forbidden(*args, **kwargs):
        pytest.fail("Quarantined evidence must not make another provider request")
    replay = [
        HumanMessage(content="A repeated old source"),
        AIMessage(content="", tool_calls=[{"name": "fs_read", "args": {}, "id": "withdrawn-call"}]),
        ToolMessage(content="Error: old fixture", tool_call_id="withdrawn-call"),
    ]
    reflector = Reflector()
    assert reflector.reflect_on_trajectory(
        replay, "task", model=type("NoCall", (), {"invoke": forbidden})(),
        source={"session_id": "f4-live", "turn": 1}, scope=str(tmp_path),
        evidence_filter=lambda evidence: store.filter_learning_evidence(evidence, str(tmp_path)),
    ) == []
    assert reflector.last_error is None
