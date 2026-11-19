"""Real MCP contracts and provider behavior around memory and tool-loop recovery."""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

pytestmark = pytest.mark.integration


def direct_actor(client, monkeypatch):
    monkeypatch.setattr(client.reviewer, "enabled", False)
    monkeypatch.setattr(client.agent, "router", None)
    monkeypatch.setattr(client.agent, "orchestrator_enabled", False)
    monkeypatch.setattr(client.agent, "_checkpoint_graph", None)
    monkeypatch.setattr(client.agent, "graph", client.agent._build_graph())


def test_real_mcp_append_accepts_the_recorded_payload_and_rejects_ambiguity(live_client, tmp_path):
    tools = {tool.name: tool for tool in live_client.agent.tools}
    target = tmp_path / ".gitignore"
    target.write_text("keep-existing/\n")
    tools["fs_read"].invoke({"path": str(target)})
    result = json.loads(tools["fs_write"].invoke({
        "path": str(target), "command": "append", "file_text": ".claude/\n.codex/\n",
    }))
    assert result["changed"] and target.read_text() == "keep-existing/\n.claude/\n.codex/\n"
    before = target.read_bytes()
    rejected = json.loads(tools["fs_write"].invoke({
        "path": str(target), "command": "append", "file_text": "one", "new_str": "two",
    }))
    assert rejected["error"] and "conflicting" in rejected["message"]
    assert target.read_bytes() == before


def test_live_greeting_has_no_task_reminder_or_tool_memory(live_client, monkeypatch):
    lookup = Mock(side_effect=AssertionError("A greeting must not retrieve tool-use episodes"))
    monkeypatch.setattr(live_client, "episodic_memory", SimpleNamespace(retrieve_similar_episodes=lookup))
    monkeypatch.setattr(live_client.reviewer, "enabled", False)
    requests = []
    original = live_client.agent._stream_once

    def observe(model, messages, *args, **kwargs):
        requests.append(list(messages))
        return original(model, messages, *args, **kwargs)

    monkeypatch.setattr(live_client.agent, "_stream_once", observe)
    monkeypatch.setattr(live_client.agent, "_invoke_tool", lambda *a, **kw: pytest.fail("Greeting invoked a tool"))
    answer = live_client.query("Hello!")
    assert answer.strip()
    lookup.assert_not_called()
    assert requests
    assert [m.content for m in requests[-1] if isinstance(m, HumanMessage)] == ["Hello!"]
    assert "<active-task>" not in str(requests)
    assert "tools_observed" not in str(requests)


def test_live_model_recovers_from_a_seeded_repeated_read(live_client, tmp_path, monkeypatch):
    direct_actor(live_client, monkeypatch)
    target = tmp_path / "codeword.txt"
    codeword = "MAPLE-" + uuid.uuid4().hex[:10]
    target.write_text(codeword)
    original = live_client.agent._stream_once
    invoke = live_client.agent._invoke_tool
    calls = []
    attempts = []

    def seed(model, messages, *args, **kwargs):
        attempts.append(1)
        if len(attempts) <= 4:
            return AIMessage(content="", tool_calls=[{
                "id": f"seed-{len(attempts)}", "name": "fs_read", "args": {"path": str(target)},
            }]), False
        return original(model, messages, *args, **kwargs)

    def observe(tool, name, args, **kwargs):
        calls.append(name)
        return invoke(tool, name, args, **kwargs)

    monkeypatch.setattr(live_client.agent, "_stream_once", seed)
    monkeypatch.setattr(live_client.agent, "_invoke_tool", observe)
    answer = live_client.query(f"Read {target} and report the exact codeword. Once you have it, answer directly.")
    assert codeword in answer, answer
    assert calls.count("fs_read") == 3
    assert not live_client.agent._last_turn_stopped


def test_live_append_preserves_existing_ignore_rules(live_client, _neutral_root, monkeypatch):
    direct_actor(live_client, monkeypatch)
    monkeypatch.setattr(live_client, "auto_approve_mode", "all")
    target = _neutral_root / "reliability-append" / ".gitignore"
    target.parent.mkdir(exist_ok=True)
    original = "# Existing project-specific rules\nprivate-data/\n!private-data/example.json\n"
    target.write_text(original)
    answer = live_client.query(
        "Edit reliability-append/.gitignore: append exactly two directory ignore patterns, .claude/ and .codex/. "
        "Preserve every existing line in its original order. Use the file tools and verify the saved result."
    )
    assert answer.strip()
    contents = target.read_text()
    assert contents.startswith(original), contents
    added = [line for line in contents[len(original):].splitlines() if line and not line.startswith("#")]
    assert added == [".claude/", ".codex/"], contents
    assert not live_client.agent._last_turn_stopped
