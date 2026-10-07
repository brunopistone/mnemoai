"""Live provider accounting and full-input estimates across save/load/resume."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from mnemoai.client import context_report, usage_tracker
from mnemoai.client.session_log import SessionLog, read_session

pytestmark = pytest.mark.integration


def test_live_usage_and_restore_estimate_include_real_mcp_schemas(
    live_client, _neutral_root, monkeypatch, record_testsuite_property,
):
    agent = live_client.agent
    monkeypatch.setattr(live_client.reviewer, "enabled", False)
    monkeypatch.setattr(agent, "router", None)
    monkeypatch.setattr(agent, "orchestrator_enabled", False)
    monkeypatch.setattr(agent, "graph", agent._build_graph())
    monkeypatch.setattr(agent, "_invoke_tool", lambda *a, **kw: pytest.fail("No tool execution requested"))
    original = agent._stream_once
    responses = []

    def observed(*args, **kwargs):
        response, reasoning = original(*args, **kwargs)
        responses.append(response)
        return response, reasoning

    monkeypatch.setattr(agent, "_stream_once", observed)
    before_usage = agent.usage.totals()
    assert live_client.query("This is a token accounting check. Do not use tools. Reply only OK.").strip()
    measured = agent._last_input_tokens
    assert measured and responses
    known = [response.usage_metadata for response in responses if response is not None and response.usage_metadata]
    assert known
    after_usage = agent.usage.totals()
    assert after_usage["input_tokens"] - before_usage["input_tokens"] == sum(u["input_tokens"] for u in known)
    assert after_usage["output_tokens"] - before_usage["output_tokens"] == sum(u["output_tokens"] for u in known)
    path = _neutral_root / "token-accounting-save.json"
    live_client.save_conversation(path=str(path))
    assert live_client.load_conversation(str(path))
    assert agent._last_input_tokens is None
    parts = context_report.collect(live_client)
    tools = sum(part.tokens for part in parts if part.group == "tools")
    estimate = live_client._count_context_tokens()
    assert tools > 0 and estimate > tools
    assert estimate == sum(part.tokens for part in parts)
    assert agent.usage.totals() == after_usage, "restoring history must not count it as new spending"
    assert "estimated next input" in live_client.usage_report()
    record_testsuite_property("live_restore_count", {
        "reported_last_input": measured, "restored_full_estimate": estimate,
        "tool_schema_estimate": tools, "stream_attempts": len(responses),
    })


def test_live_compaction_usage_and_repeated_restores(
    live_client, _neutral_root, monkeypatch, record_testsuite_property,
):
    agent = live_client.agent
    monkeypatch.setattr(live_client.reviewer, "enabled", False)
    messages = [
        HumanMessage(content="Our fictional project exports CSV and must preserve UTF-8."),
        AIMessage(content="I will preserve UTF-8 and quote embedded commas."),
        HumanMessage(content="Keep the latest export requirement."),
        AIMessage(content="The export requirement is retained."),
    ]
    source = SessionLog(cwd=str(_neutral_root))
    source.log_turn(messages)
    monkeypatch.setattr(agent, "session_log", source)
    agent.messages = messages
    model = live_client._summary_model()
    name = usage_tracker.model_name(model)
    responses = []

    async def observed(messages):
        response = await model.ainvoke(messages)
        responses.append(response)
        return response

    proxy = SimpleNamespace(model_name=name, ainvoke=observed)
    before = agent.usage.totals()
    assert asyncio.run(live_client.conversation_manager.compact(live_client, proxy, agent))
    after = agent.usage.totals()
    assert responses and all(response.usage_metadata for response in responses)
    assert after["input_tokens"] - before["input_tokens"] == sum(r.usage_metadata["input_tokens"] for r in responses)
    assert after["output_tokens"] - before["output_tokens"] == sum(r.usage_metadata["output_tokens"] for r in responses)
    assert any(row["model"] == name and row["calls"] > 0 for row in agent.usage.snapshot())
    summary = live_client.conversation_manager.summary_text
    assert summary and agent._last_input_tokens is None
    expected = live_client._count_context_tokens()
    sizes = []
    for i in range(3):
        destination = SessionLog(cwd=str(_neutral_root))
        agent.session_log = destination
        assert live_client.resume_session(str(source.path))
        sizes.append(live_client._count_context_tokens())
        assert live_client.conversation_manager.summary_text == summary
        assert read_session(destination.path)["checkpoint"]
        assert agent.usage.totals() == after
        source = destination
    assert sizes == [expected] * 3
    path = _neutral_root / "token-compacted-save.json"
    live_client.save_conversation(path=str(path))
    assert live_client.load_conversation(str(path))
    assert live_client._count_context_tokens() == expected
    assert agent.usage.totals() == after
    record_testsuite_property("live_compaction_accounting", {
        "summary_model": name, "summary_calls": len(responses),
        "reported_tokens": after["total_tokens"] - before["total_tokens"],
        "restored_estimates": sizes,
    })
