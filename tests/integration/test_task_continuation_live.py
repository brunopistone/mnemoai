"""Real model/MCP tasks must finish across execution and summary checkpoints."""

import asyncio
import json
import threading
import uuid

import pytest

from mnemoai.client.agent import subagents
from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.managers.agent_conversation_manager import AgentConversationManager

pytestmark = pytest.mark.integration


def _chain(root, length):
    paths = [root / f"{uuid.uuid4().hex}.txt" for _ in range(length)]
    tokens = [f"TOKEN_{uuid.uuid4().hex[:10]}" for _ in paths]
    for index, path in enumerate(paths):
        next_path = str(paths[index + 1]) if index + 1 < length else "END"
        path.write_text(
            f"TOKEN: {tokens[index]}\nNEXT: {next_path}\n"
            + "Background filler; only TOKEN and NEXT are relevant.\n" * 60
        )
    return paths, tokens


def test_live_parent_finishes_after_compaction_and_status_message(
    live_client, _neutral_root, monkeypatch, record_testsuite_property,
):
    paths, tokens = _chain(_neutral_root, 3)
    target = _neutral_root / "completed-chain.txt"
    tools = [t for t in live_client.tools if t.name in {"fs_read", "fs_write"}]
    agent = LangGraphAgent(
        live_client.agent._callback_free_model(), tools,
        "Complete the file task using only these tools. Do not delegate.",
        verbose=False, auto_approve_provider=lambda: "all",
    )
    agent.recursion_limit = 3
    manager = AgentConversationManager(max_tokens=32_000)
    monkeypatch.setattr(manager, "_session_blocks", lambda client=None: [])
    # _compact updates this facade's prompt; restore it when the fixture ends.
    monkeypatch.setattr(live_client, "system_prompt", live_client.system_prompt)
    reads, compactions = [], []
    invoke_tool = agent._invoke_tool

    def invoke(tool, name, args, quiet=False):
        result = invoke_tool(tool, name, args, quiet=quiet)
        if name == "fs_read":
            reads.append(args["path"])
            if len(reads) == 1:
                assert agent.accept_mid_turn("What is the status? Please keep working until the report is saved.")
        return result

    def compact(force=False):
        if not reads:
            return False
        changed = asyncio.run(manager._compact(
            live_client, live_client._summary_model(), agent, keep_recent=2,
        ))
        if changed:
            compactions.append(True)
        return changed

    agent._invoke_tool = invoke
    agent._compact_provider = compact
    answer = agent.invoke(
        f"Follow the chain starting at {paths[0]}. Read each NEXT file until END. "
        f"Create {target} using fs_write(command='create'), containing all TOKEN "
        "values, one per line, in chain order. Read each file once. Do not stop "
        "after a status update or a summary; finish saving the report."
    )
    assert target.exists(), answer
    assert target.read_text().splitlines() == tokens, answer
    assert reads == list(map(str, paths))
    assert compactions
    assert not agent._last_turn_stopped
    record_testsuite_property("parent_continuation", "real actor, MCP tools and summaries; status guidance during work")


def test_live_worker_continues_with_private_summary(live_client, _neutral_root, monkeypatch, record_property):
    paths, tokens = _chain(_neutral_root, 6)
    agent = live_client.agent
    before = list(agent.messages)
    token_count = agent._last_input_tokens
    read_tool = next(t for t in agent.tools if t.name == "fs_read")
    monkeypatch.setattr(agent, "_subagent_tools", lambda kind: [read_tool])
    monkeypatch.setattr(agent, "recursion_limit", 2)
    reads, compactions, attempts, planned = [], [], [], []
    invoke_tool = agent._invoke_tool
    tool_loop = agent._run_tool_calls

    def run_tools(calls, *args, **kwargs):
        planned.extend({"id": c["id"], "args": c["args"]} for c in calls)
        return tool_loop(calls, *args, **kwargs)

    def invoke(tool, name, args, quiet=False):
        reads.append(args["path"])
        result = invoke_tool(tool, name, args, quiet=quiet)
        attempts.append({"args": args, "result": result})
        return result

    def compact(messages, *, force=False, cancelled):
        result = live_client._compact_worker_context(messages, force=True, cancelled=cancelled)
        if result is not None:
            compactions.append(True)
        return result

    monkeypatch.setattr(agent, "_invoke_tool", invoke)
    monkeypatch.setattr(agent, "_run_tool_calls", run_tools)
    monkeypatch.setattr(agent, "_worker_compact_provider", compact)
    done = threading.Event()
    monkeypatch.setattr(agent, "_on_background_complete", lambda ident: done.set(), raising=False)
    prior = {r.agent_id for r in agent._bg_agents.list_all()}
    agent._launch_background_subagent(
        subagents.get_subagent("explore"),
        f"Follow NEXT links starting at {paths[0]} until END. Use fs_read only. "
        "Read each file exactly once; ignore filler. Return every TOKEN value in "
        "chain order. Complete the entire chain even after context summaries.",
        "private summary continuation",
    )
    if not done.wait(300):
        agent._activity.request_stop_all()
        done.wait(5)
        pytest.fail("worker did not finish its finite chain")
    record, = [r for r in agent._bg_agents.list_all() if r.agent_id not in prior]
    assert record.status == "done", record.result
    assert all(token in record.result for token in tokens), record.result
    record_property("worker_calls", json.dumps({"planned": planned, "executed": attempts}, default=str))
    assert len({c["id"] for c in planned}) == len(planned), "a completed tool call was replayed"
    # Search followed by a line read is a new operation, not a replay. Require
    # every linked file in order and forbid re-executing an identical request.
    assert list(dict.fromkeys(reads)) == list(map(str, paths))
    assert len({json.dumps(c["args"], sort_keys=True) for c in planned}) == len(planned)
    assert compactions
    assert agent.messages == before
    assert agent._last_input_tokens == token_count
