"""Live approval policy for foreground orchestration and unattended workers."""

import json
import time

import pytest

from mnemoai.utils.config import config

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "parallel,approved",
    [(False, False), (False, True), (True, True)],
    ids=["inline-decline", "inline-approve", "parallel-no-prompt"],
)
def test_live_orchestrator_preserves_foreground_approval(
    live_client, _neutral_root, monkeypatch, parallel, approved
):
    """Exercise the real scheduler, model and tools, with a controlled UI reply."""
    from mnemoai.client.agent import confirmation_gate

    agent = live_client.agent
    monkeypatch.setattr(live_client, "plan_mode_active", False)
    monkeypatch.setattr(live_client, "auto_approve_mode", "off")
    monkeypatch.setattr(agent, "_trusted_confirm_categories", set())
    monkeypatch.setattr(agent, "_max_subagent_concurrency", 2)
    monkeypatch.setitem(config._config_data, "REQUIRE_WRITE_CONFIRMATION", True)
    monkeypatch.setattr(confirmation_gate.sys.stdin, "isatty", lambda: True)
    prompts, attempts = [], []
    confirm = agent._confirm_tool

    def ui(*args):
        prompts.append(args)
        return "yes" if approved else "no"

    def observe(name, args):
        result = confirm(name, args)
        if name == "fs_write":
            attempts.append(result)
        return result

    monkeypatch.setattr(agent, "_confirm_ui", ui, raising=False)
    monkeypatch.setattr(agent, "_confirm_tool", observe)
    targets = [
        _neutral_root / f"orchestrator-{parallel}-{approved}-{i}.txt"
        for i in range(2 if parallel else 1)
    ]
    tasks = [
        {
            "description": (
                f"Call fs_write once with command='create', path='{target}', "
                "file_text='orchestrator-approved'. Make the tool call, not an example. "
                "Touch only that file. If refused, stop and report the refusal without "
                "trying another tool or repeating the request."
            ),
            "category": "full",
            "depends_on": [],
        }
        for target in targets
    ]
    results = agent._run_subtasks_scheduled(tasks)
    assert len(results) == len(targets)
    assert len(attempts) == len(targets), (
        "each real model worker must attempt its write"
    )
    assert len(prompts) == (0 if parallel else len(targets))
    permitted = approved and not parallel
    assert attempts == [permitted] * len(targets)
    for target in targets:
        if permitted:
            assert target.read_text() == "orchestrator-approved"
        else:
            assert not target.exists()
    assert not agent._is_headless()


@pytest.mark.parametrize("approved", [False, True], ids=["decline", "approve"])
def test_foreground_background_launch_uses_one_approval(
    live_client, monkeypatch, approved
):
    """A real MCP background command is launched only after the main UI decision."""
    from mnemoai.client.agent import confirmation_gate

    agent = live_client.agent
    monkeypatch.setattr(live_client, "plan_mode_active", False)
    monkeypatch.setattr(live_client, "auto_approve_mode", "off")
    monkeypatch.setattr(agent, "_trusted_confirm_categories", set())
    monkeypatch.setitem(config._config_data, "REQUIRE_BASH_CONFIRMATION", True)
    monkeypatch.setattr(confirmation_gate.sys.stdin, "isatty", lambda: True)
    prompts, launched = [], []
    invoke = agent._invoke_tool

    def ui(*args):
        prompts.append(args)
        return "yes" if approved else "no"

    def observe(tool, name, args, quiet=False):
        if name == "start_background_task":
            launched.append(args)
        return invoke(tool, name, args, quiet=quiet)

    monkeypatch.setattr(agent, "_confirm_ui", ui, raising=False)
    monkeypatch.setattr(agent, "_invoke_tool", observe)
    messages = []
    agent._run_tool_calls(
        [
            {
                "name": "start_background_task",
                "id": "live-launch",
                "args": {"command": "printf launch-approved"},
            }
        ],
        agent.tools,
        messages,
    )
    assert len(prompts) == 1 and prompts[0][2] == "bash"
    if not approved:
        assert not launched
        assert messages[0].content == "User declined to run this command."
        return

    assert len(launched) == 1
    task_id = json.loads(messages[0].content)["task_id"]
    output_tool = next(tool for tool in agent.tools if tool.name == "get_task_output")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = json.loads(output_tool.invoke({"task_id": task_id}))
        if result.get("status") == "completed":
            assert "launch-approved" in result["output"]
            break
        time.sleep(0.05)
    else:
        pytest.fail("approved background command did not complete")


@pytest.mark.parametrize("background", [False, True], ids=["foreground", "background"])
@pytest.mark.parametrize("mode", ["off", "edits"])
def test_live_delegated_write_respects_auto_mode_without_prompting(
    live_client, _neutral_root, monkeypatch, background, mode, record_testsuite_property
):
    agent = live_client.agent
    record_testsuite_property("chat_model", live_client.model_name_for_log())
    monkeypatch.setattr(live_client, "plan_mode_active", False)
    monkeypatch.setattr(live_client, "auto_approve_mode", mode)
    monkeypatch.setattr(agent, "_trusted_confirm_categories", set())
    monkeypatch.setattr(agent, "recursion_limit", 4)
    monkeypatch.setitem(config._config_data, "REQUIRE_WRITE_CONFIRMATION", True)
    monkeypatch.setitem(config._config_data, "REQUIRE_BASH_CONFIRMATION", True)
    target = _neutral_root / f"delegated-{mode}-{background}.txt"
    observed, prompts = [], []
    confirm = agent._confirm_tool

    def observe_confirmation(name, args):
        allowed = confirm(name, args)
        if name in ("fs_write", "file_edit", "execute_bash"):
            observed.append((name, allowed))
        return allowed

    def unexpected_prompt(*args, **kwargs):
        prompts.append(args)
        return False

    monkeypatch.setattr(agent, "_confirm_tool", observe_confirmation)
    monkeypatch.setattr(agent, "_prompt_confirm", unexpected_prompt)
    prompt = (
        f"Call fs_write with command='create', path='{target}', and "
        "file_text='live-delegation-ok'. Make the tool call, not a code example. "
        "Touch only that file. If the tool is refused, stop and report the refusal; "
        "do not attempt another tool or request human approval."
    )

    if background:
        prior_ids = {record.agent_id for record in agent._bg_agents.list_all()}
        agent._handle_spawn_agent(
            "general-purpose", prompt, "live permission test", run_in_background=True
        )
        records = [
            record
            for record in agent._bg_agents.list_all()
            if record.agent_id not in prior_ids
        ]
        assert len(records) == 1
        record = records[0]
        deadline = time.monotonic() + 120
        while record.status == "running" and time.monotonic() < deadline:
            time.sleep(0.05)
        if record.status == "running":
            agent._activity.request_stop_all()
            pytest.fail("background model did not finish within 120 seconds")
        assert record.status == "done", record.result
        agent.drain_background_completions()
    else:
        agent._handle_spawn_agent(
            "general-purpose", prompt, "live permission test", run_in_background=False
        )

    assert prompts == [], "a delegated agent attempted interactive approval"
    assert observed, "the live model never attempted the requested tool call"
    if mode == "off":
        assert not any(allowed for _, allowed in observed)
        assert not target.exists()
    else:
        assert any(allowed for _, allowed in observed)
        assert target.read_text() == "live-delegation-ok"
