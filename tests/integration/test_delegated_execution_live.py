"""Live approval policy for foreground orchestration and unattended workers."""

import asyncio
import json
import threading
import time

import pytest

from mnemoai.utils.config import config

pytestmark = pytest.mark.integration


def test_live_last_background_agent_triggers_one_consolidated_answer(
    live_client, monkeypatch, record_testsuite_property,
):
    """Real parent and worker models; controlled timing, actual UI delivery queue."""
    from types import SimpleNamespace

    from mnemoai.client.agent import subagents
    from mnemoai.client.agent.agent import LangGraphAgent
    from mnemoai.client.ui.tui import PinnedPromptReader, _ExitRepl

    agent = LangGraphAgent(
        live_client.agent._callback_free_model(), [],
        "Compare the delegated reports. While any required report is pending, "
        "begin your reply with INTERIM and state what is missing. Once both "
        "arrive, begin with FINAL and give both reported numbers and their sum.",
        verbose=False,
    )
    releases = [threading.Event(), threading.Event()]
    original = agent._run_one_subagent
    def gated_worker(subagent, prompt, label, **kwargs):
        if not releases[int(label)].wait(120):
            raise RuntimeError("fixture worker release timed out")
        return original(subagent, prompt, label, **kwargs)
    monkeypatch.setattr(agent, "_run_one_subagent", gated_worker)
    monkeypatch.setattr(agent, "_prompt_confirm",
                        lambda *a, **kw: pytest.fail("Background work must never prompt"))
    queries, answers = [], []
    def dispatch(text):
        queries.append(text)
        answer = agent.invoke(text)
        answers.append(answer)
        if len(queries) <= 2:
            releases[len(queries) - 1].set()
        if not any(agent.background_status()):
            return _ExitRepl
    reader = PinnedPromptReader(
        prompt_text=lambda: "> ", commands=[], dispatch=dispatch,
        background_pending=agent.has_undelivered_background,
    )
    reader._app = SimpleNamespace(invalidate=lambda: None, exit=lambda: None)
    reader._buffer.text = "unsent live-test draft"
    async def no_terminal(func, *args, **kwargs):
        func()
    monkeypatch.setattr("mnemoai.client.ui.tui.run_in_terminal", no_terminal)
    monkeypatch.setattr("mnemoai.client.ui.tui.notify.notify", lambda text: None)
    agent._on_background_complete = lambda ident: reader.notify_background_complete()
    async def run():
        reader._loop = asyncio.get_running_loop()
        reader._queue = asyncio.Queue()
        for i, number in enumerate([11, 29]):
            agent._launch_background_subagent(
                subagents.get_subagent("explore"),
                f"Connectivity fixture. No tools. Reply only REPORT_{i}: {number}.", str(i),
            )
        reader._queue.put_nowait("Compare both delegated reports, including their numbers and sum.")
        reader._pending = 1
        await asyncio.wait_for(reader._worker(), timeout=360)
        await asyncio.sleep(0)
        assert reader._queue.empty()
    try:
        asyncio.run(run())
    finally:
        agent.request_cancel()
        for release in releases:
            release.set()
    assert len(queries) == 3 and queries[1:] == ["", ""]
    assert all("INTERIM" in answer for answer in answers[:2]), answers
    assert "FINAL" in answers[-1] and all(str(n) in answers[-1] for n in [11, 29, 40]), answers
    assert agent.background_status() == (0, 0)
    assert reader._buffer.text == "unsent live-test draft"
    record_testsuite_property(
        "background_final_delivery",
        "real parent and two worker model calls; controlled staggered completion; "
        "actual UI queue automatically delivered one consolidated final answer",
    )


def test_live_background_agent_recovers_from_an_injected_incomplete_stream(
    live_client, monkeypatch, record_testsuite_property
):
    """A real provider stream is interrupted once; the same worker then completes."""
    agent = live_client.agent
    bind = agent._bind_tools
    requests, states = [], []

    class InterruptOnce:
        def __init__(self, model):
            self.model = model

        def stream(self, messages, config=None):
            requests.append(list(messages))
            states.append(agent._activity.snapshot()[-1].status)
            attempt = len(requests)
            stream = iter(self.model.stream(messages, config=config))
            try:
                for chunk in stream:
                    yield chunk
                    if attempt == 1:
                        raise ValueError("Incomplete Bedrock response stream: missing messageStop event.")
            finally:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()

    monkeypatch.setattr(agent, "_bind_tools", lambda model, tools: InterruptOnce(bind(model, tools)))
    monkeypatch.setattr(agent, "_subagent_tools", lambda subagent: [])
    monkeypatch.setattr(agent, "_empty_response_retries", 2)
    monkeypatch.setattr(agent, "_sleep_or_cancel", lambda delay: False)
    monkeypatch.setattr(agent, "_prompt_confirm",
                        lambda *a, **kw: pytest.fail("background retry must never request approval"))
    finished = threading.Event()
    monkeypatch.setattr(agent, "_on_background_complete", lambda ident: finished.set(), raising=False)
    before = {r.agent_id for r in agent._bg_agents.list_all()}
    agent._handle_spawn_agent(
        "general-purpose",
        "Connectivity fixture: do not use tools. Reply with exactly STREAM_RETRY_OK.",
        "live incomplete-stream retry", run_in_background=True,
    )
    if not finished.wait(120):
        agent.request_cancel()
        finished.wait(5)
        pytest.fail("background retry did not complete")
    record = next(r for r in agent._bg_agents.list_all() if r.agent_id not in before)
    assert record.status == "done", record.result
    assert "STREAM_RETRY_OK" in record.result
    assert len(requests) == 2 and requests[0] == requests[1]
    assert states == ["running", "running"]
    assert agent._activity.snapshot()[-1].status == "done"
    agent.drain_background_completions()
    record_testsuite_property(
        "incomplete_stream_test",
        "real configured provider + background worker; first response interrupted with the reported error",
    )


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
