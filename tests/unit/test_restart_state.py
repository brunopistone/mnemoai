"""Conversation continuity and failure handling for model-change restarts."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mnemoai import main as main_mod
from mnemoai.client import restart_state
from mnemoai.client.client import LangGraphClient
from mnemoai.client.managers.agent_conversation_manager import AgentConversationManager
from mnemoai.client.session_log import SessionLog, read_session
from mnemoai.client.ui.chat_interface import ChatInterface
from mnemoai.client.ui.tui import PinnedPromptReader
from mnemoai.utils.config import config

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(config._prompts_data, "SYSTEM_PROMPT", "NEW MODEL INSTRUCTIONS")


def client(*, recording=True):
    result = LangGraphClient.__new__(LangGraphClient)
    result.agent = SimpleNamespace(
        messages=[], system_prompt="NEW MODEL INSTRUCTIONS",
        _last_input_tokens=999, session_log=SessionLog() if recording else None,
    )
    result.system_prompt = result.agent.system_prompt
    result.conversation_manager = AgentConversationManager(max_tokens=8000)
    result.conversation_manager._session_blocks = lambda *_: []
    result.current_conversation_path = None
    result.auto_approve_mode = "off"
    result.plan_mode_active = False
    result.reviewer = None
    result.tools = []
    result.mcp_client = SimpleNamespace(shutdown=Mock())
    result._print_context_size = lambda: None
    return result


@pytest.mark.parametrize("arguments", [
    ["mnemoai", "--resume"],
    ["mnemoai", "--resume", "old-session", "--auto"],
    ["mnemoai", "--resume=old-session", "--continue", "--no-verbose"],
    ["/path with spaces/mnemoai", "--restart-state", "/old/path", "--no-verbose"],
    ["mnemoai", "--restart-state=/old/path", "--continue"],
])
def test_restart_arguments_replace_stale_session_selection(arguments):
    updated = restart_state.argv(arguments, "/new path/restart-state.json")
    assert updated[0] == arguments[0]
    assert updated[-2:] == ["--restart-state", "/new path/restart-state.json"]
    assert "--resume" not in updated and "--continue" not in updated and "--auto" not in updated
    assert not any(arg.startswith("--resume=") for arg in updated)
    assert updated.count("--no-verbose") == arguments.count("--no-verbose")
    assert updated.count("--restart-state") == 1


@pytest.mark.parametrize("recording", [True, False])
def test_loaded_file_unsaved_turns_and_tool_results_survive_repeated_restarts(tmp_path, recording):
    opened = tmp_path / "chat with spaces.json"
    opened.write_text(json.dumps({"messages": [
        {"role": "user", "content": [{"text": "original request"}]},
        {"role": "assistant", "content": [{"text": "original answer"}]},
    ]}))
    original_file = opened.read_bytes()
    active = client(recording=recording)
    assert active.load_conversation(str(opened))
    active.agent.messages.extend([
        HumanMessage(content="unsaved follow-up — café"),
        AIMessage(content="", tool_calls=[{"name": "fs_read", "args": {"path": "code.py"}, "id": "read-1"}]),
        ToolMessage(content="the file contents", tool_call_id="read-1", name="fs_read"),
        AIMessage(content="unsaved answer"),
    ])
    active.auto_approve_mode = "edits"
    expected = active.snapshot_for_restart()["messages"]
    for _ in range(4):
        path = restart_state.write(active.snapshot_for_restart(draft="unfinished input"))
        assert path.stat().st_mode & 0o777 == 0o600
        fresh = client(recording=recording)
        assert fresh.restore_after_restart(restart_state.read(path))
        assert fresh.snapshot_for_restart()["messages"] == expected
        assert fresh.current_conversation_path == str(opened)
        assert fresh.agent._last_input_tokens is None
        assert fresh.auto_approve_mode == "edits"
        assert opened.read_bytes() == original_file
        path.unlink()
        active = fresh
    active.save_conversation()
    assert json.loads(opened.read_text())["messages"][-1]["content"][0]["text"] == "unsaved answer"


def test_compacted_context_and_full_replay_survive_restart():
    previous = client()
    full = [HumanMessage(content="earlier request"), AIMessage(content="earlier response"),
            HumanMessage(content="current request"), AIMessage(content="current response")]
    previous.agent.session_log.log_turn(full)
    previous.agent.messages[:] = full[-2:]
    previous.agent.session_log.log_compaction(summary="Original goal and completed work.", kept=full[-2:])
    previous.agent.session_log.set_label("Named conversation")
    previous.conversation_manager.summary_text = "Original goal and completed work."
    previous.agent.system_prompt = "OLD MODEL INSTRUCTIONS"
    snapshot = previous.snapshot_for_restart()
    fresh = client()
    assert fresh.restore_after_restart(snapshot)
    assert fresh.agent.messages == full[-2:]
    assert "NEW MODEL INSTRUCTIONS" in fresh.agent.system_prompt
    assert "OLD MODEL INSTRUCTIONS" not in fresh.agent.system_prompt
    assert fresh.conversation_manager.summary_text == snapshot["summary"]
    assert fresh.session_label() == "Named conversation"
    restored = read_session(fresh.agent.session_log.path)
    assert len(restored["all_messages"]) == 4
    assert len(restored["messages"]) == 2
    assert restored["summary"] == snapshot["summary"]


def test_summary_only_is_not_mistaken_for_an_empty_chat():
    old = client(recording=False)
    old.conversation_manager.summary_text = "The conversation has been summarized."
    fresh = client(recording=False)
    assert fresh.restore_after_restart(old.snapshot_for_restart())
    assert fresh.conversation_manager.summary_text == old.conversation_manager.summary_text


def test_plan_mode_cannot_be_widened_by_old_auto_launch_flags():
    old = client()
    old.plan_mode_active = True
    old.auto_approve_mode = "all"  # even an inconsistent old state must stay read-only
    fresh = client()
    assert not fresh.restore_after_restart(old.snapshot_for_restart())
    assert fresh.plan_mode_active
    assert fresh.auto_approve_mode == "off"


def test_snapshot_write_failure_cleans_up_and_keeps_current_chat(tmp_path, monkeypatch, capsys):
    old = client()
    old.agent.messages = [HumanMessage(content="keep this unsaved conversation")]
    ui = ChatInterface(old)
    monkeypatch.setattr(restart_state.os, "fsync", Mock(side_effect=OSError("disk full")))
    execute = Mock()
    monkeypatch.setattr("mnemoai.client.ui.chat_interface.os.execv", execute)
    ui._restart_in_place()
    assert old.agent.messages[0].content == "keep this unsaved conversation"
    old.mcp_client.shutdown.assert_not_called()
    execute.assert_not_called()
    assert not list((tmp_path / "home" / "restarts").glob("*.json"))
    assert "still open" in capsys.readouterr().out


def test_unserializable_state_does_not_shutdown_the_client(monkeypatch):
    old = client()
    old.agent.messages = [AIMessage(content="", tool_calls=[
        {"name": "tool", "id": "1", "args": {"invalid": object()}},
    ])]
    execute = Mock()
    monkeypatch.setattr("mnemoai.client.ui.chat_interface.os.execv", execute)
    ChatInterface(old)._restart_in_place()
    execute.assert_not_called()
    old.mcp_client.shutdown.assert_not_called()


class Reexecuted(BaseException):
    pass


@pytest.mark.parametrize("section,label", [("MODEL_ID", "Chat model"), ("VISION_MODEL_ID", "Vision model")])
def test_chat_and_vision_changes_handoff_current_state(monkeypatch, section, label):
    old = client()
    old.agent.messages = [HumanMessage(content="active conversation"), AIMessage(content="latest reply")]
    old.agent.session_log.log_turn(old.agent.messages)
    ui = ChatInterface(old)
    monkeypatch.setattr("mnemoai.client.ui.chat_interface.run_model_override", lambda: SimpleNamespace(
        section=section, label=label, model_name="new-model", model_type="bedrock",
        enabled_feature=False, follows_chat=False, parameters_reset=False,
    ))
    monkeypatch.setattr(sys, "argv", ["mnemoai", "--resume", "outdated", "--auto", "--no-verbose"])
    executed = []
    def execute(binary, args):
        executed.append(args)
        raise Reexecuted
    monkeypatch.setattr("mnemoai.client.ui.chat_interface.os.execv", execute)
    with pytest.raises(Reexecuted):
        ui._dispatch("/model")
    state = restart_state.read(executed[0][-1])
    assert state["messages"][-1]["content"][0]["text"] == "latest reply"
    assert state["auto_mode"] == "off"
    assert "--resume" not in executed[0] and "--auto" not in executed[0]
    old.mcp_client.shutdown.assert_called_once()
    assert old.agent.session_log.path.exists()


def test_pending_agents_defer_restart_until_reports_are_delivered(monkeypatch):
    old = client()
    status = [1, 0]
    old.agent.background_status = lambda: tuple(status)
    ui = ChatInterface(old)
    execute = Mock(side_effect=Reexecuted)
    monkeypatch.setattr("mnemoai.client.ui.chat_interface.os.execv", execute)
    ui._restart_in_place()
    execute.assert_not_called()
    old.mcp_client.shutdown.assert_not_called()
    status[:] = [0, 1]
    ui._finish_pending_restart()
    execute.assert_not_called()
    old.agent.messages.append(HumanMessage(content="delivered final report"))
    status[:] = [0, 0]
    with pytest.raises(Reexecuted):
        ui._finish_pending_restart()
    assert restart_state.read(execute.call_args.args[1][-1])["messages"][0]["content"][0]["text"] == "delivered final report"


def test_queued_input_and_unsent_draft_are_preserved(monkeypatch):
    old = client()
    ui = ChatInterface(old)
    reader = SimpleNamespace(
        _pending=1, _queued_lines=["queued message now handled"],
        _buffer=SimpleNamespace(text="unfinished draft"),
    )
    reader.run_dialog = lambda action: action()
    ui._pinned_reader = reader
    execute = Mock(side_effect=Reexecuted)
    monkeypatch.setattr("mnemoai.client.ui.chat_interface.os.execv", execute)
    ui._restart_in_place()
    execute.assert_not_called()
    old.agent.messages.append(HumanMessage(content="queued message now handled"))
    reader._pending = 0
    reader._queued_lines.clear()
    with pytest.raises(Reexecuted):
        ui._finish_pending_restart()
    state = restart_state.read(execute.call_args.args[1][-1])
    assert state["draft"] == "unfinished draft"
    assert state["messages"][0]["content"][0]["text"] == "queued message now handled"


def test_undelivered_guidance_prevents_a_deferred_restart(monkeypatch):
    old = client()
    old.agent._mid_turn_queue = ["guidance to requeue"]
    ui = ChatInterface(old)
    ui._pending_restart_notice = ""
    execute = Mock(side_effect=Reexecuted)
    monkeypatch.setattr("mnemoai.client.ui.chat_interface.os.execv", execute)
    ui._finish_pending_restart()
    execute.assert_not_called()
    old.agent.messages.append(HumanMessage(content=old.agent._mid_turn_queue.pop()))
    with pytest.raises(Reexecuted):
        ui._finish_pending_restart()
    assert restart_state.read(execute.call_args.args[1][-1])["messages"]


def test_stale_background_wakeup_cannot_strand_a_restart(monkeypatch):
    old = client()
    ui = ChatInterface(old)
    ui._pinned_reader = SimpleNamespace(_pending=1, _queued_lines=[])
    ui._pending_restart_notice = ""
    execute = Mock(side_effect=Reexecuted)
    monkeypatch.setattr("mnemoai.client.ui.chat_interface.os.execv", execute)
    with pytest.raises(Reexecuted):
        ui._finish_pending_restart(in_dialog=True)


def test_collapsed_paste_survives_restart_as_full_draft(monkeypatch):
    old = client()
    ui = ChatInterface(old)
    reader = PinnedPromptReader(prompt_text=lambda: "> ", commands=[], dispatch=lambda _: None)
    draft = "Instructions:\n" + "long pasted content\n" * 100
    reader.restore_draft(draft)
    assert len(reader._buffer.text) < len(draft)
    ui._pinned_reader = reader
    execute = Mock(side_effect=Reexecuted)
    monkeypatch.setattr("mnemoai.client.ui.chat_interface.os.execv", execute)
    with pytest.raises(Reexecuted):
        ui._restart_in_place()
    state = restart_state.read(execute.call_args.args[1][-1])
    fresh = PinnedPromptReader(prompt_text=lambda: "> ", commands=[], dispatch=lambda _: None)
    fresh.restore_draft(state["draft"])
    assert fresh._expand_pastes(fresh._buffer.text) == draft
    assert fresh._buffer.cursor_position == len(fresh._buffer.text)


def boot(monkeypatch, target, *, startup_error=False, restore_error=False):
    target.start = Mock(side_effect=RuntimeError("startup failed") if startup_error else None)
    if restore_error:
        target.restore_after_restart = Mock(side_effect=ValueError("restore failed"))
    monkeypatch.setattr("mnemoai.client.client.LangGraphClient", lambda **kw: target)
    seen = {}
    class UI:
        def __init__(self, *a):
            pass
        def show_welcome(self):
            seen["welcome"] = True
        def run_chat_loop(self, **kwargs):
            seen["loop"] = kwargs
            seen["draft"] = getattr(self, "_restart_draft", "")
    monkeypatch.setattr("mnemoai.client.ui.chat_interface.ChatInterface", UI)
    monkeypatch.setattr(main_mod, "StartupLoader", Mock())
    monkeypatch.setattr(main_mod.screen, "fresh", lambda: None)
    return seen


@pytest.mark.parametrize("empty", [True, False])
def test_startup_uses_handoff_instead_of_old_resume_and_consumes_it(monkeypatch, empty):
    old = client()
    if not empty:
        old.agent.messages = [HumanMessage(content="current history")]
    path = restart_state.write(old.snapshot_for_restart("draft"))
    target = client()
    seen = boot(monkeypatch, target)
    picker = Mock(side_effect=AssertionError("restart must not open the picker"))
    monkeypatch.setattr(main_mod, "_resume_session", picker)
    main_mod.main(resume="obsolete-session", restart_path=str(path))
    assert target.agent.messages == old.agent.messages
    assert seen["loop"] == {"welcome": False}
    assert seen["draft"] == "draft"
    assert not path.exists()


def test_failed_startup_retains_the_handoff(monkeypatch):
    old = client(recording=False)
    old.agent.messages = [HumanMessage(content="the only copy")]
    path = restart_state.write(old.snapshot_for_restart())
    boot(monkeypatch, client(recording=False), startup_error=True)
    with pytest.raises(RuntimeError, match="startup failed"):
        main_mod.main(restart_path=str(path))
    assert restart_state.read(path)["messages"]


@pytest.mark.parametrize("outcome", ["exit", "fresh"])
def test_failed_restore_keeps_snapshot_and_does_not_open_a_fresh_chat(monkeypatch, outcome):
    old = client()
    old.agent.messages = [HumanMessage(content="recover me")]
    path = restart_state.write(old.snapshot_for_restart())
    seen = boot(monkeypatch, client(), restore_error=True)
    monkeypatch.setattr(main_mod, "_resume_session", lambda *a: outcome)
    main_mod.main(restart_path=str(path))
    assert "loop" not in seen
    assert path.exists()


def test_summary_failure_is_not_reported_as_a_successful_restore():
    old = client()
    old.conversation_manager.summary_text = "Required context"
    target = client()
    target.conversation_manager.apply_restored_summary = lambda *a: False
    with pytest.raises(ValueError, match="summary"):
        target.restore_after_restart(old.snapshot_for_restart())


def test_recovery_command_quotes_paths_and_preserves_config_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOAI_CONFIG", str(tmp_path / "config with ' quote.yaml"))
    command = shlex.split(restart_state.recovery_command(tmp_path / "restart-'quote.json"))
    assert f"MNEMOAI_CONFIG={os.environ['MNEMOAI_CONFIG']}" in command
    assert command[-2:] == [restart_state.FLAG, str(tmp_path / "restart-'quote.json")]


@pytest.mark.parametrize("deferred", [False, True])
def test_real_reexec_from_pinned_model_dialog(tmp_path, deferred):
    """Exercise both cooked-dialog and background-delivery restart boundaries."""
    result = tmp_path / "restored.json"
    source = result.with_suffix(".conversation.json")
    source.write_text("The original /load file must not be overwritten.")
    env = {
        **os.environ, "PYTHONPATH": str(REPO_ROOT / "src"),
        "RESTART_TEST_RESULT": str(result), "TERM": "xterm-256color",
    }
    if deferred:
        env["RESTART_TEST_DEFER"] = "1"
    with subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "tests/fixtures/restart_app.py"),
         "--resume=stale-session", "--auto", "--no-verbose"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ) as process:
        try:
            output, _ = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            pytest.fail(f"Restart deadlocked:\n{output}")
        assert process.returncode == 0, output
        state = json.loads(result.read_text())
        assert state["pid"] == process.pid, "must replace the process, not start another one"
    assert state["history"] == ["saved question", "unsaved answer"] + (
        ["delivered background report"] if deferred else []
    )
    assert state["summary"] == "Keep the original task."
    assert state["save_path"] == str(source)
    assert state["draft"] == "unsent draft"
    assert state["tokens"] is None
    assert "--resume=stale-session" not in state["argv"]
    assert "--auto" not in state["argv"]
    assert "--no-verbose" in state["argv"]
    assert source.read_text() == "The original /load file must not be overwritten."
    assert not list((tmp_path / "home/restarts").glob("*.json"))


def test_exec_failure_preserves_a_recoverable_snapshot(monkeypatch, capsys):
    old = client(recording=False)
    old.agent.messages = [HumanMessage(content="recover after exec failure")]
    execute = Mock(side_effect=OSError("exec unavailable"))
    monkeypatch.setattr("mnemoai.client.ui.chat_interface.os.execv", execute)
    with pytest.raises(SystemExit, match="1"):
        ChatInterface(old)._restart_in_place()
    path = execute.call_args.args[1][-1]
    assert restart_state.read(path)["messages"][0]["content"][0]["text"] == "recover after exec failure"
    assert "Recover with:" in capsys.readouterr().out


def test_interrupted_snapshot_leaves_current_client_and_no_partial_file(tmp_path, monkeypatch):
    old = client()
    monkeypatch.setattr(restart_state.os, "fsync", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        ChatInterface(old)._restart_in_place()
    old.mcp_client.shutdown.assert_not_called()
    assert not list((tmp_path / "home/restarts").glob("*.json"))


@pytest.mark.parametrize("damage", ["missing", "corrupt", "wrong_directory"])
def test_unreadable_handoff_offers_recovery_without_opening_a_blank_chat(tmp_path, monkeypatch, damage):
    old = client(recording=False)
    old.agent.messages = [HumanMessage(content="unique unsaved history")]
    path = restart_state.write(old.snapshot_for_restart())
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_text('{"restart_version":')
    else:
        data = json.loads(path.read_text())
        data["cwd"] = str(tmp_path / "another project")
        path.write_text(json.dumps(data))
    seen = boot(monkeypatch, client())
    picker = Mock(return_value="fresh")
    monkeypatch.setattr(main_mod, "_resume_session", picker)
    main_mod.main(restart_path=str(path))
    assert picker.call_args.args[1] == "pick"
    assert "loop" not in seen
    assert path.exists() == (damage != "missing")


def test_resume_that_fails_to_load_does_not_report_success(monkeypatch, capsys):
    monkeypatch.setattr("mnemoai.client.session_log.list_sessions", lambda **kw: [
        {"path": "/missing/session.jsonl", "session_id": "test-session"},
    ])
    target = SimpleNamespace(resume_session=lambda _: False)
    assert main_mod._resume_session(target, "latest") == "exit"
    assert "resumed  test-session" not in capsys.readouterr().out
