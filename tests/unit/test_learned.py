"""User controls change real injection while preserving summary and stored history."""

import json
from types import SimpleNamespace

from mnemoai.client import learned
from mnemoai.client.client import LangGraphClient
from mnemoai.client.memory.playbook_records import PlaybookEntry
from mnemoai.client.memory.playbook_store import PlaybookStore
from mnemoai.client.ui.chat_interface import ChatInterface


def client(tmp_path):
    c = LangGraphClient.__new__(LangGraphClient)
    c.playbook = PlaybookStore(str(tmp_path))
    c.playbook.append(PlaybookEntry("editing", "Read the target before editing", "test"))
    c.system_prompt = "instructions\n\n<conversation_summary>keep me</conversation_summary>"
    c.agent = SimpleNamespace(system_prompt=c.system_prompt, messages=[])
    c.refresh_playbook_context()
    return c


def run(c, command, answer=True, edit=lambda original: original):
    return learned.run(c, command, confirm=lambda _: answer, edit=edit)


def test_disable_and_restore_change_next_context_and_survive_reload(tmp_path):
    c = client(tmp_path)
    entry = c.playbook.snapshot()[0]
    assert "Read the target" in c.agent.system_prompt
    run(c, "disable " + entry["id"])
    assert "Read the target" not in c.agent.system_prompt
    assert "keep me" in c.agent.system_prompt
    c.playbook = PlaybookStore(str(tmp_path))
    c.refresh_playbook_context()
    assert "Read the target" not in c.agent.system_prompt
    run(c, "restore " + entry["id"])
    assert c.agent.system_prompt.count("Read the target") == 1
    assert "keep me" in c.agent.system_prompt


def test_off_suppresses_injection_without_modifying_entries(tmp_path):
    c = client(tmp_path)
    before = c.playbook.snapshot()
    run(c, "off")
    assert "Read the target" not in c.agent.system_prompt
    assert c.playbook.snapshot() == before
    run(c, "on")
    assert c.agent.system_prompt.count("Read the target") == 1


def test_refresh_keeps_summary_last_and_does_not_rewrite_quoted_history(tmp_path):
    c = client(tmp_path)
    historical = c._get_playbook_context()
    summary = "<conversation_summary>\nQuoted history:\n" + historical + "\n</conversation_summary>"
    c.system_prompt = "instructions\n\n" + summary
    c.agent.system_prompt = "instructions\n\n" + historical + "\n\n" + summary
    run(c, "off")
    assert c.agent.system_prompt.endswith(summary)
    assert c.agent.system_prompt == "instructions\n\n" + summary
    run(c, "on")
    assert c.agent.system_prompt.endswith(summary)


def test_cancel_and_bad_editor_json_leave_entry_unchanged(tmp_path):
    c = client(tmp_path)
    entry = c.playbook.snapshot()[0]
    run(c, "disable " + entry["id"], answer=False)
    run(c, "edit " + entry["id"], edit=lambda _: "not json")
    assert c.playbook.snapshot()[0] == entry


def test_edit_and_inspect_roundtrip(tmp_path):
    c = client(tmp_path)
    entry = c.playbook.snapshot()[0]
    run(c, "edit " + entry["id"], edit=lambda _: json.dumps({
        "context": "editing", "strategy": "Use the exact current text",
    }))
    actual = json.loads(run(c, "inspect " + entry["id"]))
    assert actual["provenance"] == "user"
    assert actual["history"][0]["strategy"] == entry["strategy"]
    assert "Use the exact current text" in c.agent.system_prompt
    assert "Read the target" not in c.agent.system_prompt


def test_clear_retains_backup(tmp_path):
    c = client(tmp_path)
    run(c, "clear")
    assert c.playbook.snapshot() == []
    assert list(tmp_path.glob("playbook.json.backup-*"))
    assert "Read the target" not in c.agent.system_prompt


def test_dispatch_does_not_send_learned_command_to_model(tmp_path, capsys):
    ui = ChatInterface.__new__(ChatInterface)
    ui.client = client(tmp_path)
    ui._dispatch("/learned")
    assert "mem-" in capsys.readouterr().out
    ui._dispatch("/learned invalid")
    assert "Usage:" in capsys.readouterr().out
