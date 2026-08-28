"""Unit tests for the session file ledger (`/files`)."""

import os
import re
import threading

import pytest

from mnemoai.client import file_ledger
from mnemoai.client.file_ledger import ATTACHED, READ, WRITTEN, FileLedger

_ANSI = re.compile(r"\033\[[0-9;]*m")


def plain(text: str) -> str:
    """The report without color, so a row can be matched as written."""
    return _ANSI.sub("", text)


@pytest.fixture
def ledger():
    return FileLedger()


class TestRecord:
    def test_one_row_per_file_regardless_of_spelling(self, ledger, tmp_path, monkeypatch):
        target = tmp_path / "x.py"
        target.write_text("x = 1\n")
        monkeypatch.chdir(tmp_path)
        ledger.record("x.py", READ)
        ledger.record("./x.py", READ)
        ledger.record(str(target), READ)
        entries = ledger.snapshot()
        assert len(entries) == 1
        assert entries[0].counts[READ] == 3

    def test_display_is_cwd_relative_when_under_it(self, ledger, tmp_path, monkeypatch):
        (tmp_path / "pkg").mkdir()
        monkeypatch.chdir(tmp_path)
        ledger.record(str(tmp_path / "pkg" / "a.py"), READ)
        assert ledger.snapshot()[0].display == os.path.join("pkg", "a.py")

    def test_display_falls_back_to_home_relative(self, ledger, monkeypatch, tmp_path):
        # A file outside the session's directory but under $HOME: the point of the
        # fallback is that it doesn't render as a full absolute path.
        home, work = tmp_path / "home", tmp_path / "work"
        home.mkdir()
        work.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(work)
        assert file_ledger._display(str(home / "notes.md")) == os.path.join("~", "notes.md")

    def test_display_keeps_an_unrelated_path_absolute(self, ledger, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.chdir(tmp_path)
        assert file_ledger._display("/etc/hosts") == "/etc/hosts"

    def test_kind_is_the_strongest_action(self, ledger):
        ledger.record("/tmp/a", READ)
        ledger.record("/tmp/a", ATTACHED)
        assert ledger.snapshot()[0].kind == ATTACHED
        ledger.record("/tmp/a", WRITTEN)
        assert ledger.snapshot()[0].kind == WRITTEN

    def test_snapshot_is_most_recent_first(self, ledger):
        ledger.record("/tmp/a", READ)
        ledger.record("/tmp/b", READ)
        ledger.record("/tmp/a", READ)  # touched again → back to the front
        assert [e.display for e in ledger.snapshot()][0].endswith("a")

    def test_unknown_action_and_empty_path_are_ignored(self, ledger):
        ledger.record("/tmp/a", "deleted")
        ledger.record("", READ)
        ledger.record("   ", READ)
        assert ledger.snapshot() == []

    def test_record_never_raises(self, ledger, monkeypatch):
        monkeypatch.setattr(file_ledger, "_resolve", lambda p: 1 / 0)
        ledger.record("/tmp/a", READ)  # must not propagate
        assert ledger.snapshot() == []

    def test_changed_paths_are_only_the_written_ones(self, ledger):
        ledger.record("/tmp/read-me", READ)
        ledger.record("/tmp/write-me", WRITTEN)
        changed = ledger.changed_paths()
        assert len(changed) == 1
        assert next(iter(changed)).endswith("write-me")

    def test_reset_forgets_everything(self, ledger):
        ledger.record("/tmp/a", WRITTEN)
        ledger.reset()
        assert ledger.snapshot() == []
        assert ledger.changed_paths() == set()
        assert ledger.overflow == 0

    def test_entries_are_capped_and_the_rest_counted(self, ledger, monkeypatch):
        monkeypatch.setattr(file_ledger, "_MAX_ENTRIES", 3)
        for i in range(6):
            ledger.record(f"/tmp/f{i}", READ)
        assert len(ledger.snapshot()) == 3
        assert ledger.overflow == 3

    def test_a_capped_file_already_known_still_counts_up(self, ledger, monkeypatch):
        monkeypatch.setattr(file_ledger, "_MAX_ENTRIES", 1)
        ledger.record("/tmp/a", READ)
        ledger.record("/tmp/b", READ)  # over the cap
        ledger.record("/tmp/a", WRITTEN)  # existing row → recorded
        assert ledger.snapshot()[0].counts[WRITTEN] == 1
        assert ledger.overflow == 1

    def test_concurrent_recording_loses_nothing(self, ledger):
        def hammer(n):
            for _ in range(50):
                ledger.record(f"/tmp/f{n}", READ)

        threads = [threading.Thread(target=hammer, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        entries = ledger.snapshot()
        assert len(entries) == 8
        assert all(e.counts[READ] == 50 for e in entries)


class TestRecordTool:
    def test_the_three_file_tools_map_to_their_path_arg(self, ledger):
        ledger.record_tool("fs_read", {"path": "/tmp/r", "mode": "LINE"})
        ledger.record_tool("fs_write", {"path": "/tmp/w", "command": "create"})
        ledger.record_tool("file_edit", {"file_path": "/tmp/e", "old_string": "a"})
        by_display = {e.display: e for e in ledger.snapshot()}
        assert by_display["/tmp/r"].counts[READ] == 1
        assert by_display["/tmp/w"].counts[WRITTEN] == 1
        assert by_display["/tmp/e"].counts[WRITTEN] == 1

    def test_tree_wide_and_unrelated_tools_are_not_recorded(self, ledger):
        ledger.record_tool("grep_search", {"path": "/tmp"})
        ledger.record_tool("glob_search", {"path": "/tmp"})
        ledger.record_tool("execute_bash", {"command": "ls"})
        ledger.record_tool("memory", {"action": "add"})
        assert ledger.snapshot() == []

    def test_a_missing_or_non_string_path_is_skipped(self, ledger):
        ledger.record_tool("fs_read", {})
        ledger.record_tool("fs_read", {"path": None})
        ledger.record_tool("fs_read", {"path": 42})
        ledger.record_tool("fs_read", None)
        ledger.record_tool("fs_read", "not-a-dict")
        assert ledger.snapshot() == []


class TestRender:
    def test_empty_ledger_says_so_and_explains(self, ledger):
        out = plain(file_ledger.render(ledger))
        assert "No files touched yet" in out
        assert "@" in out  # tells the user what lands here

    def test_groups_lead_with_what_changed(self, ledger):
        ledger.record("/tmp/read", READ)
        ledger.record("/tmp/att", ATTACHED)
        ledger.record("/tmp/edit", WRITTEN)
        out = plain(file_ledger.render(ledger))
        assert out.index("Changed (1)") < out.index("Attached with @ (1)") < out.index("Read (1)")

    def test_an_absent_group_has_no_heading(self, ledger):
        ledger.record("/tmp/read", READ)
        out = plain(file_ledger.render(ledger))
        assert "Read (1)" in out
        assert "Changed" not in out
        assert "Attached" not in out.split("Read (1)")[0]

    def test_counts_are_pluralized(self, ledger):
        ledger.record("/tmp/a", WRITTEN)
        ledger.record("/tmp/b", WRITTEN)
        ledger.record("/tmp/b", WRITTEN)
        ledger.record("/tmp/c", READ)
        out = plain(file_ledger.render(ledger))
        assert "1 edit " in out or "1 edit\n" in out.replace("  ", " ")
        assert "2 edits" in out
        assert "1 read" in out

    def test_attach_count_reads_as_a_multiplier(self, ledger):
        ledger.record("/tmp/a", ATTACHED)
        assert "attached" in plain(file_ledger.render(ledger))
        ledger.record("/tmp/a", ATTACHED)
        assert "attached 2×" in plain(file_ledger.render(ledger))

    def test_a_mixed_file_lists_both_actions(self, ledger):
        ledger.record("/tmp/a", READ)
        ledger.record("/tmp/a", WRITTEN)
        out = plain(file_ledger.render(ledger))
        assert "1 edit · 1 read" in out

    def test_long_group_elides_with_a_count(self, ledger, monkeypatch):
        monkeypatch.setattr(file_ledger, "_MAX_ROWS", 2)
        for i in range(5):
            ledger.record(f"/tmp/f{i}", READ)
        out = plain(file_ledger.render(ledger))
        assert "Read (5)" in out
        assert "… +3 more" in out

    def test_overflow_is_reported(self, ledger, monkeypatch):
        monkeypatch.setattr(file_ledger, "_MAX_ENTRIES", 2)
        for i in range(5):
            ledger.record(f"/tmp/f{i}", READ)
        out = plain(file_ledger.render(ledger))
        assert "+3 further file(s) not listed" in out

    def test_report_states_the_ledger_is_not_a_context_inventory(self, ledger):
        ledger.record("/tmp/a", READ)
        out = plain(file_ledger.render(ledger))
        assert "summarized out of the context" in out

    def test_a_deep_path_does_not_run_the_counts_off_screen(self, ledger):
        ledger.record("/tmp/" + "d/" * 60 + "deep.py", READ)
        out = plain(file_ledger.render(ledger))
        row = next(line for line in out.splitlines() if "deep.py" in line)
        # The path itself can be long; the padding must not add to it.
        assert row.rstrip().endswith("1 read")


class TestReport:
    def test_report_reads_the_agents_ledger(self, ledger):
        ledger.record("/tmp/a", WRITTEN)

        class _Agent:
            files = ledger

        class _Client:
            agent = _Agent()

        assert "Changed (1)" in plain(file_ledger.report(_Client()))

    def test_report_without_an_agent_says_so(self):
        class _Client:
            agent = None

        assert "unavailable" in file_ledger.report(_Client())
        assert "unavailable" in file_ledger.report(object())


class TestWiring:
    def test_the_agent_records_a_successful_call(self):
        from mnemoai.client.agent.agent import LangGraphAgent

        agent = LangGraphAgent.__new__(LangGraphAgent)
        agent.files = FileLedger()
        agent._record_file_activity("fs_write", {"path": "/tmp/w"})
        assert len(agent.files.snapshot()) == 1

    def test_a_bare_stub_without_a_ledger_is_tolerated(self):
        from mnemoai.client.agent.agent import LangGraphAgent

        agent = LangGraphAgent.__new__(LangGraphAgent)
        agent._record_file_activity("fs_write", {"path": "/tmp/w"})  # must not raise

    def test_files_is_a_builtin_command(self):
        from mnemoai.client.ui.chat_interface import ChatInterface
        from mnemoai.client.user_commands import BUILTIN_COMMANDS

        assert "files" in BUILTIN_COMMANDS
        assert any(cmd == "/files" for cmd, _ in ChatInterface._COMMANDS)
        assert any(
            cmd == "/files"
            for _, items in ChatInterface._COMMAND_GROUPS
            for cmd, _ in items
        )
