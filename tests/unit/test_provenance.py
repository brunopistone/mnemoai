"""Unit tests for the change index behind `/why`."""

import json
import os
import re
import time

import pytest

from mnemoai.client import provenance
from mnemoai.client.provenance import Change, ProvenanceLog

_ANSI = re.compile(r"\033\[[0-9;]*m")


def plain(text: str) -> str:
    """The report without color, so a row can be matched as written."""
    return _ANSI.sub("", text)


def change(**kw) -> Change:
    """A record with sensible defaults, overridden per test."""
    fields = {
        "path": "/tmp/a.py",
        "tool": "file_edit",
        "ts": time.time(),
        "turn": 3,
        "session": "s1",
        "prompt": "make the footer wrap",
        "detail": "+2 -1",
    }
    fields.update(kw)
    return Change(**fields)


@pytest.fixture
def log(tmp_path, monkeypatch):
    """A log writing into a tmp index (never the developer's app home)."""
    index = tmp_path / "prov.jsonl"
    monkeypatch.setattr(provenance, "provenance_path", lambda *a, **k: index)
    return ProvenanceLog()


def rows(log) -> list:
    return [json.loads(line) for line in open(log.path).read().splitlines()]


class TestRecord:
    def test_the_two_changing_tools_are_recorded(self, log):
        log.record("fs_write", {"path": "/tmp/w", "command": "create", "file_text": "a\n"})
        log.record("file_edit", {"file_path": "/tmp/e", "old_string": "a", "new_string": "b"})
        assert [r["tool"] for r in rows(log)] == ["fs_write", "file_edit"]

    def test_a_read_is_not_a_change(self, log):
        # `/why` explains what a file IS; reading it changed nothing to explain.
        log.record("fs_read", {"path": "/tmp/r"})
        log.record("grep_search", {"path": "/tmp"})
        log.record("execute_bash", {"command": "ls"})
        assert not os.path.exists(log.path)

    def test_two_spellings_of_one_file_share_a_key(self, log, tmp_path, monkeypatch):
        target = tmp_path / "x.py"
        target.write_text("x = 1\n")
        monkeypatch.chdir(tmp_path)
        log.record("file_edit", {"file_path": "x.py"})
        log.record("file_edit", {"file_path": "./x.py"})
        log.record("file_edit", {"file_path": str(target)})
        assert len({r["path"] for r in rows(log)}) == 1

    def test_the_turn_and_session_come_from_the_caller(self, log):
        log.record("file_edit", {"file_path": "/tmp/e"}, turn=12, session="abc")
        record = rows(log)[0]
        assert record["turn"] == 12
        assert record["session"] == "abc"

    def test_injected_context_is_stripped_from_the_quoted_prompt(self, log):
        prompt = (
            "[Episodic Memory - Similar Past Interactions]\n1. tools: fs_read\n\n"
            "<steering>be brief</steering>\nfix the footer"
        )
        log.record("file_edit", {"file_path": "/tmp/e"}, prompt=prompt)
        assert rows(log)[0]["prompt"] == "fix the footer"

    def test_a_long_prompt_is_clipped_and_says_so(self, log):
        log.record("file_edit", {"file_path": "/tmp/e"}, prompt="x" * 400)
        stored = rows(log)[0]["prompt"]
        assert len(stored) <= provenance._MAX_PROMPT_CHARS
        assert stored.endswith("…")

    def test_a_missing_or_non_string_path_is_skipped(self, log):
        log.record("file_edit", {})
        log.record("file_edit", {"file_path": None})
        log.record("file_edit", {"file_path": 42})
        log.record("file_edit", {"file_path": "   "})
        log.record("file_edit", None)
        log.record("file_edit", "not-a-dict")
        assert not os.path.exists(log.path)

    def test_record_never_raises(self, log, monkeypatch):
        monkeypatch.setattr(provenance, "resolve_path", lambda p: 1 / 0)
        log.record("file_edit", {"file_path": "/tmp/e"})  # must not propagate

    def test_an_unopenable_index_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr(provenance, "provenance_path", lambda *a, **k: 1 / 0)
        log = ProvenanceLog()
        assert log.path is None
        log.record("file_edit", {"file_path": "/tmp/e"})  # must not raise

    def test_stored_details_carry_no_terminal_escapes(self, log):
        # What is stored must outlive the report that displays it: colors belong
        # to the renderer, not to the record.
        log.record("fs_write", {"path": "/tmp/w", "command": "create", "file_text": "a\n"})
        log.record("file_edit", {"file_path": "/tmp/e", "old_string": "a", "new_string": "b"})
        assert all("\033" not in json.dumps(r) for r in rows(log))


class TestDescribeChange:
    def test_a_replacement_counts_both_sides(self):
        assert provenance.describe_change(
            "file_edit", {"old_string": "a\nb\n", "new_string": "x\ny\nz\n"}
        ) == "+3 -2"

    def test_an_insertion_names_its_line(self):
        assert provenance.describe_change(
            "fs_write", {"command": "insert", "insert_line": 12, "new_str": "a\nb\n"}
        ) == "+2 at line 12"

    def test_a_create_reports_the_size_written(self):
        assert provenance.describe_change(
            "fs_write", {"command": "create", "file_text": "a\nb\nc\n"}
        ) == "wrote 3 lines"

    def test_an_append_says_so(self):
        assert provenance.describe_change(
            "fs_write", {"command": "append", "new_str": "one line"}
        ) == "+1 appended"

    def test_an_unterminated_last_line_still_counts(self):
        assert provenance._lines("a\nb") == 2
        assert provenance._lines("a\n") == 1
        assert provenance._lines("") == 0

    def test_an_unrecognized_call_degrades_to_a_word(self):
        assert provenance.describe_change("fs_write", {"command": "weird"}) == "weird"
        assert provenance.describe_change("fs_write", {}) == "fs_write"
        assert provenance.describe_change("file_edit", {}) == "edited"
        assert provenance.describe_change("fs_write", None) == "fs_write"


class TestChanges:
    def test_records_come_back_newest_first(self, log):
        for i in range(3):
            log.record("file_edit", {"file_path": f"/tmp/f{i}"}, turn=i)
        found = provenance.changes()
        assert [c.turn for c in found] == [2, 1, 0]

    def test_a_target_filters_by_resolved_key(self, log, tmp_path, monkeypatch):
        (tmp_path / "a.py").write_text("a\n")
        (tmp_path / "b.py").write_text("b\n")
        monkeypatch.chdir(tmp_path)
        log.record("file_edit", {"file_path": "a.py"})
        log.record("file_edit", {"file_path": "b.py"})
        # A different spelling of the same file must still match.
        found = provenance.changes(str(tmp_path / "a.py"))
        assert len(found) == 1

    def test_a_missing_index_is_empty_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            provenance, "provenance_path", lambda *a, **k: tmp_path / "absent.jsonl"
        )
        assert provenance.changes() == []

    def test_a_corrupt_line_is_skipped_not_fatal(self, log):
        log.record("file_edit", {"file_path": "/tmp/e"})
        with open(log.path, "a") as fh:
            fh.write("not json\n")
            fh.write(json.dumps({"t": "other"}) + "\n")
            fh.write(json.dumps({"t": "change"}) + "\n")  # no path
            fh.write(json.dumps({"t": "change", "path": "/tmp/x", "ts": "nope"}) + "\n")
        found = provenance.changes()
        assert len(found) == 2
        assert found[0].ts == 0.0  # the unparseable timestamp, not a crash

    def test_a_change_survives_a_rewind(self, log):
        # A rewind moves the conversation only — the edit is still on disk, so the
        # prompt that caused it is still the true answer and must stay findable.
        log.record("file_edit", {"file_path": "/tmp/e"}, turn=4, prompt="add a footer")
        before = provenance.changes()
        assert len(before) == 1
        # Nothing in this module offers a withdrawal, by design.
        assert not any(
            hasattr(provenance, name) for name in ("withdraw", "forget", "remove")
        )
        assert provenance.changes() == before


class TestTrim:
    def test_the_index_is_bounded_by_size_keeping_the_newest(self, log, monkeypatch):
        monkeypatch.setattr(provenance, "_MAX_BYTES", 400)
        for i in range(40):
            log.record("file_edit", {"file_path": f"/tmp/f{i}"}, turn=i)
        assert os.path.getsize(log.path) <= 800  # bounded, within one half-cycle
        turns = [c.turn for c in provenance.changes()]
        assert turns[0] == 39  # the newest change is never the one dropped
        assert len(turns) < 40

    def test_no_temp_file_is_left_behind(self, log, monkeypatch, tmp_path):
        monkeypatch.setattr(provenance, "_MAX_BYTES", 200)
        for i in range(20):
            log.record("file_edit", {"file_path": f"/tmp/f{i}"})
        assert not (tmp_path / "prov.jsonl.tmp").exists()


class TestRenderFile:
    def test_no_record_says_so_and_explains_why(self):
        out = plain(provenance.render_file([], "a.py"))
        assert "No recorded change to a.py" in out
        assert "since it began keeping the index" in out

    def test_the_prompt_is_shown_under_its_change(self):
        out = plain(provenance.render_file([change()], "a.py", "s1"))
        assert "Why a.py looks like this" in out
        assert "make the footer wrap" in out
        assert "+2 -1" in out

    def test_this_session_leads_however_the_records_are_ordered(self):
        # It is the session the reader is holding, so it goes first even when an
        # earlier session's record happens to be the most recently written one.
        earlier = change(session="s0", turn=2, ts=time.time() - 90000)
        mine = change(session="s1", turn=9)
        for items in ([mine, earlier], [earlier, mine]):
            out = plain(provenance.render_file(items, "a.py", "s1"))
            assert out.index("This session") < out.index("Earlier · session s0")

    def test_earlier_sessions_keep_their_own_order(self):
        items = [change(session="s9"), change(session="s8"), change(session="s9")]
        out = plain(provenance.render_file(items, "a.py", "s1"))
        assert out.index("session s9") < out.index("session s8")

    def test_an_earlier_group_names_the_session_so_it_can_be_reopened(self):
        out = plain(provenance.render_file([change(session="old-id")], "a.py", "s1"))
        assert "session old-id" in out

    def test_a_record_with_no_session_still_renders(self):
        out = plain(provenance.render_file([change(session="")], "a.py", "s1"))
        assert "Earlier" in out

    def test_the_turn_number_locates_the_turn(self):
        out = plain(provenance.render_file([change(turn=7)], "a.py", "s1"))
        assert "turn 7" in out

    def test_a_long_history_is_capped_with_the_true_count(self, monkeypatch):
        monkeypatch.setattr(provenance, "_MAX_ROWS", 2)
        items = [change(turn=i) for i in range(9)]
        out = plain(provenance.render_file(items, "a.py", "s1"))
        assert "9 recorded changes, 2 shown" in out

    def test_one_change_is_singular(self):
        assert "1 recorded change." in plain(provenance.render_file([change()], "a.py"))


class TestRenderOverview:
    def test_an_empty_project_says_what_would_land_here(self):
        out = plain(provenance.render_overview([]))
        assert "No changes recorded for this project yet" in out
        assert "/why" in out

    def test_one_row_per_file_with_a_count(self):
        items = [
            change(path="/tmp/a.py", ts=200.0),
            change(path="/tmp/a.py", ts=100.0),
            change(path="/tmp/b.py", ts=150.0),
        ]
        out = plain(provenance.render_overview(items, "s1"))
        assert "2 changes" in out
        assert "1 change" in out

    def test_files_are_ordered_by_most_recent_change(self):
        items = [change(path="/tmp/b.py", ts=300.0), change(path="/tmp/a.py", ts=100.0)]
        out = plain(provenance.render_overview(items, "s1"))
        assert out.index("b.py") < out.index("a.py")

    def test_this_session_is_preferred_over_the_projects_history(self):
        items = [
            change(path="/tmp/mine.py", session="s1"),
            change(path="/tmp/theirs.py", session="s0"),
        ]
        out = plain(provenance.render_overview(items, "s1"))
        assert "Changed in this session" in out
        assert "mine.py" in out
        assert "theirs.py" not in out

    def test_it_falls_back_when_this_session_changed_nothing(self):
        # An empty report would be the least useful true answer available.
        items = [change(path="/tmp/theirs.py", session="s0")]
        out = plain(provenance.render_overview(items, "s1"))
        assert "Changed in earlier sessions here" in out
        assert "theirs.py" in out

    def test_the_file_list_is_capped_with_a_count(self, monkeypatch):
        monkeypatch.setattr(provenance, "_MAX_FILES", 2)
        items = [change(path=f"/tmp/f{i}.py", ts=float(i)) for i in range(7)]
        out = plain(provenance.render_overview(items, "s1"))
        assert "… +5 more" in out

    def test_a_deep_path_does_not_run_the_counts_off_screen(self):
        items = [change(path="/tmp/" + "d/" * 60 + "deep.py")]
        out = plain(provenance.render_overview(items, "s1"))
        row = next(line for line in out.splitlines() if "deep.py" in line)
        # The path itself can be long; the padding must not add to it.
        assert row.rstrip().endswith(provenance._stamp(items[0], "s1"))


class TestReport:
    def _client(self, session_id="s1"):
        class _Log:
            pass

        log = _Log()
        log.session_id = session_id

        class _Agent:
            session_log = log

        class _Client:
            agent = _Agent()

        return _Client()

    def test_a_bare_command_lists_the_files(self, log):
        log.record("file_edit", {"file_path": "/tmp/e"}, session="s1", prompt="do it")
        out = plain(provenance.report(self._client()))
        assert "Why these files look like this" in out
        assert "do it" in out

    def test_a_path_argument_reports_that_file(self, log, tmp_path, monkeypatch):
        (tmp_path / "a.py").write_text("a\n")
        monkeypatch.chdir(tmp_path)
        log.record("file_edit", {"file_path": "a.py"}, session="s1", prompt="rename it")
        out = plain(provenance.report(self._client(), "a.py"))
        assert "Why a.py looks like this" in out
        assert "rename it" in out

    def test_a_client_without_an_agent_still_reports(self, log):
        assert "No changes recorded" in plain(provenance.report(object()))

    def test_a_failure_becomes_a_message_not_a_traceback(self, monkeypatch):
        monkeypatch.setattr(provenance, "changes", lambda *a, **k: 1 / 0)
        assert provenance.report(self._client()) == "Change history is unavailable."

    def test_the_report_writes_only_its_own_index(self):
        # A guard, not a formality: `/why` must stay a report. Every file this
        # module opens for writing is inspected here, so no other write can slip in.
        source = open(provenance.__file__).read()
        assert re.findall(r'open\([^)]*"(w|a)"', source) == ["a", "w"]  # append, trim

    def test_the_report_reaches_for_no_model_and_no_config(self):
        # It answers from the index alone: no LLM call, so it costs no turn, and no
        # config read, so it behaves the same on an install that has never been set up.
        source = open(provenance.__file__).read()
        imports = re.findall(r"^(?:from|import)\s+(\S+)", source, re.MULTILINE)
        assert sorted(imports) == [
            "json",
            "mnemoai.client.file_ledger",
            "mnemoai.client.ui.turn_view",
            "mnemoai.utils.logger",
            "mnemoai.utils.paths",
            "os",
            "time",
            "typing",
        ]


class TestWiring:
    def test_the_agent_records_a_change_from_the_tool_chokepoint(self, log):
        from mnemoai.client.agent.agent import LangGraphAgent

        agent = LangGraphAgent.__new__(LangGraphAgent)
        agent.provenance = log

        class _SessionLog:
            next_turn = 5
            session_id = "sid"

        agent.session_log = _SessionLog()
        agent._turn_prompt = "add the footer"
        agent._record_file_activity("file_edit", {"file_path": "/tmp/e"})
        record = rows(log)[0]
        assert record["turn"] == 5
        assert record["session"] == "sid"
        assert record["prompt"] == "add the footer"

    def test_a_stub_without_an_index_is_tolerated(self):
        from mnemoai.client.agent.agent import LangGraphAgent

        agent = LangGraphAgent.__new__(LangGraphAgent)
        agent._record_file_activity("file_edit", {"file_path": "/tmp/e"})  # no raise

    def test_the_turn_number_is_the_one_this_turn_will_be_logged_as(self):
        # `_turn` only advances when a turn is FLUSHED, so a mid-turn writer has to
        # ask for the number in advance or every change points at the turn before it.
        from mnemoai.client.session_log import SessionLog

        log = SessionLog.__new__(SessionLog)
        log._turn = 4
        assert log.next_turn == 5

    def test_the_marker_counts_from_the_ledger_and_tolerates_no_agent(self):
        from mnemoai.client.file_ledger import WRITTEN, FileLedger
        from mnemoai.client.ui.chat_interface import ChatInterface

        ui = ChatInterface.__new__(ChatInterface)
        ui.client = type("_C", (), {"agent": None})()
        assert ui._files_mark() == 0
        assert ui._files_changed_since(0) == 0  # no ledger → no marker, no crash

        ledger = FileLedger()
        ui.client = type("_C", (), {"agent": type("_A", (), {"files": ledger})()})()
        mark = ui._files_mark()
        ledger.record("/tmp/a", WRITTEN)
        assert ui._files_changed_since(mark) == 1

    def test_a_ledger_that_raises_costs_the_marker_not_the_turn(self):
        from mnemoai.client.ui.chat_interface import ChatInterface

        class _Ledger:
            def mark(self):
                raise RuntimeError("boom")

            def changed_since(self, mark):
                raise RuntimeError("boom")

        ui = ChatInterface.__new__(ChatInterface)
        ui.client = type("_C", (), {"agent": type("_A", (), {"files": _Ledger()})()})()
        assert ui._files_mark() == 0
        assert ui._files_changed_since(0) == 0

    def test_why_is_a_builtin_command(self):
        from mnemoai.client.ui.chat_interface import ChatInterface
        from mnemoai.client.user_commands import BUILTIN_COMMANDS

        assert "why" in BUILTIN_COMMANDS
        assert any(cmd == "/why" for cmd, _ in ChatInterface._COMMANDS)
        assert any(
            cmd.startswith("/why")
            for _, items in ChatInterface._COMMAND_GROUPS
            for cmd, _ in items
        )
