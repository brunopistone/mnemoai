"""Unit tests for naming a session (`/rename`).

The picker labels every row with the session's first prompt — the one thing a
resumed conversation and its parent have in common — so after a few sessions in
one project the rows stop being distinguishable. A name fixes that, and the
invariants that keep it honest are:

  * append-only like every other record, so renaming twice leaves two records
    and the LAST one wins (the file is never rewritten);
  * a `label` record is not a turn, so naming a session you then abandon must
    not keep its empty file alive in the picker;
  * a name survives a `--resume` (same conversation, new file) but a `/branch`
    fork deliberately does NOT inherit it — otherwise the fork and its parent
    are once again two identical rows.

Pure file logic plus a stub client — no LLM, no TTY.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from mnemoai import main as main_mod
from mnemoai.client import session_log as slog
from mnemoai.client.client import LangGraphClient
from mnemoai.client.ui.chat_interface import ChatInterface
from mnemoai.utils import paths


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))
    monkeypatch.setattr(paths, "_profile_name", lambda: "tester")
    return tmp_path


def _turn(q="q", a="a"):
    return [HumanMessage(content=q), AIMessage(content=a)]


class TestSetLabel:
    def test_records_and_reads_back(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn())
        assert log.set_label("refactor the router") is True
        assert slog.read_session(log.path)["label"] == "refactor the router"

    def test_last_one_wins(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn())
        log.set_label("first name")
        log.set_label("second name")
        # Appended, never rewritten — so reading has to resolve the conflict.
        assert slog.read_session(log.path)["label"] == "second name"

    def test_an_empty_title_clears_the_name(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn())
        log.set_label("a name")
        log.set_label("")
        assert slog.read_session(log.path)["label"] == ""

    def test_a_long_title_is_bounded(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn())
        log.set_label("x" * 500)
        assert len(slog.read_session(log.path)["label"]) == slog._LABEL_CHARS

    def test_no_transcript_means_no_label(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.path = None  # recording disabled
        assert log.set_label("nope") is False

    def test_unnamed_sessions_read_back_empty(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn())
        assert slog.read_session(log.path)["label"] == ""


class TestLabelIsNotATurn:
    def test_a_name_alone_does_not_make_a_session_resumable(self, home, monkeypatch):
        monkeypatch.chdir(home)
        log = slog.SessionLog()
        log.set_label("named but empty")
        assert slog.read_session(log.path)["turns"] == 0
        assert slog.list_sessions() == []  # nothing to restore

    def test_a_named_but_turnless_file_is_still_discarded(self, home, monkeypatch):
        monkeypatch.chdir(home)
        log = slog.SessionLog()
        log.set_label("named but empty")
        path = log.path
        assert log.discard_if_empty() is True
        assert not path.exists()


class TestListSessions:
    def test_rows_carry_the_label(self, home, monkeypatch):
        monkeypatch.chdir(home)
        log = slog.SessionLog()
        log.log_turn(_turn("what is this project"))
        log.set_label("project tour")
        rows = slog.list_sessions()
        assert len(rows) == 1
        assert rows[0]["label"] == "project tour"
        assert rows[0]["preview"]  # the preview is still there as the fallback


class TestPickerRow:
    def _row(self, **over):
        row = {"modified": 0, "exchanges": 3, "preview": "the first prompt"}
        row.update(over)
        return row

    def test_the_name_replaces_the_preview(self):
        label = main_mod._format_session_label(self._row(label="ship 1.12"))
        assert "ship 1.12" in label and "the first prompt" not in label

    def test_without_a_name_the_preview_is_used(self):
        assert "the first prompt" in main_mod._format_session_label(self._row())

    def test_a_blank_name_falls_back(self):
        assert "the first prompt" in main_mod._format_session_label(self._row(label="   "))

    def test_the_branch_tag_survives_a_name(self):
        label = main_mod._format_session_label(
            self._row(label="alt approach", branched_from={"through_turn": 4})
        )
        assert "alt approach" in label and "branch @ turn 4" in label


class TestNameSurvivesAResume:
    def test_seeding_a_resumed_session_carries_the_name(self, home, monkeypatch):
        monkeypatch.chdir(home)
        first = slog.SessionLog()
        first.log_turn(_turn("original question"))
        first.set_label("the long investigation")

        data = slog.read_session(first.path)
        # What resume_session does: a new file, seeded with the old conversation.
        second = slog.SessionLog()
        client = LangGraphClient.__new__(LangGraphClient)
        client.agent = type("A", (), {"session_log": second})()
        client._seed_session_log(
            [HumanMessage(content="original question"), AIMessage(content="a")],
            source=str(first.path),
            label=data["label"],
        )
        # A resume continues the SAME conversation, so losing the name on the
        # first resume would defeat the point of naming it.
        assert slog.read_session(second.path)["label"] == "the long investigation"

    def test_a_branch_does_not_inherit_the_name(self, home, monkeypatch):
        monkeypatch.chdir(home)
        source = slog.SessionLog()
        source.log_turn(_turn("q1", "a1"))
        source.log_turn(_turn("q2", "a2"))
        source.set_label("the main thread")

        fork = slog.branch_session(source.path, through_turn=1)
        assert fork is not None
        # A fork already shares its parent's opening prompt; inheriting the name
        # too would make the two rows indistinguishable again.
        assert slog.read_session(fork)["label"] == ""


class TestClientApi:
    def _client(self, log):
        client = LangGraphClient.__new__(LangGraphClient)
        client.agent = type("A", (), {"session_log": log})()
        return client

    def test_rename_and_read_back(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn())
        client = self._client(log)
        assert client.rename_session("a good name") is True
        assert client.session_label() == "a good name"

    def test_whitespace_is_normalized(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn())
        client = self._client(log)
        client.rename_session("  too   many\n spaces  ")
        assert client.session_label() == "too many spaces"

    def test_no_recording_is_reported_not_crashed(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.path = None
        client = self._client(log)
        assert client.rename_session("x") is False
        assert client.session_label() == ""

    def test_no_session_log_at_all(self):
        client = LangGraphClient.__new__(LangGraphClient)
        client.agent = type("A", (), {"session_log": None})()
        assert client.rename_session("x") is False
        assert client.session_label() == ""


class TestRenameCommand:
    class _Stub:
        def __init__(self, label="", ok=True):
            self.label = label
            self.ok = ok
            self.renamed = None

        def session_label(self):
            return self.label

        def rename_session(self, title):
            self.renamed = title
            return self.ok

    def _ci(self, client):
        ci = ChatInterface.__new__(ChatInterface)
        ci.client = client
        return ci

    def test_bare_command_shows_the_current_name(self, capsys):
        ci = self._ci(self._Stub(label="tour"))
        ci._dispatch("/rename")
        assert "tour" in capsys.readouterr().out

    def test_bare_command_without_a_name_explains_the_default(self, capsys):
        ci = self._ci(self._Stub())
        ci._dispatch("/rename")
        assert "first prompt" in capsys.readouterr().out

    def test_setting_a_name(self, capsys):
        client = self._Stub()
        self._ci(client)._dispatch("/rename  the big refactor ")
        assert client.renamed == "the big refactor"
        assert "the big refactor" in capsys.readouterr().out

    def test_clear_removes_an_existing_name(self, capsys):
        client = self._Stub(label="tour")
        self._ci(client)._dispatch("/rename clear")
        assert client.renamed == ""
        assert "cleared" in capsys.readouterr().out.lower()

    def test_clear_with_nothing_to_clear(self, capsys):
        client = self._Stub()
        self._ci(client)._dispatch("/rename clear")
        assert client.renamed is None
        assert "no name to clear" in capsys.readouterr().out

    def test_recording_off_says_so(self, capsys):
        client = self._Stub(ok=False)
        self._ci(client)._dispatch("/rename something")
        # Not a silent no-op: the user typed a command and nothing was written.
        assert "nothing is being recorded" in capsys.readouterr().out


class TestCommandSurface:
    def test_rename_is_documented_and_autocompleted(self):
        # A command missing from either list is invisible to the user.
        completed = {cmd for cmd, _desc in ChatInterface._COMMANDS}
        assert {"/rename", "/hooks", "/doctor"} <= completed
        documented = {
            cmd.split()[0]
            for _group, entries in ChatInterface._COMMAND_GROUPS
            for cmd, _desc in entries
        }
        assert {"/rename", "/hooks", "/doctor"} <= documented
