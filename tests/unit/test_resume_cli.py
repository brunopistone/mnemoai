"""Unit tests for the `--resume` CLI flow (`main._resume_session`).

The contract that matters: `--resume` is a request to RESUME. If the user
cancels the picker (Esc) or names a session that doesn't exist, the app must
**exit** rather than fall through into a fresh chat — silently starting a new
conversation is a surprise, and it leaves an unwanted empty session behind.
`_resume_session` signals that by returning "exit" (vs "resumed" / "fresh").

Pure logic — no client, no LLM; the client is a stub and the picker is patched.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from mnemoai import main as main_mod
from mnemoai.client import session_log as slog
from mnemoai.utils import paths


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))
    monkeypatch.setattr(paths, "_profile_name", lambda: "tester")
    monkeypatch.chdir(tmp_path)  # sessions are scoped to cwd
    return tmp_path


class _Client:
    def __init__(self):
        self.resumed = None

    def resume_session(self, path):
        self.resumed = path
        return True


def _make_session(prompt="a question"):
    log = slog.SessionLog()
    log.log_turn([HumanMessage(content=prompt), AIMessage(content="an answer")])
    return log


class TestCancelAborts:
    def test_cancelling_the_picker_exits(self, home, monkeypatch):
        # THE reported bug: Esc used to fall through and start a fresh session.
        _make_session()
        _make_session("second")
        monkeypatch.setattr(main_mod, "print_error", lambda *_: None)
        monkeypatch.setattr(
            "mnemoai.client.ui.tui.select_from_list", lambda *a, **k: None
        )
        client = _Client()
        assert main_mod._resume_session(client, "pick") == "exit"
        assert client.resumed is None

    def test_unknown_session_id_exits(self, home, monkeypatch):
        _make_session()
        monkeypatch.setattr(main_mod, "print_error", lambda *_: None)
        client = _Client()
        assert main_mod._resume_session(client, "does-not-exist") == "exit"
        assert client.resumed is None

    def test_single_session_still_shows_the_picker(self, home, monkeypatch):
        # With one session it must still be cancellable — auto-resuming would
        # deny the user any chance to back out.
        _make_session()
        monkeypatch.setattr(main_mod, "print_error", lambda *_: None)
        seen = {}

        def _picker(title, options, **kw):
            seen["count"] = len(options)
            return None

        monkeypatch.setattr("mnemoai.client.ui.tui.select_from_list", _picker)
        client = _Client()
        assert main_mod._resume_session(client, "pick") == "exit"
        assert seen["count"] == 1


class TestProceeds:
    def test_no_sessions_proceeds_with_a_fresh_one(self, home, monkeypatch):
        # Nothing to resume isn't a failure — a fresh session is the right result.
        monkeypatch.setattr(main_mod, "print_error", lambda *_: None)
        client = _Client()
        assert main_mod._resume_session(client, "pick") == "fresh"
        assert client.resumed is None

    def test_latest_resumes_newest(self, home, monkeypatch):
        import os
        import time

        older = _make_session("older question")
        newer = _make_session("newer question")
        os.utime(older.path, (time.time() - 500,) * 2)
        client = _Client()
        assert main_mod._resume_session(client, "latest") == "resumed"
        assert client.resumed == str(newer.path)

    def test_explicit_id_resumes_that_session(self, home):
        target = _make_session("pick me")
        _make_session("not me")
        client = _Client()
        assert main_mod._resume_session(client, target.session_id) == "resumed"
        assert client.resumed == str(target.path)

    def test_partial_id_suffix_resolves(self, home):
        target = _make_session("pick me")
        client = _Client()
        assert main_mod._resume_session(client, target.session_id[-6:]) == "resumed"
        assert client.resumed == str(target.path)

    def test_picker_choice_is_resumed(self, home, monkeypatch):
        a = _make_session("first")
        _make_session("second")
        monkeypatch.setattr(
            "mnemoai.client.ui.tui.select_from_list", lambda *args, **kw: str(a.path)
        )
        client = _Client()
        assert main_mod._resume_session(client, "pick") == "resumed"
        assert client.resumed == str(a.path)


class TestLabel:
    def test_label_has_age_turns_and_preview(self):
        import time

        label = main_mod._format_session_label(
            {"modified": time.time() - 120, "turns": 3, "preview": "do the thing"}
        )
        assert "2m ago" in label and "3 turns" in label and "do the thing" in label

    def test_singular_turn(self):
        import time

        label = main_mod._format_session_label(
            {"modified": time.time(), "turns": 1, "preview": "x"}
        )
        assert "1 turn " in label and "1 turns" not in label


class TestBannerOrdering:
    """The restored transcript must render BELOW the banner, next to the prompt.

    Reported bug: the replayed conversation appeared ABOVE the logo/command box,
    because the banner was printed by run_chat_loop AFTER the resume. The banner
    is now shown from inside _resume_session, before the replay.
    """

    def test_banner_is_shown_before_the_transcript(self, home, monkeypatch):
        order = []

        class _UI:
            def show_welcome(self):
                order.append("banner")

        class _C:
            def resume_session(self, path):
                order.append("transcript")
                return True

        target = _make_session("q")
        assert (
            main_mod._resume_session(_C(), target.session_id, _UI()) == "resumed"
        )
        assert order == ["banner", "transcript"]

    def test_no_banner_when_nothing_to_resume(self, home, monkeypatch):
        # "fresh" must let the CALLER print the banner, or it'd never appear.
        monkeypatch.setattr(main_mod, "print_error", lambda *_: None)
        shown = []

        class _UI:
            def show_welcome(self):
                shown.append(1)

        class _C:
            def resume_session(self, path):
                return True

        assert main_mod._resume_session(_C(), "pick", _UI()) == "fresh"
        assert shown == []
