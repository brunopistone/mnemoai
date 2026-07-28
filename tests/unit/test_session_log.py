"""Unit tests for per-directory session transcripts (`--resume`).

Sessions are append-only JSONL scoped to the launch directory. The invariants
that matter:

  * per-DIRECTORY isolation — `--resume` in one project must never offer
    another project's sessions;
  * append-only, so it survives compaction REPLACING `agent.messages` (a mirror
    of the live list would lose the summarized-away history);
  * a truncated final line (process killed mid-write) must not make the whole
    session unreadable;
  * newest-first listing with a cap on what's OFFERED — deletion is age-based
    only, so a cap can never silently drop a session the user wanted;
  * `/save` files live elsewhere and are never swept.

Pure file logic — no LLM, no network.
"""

import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from mnemoai.client import session_log as slog
from mnemoai.utils import paths


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))
    monkeypatch.setattr(paths, "_profile_name", lambda: "tester")
    return tmp_path


def _turn(q, a):
    return [HumanMessage(content=q), AIMessage(content=a)]


class TestSanitizeCwd:
    def test_readable_name(self):
        assert paths.sanitize_cwd("/Users/x/dev/proj") == "Users-x-dev-proj"

    def test_empty_falls_back(self):
        assert paths.sanitize_cwd("") == "root"
        assert paths.sanitize_cwd(None) == "root"

    def test_long_paths_are_bounded_and_dont_collide(self):
        # Two deep paths sharing a long prefix must NOT map to the same dir —
        # otherwise one project's sessions leak into another's picker.
        base = "/Users/x/" + "verydeepdirectory/" * 20
        a, b = paths.sanitize_cwd(base + "alpha"), paths.sanitize_cwd(base + "beta")
        assert a != b
        assert len(a) <= paths._MAX_SANITIZED_CWD + 9  # truncated + "-<8 hex>"

    def test_no_path_separators_survive(self):
        assert "/" not in paths.sanitize_cwd("/a/b/c")


class TestWriteAndRead:
    def test_round_trip(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q1", "a1"))
        log.log_turn(_turn("q2", "a2"))
        data = slog.read_session(log.path)
        assert data["turns"] == 2
        assert [m["role"] for m in data["messages"]] == [
            "user", "assistant", "user", "assistant",
        ]

    def test_meta_record_written(self, home):
        log = slog.SessionLog(cwd="/proj/a", model="opus-5")
        log.log_turn(_turn("q", "a"))
        meta = slog.read_session(log.path)["meta"]
        assert meta["model"] == "opus-5" and meta["cwd"] == "/proj/a"
        assert meta["session_id"] == log.session_id

    def test_compaction_marker_does_not_lose_transcript(self, home):
        # The point of append-only: compaction shrinks the LIVE context, but the
        # transcript must still hold everything that was said.
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("early", "early-answer"))
        log.log_compaction()
        log.log_turn(_turn("late", "late-answer"))
        joined = str(slog.read_session(log.path)["messages"])
        assert "early" in joined and "late" in joined
        assert slog.read_session(log.path)["turns"] == 2

    def test_truncated_last_line_is_tolerated(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q", "a"))
        with open(log.path, "a") as f:
            f.write('{"t":"turn","messa')  # killed mid-write
        assert slog.read_session(log.path)["turns"] == 1

    def test_empty_turn_writes_nothing(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn([])
        assert slog.read_session(log.path)["turns"] == 0

    def test_write_failure_is_silent(self, home, monkeypatch):
        # A log must never break the turn the user is waiting on.
        log = slog.SessionLog(cwd="/proj/a")
        log.path = home / "nonexistent-dir" / "x.jsonl"
        log.log_turn(_turn("q", "a"))  # must not raise


class TestListing:
    def test_per_directory_isolation(self, home):
        a = slog.SessionLog(cwd="/proj/alpha")
        a.log_turn(_turn("alpha question", "x"))
        b = slog.SessionLog(cwd="/proj/beta")
        b.log_turn(_turn("beta question", "y"))
        assert len(slog.list_sessions("/proj/alpha")) == 1
        assert len(slog.list_sessions("/proj/beta")) == 1
        assert "alpha" in slog.list_sessions("/proj/alpha")[0]["preview"]

    def test_newest_first(self, home):
        first = slog.SessionLog(cwd="/proj/a")
        first.log_turn(_turn("older", "x"))
        second = slog.SessionLog(cwd="/proj/a")
        second.log_turn(_turn("newer", "y"))
        import os

        os.utime(first.path, (time.time() - 500,) * 2)
        previews = [s["preview"] for s in slog.list_sessions("/proj/a")]
        assert previews[0] == "newer" and previews[1] == "older"

    def test_sessions_with_no_turn_are_skipped(self, home):
        slog.SessionLog(cwd="/proj/a")  # meta only — resuming it restores nothing
        assert slog.list_sessions("/proj/a") == []

    def test_limit_caps_what_is_offered(self, home):
        for i in range(5):
            log = slog.SessionLog(cwd="/proj/a")
            log.log_turn(_turn(f"q{i}", "a"))
        assert len(slog.list_sessions("/proj/a", limit=2)) == 2
        # …and capping the LISTING must not delete anything.
        assert len(slog.list_sessions("/proj/a", limit=99)) == 5

    def test_latest_session(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("only one", "a"))
        assert slog.latest_session("/proj/a")["preview"] == "only one"

    def test_latest_none_when_empty(self, home):
        assert slog.latest_session("/proj/empty") is None

    def test_preview_flattens_and_clips(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("line one\n\n   line two" + "x" * 400, "a"))
        preview = slog.list_sessions("/proj/a")[0]["preview"]
        assert "\n" not in preview and preview.endswith("…")
        assert len(preview) <= slog._PREVIEW_CHARS + 1


class TestEmptySessionCleanup:
    """A turn-less session file is junk on disk.

    The `meta` record is written at startup, before we know whether the user will
    type anything — so quitting immediately (or cancelling a `--resume` picker)
    used to leave a file behind. Those were already hidden from the picker, which
    is why a directory could show 3 files but only 2 entries; they're now removed
    at exit instead of lingering until they age out.
    """

    def test_empty_session_is_discarded(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        path = log.path
        assert path.exists()
        assert log.discard_if_empty() is True
        assert not path.exists()

    def test_session_with_a_turn_is_kept(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q", "a"))
        assert log.discard_if_empty() is False
        assert log.path.exists()

    def test_discard_is_idempotent(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        assert log.discard_if_empty() is True
        assert log.discard_if_empty() is False  # already gone, no raise

    def test_turnless_file_written_by_another_process_is_left_alone(self, home):
        # Defensive: only unlink a file we can confirm has no turns.
        log = slog.SessionLog(cwd="/proj/a")
        other = slog.SessionLog(cwd="/proj/a")
        other.log_turn(_turn("their question", "a"))
        # Point log at the OTHER (non-empty) file; it must refuse to delete it.
        log.path = other.path
        assert log.discard_if_empty() is False
        assert other.path.exists()


class TestExpiry:
    def test_old_sessions_swept_recent_kept(self, home):
        import os

        old = slog.SessionLog(cwd="/proj/a")
        old.log_turn(_turn("stale", "a"))
        fresh = slog.SessionLog(cwd="/proj/a")
        fresh.log_turn(_turn("current", "a"))
        os.utime(old.path, (time.time() - 40 * 86400,) * 2)

        assert paths.sweep_old_sessions(30) == 1
        assert not old.path.exists() and fresh.path.exists()

    def test_zero_disables_sweep(self, home):
        import os

        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q", "a"))
        os.utime(log.path, (time.time() - 999 * 86400,) * 2)
        assert paths.sweep_old_sessions(0) == 0
        assert log.path.exists()

    def test_sweeps_every_project_not_just_cwd(self, home):
        # A directory you stopped working in must still age out.
        import os

        for proj in ("/proj/a", "/proj/b"):
            log = slog.SessionLog(cwd=proj)
            log.log_turn(_turn("q", "a"))
            os.utime(log.path, (time.time() - 60 * 86400,) * 2)
        assert paths.sweep_old_sessions(30) == 2

    def test_saved_conversations_are_never_swept(self, home):
        # /save is the durable, user-curated path — expiry must not touch it.
        conv = paths.conversations_dir() / "conversation_20200101_000000.json"
        conv.write_text("[]")
        import os

        os.utime(conv, (time.time() - 999 * 86400,) * 2)
        paths.sweep_old_sessions(30)
        assert conv.exists()


class TestPreviewShowsWhatTheUserTyped:
    """The picker label must be the user's words, not injected context.

    Every one of these prefixes is prepended by the client BEFORE the agent
    stores the message, so an unfiltered preview labelled unrelated sessions
    identically (several rows all reading `[Episodic Memory …] 1. "hello" → …`)
    and none of them could be told apart.
    """

    def test_episodic_block_is_stripped(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(
            _turn(
                '[Episodic Memory - Similar Past Tasks]\n1. "hello" → no tools '
                "(similarity: 0.79)\n\n\nRefactor the router",
                "ok",
            )
        )
        assert slog.list_sessions(cwd="/proj/a")[0]["preview"] == "Refactor the router"

    def test_steering_block_is_stripped(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("<steering>always use tabs</steering>\ndo the thing", "ok"))
        assert slog.list_sessions(cwd="/proj/a")[0]["preview"] == "do the thing"

    def test_plan_mode_banner_is_stripped(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(
            _turn("<plan-mode-active>read only</plan-mode-active>\nplan the work", "ok")
        )
        assert slog.list_sessions(cwd="/proj/a")[0]["preview"] == "plan the work"

    def test_falls_through_to_the_next_real_prompt(self, home):
        # A first turn that is PURE injection yields nothing; the label must come
        # from the next real prompt rather than giving up and showing "(no prompt)".
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("[Episodic Memory - Similar Past Tasks]\n1. x", "ok"))
        log.log_turn(_turn("the actual question", "ok"))
        assert slog.list_sessions(cwd="/proj/a")[0]["preview"] == "the actual question"

    def test_background_subagent_report_is_not_a_prompt(self, home):
        # Auto-delivered as a user message, but it's the agent talking to itself.
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(
            _turn("Your background sub-agent 'explore-2' finished: …", "noted")
        )
        log.log_turn(_turn("now summarize it", "ok"))
        assert slog.list_sessions(cwd="/proj/a")[0]["preview"] == "now summarize it"

    def test_plain_prompt_is_unchanged(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("just a normal question", "ok"))
        assert slog.list_sessions(cwd="/proj/a")[0]["preview"] == "just a normal question"


class TestResumedHistoryIsRecorded:
    """A restored conversation must be copied into the NEW session file.

    Otherwise the new file holds only what happened after the restore, so
    resuming it later replays a fragment of a conversation the user can see in
    full on screen — and each resume-of-a-resume truncates the chain further.
    """

    def _seed(self, log, data):
        from mnemoai.client.agent.message_codec import (
            convert_strands_messages_to_langchain,
        )

        log.seed_history(convert_strands_messages_to_langchain(data["messages"]))

    def test_seeded_history_is_replayed(self, home):
        first = slog.SessionLog(cwd="/proj/a")
        first.log_turn(_turn("the original question", "the original answer"))

        second = slog.SessionLog(cwd="/proj/a")
        self._seed(second, slog.read_session(first.path))
        second.log_turn(_turn("follow up", "ok"))

        replayed = str(slog.read_session(second.path)["messages"])
        assert "the original question" in replayed
        assert "follow up" in replayed

    def test_resume_of_a_resume_keeps_everything(self, home):
        a = slog.SessionLog(cwd="/proj/a")
        a.log_turn(_turn("turn one", "a1"))
        b = slog.SessionLog(cwd="/proj/a")
        self._seed(b, slog.read_session(a.path))
        b.log_turn(_turn("turn two", "a2"))
        c = slog.SessionLog(cwd="/proj/a")
        self._seed(c, slog.read_session(b.path))
        c.log_turn(_turn("turn three", "a3"))

        replayed = str(slog.read_session(c.path)["messages"])
        assert "turn one" in replayed and "turn two" in replayed
        assert "turn three" in replayed

    def test_the_source_session_is_never_mutated(self, home):
        # Resuming must leave the original resumable and unchanged, so the user
        # can resume the same point twice.
        a = slog.SessionLog(cwd="/proj/a")
        a.log_turn(_turn("original", "answer"))
        before = a.path.read_text()

        b = slog.SessionLog(cwd="/proj/a")
        self._seed(b, slog.read_session(a.path))
        b.log_turn(_turn("new work", "ok"))

        assert a.path.read_text() == before

    def test_seeded_history_does_not_inflate_the_turn_count(self, home):
        a = slog.SessionLog(cwd="/proj/a")
        for i in range(3):
            a.log_turn(_turn(f"q{i}", "a"))
        b = slog.SessionLog(cwd="/proj/a")
        self._seed(b, slog.read_session(a.path))
        b.log_turn(_turn("one new question", "a"))
        # "turns" means turns taken in THIS session, so the picker doesn't claim
        # the user asked four things here.
        assert slog.read_session(b.path)["turns"] == 1

    def test_a_resume_nobody_typed_into_is_not_offered(self, home):
        # It's a byte-for-byte duplicate of the session it restored, so listing
        # it would grow the picker by one row on every no-op resume.
        a = slog.SessionLog(cwd="/proj/a")
        a.log_turn(_turn("real question", "a"))
        b = slog.SessionLog(cwd="/proj/a")
        self._seed(b, slog.read_session(a.path))

        listed = slog.list_sessions(cwd="/proj/a")
        assert len(listed) == 1
        assert listed[0]["path"] == str(a.path)

    def test_an_unused_resume_is_discarded_at_exit(self, home):
        a = slog.SessionLog(cwd="/proj/a")
        a.log_turn(_turn("real question", "a"))
        b = slog.SessionLog(cwd="/proj/a")
        self._seed(b, slog.read_session(a.path))
        assert b.discard_if_empty() is True
        assert a.path.exists()  # the source is untouched

    def test_a_resume_that_was_used_is_kept(self, home):
        a = slog.SessionLog(cwd="/proj/a")
        a.log_turn(_turn("real question", "a"))
        b = slog.SessionLog(cwd="/proj/a")
        self._seed(b, slog.read_session(a.path))
        b.log_turn(_turn("something new", "a"))
        assert b.discard_if_empty() is False
        assert b.path.exists()
