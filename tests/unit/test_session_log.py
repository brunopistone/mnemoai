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
