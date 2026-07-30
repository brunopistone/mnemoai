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
from mnemoai.client.agent.message_codec import convert_strands_messages_to_langchain
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


class TestCompactionBoundaryIsRecorded:
    """A compaction writes a marker, and the marker changes nothing else.

    The transcript is append-only, so the turns compaction summarizes away are
    still on disk and `--resume` restores them regardless — the marker is purely
    informational, recording WHERE the live context was shrunk. It had been dead
    code (defined, never called from the compaction path).
    """

    def test_marker_does_not_inflate_the_turn_count(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q1", "a1"))
        log.log_compaction()
        log.log_turn(_turn("q2", "a2"))
        # The picker shows "N turns"; a compaction is not a turn the user took.
        assert slog.read_session(log.path)["turns"] == 2

    def test_history_either_side_of_a_compaction_survives(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("before compaction", "a1"))
        log.log_compaction()
        log.log_turn(_turn("after compaction", "a2"))
        blob = str(slog.read_session(log.path)["messages"])
        assert "before compaction" in blob and "after compaction" in blob

    def test_a_marker_only_session_is_still_treated_as_empty(self, home):
        # Otherwise a launch that only compacted would be offered as resumable
        # and restore nothing.
        log = slog.SessionLog(cwd="/proj/a")
        log.log_compaction()
        assert slog.list_sessions(cwd="/proj/a") == []
        assert log.discard_if_empty() is True

    def test_the_compaction_path_calls_it(self, home, monkeypatch):
        # Guard against it going dead again: assert the wiring, not just the API.
        import inspect

        from mnemoai.client.managers import agent_conversation_manager as acm

        src = inspect.getsource(acm)
        assert "log_compaction()" in src, "compaction no longer records a boundary"


class TestTurnSummariesLabelEachTurn:
    """The ``/branch`` picker needs a label per turn, stripped the same way the
    ``--resume`` picker strips (else every row reads as injected context)."""

    def test_one_entry_per_turn_in_order(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("first question", "a1"))
        log.log_turn(_turn("second question", "a2"))
        rows = slog.turn_summaries(log.path)
        assert [r["n"] for r in rows] == [1, 2]
        assert rows[0]["preview"] == "first question"
        assert rows[1]["preview"] == "second question"

    def test_injected_context_is_stripped_from_the_label(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn([
            HumanMessage(content="<steering>x</steering>\nreal prompt"),
            AIMessage(content="a"),
        ])
        assert slog.turn_summaries(log.path)[0]["preview"] == "real prompt"

    def test_restored_history_is_not_a_branchable_turn(self, home):
        # Branching "at" inherited history would just duplicate the parent.
        log = slog.SessionLog(cwd="/proj/a")
        log.seed_history(_turn("inherited", "a0"), source="/old.jsonl")
        log.log_turn(_turn("mine", "a1"))
        rows = slog.turn_summaries(log.path)
        assert [r["preview"] for r in rows] == ["mine"]

    def test_a_session_with_no_turns_has_no_rows(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        assert slog.turn_summaries(log.path) == []


class TestBranchingCopiesAndNeverMutates:
    """A branch is a COPY: the original must stay resumable exactly as it was, so
    a branch that goes nowhere costs nothing."""

    def _three_turns(self):
        log = slog.SessionLog(cwd="/proj/a")
        for i in (1, 2, 3):
            log.log_turn(_turn(f"question {i}", f"answer {i}"))
        return log

    def test_branching_truncates_at_the_chosen_turn(self, home):
        log = self._three_turns()
        new = slog.branch_session(log.path, through_turn=2)
        blob = str(slog.read_session(new)["messages"])
        assert "question 1" in blob and "question 2" in blob
        assert "question 3" not in blob

    def test_the_source_file_is_untouched(self, home):
        log = self._three_turns()
        before = log.path.read_bytes()
        slog.branch_session(log.path, through_turn=1)
        assert log.path.read_bytes() == before

    def test_the_original_stays_fully_resumable(self, home):
        log = self._three_turns()
        slog.branch_session(log.path, through_turn=1)
        data = slog.read_session(log.path)
        assert data["turns"] == 3
        assert "question 3" in str(data["messages"])

    def test_a_branch_is_a_new_file(self, home):
        log = self._three_turns()
        new = slog.branch_session(log.path, through_turn=2)
        assert new is not None and new != log.path

    def test_no_limit_copies_the_whole_session(self, home):
        log = self._three_turns()
        for limit in (None, 0, 99):
            blob = str(slog.read_session(slog.branch_session(log.path, limit))["messages"])
            assert "question 3" in blob

    def test_the_branch_records_where_it_came_from(self, home):
        log = self._three_turns()
        new = slog.branch_session(log.path, through_turn=2)
        recs = list(slog._iter_records(new))
        restore = next(r for r in recs if r.get("t") == "restore")
        assert restore["branched_from"]["through_turn"] == 2
        assert restore["branched_from"]["session_id"] == log.session_id

    def test_copied_history_does_not_count_as_the_branchs_own_turns(self, home):
        # The branch's turn counter means "turns taken IN the branch", so a fresh
        # branch has none — and it isn't offered as a duplicate in the picker.
        log = self._three_turns()
        new = slog.branch_session(log.path, through_turn=2)
        assert slog.read_session(new)["turns"] == 0
        assert slog.turn_summaries(new) == []

    def test_branching_an_empty_session_yields_nothing(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        assert slog.branch_session(log.path, through_turn=1) is None

    def test_branching_a_branch_keeps_the_inherited_history(self, home):
        log = self._three_turns()
        first = slog.branch_session(log.path, through_turn=2)
        second = slog.branch_session(first)
        blob = str(slog.read_session(second)["messages"])
        assert "question 1" in blob and "question 2" in blob


class TestReopeningABranchKeepsAppending:
    def test_it_writes_into_the_existing_file(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q1", "a1"))
        new = slog.branch_session(log.path, through_turn=1)

        reopened = slog.SessionLog.reopen(new)
        assert reopened.path == new  # no third file
        reopened.log_turn(_turn("branch turn", "a2"))
        blob = str(slog.read_session(new)["messages"])
        assert "q1" in blob and "branch turn" in blob

    def test_no_second_meta_record_is_written(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q1", "a1"))
        new = slog.branch_session(log.path, through_turn=1)
        slog.SessionLog.reopen(new).log_turn(_turn("q2", "a2"))
        metas = [r for r in slog._iter_records(new) if r.get("t") == "meta"]
        assert len(metas) == 1

    def test_the_turn_counter_continues_instead_of_restarting(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q1", "a1"))
        new = slog.branch_session(log.path, through_turn=1)
        r1 = slog.SessionLog.reopen(new)
        r1.log_turn(_turn("b1", "x"))
        # Reopen AGAIN (a later /branch or restart) — numbering must not reset.
        r2 = slog.SessionLog.reopen(new)
        r2.log_turn(_turn("b2", "y"))
        ns = [r["n"] for r in slog._iter_records(new) if r.get("t") == "turn"]
        assert ns == [1, 2]

    def test_a_reopened_branch_is_offered_once_it_has_a_turn(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q1", "a1"))
        new = slog.branch_session(log.path, through_turn=1)
        slog.SessionLog.reopen(new).log_turn(_turn("in the branch", "a2"))
        previews = [s["preview"] for s in slog.list_sessions(cwd="/proj/a")]
        assert any("in the branch" in p or "q1" in p for p in previews)


class TestABranchIsDistinguishableInThePicker:
    """A fork inherits its parent's opening prompt, so the preview alone renders
    the two as identical rows — the same indistinguishable-sessions bug fixed for
    injected prefixes in 1.8.1."""

    def test_the_listing_carries_the_branch_point(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("shared opening", "a1"))
        log.log_turn(_turn("second", "a2"))
        new = slog.branch_session(log.path, through_turn=1)
        slog.SessionLog.reopen(new).log_turn(_turn("in branch", "a3"))

        rows = {r["path"]: r for r in slog.list_sessions(cwd="/proj/a")}
        assert rows[str(new)]["branched_from"]["through_turn"] == 1
        assert not rows[str(log.path)]["branched_from"]  # the parent is not a fork

    def test_read_session_exposes_the_provenance(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q", "a"))
        new = slog.branch_session(log.path, through_turn=1)
        assert slog.read_session(new)["branched_from"]["session_id"] == log.session_id

    def test_an_ordinary_resume_is_not_labelled_a_branch(self, home):
        # seed_history writes a `restore` record too; only a fork sets provenance.
        log = slog.SessionLog(cwd="/proj/a")
        log.seed_history(_turn("restored", "a"), source="/old.jsonl")
        log.log_turn(_turn("q", "a"))
        assert slog.read_session(log.path)["branched_from"] == {}

    def test_the_picker_row_tags_a_branch(self, home):
        from mnemoai.main import _format_session_label

        row = _format_session_label({
            "modified": time.time(), "turns": 2, "preview": "shared opening",
            "branched_from": {"through_turn": 3, "session_id": "x"},
        })
        assert "branch @ turn 3" in row

    def test_the_picker_row_is_unchanged_for_a_normal_session(self, home):
        from mnemoai.main import _format_session_label

        row = _format_session_label({
            "modified": time.time(), "turns": 2, "preview": "hello",
        })
        assert "branch" not in row and "hello" in row


class TestAResumeChainIsOneRowNotFour:
    """Resuming writes a NEW file seeded with the whole prior conversation, so a
    chat resumed three times exists as four files — each a strict SUPERSET of the
    last, all sharing one opening prompt. Listing per-file showed four
    near-identical rows for ONE conversation, and the newest (longest) one
    reported the FEWEST turns because `turns` excludes inherited history.
    """

    def _chain(self, home, links=3):
        """A session resumed ``links`` times; returns the files oldest→newest."""
        first = slog.SessionLog(cwd="/proj/a")
        first.log_turn(_turn("the original question", "a1"))
        files = [first.path]
        prev = first
        for i in range(links):
            nxt = slog.SessionLog(cwd="/proj/a")
            # What client.resume_session does: decode the transcript back to
            # LangChain messages, then seed them into the fresh session's log.
            restored = convert_strands_messages_to_langchain(
                slog.read_session(prev.path)["messages"]
            )
            nxt.seed_history(restored, source=str(prev.path))
            nxt.log_turn(_turn(f"follow-up {i}", f"b{i}"))
            files.append(nxt.path)
            prev = nxt
        return files

    def test_only_the_tip_of_the_chain_is_offered(self, home):
        files = self._chain(home)
        rows = slog.list_sessions(cwd="/proj/a")
        assert len(rows) == 1
        assert rows[0]["path"] == str(files[-1])

    def test_the_row_is_sized_by_the_whole_conversation(self, home):
        # 1 original + 3 follow-ups = 4 real exchanges, even though the tip file
        # only recorded ONE turn of its own.
        self._chain(home)
        row = slog.list_sessions(cwd="/proj/a")[0]
        assert row["turns"] == 1
        assert row["exchanges"] == 4

    def test_the_label_shows_the_true_length(self, home):
        from mnemoai.main import _format_session_label

        self._chain(home)
        row = slog.list_sessions(cwd="/proj/a")[0]
        label = _format_session_label(row)
        assert "4 turns" in label
        assert "1 turn " not in label
        assert "(continued)" in label

    def test_nothing_from_the_hidden_links_is_lost(self, home):
        # The whole justification for hiding them.
        self._chain(home)
        blob = str(slog.list_sessions(cwd="/proj/a")[0]["messages"])
        assert "the original question" in blob
        for i in range(3):
            assert f"follow-up {i}" in blob

    def test_an_unresumed_session_is_still_offered(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("standalone", "a"))
        rows = slog.list_sessions(cwd="/proj/a")
        assert [r["path"] for r in rows] == [str(log.path)]
        assert "(continued)" not in __import__(
            "mnemoai.main", fromlist=["x"]
        )._format_session_label(rows[0])

    def test_two_independent_conversations_stay_two_rows(self, home):
        a = slog.SessionLog(cwd="/proj/a")
        a.log_turn(_turn("first chat", "x"))
        b = slog.SessionLog(cwd="/proj/a")
        b.log_turn(_turn("second chat", "y"))
        assert len(slog.list_sessions(cwd="/proj/a")) == 2

    def test_a_branch_is_NOT_collapsed(self, home):
        # A fork DIVERGES from its parent (both are real conversations); only a
        # resume supersedes. Collapsing a branch would hide the work it forked from.
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("shared start", "a1"))
        log.log_turn(_turn("the path not taken", "a2"))
        fork = slog.branch_session(log.path, through_turn=1)
        slog.SessionLog.reopen(fork).log_turn(_turn("the other direction", "b1"))

        rows = slog.list_sessions(cwd="/proj/a")
        assert len(rows) == 2
        paths = {r["path"] for r in rows}
        assert str(log.path) in paths and str(fork) in paths

    def test_a_load_of_a_saved_conversation_does_not_hide_a_session(self, home):
        # /load seeds from a conversations/*.json file, which is not a session —
        # a non-session source must never suppress anything.
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("live work", "a"))
        loaded = slog.SessionLog(cwd="/proj/a")
        loaded.seed_history(_turn("from a saved file", "b"), source="/tmp/conv.json")
        loaded.log_turn(_turn("after the load", "c"))
        assert len(slog.list_sessions(cwd="/proj/a")) == 2

    def test_the_limit_applies_after_collapsing(self, home):
        # Otherwise a chain could consume the whole budget and hide real chats.
        self._chain(home, links=4)
        other = slog.SessionLog(cwd="/proj/a")
        other.log_turn(_turn("a different chat", "z"))
        rows = slog.list_sessions(cwd="/proj/a", limit=2)
        assert len(rows) == 2
        previews = {r["preview"] for r in rows}
        assert "a different chat" in previews


class TestExchangeCountingIgnoresInjectedMessages:
    """`exchanges` sizes the row, so it must count only real prompts — a
    tool-result message and a background sub-agent report both carry
    ``role: user``."""

    def test_a_plain_prompt_counts(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("hello", "hi"))
        assert slog.read_session(log.path)["exchanges"] == 1

    def test_an_episodic_prefixed_prompt_counts_once(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn([
            HumanMessage(content='[Episodic Memory - x]\n1. "y"\n\nthe real ask'),
            AIMessage(content="a"),
        ])
        assert slog.read_session(log.path)["exchanges"] == 1

    def test_a_steering_only_message_does_not_count(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn([
            HumanMessage(content="<steering>rules</steering>"),
            AIMessage(content="a"),
        ])
        assert slog.read_session(log.path)["exchanges"] == 0

    def test_a_background_subagent_report_does_not_count(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn([
            HumanMessage(content="Your background sub-agent finished: …"),
            AIMessage(content="a"),
        ])
        assert slog.read_session(log.path)["exchanges"] == 0


class TestAnExplicitIdStillReachesACollapsedLink:
    """The menu collapses a resume chain to its tip, but naming an id asks for
    THAT exact point — so hiding a row must not make it unresolvable. Regression:
    `--resume <id>` of a superseded session started failing with "No session
    matching …" once collapsing landed.
    """

    def _chain(self):
        first = slog.SessionLog(cwd="/proj/a")
        first.log_turn(_turn("original", "a1"))
        second = slog.SessionLog(cwd="/proj/a")
        second.seed_history(
            convert_strands_messages_to_langchain(
                slog.read_session(first.path)["messages"]
            ),
            source=str(first.path),
        )
        second.log_turn(_turn("follow-up", "a2"))
        return first, second

    def test_collapsing_hides_the_parent_from_the_menu(self, home):
        first, second = self._chain()
        assert [r["path"] for r in slog.list_sessions(cwd="/proj/a")] == [
            str(second.path)
        ]

    def test_uncollapsed_listing_still_contains_every_link(self, home):
        first, second = self._chain()
        paths = {
            r["path"]
            for r in slog.list_sessions(cwd="/proj/a", collapse_chains=False)
        }
        assert paths == {str(first.path), str(second.path)}

    def test_the_hidden_link_keeps_its_own_shorter_history(self, home):
        # Resolving the id must restore that POINT, not the tip.
        first, _ = self._chain()
        rows = slog.list_sessions(cwd="/proj/a", collapse_chains=False)
        parent = next(r for r in rows if r["path"] == str(first.path))
        blob = str(parent["messages"])
        assert "original" in blob and "follow-up" not in blob

    def test_the_resume_path_resolves_against_all_sessions(self, home):
        # Guard the wiring: reading the menu list here is what broke it.
        import inspect

        from mnemoai import main as m

        src = inspect.getsource(m._resume_session)
        assert "collapse_chains=False" in src
