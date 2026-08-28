"""Unit tests for taking back the last exchange (`/rewind`).

A rewind is an undo of the CONVERSATION, and everything interesting about it is a
boundary:

  * it finds the last thing the USER typed — not the last `role: user` message,
    which can be a tool result, an auto-delivered sub-agent report, or a prompt
    that is nothing but injected context;
  * the transcript is append-only, so the withdrawal is a RECORD, and every
    reader (`read_session`, `turn_summaries`, `branch_session`) has to honor it —
    a `/branch` that could still fork at a withdrawn turn would hand the
    conversation straight back;
  * a compaction that already stands for the turn can't be undone, so the rewind
    is refused rather than half-applied;
  * files on disk are never touched, and the notice says so.

Pure logic plus a stub client — no LLM, no TTY.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mnemoai.client import rewind
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


def _texts(encoded):
    """The text of each logged (strands-encoded) message, in order."""
    return [slog._message_text(m) for m in encoded]


class TestBoundary:
    def test_finds_the_last_prompt(self):
        messages = [
            HumanMessage(content="first"),
            AIMessage(content="answer"),
            HumanMessage(content="second"),
            AIMessage(content="answer"),
        ]
        assert rewind.boundary(messages) == 2

    def test_a_tool_result_is_not_a_prompt(self):
        # A ToolMessage rides in the user turn on the wire; the boundary must be
        # the prompt, or a rewind would drop only the tail of the turn.
        messages = [
            HumanMessage(content="do the thing"),
            AIMessage(content=""),
            ToolMessage(content="tool output", tool_call_id="1"),
            AIMessage(content="done"),
        ]
        assert rewind.boundary(messages) == 0

    def test_an_injection_only_message_is_skipped(self):
        # A delivery-only turn's prompt is pure injection: the steering block plus
        # nothing. It is not a prompt anyone typed, so it can't be the boundary.
        messages = [
            HumanMessage(content="the real question"),
            AIMessage(content="answer"),
            HumanMessage(content="<steering>always use tabs</steering>"),
            AIMessage(content="answer"),
        ]
        assert rewind.boundary(messages) == 0

    def test_an_auto_delivered_report_is_skipped(self):
        messages = [
            HumanMessage(content="the real question"),
            AIMessage(content="answer"),
            HumanMessage(content="Your background sub-agent finished: …"),
            AIMessage(content="here is what it found"),
        ]
        assert rewind.boundary(messages) == 0

    def test_injection_prefixed_prompt_still_counts(self):
        messages = [
            HumanMessage(content="<steering>tabs</steering>fix the parser"),
            AIMessage(content="answer"),
        ]
        assert rewind.boundary(messages) == 0

    def test_no_prompt_at_all(self):
        assert rewind.boundary([AIMessage(content="a")]) == -1
        assert rewind.boundary([]) == -1


class TestPreview:
    def test_strips_injection_and_flattens(self):
        msg = HumanMessage(content="<steering>tabs</steering>fix   the\nparser")
        assert rewind.preview(msg) == "fix the parser"

    def test_clips_with_a_marker(self):
        msg = HumanMessage(content="x" * 200)
        assert len(rewind.preview(msg)) == rewind._PREVIEW_CHARS
        assert rewind.preview(msg).endswith("…")

    def test_block_content(self):
        msg = HumanMessage(content=[{"type": "text", "text": "hello there"}])
        assert rewind.preview(msg) == "hello there"


class TestRender:
    def test_names_what_went_and_where(self):
        out = rewind.render("fix the parser", 6, recorded=True, sinks=[])
        assert "fix the parser" in out
        assert "6 messages" in out
        assert "transcript" in out

    def test_without_a_transcript_it_does_not_claim_one(self):
        out = rewind.render("q", 2, recorded=False, sinks=[])
        assert "transcript" not in out

    def test_singular_message(self):
        assert "1 message dropped" in rewind.render("q", 1, recorded=False, sinks=[])

    def test_always_says_files_are_untouched(self):
        # The whole scope of the feature: a user who reads this must not expect
        # their edits to have been rolled back.
        assert "untouched" in rewind.render("q", 2, recorded=True, sinks=[])

    def test_names_the_learning_it_cannot_reach(self):
        out = rewind.render("q", 2, recorded=True, sinks=["episodic memory"])
        assert "episodic memory" in out

    def test_says_nothing_about_learning_that_is_off(self):
        assert "learned" not in rewind.render("q", 2, recorded=True, sinks=[])


class TestLearningSinks:
    def _client(self, **attrs):
        client = LangGraphClient.__new__(LangGraphClient)
        for name in ("episodic_memory", "playbook", "reflector"):
            setattr(client, name, attrs.get(name))
        return client

    def test_only_what_is_enabled(self):
        assert rewind.learning_sinks(self._client()) == []
        assert rewind.learning_sinks(self._client(episodic_memory=object())) == [
            "episodic memory"
        ]

    def test_the_playbook_needs_both_halves(self):
        # The reflector is what writes to it; a store nothing feeds learned nothing.
        assert rewind.learning_sinks(self._client(playbook=object())) == []
        assert rewind.learning_sinks(
            self._client(playbook=object(), reflector=object())
        ) == ["the playbook"]


class _StubAgent:
    def __init__(self, messages, session_log=None):
        self.messages = messages
        self.session_log = session_log
        self._last_input_tokens = 12345


def _client(agent):
    client = LangGraphClient.__new__(LangGraphClient)
    client.agent = agent
    client.episodic_memory = None
    client.playbook = None
    client.reflector = None
    client.previous_query = "the withdrawn prompt"
    client.previous_response = "the withdrawn answer"
    client.previous_messages = list(getattr(agent, "messages", []) or [])
    return client


class TestWithdraw:
    def test_drops_the_last_exchange(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("first", "a1"))
        log.log_turn(_turn("second", "a2"))
        agent = _StubAgent(
            [*_turn("first", "a1"), *_turn("second", "a2")], session_log=log
        )
        out = _client(agent).rewind_turn()
        assert [m.content for m in agent.messages] == ["first", "a1"]
        assert "second" in out

    def test_drops_the_whole_turn_not_just_the_prompt(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("first", "a1"))
        log.log_turn(_turn("second", "a2"))
        messages = [
            *_turn("first", "a1"),
            HumanMessage(content="second"),
            AIMessage(content=""),
            ToolMessage(content="output", tool_call_id="1"),
            AIMessage(content="a2"),
        ]
        agent = _StubAgent(messages, session_log=log)
        _client(agent).rewind_turn()
        assert len(agent.messages) == 2

    def test_forgets_the_cached_context_size(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn())
        agent = _StubAgent(_turn(), session_log=log)
        _client(agent).rewind_turn()
        # The count measured a conversation that no longer exists — the same
        # invariant every path that REPLACES live history carries.
        assert agent._last_input_tokens is None

    def test_clears_the_pending_episode(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn())
        agent = _StubAgent(_turn(), session_log=log)
        client = _client(agent)
        client.rewind_turn()
        # Legacy episodic storage evaluates the PREVIOUS exchange on the next
        # prompt; left in place, the withdrawn turn would be learned anyway.
        assert client.previous_query is None
        assert client.previous_response is None
        assert client.previous_messages is None

    def test_records_the_withdrawal_in_the_transcript(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("first", "a1"))
        log.log_turn(_turn("second", "a2"))
        agent = _StubAgent(
            [*_turn("first", "a1"), *_turn("second", "a2")], session_log=log
        )
        _client(agent).rewind_turn()
        data = slog.read_session(log.path)
        assert data["turns"] == 1
        assert _texts(data["all_messages"]) == ["first", "a1"]

    def test_works_without_a_transcript(self, home):
        # SESSION_MAX_AGE_DAYS: 0 — nothing recorded, so nothing to keep in step.
        agent = _StubAgent(_turn("only", "a"), session_log=None)
        out = _client(agent).rewind_turn()
        assert agent.messages == []
        assert "transcript" not in out

    def test_nothing_to_rewind(self):
        assert "no turns yet" in _client(_StubAgent([])).rewind_turn()

    def test_no_prompt_left(self):
        agent = _StubAgent([AIMessage(content="orphaned answer")])
        assert "summarized" in _client(agent).rewind_turn()

    def test_refuses_after_a_compaction(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("first", "a1"))
        log.log_compaction(summary="…", kept=_turn("first", "a1"))
        agent = _StubAgent(_turn("first", "a1"), session_log=log)
        out = _client(agent).rewind_turn()
        assert "Can't rewind" in out
        # …and refused means unchanged, not partly applied.
        assert len(agent.messages) == 2
        assert slog.read_session(log.path)["turns"] == 1


class TestWithdrawnTurnRecord:
    def test_read_session_drops_it_from_both_lists(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q1", "a1"))
        log.log_turn(_turn("q2", "a2"))
        log.log_rewind(2)
        data = slog.read_session(log.path)
        assert data["turns"] == 1
        assert data["exchanges"] == 1
        assert _texts(data["messages"]) == ["q1", "a1"]
        assert _texts(data["all_messages"]) == ["q1", "a1"]

    def test_the_text_stays_on_disk(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q1", "a1"))
        log.log_rewind(1)
        # Append-only is the point: the withdrawal is a record, not a redaction.
        assert "q1" in log.path.read_text(encoding="utf-8")

    def test_turn_summaries_skips_it_and_renumbers(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q1", "a1"))
        log.log_turn(_turn("q2", "a2"))
        log.log_rewind(2)
        log.log_turn(_turn("q3", "a3"))
        rows = slog.turn_summaries(log.path)
        assert [r["preview"] for r in rows] == ["q1", "q3"]
        assert [r["n"] for r in rows] == [1, 2]

    def test_a_branch_cannot_resurrect_it(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q1", "a1"))
        log.log_turn(_turn("q2", "a2"))
        log.log_rewind(2)
        fork = slog.branch_session(log.path)
        assert fork is not None
        assert _texts(slog.read_session(fork)["all_messages"]) == ["q1", "a1"]

    def test_the_counter_steps_back_so_the_number_is_reused(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q1", "a1"))
        log.log_turn(_turn("q2", "a2"))
        log.log_rewind(2)
        log.log_turn(_turn("q3", "a3"))
        # Two records now carry n=2, so the withdrawal has to match the most
        # recent turn that still has it — the replacement must survive.
        contents = _texts(slog.read_session(log.path)["all_messages"])
        assert contents == ["q1", "a1", "q3", "a3"]

    def test_a_session_whose_only_turn_was_withdrawn_is_discarded(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn())
        log.log_rewind(1)
        path = log.path
        assert log.discard_if_empty() is True
        assert not path.exists()

    def test_a_rewind_without_a_transcript_is_not_recorded(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.path = None
        assert log.log_rewind(1) is False

    def test_a_rewind_record_with_no_number_takes_the_last_turn(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q1", "a1"))
        log.log_turn(_turn("q2", "a2"))
        log._append({"t": "rewind", "ts": 0})  # tolerant: hand-written / older shape
        assert slog.read_session(log.path)["turns"] == 1


class TestWithdrawingInheritedHistory:
    """A resumed conversation arrives as ONE `restore` blob.

    So its last exchange has no `turn` record to skip, and a rewind that only
    truncated live history would hold for the run and then be undone by the next
    `--resume` — the state that comes back has to be the state the user had.
    """

    def _resumed(self, home, history):
        log = slog.SessionLog(cwd="/proj/a")
        log.seed_history(history, source="/prior/session.jsonl")
        return log

    def test_pins_the_surviving_history(self, home):
        log = self._resumed(home, [*_turn("q1", "a1"), *_turn("q2", "a2")])
        agent = _StubAgent([*_turn("q1", "a1"), *_turn("q2", "a2")], session_log=log)
        out = _client(agent).rewind_turn()
        assert "transcript" in out  # it DID reach the transcript
        data = slog.read_session(log.path)
        assert _texts(data["messages"]) == ["q1", "a1"]
        assert data["checkpoint"] is True
        # Both lists narrow, as they do when a turn record is withdrawn by number:
        # a rewind means the exchange did not happen, so the replay, the picker
        # preview and `exchanges` must not show it either.
        assert _texts(data["all_messages"]) == ["q1", "a1"]
        assert data["exchanges"] == 1
        # The raw text is still on disk, though — an undo, not a redaction.
        assert "q2" in log.path.read_text(encoding="utf-8")

    def test_the_withdrawal_survives_a_resume(self, home):
        log = self._resumed(home, [*_turn("q1", "a1"), *_turn("q2", "a2")])
        agent = _StubAgent([*_turn("q1", "a1"), *_turn("q2", "a2")], session_log=log)
        _client(agent).rewind_turn()
        # Resuming the file restores `messages`, which is what the user was left
        # with — not the raw blob that still holds the withdrawn exchange.
        restored = slog.read_session(log.path)["messages"]
        again = slog.SessionLog(cwd="/proj/a")
        again.seed_history(
            [HumanMessage(content=slog._message_text(restored[0])), AIMessage(content="a1")],
            source=str(log.path),
            kept=None,
        )
        assert _texts(slog.read_session(again.path)["all_messages"]) == ["q1", "a1"]

    def test_it_does_not_withdraw_a_turn_by_accident(self, home):
        # The record names no turn, so the tolerant "no number → the last turn"
        # fallback must not fire: the turn taken in this session survives.
        log = self._resumed(home, _turn("inherited", "a0"))
        log.log_turn(_turn("mine", "a1"))
        agent = _StubAgent(
            [*_turn("inherited", "a0"), *_turn("mine", "a1")], session_log=log
        )
        _client(agent).rewind_turn()  # withdraws "mine" — a real turn record
        assert slog.read_session(log.path)["turns"] == 0
        agent.messages = [*_turn("inherited", "a0")]
        _client(agent).rewind_turn()  # now reaches inherited history
        data = slog.read_session(log.path)
        assert data["messages"] == []
        assert data["checkpoint"] is True
        # "mine" was withdrawn by number, "inherited" by the rebase — the second
        # rewind must not have re-withdrawn the first one's record on top.
        assert data["turns"] == 0
        raw = log.path.read_text(encoding="utf-8")
        assert raw.count('"t": "rewind"') == 2

    def test_the_file_is_not_discarded_at_exit(self, home):
        # It holds no turn of its own, but it is the ONLY record of the
        # withdrawal — discarding it would resurrect the exchange.
        log = self._resumed(home, [*_turn("q1", "a1"), *_turn("q2", "a2")])
        agent = _StubAgent([*_turn("q1", "a1"), *_turn("q2", "a2")], session_log=log)
        _client(agent).rewind_turn()
        assert log.discard_if_empty() is False
        assert log.path.exists()

    def test_a_branch_forks_the_rebased_state(self, home):
        log = self._resumed(home, [*_turn("q1", "a1"), *_turn("q2", "a2")])
        agent = _StubAgent([*_turn("q1", "a1"), *_turn("q2", "a2")], session_log=log)
        _client(agent).rewind_turn()
        log.log_turn(_turn("q3", "a3"))
        fork = slog.branch_session(log.path)
        assert fork is not None
        assert _texts(slog.read_session(fork)["messages"]) == ["q1", "a1", "q3", "a3"]

    def test_an_unencodable_window_records_nothing(self, home):
        log = self._resumed(home, _turn("q1", "a1"))
        before = log.path.read_text(encoding="utf-8")
        assert log.log_rewind(kept=[object()]) is False
        assert log.path.read_text(encoding="utf-8") == before

    def test_an_empty_window_is_a_valid_record(self, home):
        # Everything withdrawn: `messages` is empty, which is not "no checkpoint".
        log = self._resumed(home, _turn("q1", "a1"))
        assert log.log_rewind(kept=[]) is True
        data = slog.read_session(log.path)
        assert data["messages"] == []
        assert data["checkpoint"] is True


class TestLastLiveTurn:
    def test_reports_the_last_turn(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q1", "a1"))
        log.log_turn(_turn("q2", "a2"))
        target = slog.last_live_turn(log.path)
        assert target["n"] == 2
        assert target["preview"] == "q2"
        assert target["compacted"] is False

    def test_skips_an_already_withdrawn_turn(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q1", "a1"))
        log.log_turn(_turn("q2", "a2"))
        log.log_rewind(2)
        # Two rewinds in a row walk back two turns, not the same one twice.
        assert slog.last_live_turn(log.path)["preview"] == "q1"

    def test_nothing_left(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn())
        log.log_rewind(1)
        assert slog.last_live_turn(log.path) is None

    def test_a_checkpoint_after_the_turn_blocks_it(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q1", "a1"))
        log.log_compaction(summary="…", kept=_turn("q1", "a1"))
        assert slog.last_live_turn(log.path)["compacted"] is True

    def test_a_checkpoint_just_before_the_turn_blocks_it(self, home):
        # A mid-turn compaction is written before the turn record it belongs to,
        # and is indistinguishable from a post-turn one — so both count.
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q1", "a1"))
        log.log_compaction(summary="…", kept=_turn("q1", "a1"))
        log.log_turn(_turn("q2", "a2"))
        assert slog.last_live_turn(log.path)["compacted"] is True

    def test_an_older_checkpoint_does_not_block(self, home):
        log = slog.SessionLog(cwd="/proj/a")
        log.log_turn(_turn("q1", "a1"))
        log.log_compaction(summary="…", kept=_turn("q1", "a1"))
        log.log_turn(_turn("q2", "a2"))
        log.log_turn(_turn("q3", "a3"))
        assert slog.last_live_turn(log.path)["compacted"] is False

    def test_a_seeded_checkpoint_does_not_block_the_first_turn(self, home):
        # seed_history re-states the checkpoint that was CARRIED IN, so the first
        # turn of a resumed session is still withdrawable.
        log = slog.SessionLog(cwd="/proj/a")
        log.seed_history(_turn("earlier", "a0"), source="x", summary="…", kept=[])
        log.log_turn(_turn("q1", "a1"))
        assert slog.last_live_turn(log.path)["compacted"] is False


class TestCommandSurface:
    class _Stub:
        def __init__(self):
            self.calls = []

        def rewind_turn(self):
            self.calls.append("rewind_turn")
            return "⟲ withdrew your last prompt"

    def test_dispatch_answers_locally(self, capsys):
        ci = ChatInterface.__new__(ChatInterface)
        ci.client = self._Stub()
        assert ci._dispatch("/rewind") is None
        assert ci.client.calls == ["rewind_turn"]
        assert "withdrew" in capsys.readouterr().out

    def test_documented_and_autocompleted(self):
        assert "/rewind" in {cmd for cmd, _ in ChatInterface._COMMANDS}
        documented = {
            cmd.split()[0]
            for _group, entries in ChatInterface._COMMAND_GROUPS
            for cmd, _desc in entries
        }
        assert "/rewind" in documented
