"""Unit tests for the ACE reflector + playbook store (client/memory/).

Focus: per-turn scoping. Reflection runs after EVERY turn with the whole
session in ``messages``, so without scoping each earlier tool call is
re-analyzed every turn -- inflating metrics and re-bumping the confidence of
strategies that were only ever learned once.
"""

from types import SimpleNamespace

import pytest

from mnemoai.client.memory.playbook_store import PlaybookStore
from mnemoai.client.memory.reflector import (
    PlaybookEntry,
    Reflector,
    current_turn_messages,
)


def _human(text):
    return SimpleNamespace(type="human", content=text)


def _ai_tool_call(name, args, call_id):
    return SimpleNamespace(
        type="ai",
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


def _tool_result(name, call_id, content):
    return SimpleNamespace(
        type="tool", name=name, tool_call_id=call_id, content=content
    )


def _failing_turn(seq):
    """One turn whose tool call fails with a path error (a FAILURE_PATTERNS hit)."""
    return [
        _human(f"do thing {seq}"),
        _ai_tool_call("fs_read", {"path": f"/nope/{seq}.py"}, f"c{seq}"),
        _tool_result(
            "fs_read", f"c{seq}", "Error: file not found: no such file or directory"
        ),
    ]


class TestCurrentTurnMessages:
    def test_slices_from_last_human_message(self):
        msgs = _failing_turn(1) + _failing_turn(2)
        assert current_turn_messages(msgs) == msgs[3:]

    def test_returns_all_when_no_human_message(self):
        msgs = [_ai_tool_call("fs_read", {}, "c1"), _tool_result("fs_read", "c1", "ok")]
        assert current_turn_messages(msgs) == msgs

    def test_handles_dict_role_messages(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "b"},
        ]
        assert current_turn_messages(msgs) == msgs[2:]

    def test_empty_input(self):
        assert current_turn_messages([]) == []


class TestReflectOnTrajectory:
    def test_only_current_turn_is_analyzed(self, tmp_path):
        reflector = Reflector(persist_path=str(tmp_path))
        history = _failing_turn(1) + _failing_turn(2) + _failing_turn(3)

        reflector.reflect_on_trajectory(messages=history, task="do thing 3")

        # 3 failing tool calls in history, but only the last turn's one counts.
        assert reflector.metrics["total_tool_calls"] == 1
        assert reflector.metrics["failed_calls"] == 1

    def test_metrics_do_not_inflate_across_turns(self, tmp_path):
        # Simulates the real call pattern: reflect after every turn, passing the
        # whole session each time. Was O(turns^2) tool calls counted.
        reflector = Reflector(persist_path=str(tmp_path))
        history = []
        for seq in range(1, 5):
            history += _failing_turn(seq)
            reflector.reflect_on_trajectory(messages=history, task=f"do thing {seq}")

        assert reflector.metrics["total_tool_calls"] == 4

    def test_opt_out_analyzes_whole_history(self, tmp_path):
        reflector = Reflector(persist_path=str(tmp_path))
        history = _failing_turn(1) + _failing_turn(2)

        reflector.reflect_on_trajectory(
            messages=history, task="do thing 2", scope_to_last_turn=False
        )

        assert reflector.metrics["total_tool_calls"] == 2

    def test_failure_produces_a_playbook_entry(self, tmp_path):
        reflector = Reflector(persist_path=str(tmp_path))
        entries = reflector.reflect_on_trajectory(
            messages=_failing_turn(1), task="do thing 1"
        )
        assert entries
        assert entries[0].outcome == "failure"


class TestFindToolResult:
    def test_repeated_tool_does_not_borrow_the_first_result(self, tmp_path):
        reflector = Reflector(persist_path=str(tmp_path))
        messages = [
            _human("read both"),
            _ai_tool_call("fs_read", {"path": "a.py"}, "c1"),
            _tool_result("fs_read", "c1", "contents of a"),
            _ai_tool_call("fs_read", {"path": "b.py"}, "c2"),
            _tool_result("fs_read", "c2", "contents of b"),
        ]
        assert reflector._find_tool_result("c2", "fs_read", messages) == "contents of b"

    def test_missing_result_is_empty_not_a_sibling(self, tmp_path):
        reflector = Reflector(persist_path=str(tmp_path))
        messages = [
            _human("read both"),
            _ai_tool_call("fs_read", {"path": "a.py"}, "c1"),
            _tool_result("fs_read", "c1", "contents of a"),
            _ai_tool_call("fs_read", {"path": "b.py"}, "c2"),
        ]
        assert reflector._find_tool_result("c2", "fs_read", messages) == ""

    def test_name_fallback_when_provider_gives_no_id(self, tmp_path):
        reflector = Reflector(persist_path=str(tmp_path))
        messages = [
            _human("read it"),
            _ai_tool_call("fs_read", {"path": "a.py"}, ""),
            _tool_result("fs_read", "", "contents of a"),
        ]
        assert reflector._find_tool_result("", "fs_read", messages) == "contents of a"


class TestPlaybookConfidence:
    @pytest.fixture
    def store(self, tmp_path):
        return PlaybookStore(persist_path=str(tmp_path))

    def test_duplicate_strategy_bumps_confidence_once_per_append(self, store):
        def entry():
            return PlaybookEntry(
                context="reading files",
                strategy="check the path exists first",
                source="test",
                outcome="failure",
                confidence=0.9,
            )

        store.append(entry())
        assert len(store.entries) == 1
        assert store.entries[0]["confidence"] == pytest.approx(0.9)

        store.append(entry())
        assert len(store.entries) == 1
        assert store.entries[0]["confidence"] == pytest.approx(1.0)

    def test_confidence_is_capped_at_one(self, store):
        for _ in range(5):
            store.append(
                PlaybookEntry(
                    context="c", strategy="s", source="test", confidence=0.9
                )
            )
        assert store.entries[0]["confidence"] == pytest.approx(1.0)

    def test_entries_persist_and_reload(self, tmp_path):
        store = PlaybookStore(persist_path=str(tmp_path))
        store.append(PlaybookEntry(context="c", strategy="s", source="test"))
        reloaded = PlaybookStore(persist_path=str(tmp_path))
        assert len(reloaded.entries) == 1
        assert reloaded.entries[0]["strategy"] == "s"
