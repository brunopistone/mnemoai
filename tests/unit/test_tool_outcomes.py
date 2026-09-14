"""Unit tests for tool-outcome recording (``tool_patterns``).

A success RATE needs both outcomes. ``record_tool_outcome`` was only ever called
with a hardcoded ``True``, from inside the episode-storage path — which by
construction runs only after a success — so every sample was a success and the
ratio was 1.00 for every tool forever: a frequency count wearing a success label.
One real profile held 39,375 samples, 39,375 successes, not one failure.

Three halves are covered here: the call sites now record a failed turn too, the
counter's ``success`` argument can no longer be defaulted away, and the injected
``<profile>`` block refuses to name a "best tool" for an intent whose rates
cannot rank (``max`` over all-1.00 rates returns whichever tool came first).
"""

import inspect
import json
import os
import tempfile

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mnemoai.client.managers.user_profile_manager import UserProfileManager
from mnemoai.client.ui import chat_interface as ci_mod
from mnemoai.client.ui.chat_interface import ChatInterface


def _manager(**profile):
    m = UserProfileManager(profile_path=os.path.join(tempfile.mkdtemp(), "p.json"))
    m.profile.update(profile)
    return m


def _tools(*names):
    return [{"name": n, "args": {}, "id": f"id-{n}"} for n in names]


def _stats(manager, intent, tool):
    return manager.profile["tool_patterns"][intent][tool]


class TestTheOutcomeMustBeSupplied:
    def test_success_has_no_default(self):
        # The whole defect in one line: a default let the only call site pass the
        # literal True and read like it was recording an outcome.
        success = inspect.signature(UserProfileManager.record_tool_outcome).parameters[
            "success"
        ]
        assert success.default is inspect.Parameter.empty

    def test_a_failure_moves_the_rate_off_one(self):
        m = _manager()
        m.record_tool_outcome("debug", _tools("fs_read"), True)
        m.record_tool_outcome("debug", _tools("fs_read"), False)
        # Only the two counters are stored; the ratio is derived at read time.
        assert _stats(m, "debug", "fs_read") == {"success": 1, "total": 2}

    def test_a_failed_turn_counts_against_every_tool_it_used(self):
        # Turn-level attribution, stated in the docstring: the turn ended badly, and
        # which of the eight calls caused it isn't knowable here.
        m = _manager()
        used = _tools("fs_read", "grep_search", "file_edit", "execute_bash")
        m.record_tool_outcome("debug", used, False)
        for tool in ("fs_read", "grep_search", "file_edit", "execute_bash"):
            assert _stats(m, "debug", tool) == {"success": 0, "total": 1}

    def test_no_tools_records_nothing(self):
        m = _manager()
        m.record_tool_outcome("debug", [], False)
        assert m.profile.get("tool_patterns", {}) == {}


class TestTheCallSitesRecordBothOutcomes:
    """Drives the real ``ChatInterface`` storage paths, not ``record_tool_outcome``
    directly: the bug was never in the counter, it was in who called it."""

    @pytest.fixture(autouse=True)
    def _profiling_on(self, monkeypatch):
        monkeypatch.setattr(
            ci_mod.config,
            "get",
            lambda k, d=None: {"USE_PROFILING": True} if k == "PROFILE" else d,
        )

    def _interface(self):
        ci = ChatInterface.__new__(ChatInterface)
        stored = []
        ci.client = type(
            "_C",
            (),
            {
                "episodic_memory": type(
                    "_E",
                    (),
                    {"store_episode": lambda self, **kw: stored.append(kw)},
                )(),
                "profile_manager": _manager(),
                "agent": None,
                "previous_query": None,
                "previous_response": None,
                "previous_messages": None,
            },
        )()
        return ci, stored

    def _turn(self, answer):
        return [
            HumanMessage(content="fix the parser"),
            AIMessage(
                content="",
                tool_calls=[{"name": "fs_read", "args": {}, "id": "1"}],
            ),
            ToolMessage(content="ok", tool_call_id="1"),
            AIMessage(content=answer),
        ]

    def test_the_immediate_path_records_a_failure(self):
        ci, stored = self._interface()
        ci.client.agent = type("_A", (), {"messages": None})()
        answer = "Error: could not read the file"
        ci.client.agent.messages = self._turn(answer)

        ci._ChatInterface__store_current_episode_immediately("fix the parser", answer)

        # No episode — the turn failed — but the outcome is on the books.
        assert stored == []
        assert _stats(ci.client.profile_manager, "debug", "fs_read") == {
            "success": 0,
            "total": 1,
        }

    def test_the_immediate_path_still_records_a_success(self):
        ci, stored = self._interface()
        answer = "The parser now handles the trailing comma."
        ci.client.agent = type("_A", (), {"messages": self._turn(answer)})()

        ci._ChatInterface__store_current_episode_immediately("fix the parser", answer)

        assert len(stored) == 1 and stored[0]["outcome"] == "success"
        assert _stats(ci.client.profile_manager, "debug", "fs_read")["success"] == 1

    def _previous(self, ci, answer):
        ci.client.previous_query = "why does it not work"
        ci.client.previous_response = answer
        ci.client.previous_messages = [
            {"role": "user", "content": [{"text": "add a parser to utils"}]},
            {"role": "user", "content": [{"text": "why does it not work"}]},
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"name": "fs_read", "input": {}, "toolUseId": "1"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "1", "content": [{"text": "ok"}]}}
                ],
            },
            {"role": "assistant", "content": [{"text": answer}]},
        ]

    def test_the_legacy_path_records_a_failure(self):
        ci, stored = self._interface()
        self._previous(ci, "I changed the tokenizer.")

        # A correction in the next prompt is what marks the previous turn failed.
        ci._ChatInterface__store_episode_in_episodic_memory("no, that's wrong")

        assert stored == []
        patterns = ci.client.profile_manager.profile["tool_patterns"]
        assert patterns["implement"]["fs_read"] == {"success": 0, "total": 1}

    def test_a_failure_is_filed_under_the_intent_a_success_would_have_been(self):
        # The success path keys on the conversation's FIRST prompt ("add a parser"
        # → implement), so a failure keyed on the latest one ("why does it not
        # work" → learn) would split one task's outcomes across two buckets.
        ci, _ = self._interface()
        self._previous(ci, "I changed the tokenizer.")
        ci._ChatInterface__store_episode_in_episodic_memory("no, that's wrong")
        failed = ci.client.profile_manager.profile["tool_patterns"]

        ok, stored = self._interface()
        self._previous(ok, "I changed the tokenizer.")
        ok._ChatInterface__store_episode_in_episodic_memory("thanks, perfect")
        succeeded = ok.client.profile_manager.profile["tool_patterns"]

        assert stored, "the success path should still store its episode"
        assert list(failed) == list(succeeded) == ["implement"]

    def test_a_toolless_turn_records_nothing(self):
        # There is no tool to attribute the outcome to.
        ci, _ = self._interface()
        answer = "Error: unable to help with that"
        ci.client.agent = type(
            "_A",
            (),
            {"messages": [HumanMessage(content="fix it"), AIMessage(content=answer)]},
        )()

        ci._ChatInterface__store_current_episode_immediately("fix it", answer)

        assert ci.client.profile_manager.profile.get("tool_patterns", {}) == {}

    def test_nothing_is_recorded_with_profiling_off(self, monkeypatch):
        monkeypatch.setattr(
            ci_mod.config,
            "get",
            lambda k, d=None: {"USE_PROFILING": False} if k == "PROFILE" else d,
        )
        ci, _ = self._interface()
        answer = "Error: could not read the file"
        ci.client.agent = type("_A", (), {"messages": self._turn(answer)})()

        ci._ChatInterface__store_current_episode_immediately("fix the parser", answer)

        assert ci.client.profile_manager.profile.get("tool_patterns", {}) == {}

    def test_a_profiling_failure_does_not_break_the_turn(self):
        # The answer is already on screen; a counter write must not undo the turn.
        ci, _ = self._interface()
        answer = "Error: could not read the file"
        ci.client.agent = type("_A", (), {"messages": self._turn(answer)})()

        def _boom(*a, **kw):
            raise RuntimeError("disk full")

        ci.client.profile_manager.record_tool_outcome = _boom
        ci._ChatInterface__store_current_episode_immediately("fix the parser", answer)


class TestAnUnlabelledProfileIsRepaired:
    """The code fix can't undo what a year of hardcoded ``True`` wrote to disk."""

    def _profile(self, tool_patterns):
        path = os.path.join(tempfile.mkdtemp(), "p.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "interaction_count": 900,
                    "verbosity": 0.42,
                    "technical_level": 0.55,
                    "tool_patterns": tool_patterns,
                    "_legacy_migrated": True,
                    "_recount_repaired": True,
                    "_tech_signal_repaired": True,
                },
                f,
            )
        return path

    def _all_success(self, **totals):
        return {
            "debug": {
                name: {"success": n, "total": n}
                for name, n in totals.items()
            }
        }

    def test_a_profile_with_no_failure_anywhere_is_cleared(self):
        m = UserProfileManager(
            profile_path=self._profile(self._all_success(fs_read=39000, git_safe=375))
        )
        assert m.profile["tool_patterns"] == {}

    def test_a_profile_that_records_failures_is_kept(self):
        patterns = self._all_success(fs_read=10)
        patterns["debug"]["execute_bash"] = {"success": 4, "total": 7}
        m = UserProfileManager(profile_path=self._profile(patterns))
        assert m.profile["tool_patterns"] == patterns

    def test_the_repair_runs_once(self):
        path = self._profile(self._all_success(fs_read=9))
        UserProfileManager(profile_path=path)
        again = UserProfileManager(profile_path=path)
        again.record_tool_outcome("debug", _tools("fs_read"), True)
        # A second pass would clear the honest sample just recorded.
        assert _stats(again, "debug", "fs_read")["total"] == 1

    def test_an_empty_profile_is_only_flagged(self):
        m = UserProfileManager(profile_path=self._profile({}))
        assert m.profile["tool_patterns"] == {}
        assert m.profile["_outcomes_repaired"] is True

    def test_a_fresh_profile_needs_no_repair(self):
        m = UserProfileManager(profile_path=os.path.join(tempfile.mkdtemp(), "n.json"))
        assert m.profile.get("_outcomes_repaired") is True

    @pytest.mark.parametrize(
        "patterns",
        [
            {"debug": "not a dict"},
            {"debug": {"fs_read": "not a dict"}},
            {"debug": {"fs_read": {"success": None, "total": None}}},
            {"debug": {"fs_read": {}}},
        ],
    )
    def test_a_malformed_profile_does_not_crash_the_repair(self, patterns):
        m = UserProfileManager(profile_path=self._profile(patterns))
        assert m.profile.get("_outcomes_repaired") is True


class TestTheProfileBlockNeedsSomethingToRank:
    """``Tools:`` is the only consumer of ``tool_patterns``, and it claims a best
    tool. With every rate at 1.00 ``max`` returns whichever the dict yielded
    first — a hint chosen by nothing, injected every turn."""

    def _summary(self, patterns):
        m = _manager(interaction_count=50, tool_patterns=patterns)
        return m.get_profile_summary()

    def test_no_best_tool_is_named_when_nothing_ever_failed(self):
        summary = self._summary(
            {
                "debug": {
                    "fs_read": {"success": 9, "total": 9},
                    "execute_bash": {"success": 4, "total": 4},
                }
            }
        )
        assert "Tools: learning" in summary

    def test_a_recorded_failure_makes_the_ranking_mean_something(self):
        summary = self._summary(
            {
                "debug": {
                    "fs_read": {"success": 9, "total": 9},
                    "execute_bash": {"success": 2, "total": 6},
                }
            }
        )
        assert "Tools: debug: fs_read" in summary

    def test_it_names_the_better_tool_not_the_first(self):
        # Insertion order is deliberately the losing one.
        summary = self._summary(
            {
                "debug": {
                    "execute_bash": {"success": 2, "total": 6},
                    "fs_read": {"success": 8, "total": 9},
                }
            }
        )
        assert "Tools: debug: fs_read" in summary

    def test_an_intent_that_can_rank_is_not_hidden_by_one_that_cannot(self):
        summary = self._summary(
            {
                "debug": {"fs_read": {"success": 9, "total": 9}},
                "implement": {
                    "file_edit": {"success": 3, "total": 5},
                    "fs_write": {"success": 1, "total": 4},
                },
            }
        )
        assert "Tools: implement: file_edit" in summary
