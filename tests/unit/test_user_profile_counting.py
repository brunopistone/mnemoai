"""Unit tests for ``UserProfileManager`` counting accuracy (client/managers/).

Focus: per-turn scoping, the same bug class ``test_reflector_scope`` covers for
the reflector. Profiling runs after EVERY turn while ``agent.messages`` holds the
whole session, so passing all of it re-counted every earlier prompt:
``interaction_count`` grew as N²/2 (an observed real profile reached 62,977 for a
few hundred turns) and each trait's EMA washed out toward a degenerate value
because the same messages kept being folded in.

Also covers the one-shot repair of a profile already inflated on disk — the code
fix can't undo what was written before it.
"""

import json
import os
import tempfile

from mnemoai.client.managers.user_profile_manager import UserProfileManager
from mnemoai.client.memory.reflector import current_turn_messages


def _turn(n):
    return [
        {"role": "user", "content": [{"text": f"question {n} about python"}]},
        {"role": "assistant", "content": [{"text": "an answer"}]},
    ]


def _manager():
    return UserProfileManager(
        profile_path=os.path.join(tempfile.mkdtemp(), "p.json")
    )


class TestEachTurnIsCountedOnce:
    def test_ten_turns_count_as_ten(self):
        m = _manager()
        history = []
        for n in range(1, 11):
            history.extend(_turn(n))
            m.analyze_conversation(current_turn_messages(history))
        assert m.profile["interaction_count"] == 10

    def test_the_unscoped_call_is_what_inflated_it(self):
        # Pins the mechanism rather than just the fix: N(N+1)/2 = 55 for 10 turns.
        m = _manager()
        history = []
        for n in range(1, 11):
            history.extend(_turn(n))
            m.analyze_conversation(history)
        assert m.profile["interaction_count"] == 55

    def test_current_turn_messages_keeps_only_the_last_exchange(self):
        history = []
        for n in range(1, 4):
            history.extend(_turn(n))
        scoped = current_turn_messages(history)
        assert len(scoped) == 2
        assert "question 3" in str(scoped)
        assert "question 1" not in str(scoped)

    def test_the_client_scopes_before_profiling(self):
        # Guards the wiring: the fix lives at the call site, so a behavioral test
        # of the manager alone would stay green if it regressed.
        import inspect

        from mnemoai.client import client as client_mod

        src = inspect.getsource(client_mod.LangGraphClient.query)
        assert "current_turn_messages(self.agent.messages)" in src


class TestAnInflatedProfileIsRepaired:
    """Modelled on a real profile: 62,977 interactions, ``technical_level``
    washed down to 0.0002 while mid-range traits still held signal."""

    def _profile(self, **overrides):
        path = os.path.join(tempfile.mkdtemp(), "p.json")
        data = {
            "created_at": "2026-01-01T00:00:00",
            "last_updated": "2026-01-01T00:00:00",
            "interaction_count": 62977,
            "verbosity": 0.42,
            "directness": 0.28,
            "technical_level": 0.0002,
            "abstraction": 0.5,
            "top_domains": [],
            "tool_patterns": {},
            "_legacy_migrated": True,
        }
        data.update(overrides)
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_the_count_is_brought_back_to_a_plausible_number(self):
        m = UserProfileManager(profile_path=self._profile())
        # sqrt(2 * 62977) ≈ 354 turns.
        assert 300 < m.profile["interaction_count"] < 400

    def test_a_saturated_trait_is_reset_so_it_can_relearn(self):
        m = UserProfileManager(profile_path=self._profile())
        assert m.profile["technical_level"] == 0.5

    def test_an_unsaturated_trait_is_preserved(self):
        # It still carries real signal; resetting it would discard learning.
        m = UserProfileManager(profile_path=self._profile())
        assert m.profile["verbosity"] == 0.42
        assert m.profile["directness"] == 0.28

    def test_the_repair_runs_only_once(self):
        path = self._profile()
        first = UserProfileManager(profile_path=path).profile["interaction_count"]
        second = UserProfileManager(profile_path=path).profile["interaction_count"]
        assert first == second, "repair re-applied and shrank the count again"

    def test_a_healthy_profile_is_untouched(self):
        path = self._profile(interaction_count=42, technical_level=0.0002)
        m = UserProfileManager(profile_path=path)
        assert m.profile["interaction_count"] == 42
        # Below the threshold nothing is assumed corrupt — a low trait may be real.
        assert m.profile["technical_level"] == 0.0002

    def test_a_fresh_profile_needs_no_repair(self):
        m = UserProfileManager(profile_path=os.path.join(tempfile.mkdtemp(), "n.json"))
        assert m.profile["interaction_count"] == 0
        assert m.profile.get("_recount_repaired") is True

    def test_a_non_numeric_count_does_not_crash_the_repair(self):
        m = UserProfileManager(profile_path=self._profile(interaction_count="lots"))
        assert m.profile.get("_recount_repaired") is True
