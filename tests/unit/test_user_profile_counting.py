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

    def test_tool_results_are_not_interactions(self):
        # A ToolMessage encodes to role "user" (message_codec), so filtering on the
        # role alone counted every tool RESULT: one turn with 3 tool calls scored 4.
        m = _manager()
        turn = [{"role": "user", "content": [{"text": "fix the bug in python"}]}]
        for i in range(3):
            turn.append({"role": "assistant", "content": [{"toolUse": {"name": "fs_read"}}]})
            turn.append({
                "role": "user",
                "content": [{"toolResult": {"toolUseId": f"c{i}",
                                            "content": [{"text": "file body"}]}}],
            })
        turn.append({"role": "assistant", "content": [{"text": "done"}]})
        m.analyze_conversation(turn)
        assert m.profile["interaction_count"] == 1

    def test_a_tool_heavy_turn_yields_one_ema_sample(self):
        # The >= 5 "enough data" gate in get_profile_summary must not trip after a
        # single tool-heavy turn.
        m = _manager()
        for n in range(1, 4):
            turn = [{"role": "user", "content": [{"text": f"prompt {n} about python"}]}]
            for i in range(4):
                turn.append({
                    "role": "user",
                    "content": [{"toolResult": {"toolUseId": f"c{n}{i}"}}],
                })
            m.analyze_conversation(turn)
        assert m.profile["interaction_count"] == 3

    def test_the_turn_boundary_is_a_prompt_not_a_tool_result(self):
        # current_turn_messages must not cut the turn at its last tool result.
        turn = [{"role": "user", "content": [{"text": "do the thing"}]}]
        for i in range(3):
            turn.append({"role": "assistant", "content": [{"toolUse": {"name": "x"}}]})
            turn.append({"role": "user", "content": [{"toolResult": {"toolUseId": f"c{i}"}}]})
        turn.append({"role": "assistant", "content": [{"text": "done"}]})
        assert len(current_turn_messages(turn)) == len(turn)

    def _client(self, messages):
        """A client wired just enough to run ``_profile_turn`` for real."""
        from mnemoai.client.client import LangGraphClient

        c = LangGraphClient.__new__(LangGraphClient)
        c.agent = type("_A", (), {"messages": messages})()
        c.profile_manager = _manager()
        return c

    def test_the_client_counts_one_turn_once(self, monkeypatch):
        # Drives the REAL call path (client._profile_turn), not a hand-scoped
        # input: the earlier version of this test did its own scoping, so it
        # passed even with the fix removed.
        from langchain_core.messages import AIMessage, HumanMessage

        from mnemoai.client import client as client_mod

        monkeypatch.setattr(
            client_mod.config, "get",
            lambda k, d=None: {"USE_PROFILING": True} if k == "PROFILE" else d,
        )
        history = []
        for n in range(1, 6):
            history += [HumanMessage(content=f"prompt {n} about python"),
                        AIMessage(content="an answer")]
        c = self._client(history)
        # Five turns of accumulated history; profiling each turn in sequence.
        for n in range(1, 6):
            c.agent.messages = history[: n * 2]
            c._profile_turn()
        assert c.profile_manager.profile["interaction_count"] == 5

    def test_profiling_is_skipped_when_disabled(self, monkeypatch):
        from langchain_core.messages import HumanMessage

        from mnemoai.client import client as client_mod

        monkeypatch.setattr(
            client_mod.config, "get",
            lambda k, d=None: {"USE_PROFILING": False} if k == "PROFILE" else d,
        )
        c = self._client([HumanMessage(content="hello there")])
        c._profile_turn()
        assert c.profile_manager.profile["interaction_count"] == 0

    def test_a_profiling_failure_never_breaks_the_turn(self, monkeypatch):
        # The answer has already been streamed to the user by this point.
        from langchain_core.messages import HumanMessage

        from mnemoai.client import client as client_mod

        monkeypatch.setattr(
            client_mod.config, "get",
            lambda k, d=None: {"USE_PROFILING": True} if k == "PROFILE" else d,
        )
        c = self._client([HumanMessage(content="hello there")])

        def _boom(_messages):
            raise KeyError("interaction_count")

        c.profile_manager.analyze_conversation = _boom
        c._profile_turn()  # must not raise


class TestTraitsAreScoredOnWhatTheUserTyped:
    """Every trait is scored from the message text, so scoring the RAW message
    inverted the signal: with a 5-entry episodic block prepended, "fix it"
    (6 chars) measured 496 and pushed `verbosity` UP toward "detailed". It fired
    on every turn episodic memory injected.
    """

    EPISODIC = "[Episodic Memory - Similar Past Tasks]\n" + "".join(
        f'{i}. "an earlier task" \u2192 execute_bash, fs_read, file_edit '
        f"(similarity: 0.7{i})\n"
        for i in range(1, 6)
    )

    def _verbosity_after(self, text):
        m = _manager()
        before = m.profile["verbosity"]
        m.analyze_conversation([{"role": "user", "content": [{"text": text}]}])
        return before, m.profile["verbosity"]

    def test_a_terse_prompt_lowers_verbosity_despite_injection(self):
        before, after = self._verbosity_after(self.EPISODIC + "\nfix it")
        assert after < before, "injected context made a terse prompt look verbose"

    def test_injection_does_not_change_the_score(self):
        # The whole point: the same prompt must score the same either way.
        _, with_block = self._verbosity_after(self.EPISODIC + "\nfix it")
        _, without = self._verbosity_after("fix it")
        assert with_block == without

    def test_a_genuinely_verbose_prompt_still_raises_verbosity(self):
        before, after = self._verbosity_after(
            self.EPISODIC + "\n" + "please explain this in detail " * 20
        )
        assert after > before

    def test_a_steering_block_is_stripped_too(self):
        _, with_block = self._verbosity_after("<steering>be nice</steering>\nfix it")
        _, without = self._verbosity_after("fix it")
        assert with_block == without

    def test_extract_returns_only_the_typed_text(self):
        m = _manager()
        msg = {"role": "user", "content": [{"text": self.EPISODIC + "\nfix it"}]}
        assert m._extract_text_content(msg) == "fix it"


class TestALegacyProfileDoesNotBreakATurn:
    """``interaction_count`` predates some profiles. ``+=`` on a bare key raised
    KeyError inside ``client.query``'s try block — replacing an answer the user had
    ALREADY seen streamed with "Something went wrong"."""

    def _legacy(self, **fields):
        path = os.path.join(tempfile.mkdtemp(), "legacy.json")
        with open(path, "w") as f:
            json.dump({"created_at": "x", "_legacy_migrated": True, **fields}, f)
        return UserProfileManager(profile_path=path)

    def test_a_profile_with_no_count_field_still_counts(self):
        m = self._legacy()
        m.analyze_conversation([{"role": "user", "content": [{"text": "hello there"}]}])
        assert m.profile["interaction_count"] == 1

    def test_a_null_count_is_treated_as_zero(self):
        m = self._legacy(interaction_count=None)
        m.analyze_conversation([{"role": "user", "content": [{"text": "hello there"}]}])
        assert m.profile["interaction_count"] == 1

    def test_a_non_numeric_count_is_treated_as_zero(self):
        m = self._legacy(interaction_count="lots")
        m.analyze_conversation([{"role": "user", "content": [{"text": "hello there"}]}])
        assert m.profile["interaction_count"] == 1


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

    def test_the_count_is_reset_rather_than_guessed(self):
        # Inverting N²/2 assumes the old increment was 1/turn when it was
        # 1 + (tool results), so the estimate overshoots by sqrt(1+k) — 2-3x for a
        # tool-heavy user. Its only consumer is the "enough data" gate, so a wrong
        # number is worse than starting over.
        m = UserProfileManager(profile_path=self._profile())
        assert m.profile["interaction_count"] == 0

    def test_the_count_re_accrues_one_per_turn_after_the_reset(self):
        m = UserProfileManager(profile_path=self._profile())
        for n in range(3):
            m.analyze_conversation(
                [{"role": "user", "content": [{"text": f"prompt {n} about python"}]}]
            )
        assert m.profile["interaction_count"] == 3

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
