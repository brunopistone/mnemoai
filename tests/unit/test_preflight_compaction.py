"""Unit tests for the client's mid-loop compaction hook (_compact_now).

Layer B of the context-management work: before a model call, compact when the
accumulated history exceeds the high-water mark (default 80% of
MAX_CONVERSATION_TOKENS). ``force=True`` (the overflow backstop) skips the check.
Uses a bare client + fake manager so no LLM/event loop is needed.
"""

import mnemoai.client.client as client_mod
from mnemoai.client.client import LangGraphClient


class _FakeManager:
    def __init__(self, token_count, evict_returns=False, token_after_evict=None):
        self.max_tokens = 1000
        self._token_count = token_count
        # Token count reported after an eviction pass (defaults to unchanged).
        self._token_after_evict = (
            token_after_evict if token_after_evict is not None else token_count
        )
        self._evict_returns = evict_returns
        self._evicted = False
        self.evict_calls = 0
        self.compacted = []  # (keep_recent,) per _compact call

    def count_tokens(self, msgs):
        return self._token_after_evict if self._evicted else self._token_count

    def evict_old_tool_results(self, agent):
        self.evict_calls += 1
        if self._evict_returns:
            self._evicted = True
        return self._evict_returns

    async def _compact(self, client, model, agent, keep_recent=6):
        self.compacted.append(keep_recent)
        return True


class _FakeAgent:
    def __init__(self, last_input_tokens=None):
        self.messages = [1, 2, 3]  # non-empty; content irrelevant (count is faked)
        self._last_input_tokens = last_input_tokens


def _client(token_count, high_water=None, evict_returns=False, token_after_evict=None,
            last_input_tokens=None):
    c = LangGraphClient.__new__(LangGraphClient)
    c.agent = _FakeAgent(last_input_tokens=last_input_tokens)
    c.model = object()
    c.conversation_manager = _FakeManager(
        token_count, evict_returns=evict_returns, token_after_evict=token_after_evict
    )
    return c


def _patch_config(monkeypatch, **llm):
    def _get(key, default=None):
        if key == "LLM":
            return llm
        return default

    monkeypatch.setattr(client_mod.config, "get", _get)


class TestCompactNow:
    def test_compacts_when_over_high_water(self, monkeypatch):
        # high_water defaults to 80% of max_tokens (800); 900 > 800 → compact.
        _patch_config(monkeypatch)
        c = _client(token_count=900)
        assert c._compact_now() is True
        assert c.conversation_manager.compacted == [6]  # default keep_recent

    def test_noop_under_high_water(self, monkeypatch):
        _patch_config(monkeypatch)
        c = _client(token_count=100)  # < 800
        assert c._compact_now() is False
        assert c.conversation_manager.compacted == []

    def test_explicit_high_water_override(self, monkeypatch):
        _patch_config(monkeypatch, COMPACT_HIGH_WATER_TOKENS=50)
        c = _client(token_count=100)  # > 50 → compact
        assert c._compact_now() is True

    def test_zero_high_water_disables_proactive(self, monkeypatch):
        _patch_config(monkeypatch, COMPACT_HIGH_WATER_TOKENS=0)
        c = _client(token_count=999999)
        assert c._compact_now() is False
        assert c.conversation_manager.compacted == []

    def test_force_skips_budget_and_uses_small_window(self, monkeypatch):
        # force=True compacts regardless of token count, keeping a smaller window.
        _patch_config(monkeypatch, COMPACT_HIGH_WATER_TOKENS=0)
        c = _client(token_count=1)
        assert c._compact_now(force=True) is True
        assert c.conversation_manager.compacted == [2]  # aggressive keep_recent

    def test_no_agent_returns_false(self, monkeypatch):
        _patch_config(monkeypatch)
        c = LangGraphClient.__new__(LangGraphClient)
        c.agent = None
        assert c._compact_now() is False


class TestToolEvictionShortCircuit:
    """Over the high-water mark, the cheap tool-result eviction runs first; if it
    brings the estimate back under budget, the expensive summary is skipped."""

    def test_eviction_under_budget_skips_summary(self, monkeypatch):
        # 900 > 800; eviction drops it to 100 (< 800) → no _compact summary.
        _patch_config(monkeypatch)
        c = _client(token_count=900, evict_returns=True, token_after_evict=100)
        assert c._compact_now() is True
        assert c.conversation_manager.evict_calls == 1
        assert c.conversation_manager.compacted == []  # summary NOT run

    def test_eviction_insufficient_falls_through_to_summary(self, monkeypatch):
        # 900 > 800; eviction only drops it to 850 (still > 800) → full summary.
        _patch_config(monkeypatch)
        c = _client(token_count=900, evict_returns=True, token_after_evict=850)
        assert c._compact_now() is True
        assert c.conversation_manager.evict_calls == 1
        assert c.conversation_manager.compacted == [6]

    def test_no_eviction_still_summarizes(self, monkeypatch):
        # Nothing to evict → straight to the full summary (unchanged behavior).
        _patch_config(monkeypatch)
        c = _client(token_count=900, evict_returns=False)
        assert c._compact_now() is True
        assert c.conversation_manager.compacted == [6]

    def test_force_bypasses_eviction(self, monkeypatch):
        # force=True is the overflow backstop — goes straight to summary.
        _patch_config(monkeypatch, COMPACT_HIGH_WATER_TOKENS=0)
        c = _client(token_count=1, evict_returns=True)
        assert c._compact_now(force=True) is True
        assert c.conversation_manager.evict_calls == 0
        assert c.conversation_manager.compacted == [2]


class TestTriggerPrefersGroundTruth:
    """The trigger prefers the provider's exact input_tokens (_last_input_tokens,
    ground truth) over the manager's estimate, which over-counts non-OpenAI
    models ~2x and would fire compaction far too early."""

    def test_uses_actual_not_inflated_estimate(self, monkeypatch):
        _patch_config(monkeypatch)
        # Estimate says 5000 (>800 high-water) but the provider's real count is
        # 100 (<800). Must NOT compact — trust the ground truth.
        c = _client(token_count=5000, last_input_tokens=100)
        assert c._compact_now() is False
        assert c.conversation_manager.compacted == []

    def test_actual_over_high_water_compacts(self, monkeypatch):
        _patch_config(monkeypatch)
        # Real count 900 > 800 → compact, regardless of a low estimate.
        c = _client(token_count=10, last_input_tokens=900)
        assert c._compact_now() is True
        assert c.conversation_manager.compacted == [6]

    def test_estimate_fallback_when_no_actual(self, monkeypatch):
        _patch_config(monkeypatch)
        # No turn has run yet (actual is None) → fall back to the estimate.
        c = _client(token_count=900, last_input_tokens=None)
        assert c._compact_now() is True


class TestSummaryModel:
    """_summary_model returns a reasoning-disabled variant by default (built once,
    provider-agnostic), reuses the main model when SUMMARIZATION_THINK is on, and
    falls back to the main model if the variant can't be built."""

    def _client_with_ctrl(self, monkeypatch, think, ctrl):
        _patch_config(monkeypatch, SUMMARIZATION_THINK=think)
        c = LangGraphClient.__new__(LangGraphClient)
        c.model = "MAIN"
        c.llm_controller = ctrl
        return c

    def test_default_builds_non_reasoning_variant(self, monkeypatch):
        class _Ctrl:
            def build_non_reasoning_model(self):
                return "NO_THINK"

        c = self._client_with_ctrl(monkeypatch, think=False, ctrl=_Ctrl())
        assert c._summary_model() == "NO_THINK"
        # Cached: a second call reuses it (no rebuild).
        assert c._summary_model() == "NO_THINK"

    def test_summarization_think_reuses_main_model(self, monkeypatch):
        class _Ctrl:
            def build_non_reasoning_model(self):
                raise AssertionError("must not build when thinking is kept on")

        c = self._client_with_ctrl(monkeypatch, think=True, ctrl=_Ctrl())
        assert c._summary_model() == "MAIN"

    def test_falls_back_to_main_on_build_failure(self, monkeypatch):
        class _Ctrl:
            def build_non_reasoning_model(self):
                raise RuntimeError("provider error")

        c = self._client_with_ctrl(monkeypatch, think=False, ctrl=_Ctrl())
        assert c._summary_model() == "MAIN"
