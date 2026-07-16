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
    messages = [1, 2, 3]  # non-empty; content irrelevant (count is faked)


def _client(token_count, high_water=None, evict_returns=False, token_after_evict=None):
    c = LangGraphClient.__new__(LangGraphClient)
    c.agent = _FakeAgent()
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
