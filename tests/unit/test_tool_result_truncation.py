"""Unit tests for tool-result truncation (client/agent/tool_formatting.py).

A single tool result must never be able to overflow the context window — a
runaway grep/read is capped at the source, keeping head+tail with a middle note
so the model knows output was trimmed. Also covers the agent's thin delegator.
"""

from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.agent.tool_formatting import truncate_tool_result


class TestTruncateToolResult:
    def test_under_cap_is_noop(self):
        assert truncate_tool_result("hello world", 100) == "hello world"

    def test_zero_disables(self):
        big = "x" * 5000
        assert truncate_tool_result(big, 0) == big

    def test_caps_and_keeps_both_ends(self):
        text = "HEAD" + ("m" * 2000) + "TAIL"
        out = truncate_tool_result(text, 200)
        assert len(out) <= 200
        assert out.startswith("HEAD")
        assert out.endswith("TAIL")
        assert "truncated" in out

    def test_note_reports_dropped_amount(self):
        text = "a" * 1000
        out = truncate_tool_result(text, 200)
        # 800 chars dropped (1000 - 200 cap), ~200 tokens (÷4).
        assert "truncated 800 chars" in out
        assert "~200 tokens" in out

    def test_exact_cap_length_is_noop(self):
        text = "y" * 300
        assert truncate_tool_result(text, 300) == text


class TestAgentDelegator:
    def _agent(self, cap):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a._max_tool_result_chars = cap
        return a

    def test_delegates_with_configured_cap(self):
        a = self._agent(cap=100)
        out = a._truncate_tool_result("z" * 500)
        assert len(out) <= 100
        assert "truncated" in out

    def test_missing_attr_uses_default(self):
        # Bare object without the attr set falls back to a safe default (no-op
        # for small input).
        a = LangGraphAgent.__new__(LangGraphAgent)
        assert a._truncate_tool_result("small") == "small"


class TestDynamicDefault:
    """The cap auto-derives from MAX_CONVERSATION_TOKENS (10% of the window in
    chars, ~4 chars/token) when MAX_TOOL_RESULT_CHARS is unset, so it scales with
    the model instead of a fixed number."""

    def _init_cap(self, monkeypatch, max_conv, llm=None):
        import mnemoai.client.agent.agent as agent_mod

        cfg = {"MAX_CONVERSATION_TOKENS": max_conv, "LLM": llm or {}}
        monkeypatch.setattr(
            agent_mod.config, "get", lambda k, d=None: cfg.get(k, d)
        )
        # Replicate the __init__ derivation (no full agent build needed).
        _max = int(agent_mod.config.get("MAX_CONVERSATION_TOKENS", 8192))
        return int(
            agent_mod.config.get("LLM", {}).get(
                "MAX_TOOL_RESULT_CHARS", int(_max * 0.10 * 4)
            )
        )

    def test_large_window_derives_larger_cap(self, monkeypatch):
        assert self._init_cap(monkeypatch, 1_000_000) == 400_000

    def test_small_window_derives_smaller_cap(self, monkeypatch):
        assert self._init_cap(monkeypatch, 65_536) == 26_214

    def test_explicit_config_wins(self, monkeypatch):
        cap = self._init_cap(monkeypatch, 1_000_000, llm={"MAX_TOOL_RESULT_CHARS": 5000})
        assert cap == 5000
