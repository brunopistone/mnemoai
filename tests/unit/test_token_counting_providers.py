"""Provider-aware, never-undercount token counting (utils.tokenization).
"""

import mnemoai.utils.tokenization as tok


def _set_type(monkeypatch, model_type):
    monkeypatch.setattr(
        tok.config, "get",
        lambda k, d=None: {"TYPE": model_type} if k == "MODEL_ID" else (d or {}),
    )


TEXT = "def f(x):\n    return {'a': 1, 'b': [2, 3]}  # some code + json\n" * 20
_ENCODER_NAME = "o200k_base"

class TestProviderMultipliers:
    def test_openai_is_exact_tiktoken(self, monkeypatch):
        _set_type(monkeypatch, "openai")
        import tiktoken
        exact = len(tiktoken.get_encoding(_ENCODER_NAME).encode(TEXT))
        assert tok.count_tokens(TEXT) == exact  # multiplier 1.0

    def test_anthropic_never_undercounts(self, monkeypatch):
        _set_type(monkeypatch, "anthropic")
        import tiktoken
        base = len(tiktoken.get_encoding(_ENCODER_NAME).encode(TEXT))
        n = tok.count_tokens(TEXT)
        assert n > base                    # scaled up, never below the basis
        assert n >= int(base * 1.4)        # meaningfully conservative (~1.5x)

    def test_mantle_treated_like_claude(self, monkeypatch):
        _set_type(monkeypatch, "mantle")
        import tiktoken
        base = len(tiktoken.get_encoding(_ENCODER_NAME).encode(TEXT))
        assert tok.count_tokens(TEXT) >= int(base * 1.4)

    def test_bedrock_conservative(self, monkeypatch):
        _set_type(monkeypatch, "bedrock")
        import tiktoken
        base = len(tiktoken.get_encoding(_ENCODER_NAME).encode(TEXT))
        assert tok.count_tokens(TEXT) > base  # scaled up

    def test_empty_is_zero(self, monkeypatch):
        _set_type(monkeypatch, "anthropic")
        assert tok.count_tokens("") == 0

    def test_config_override_multiplier(self, monkeypatch):
        # LLM.TOKEN_COUNTING.<TYPE>_MULTIPLIER overrides the default.
        def _get(k, d=None):
            if k == "MODEL_ID":
                return {"TYPE": "anthropic"}
            if k == "LLM":
                return {"TOKEN_COUNTING": {"ANTHROPIC_MULTIPLIER": 2.0}}
            return d or {}
        monkeypatch.setattr(tok.config, "get", _get)
        import tiktoken
        base = len(tiktoken.get_encoding(_ENCODER_NAME).encode(TEXT))
        assert tok.count_tokens(TEXT) == int(base * 2.0)


class TestCaptureInputTokens:
    """The agent records the provider's exact input_tokens (usage_metadata) as
    ground truth for the current context size."""

    def _agent(self):
        from mnemoai.client.agent.agent import LangGraphAgent
        a = LangGraphAgent.__new__(LangGraphAgent)
        a._last_input_tokens = None
        return a

    def test_captures_from_usage_metadata(self):
        from langchain_core.messages import AIMessage
        a = self._agent()
        resp = AIMessage(content="hi", usage_metadata={
            "input_tokens": 1147426, "output_tokens": 10, "total_tokens": 1147436})
        a._capture_input_tokens(resp)
        assert a._last_input_tokens == 1147426  # exact provider count

    def test_no_usage_metadata_leaves_prior_value(self):
        from langchain_core.messages import AIMessage
        a = self._agent()
        a._last_input_tokens = 500
        a._capture_input_tokens(AIMessage(content="hi"))  # no usage_metadata
        assert a._last_input_tokens == 500  # unchanged, not clobbered to None

    def test_bad_response_is_safe(self):
        a = self._agent()
        a._capture_input_tokens(object())  # no usage_metadata attr → no crash
        assert a._last_input_tokens is None
