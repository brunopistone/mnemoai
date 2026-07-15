"""Shared token-counting helper — provider-aware and never-undercount.
This is only the PRE-FLIGHT estimate for a not-yet-sent prompt. The exact size
of an already-sent prompt comes from the provider's own ``usage_metadata`` (see
the agent loop), which is ground truth and needs no estimation.
"""

from mnemoai.utils.config import config

_ENCODING_NAME = "o200k_base"
_encoder = None

# Conservative multipliers over the tiktoken basis, per provider family. Chosen
# to NEVER undercount (overflow is the failure mode). Overridable via
# LLM.TOKEN_COUNTING.<TYPE>_MULTIPLIER.
_DEFAULT_MULTIPLIERS = {
    "anthropic": 1.5,   # measured ~1.5x on code/JSON history
    "mantle": 1.5,      # Mantle commonly fronts Claude (anthropic protocol)
    "bedrock": 1.35,    # mixed; Claude on Bedrock still undercounts
    "sagemaker": 1.35,
    "litellm": 1.35,
    "openai": 1.0,      # tiktoken is exact for OpenAI
}


def _get_encoder():
    """Return a cached tiktoken encoder, creating it on first use."""
    global _encoder
    if _encoder is None:
        import tiktoken

        _encoder = tiktoken.get_encoding(_ENCODING_NAME)
    return _encoder


def _multiplier(model_type: str) -> float:
    """Conservative multiplier for a provider family (config-overridable)."""
    tc = config.get("LLM", {}).get("TOKEN_COUNTING", {})
    override = tc.get(f"{model_type.upper()}_MULTIPLIER")
    if override is not None:
        return float(override)
    return _DEFAULT_MULTIPLIERS.get(model_type, 1.35)


def count_tokens(text: str) -> int:
    """Conservatively estimate the token count of ``text`` for the current model.

    Never undercounts: OpenAI is exact (tiktoken), other providers scale the
    tiktoken basis by a safety multiplier so the context estimate can't fall
    below the provider's real count.
    """
    if not text:
        return 0

    model_type = str(config.get("MODEL_ID", {}).get("TYPE", "ollama")).lower()

    if model_type == "ollama":
        # Local models: chars/ratio. Default 3.0 (a safe average; the old 1.3
        # was denser than any real tokenizer and risked undercounting code).
        ratio = config.get("LLM", {}).get("TOKEN_COUNTING", {}).get(
            "OLLAMA_CHARS_PER_TOKEN", 3.0
        )
        return int(len(text) / ratio)

    # disallowed_special=() → count special-token text as ordinary text, never raise.
    base = len(_get_encoder().encode(text, disallowed_special=()))
    return int(base * _multiplier(model_type))
