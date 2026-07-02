"""Shared token-counting helper."""

from mnemoai.utils.config import config

# Tiktoken model whose BPE we borrow purely as a length estimator.
_ENCODER_MODEL = "gpt-4"

# Lazily-initialized tiktoken encoder (created on first non-Ollama count).
_encoder = None


def _get_encoder():
    """Return a cached tiktoken encoder, creating it on first use."""
    global _encoder
    if _encoder is None:
        import tiktoken

        _encoder = tiktoken.encoding_for_model(_ENCODER_MODEL)
    return _encoder


def count_tokens(text: str) -> int:
    """Estimate the token count of ``text`` for the configured model.

    Args:
        text: The text to measure.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0

    model_type = config.get("MODEL_ID", {}).get("TYPE", "ollama")

    if model_type == "ollama":
        # Ollama approximation: ~1.3 chars per token (configurable).
        multiplier = (
            config.get("LLM", {}).get("TOKEN_COUNTING", {}).get("OLLAMA_APPROXIMATION", 1.3)
        )
        return int(len(text) / multiplier)

    # disallowed_special=() → count special-token text as ordinary text, never raise.
    return len(_get_encoder().encode(text, disallowed_special=()))
