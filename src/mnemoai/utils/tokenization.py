"""Shared token-counting helper.

Lives in ``utils`` so both the client and the MCP server import it from a neutral
place — previously the client reached into ``server.tools`` for ``count_tokens``,
coupling the two layers across the MCP boundary.

Counting is a length *estimate*, not an encoding for the model:
- Ollama (and other char-based providers): character-count ÷ approximation factor.
- Everything else (OpenAI/Bedrock/SageMaker/…): tiktoken's ``gpt-4`` encoder with
  ``disallowed_special=()`` so literal special-token text (e.g. a file containing
  ``<|endoftext|>``) is counted as ordinary text rather than raising.

The encoder is created lazily on first use so importing this module has no
side effects and stays config-independent (keeps it unit-testable).
"""

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
