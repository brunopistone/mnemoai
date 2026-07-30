"""Regression: token counting must not crash on special-token text.

tiktoken's encode() raises on text containing a special token like
"<|endoftext|>" unless disallowed_special=() is passed. The token-COUNTING
helpers (used by fs_read's size check, RAG chunking, episodic memory, and
conversation compaction) treat such text as ordinary content, so they must pass
disallowed_special=() and never raise — otherwise reading a file that literally
contains "<|endoftext|>" crashes fs_read.
"""

import tiktoken

SPECIAL = "config STOP:\n  - <|endoftext|>\n  - <|im_start|>\n  - <|im_end|>"


def test_raw_tiktoken_default_raises_but_disallowed_empty_does_not():
    enc = tiktoken.encoding_for_model("gpt-4")
    # Sanity: the default DOES raise (this is the bug condition we guard against).
    import pytest

    with pytest.raises(ValueError):
        enc.encode(SPECIAL)
    # The fix: counted as ordinary text, no raise.
    assert len(enc.encode(SPECIAL, disallowed_special=())) > 0


def test_tool_manager_count_tokens_handles_special(monkeypatch):
    import mnemoai.server.tools.tools_manager as tm
    from mnemoai.server.tools.tools_manager import ToolManager

    monkeypatch.setattr(
        tm.config, "get",
        lambda k, d=None: {"TYPE": "openai"} if k == "MODEL_ID" else d,
    )
    mgr = ToolManager()
    n = mgr.count_tokens(SPECIAL)  # must NOT raise
    assert n > 0


def test_chunking_helper_count_tokens_handles_special():
    from mnemoai.server.tools.readers.chunking_helper import __count_tokens as ct

    assert ct(SPECIAL) > 0  # must not raise / not fall back silently


class TestChunkingUsesARealEncoding:
    """The chunker asked tiktoken for encoding ``"gpt-4"`` — a MODEL name, not an
    encoding name — so ``get_encoding`` raised on EVERY call and the bare
    ``except`` silently returned ``len(text)//4``. Chunk sizes were ~12% off and
    nothing ever reported a problem, which is why it survived: the fallback exists
    for genuinely unavailable tokenizers, and it masked a permanent failure.
    """

    def _count(self, text):
        # Fetched via getattr, not `from … import __count_tokens`: inside a class
        # body a leading-dunder name is mangled to `_ClassName__count_tokens`.
        import mnemoai.server.tools.readers.chunking_helper as ch

        return getattr(ch, "__count_tokens")(text)

    def test_the_configured_encoding_name_exists(self):
        from mnemoai.server.tools.readers import chunking_helper as ch

        tiktoken.get_encoding(ch._ENCODING_NAME)  # raised ValueError before

    def test_the_count_is_tiktokens_not_the_crude_fallback(self):
        from mnemoai.server.tools.readers import chunking_helper as ch

        text = "The quick brown fox jumps over the lazy dog. " * 20
        expected = len(
            tiktoken.get_encoding(ch._ENCODING_NAME).encode(
                text, disallowed_special=()
            )
        )
        assert self._count(text) == expected
        assert self._count(text) != len(text) // 4  # the old silent answer

    def test_it_shares_one_encoding_with_the_rest_of_the_app(self):
        # Two encodings would make chunk sizes disagree with the context budget
        # they're measured against.
        from mnemoai.server.tools.readers import chunking_helper as ch
        from mnemoai.utils import tokenization

        assert ch._ENCODING_NAME == tokenization._ENCODING_NAME

    def test_empty_text_is_zero(self):
        assert self._count("") == 0


def test_episodic_memory_count_and_truncate_handle_special(monkeypatch):
    import mnemoai.client.memory.episodic_memory as em
    from mnemoai.client.memory.episodic_memory import EpisodicMemoryManager

    a = EpisodicMemoryManager.__new__(EpisodicMemoryManager)
    a.encoder = tiktoken.encoding_for_model("gpt-4")
    # Force the tiktoken branch (non-ollama).
    monkeypatch.setattr(
        em.config, "get",
        lambda k, d=None: {"TYPE": "openai"} if k == "MODEL_ID" else d,
    )
    assert a.count_tokens(SPECIAL) > 0  # count must not raise
    out = a._truncate_to_tokens(SPECIAL, max_tokens=5)  # truncate must not raise
    assert isinstance(out, str)


def test_conversation_manager_count_tokens_handles_special(monkeypatch):
    import mnemoai.client.managers.agent_conversation_manager as acm
    from mnemoai.client.managers.agent_conversation_manager import (
        AgentConversationManager,
    )

    monkeypatch.setattr(
        acm.config, "get",
        lambda k, d=None: {"TYPE": "openai"} if k == "MODEL_ID" else d,
    )
    mgr = AgentConversationManager(max_tokens=1000)
    mgr.encoder = tiktoken.encoding_for_model("gpt-4")
    # count_tokens takes a list of message dicts; put the special token inside.
    assert mgr.count_tokens([{"role": "user", "content": SPECIAL}]) > 0
