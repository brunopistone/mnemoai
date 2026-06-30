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
