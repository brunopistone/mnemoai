"""DOC_MAX_TOKENS is derived as 25% of MAX_CONVERSATION_TOKENS.

The configurator (/config, /model, /params) keeps the document read cap at a
quarter of the context window rather than letting it be hand-tuned. This tests
the pure text transform used by all three flows.
"""

from mnemoai.utils.configurator import _get_top_level, _sync_doc_max_tokens


def test_overwrites_existing_to_quarter():
    text = (
        "MODEL_ID:\n  NAME: x\n"
        "MAX_CONVERSATION_TOKENS: 65536\n"
        "DOC_MAX_TOKENS: 999\n"
    )
    out = _sync_doc_max_tokens(text)
    assert _get_top_level(out, "DOC_MAX_TOKENS") == "16384"  # 65536 // 4


def test_appends_when_missing():
    out = _sync_doc_max_tokens("MAX_CONVERSATION_TOKENS: 200000\n")
    assert _get_top_level(out, "DOC_MAX_TOKENS") == "50000"  # 200000 // 4


def test_noop_when_context_window_missing():
    text = "FOO: bar\n"
    assert _sync_doc_max_tokens(text) == text


def test_noop_when_context_window_unparseable():
    text = "MAX_CONVERSATION_TOKENS: notanumber\nDOC_MAX_TOKENS: 5\n"
    assert _sync_doc_max_tokens(text) == text


def test_small_context_window_floors_at_one():
    out = _sync_doc_max_tokens("MAX_CONVERSATION_TOKENS: 2\n")
    assert _get_top_level(out, "DOC_MAX_TOKENS") == "1"  # max(1, 2 // 4)
