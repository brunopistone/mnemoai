"""Unit tests for the tool-pair sanitizer (LangGraphAgent._sanitize_tool_pairs).

A tool exchange must stay paired: every assistant tool_call needs a matching
ToolMessage and vice-versa. An orphan on either side makes strict providers
(OpenAI Responses API) reject the whole request:
  - orphaned result -> "No tool call found for function call output …"
  - orphaned call   -> "No tool output found for function call …"
Once an orphan is persisted in agent.messages it breaks every subsequent turn,
so the sanitizer repairs the list before each model call (and in compaction).
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mnemoai.client.agent.agent import LangGraphAgent

S = LangGraphAgent._sanitize_tool_pairs


def _call(*ids, text=""):
    m = AIMessage(content=text)
    m.tool_calls = [
        {"name": "x", "args": {}, "id": i, "type": "tool_call"} for i in ids
    ]
    return m


def _result(call_id, content="ok"):
    return ToolMessage(content=content, tool_call_id=call_id, name="x")


def _types(msgs):
    return [type(m).__name__ for m in msgs]


def test_valid_pair_is_unchanged():
    msgs = [_call("c0"), _result("c0"), AIMessage(content="done")]
    out = S(msgs)
    assert _types(out) == ["AIMessage", "ToolMessage", "AIMessage"]
    assert out[0].tool_calls[0]["id"] == "c0"


def test_orphaned_call_with_no_text_is_dropped():
    msgs = [HumanMessage(content="hi"), _call("c0")]
    out = S(msgs)
    assert _types(out) == ["HumanMessage"]


def test_orphaned_call_with_text_keeps_text_strips_call():
    msgs = [_call("c0", text="partial answer")]
    out = S(msgs)
    assert len(out) == 1
    assert out[0].content == "partial answer"
    assert out[0].tool_calls == []


def test_orphaned_result_is_dropped():
    msgs = [HumanMessage(content="hi"), _result("c9"), AIMessage(content="x")]
    out = S(msgs)
    assert _types(out) == ["HumanMessage", "AIMessage"]


def test_partial_orphan_keeps_only_matched_call():
    # One assistant turn with two calls; only one has a result.
    msgs = [_call("a", "b"), _result("a")]
    out = S(msgs)
    assert len(out) == 2
    assert [c["id"] for c in out[0].tool_calls] == ["a"]
    assert isinstance(out[1], ToolMessage)


def test_clean_conversation_passes_through():
    msgs = [
        SystemMessage(content="sys"),
        HumanMessage(content="q"),
        AIMessage(content="a"),
    ]
    out = S(msgs)
    assert _types(out) == ["SystemMessage", "HumanMessage", "AIMessage"]


def test_does_not_mutate_input():
    original = _call("a", "b")
    msgs = [original, _result("a")]
    S(msgs)
    # Original message still has BOTH calls (sanitizer returns copies).
    assert [c["id"] for c in original.tool_calls] == ["a", "b"]


def test_multiple_orphans_mixed_with_valid():
    msgs = [
        HumanMessage(content="q"),
        _call("good"),
        _result("good"),
        _call("orphan_call"),          # no result -> dropped
        _result("orphan_result"),       # no call -> dropped
        AIMessage(content="final"),
    ]
    out = S(msgs)
    # good pair + the two bracketing messages survive; both orphans gone.
    assert _types(out) == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert out[1].tool_calls[0]["id"] == "good"
