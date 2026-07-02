"""Repair orphaned tool-call/result pairs in a message history (pure).

Extracted from ``agent.py``: no agent state. ``LangGraphAgent`` keeps a thin
``_sanitize_tool_pairs`` delegator so its historical surface (used by the unit
tests and by ``AgentConversationManager`` via ``getattr``) is unchanged.
"""

from typing import List

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def sanitize_tool_pairs(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Drop orphaned tool calls/results so strict providers don't 400.

    Every assistant ``tool_call`` needs a following ``ToolMessage`` with the same
    id and vice versa; an orphan (from a cut-short turn or a compaction slice)
    makes providers like the OpenAI Responses API reject the request. Keeps only
    calls whose id has a result and drops results with no surviving call; an
    assistant message left with no calls and no text is dropped. Returns a new
    list (inputs not mutated); a clean history passes through unchanged.
    """
    result_ids = {
        m.tool_call_id
        for m in messages
        if isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None)
    }

    # First pass: fix assistant messages, tracking which call ids survive.
    kept_call_ids: set = set()
    intermediate: List[BaseMessage] = []
    for msg in messages:
        calls = getattr(msg, "tool_calls", None)
        if isinstance(msg, AIMessage) and calls:
            good = [c for c in calls if c.get("id") in result_ids]
            if len(good) == len(calls):
                kept_call_ids.update(c.get("id") for c in good)
                intermediate.append(msg)
                continue
            if good:
                kept_call_ids.update(c.get("id") for c in good)
                intermediate.append(msg.model_copy(update={"tool_calls": good}))
                continue
            # No surviving calls: keep the turn only if it has visible text.
            if str(msg.content).strip():
                intermediate.append(msg.model_copy(update={"tool_calls": []}))
            # else drop the message entirely (it was only orphaned calls)
            continue
        intermediate.append(msg)

    # Second pass: drop tool results whose originating call didn't survive.
    cleaned: List[BaseMessage] = [
        m
        for m in intermediate
        if not (
            isinstance(m, ToolMessage)
            and getattr(m, "tool_call_id", None) not in kept_call_ids
        )
    ]
    return cleaned
