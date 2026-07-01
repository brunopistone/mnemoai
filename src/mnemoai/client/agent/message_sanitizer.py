"""Repair orphaned tool-call/result pairs in a message history (pure).

Extracted from ``agent.py``: no agent state. ``LangGraphAgent`` keeps a thin
``_sanitize_tool_pairs`` delegator so its historical surface (used by the unit
tests and by ``AgentConversationManager`` via ``getattr``) is unchanged.
"""

from typing import List

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def sanitize_tool_pairs(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Drop orphaned tool calls/results so the provider doesn't 400.

    A tool exchange must stay paired: every assistant ``tool_call`` needs a
    following ``ToolMessage`` with the same id, and every ``ToolMessage`` needs a
    preceding assistant call with that id. An orphan on either side makes strict
    providers (e.g. the OpenAI Responses API) reject the whole request:

    * orphaned **result** → "No tool call found for function call output …"
    * orphaned **call**   → "No tool output found for function call …"

    Orphans arise when a turn is cut short (recursion limit, stream error,
    interrupt) or when history is sliced (compaction) in a way the boundary guard
    didn't catch — and once one is persisted in ``agent.messages`` it breaks
    *every* subsequent turn. This is a defensive, id-based repair:

    * Collect the ids of all tool results present.
    * From each assistant message, keep only tool_calls whose id has a result; if
      that empties the calls and the message has no visible text, drop the message
      entirely (it carried nothing but orphaned calls).
    * Drop any tool result whose id has no surviving originating call.

    Returns a new list; inputs are not mutated. A clean history passes through
    unchanged.
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
