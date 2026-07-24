"""Repair orphaned tool-call/result pairs in a message history (pure).

Extracted from ``agent.py``: no agent state. ``LangGraphAgent`` keeps a thin
``_sanitize_tool_pairs`` delegator so its historical surface (used by the unit
tests and by ``AgentConversationManager`` via ``getattr``) is unchanged.
"""

from typing import List

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def _reasoning_block_fix(block: dict):
    """Repair (or drop) one provably-invalid reasoning content block; provider-
    agnostic. Returns the (possibly new) block, or ``None`` to drop it. Returns
    the SAME object when nothing needed changing.

    The three providers that put reasoning in the CONTENT LIST each have a
    distinct replay constraint, so "malformed" is defined per shape:

    - **Anthropic** ``{type: "thinking"}`` — the API requires the inner
      ``thinking`` field to be PRESENT (``messages.N.content.M.thinking.thinking:
      Field required``). With summarized/omitted thinking, langchain-anthropic's
      streaming accumulates a ``signature_delta`` into a ``thinking`` block that
      has the (load-bearing, server-validated) ``signature`` but LOSES the inner
      text, yielding ``{type:thinking, signature:…}`` with no ``thinking`` key.
      A thinking-enabled assistant turn carrying ``tool_use`` MUST also LEAD with
      its thinking block, so DROPPING it (promoting tool_use to first) trades one
      400 for another. So we **normalize, not drop**: re-inject ``thinking: ""``
      (what Anthropic originally sent), keeping the signature and block order.
      Only DROP a thinking block with NEITHER text NOR signature (an unsignable
      stub Anthropic never actually emits).
    - **Bedrock** ``{type: "reasoning_content"}`` — drop only when the inner text
      is empty AND there's no signature (Bedrock needs the signature; keep it).
    - **OpenAI Responses** ``{type: "reasoning"}`` — may legitimately carry no
      summary while holding ``id``/``encrypted_content`` needed for the reasoning
      chain; NEVER drop those. Drop only a bare ``{type:"reasoning"}`` stub.
    """
    btype = block.get("type")
    if btype == "thinking":
        has_text = bool(str(block.get("thinking", "")).strip())
        has_sig = bool(block.get("signature"))
        if has_text:
            return block  # healthy
        if has_sig:
            # Signature present but text lost in accumulation → restore the
            # empty inner field so the schema is satisfied and order preserved.
            if "thinking" in block and block["thinking"] == "":
                return block  # already normalized
            return {**block, "thinking": ""}
        return None  # no text, no signature → unsendable stub, drop
    if btype == "reasoning_content":  # Bedrock
        rc = block.get("reasoning_content")
        text = rc.get("text", "") if isinstance(rc, dict) else ""
        sig = (rc.get("signature") if isinstance(rc, dict) else None) or block.get(
            "signature"
        )
        if str(text).strip() or sig:
            return block
        return None
    if btype == "reasoning":  # OpenAI Responses
        summary = block.get("summary")
        has_summary = isinstance(summary, list) and any(
            str((s or {}).get("text", "")).strip()
            for s in summary
            if isinstance(s, dict)
        )
        if has_summary or block.get("id") or block.get("encrypted_content"):
            return block
        return None
    return block  # not a reasoning block


def strip_malformed_reasoning(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Repair provably-invalid reasoning blocks in assistant messages (pure).

    Provider-agnostic egress guard: an assistant turn re-fed to the model can
    carry a reasoning block that the provider rejects on replay (see
    :func:`_reasoning_block_fix` for the per-provider rules). Normalizes or drops
    only those blocks, leaving everything else — and any message whose content is
    a plain string or whose reasoning lives in ``additional_kwargs`` (Ollama,
    LiteLLM) — untouched. Returns a new list; a clean history passes through with
    the SAME message objects (no needless copies). Never mutates inputs.
    """
    out: List[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and isinstance(msg.content, list):
            fixed = _strip_malformed_thinking(msg.content)
            if fixed is not msg.content:
                msg = msg.model_copy(update={"content": fixed})
        out.append(msg)
    return out


def _strip_malformed_thinking(content: object) -> object:
    """Normalize/drop provably-invalid reasoning blocks in one message's content.

    Delegates each block to :func:`_reasoning_block_fix` (Anthropic ``thinking``,
    Bedrock ``reasoning_content``, OpenAI ``reasoning``). Non-list content passes
    through unchanged. Returns the SAME object when nothing changed (so callers
    can cheaply skip a ``model_copy``). Kept as the shared helper both
    :func:`sanitize_tool_pairs` and :func:`strip_malformed_reasoning` use, so the
    main path and the worker/egress path agree exactly.
    """
    if not isinstance(content, list):
        return content
    cleaned = []
    changed = False
    for block in content:
        if isinstance(block, dict) and block.get("type") in (
            "thinking",
            "reasoning_content",
            "reasoning",
        ):
            fixed = _reasoning_block_fix(block)
            if fixed is None:
                changed = True
                continue
            if fixed is not block:
                changed = True
                cleaned.append(fixed)
                continue
        cleaned.append(block)
    return cleaned if changed else content


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

    # Pre-pass: drop malformed (empty-inner-text) thinking blocks from assistant
    # messages so an Anthropic request isn't rejected wholesale (see
    # _strip_malformed_thinking). Only rebuilds a message whose content changed
    # (the helper returns the same object when nothing needed dropping).
    repaired: List[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            fixed = _strip_malformed_thinking(msg.content)
            if fixed is not msg.content:
                msg = msg.model_copy(update={"content": fixed})
        repaired.append(msg)
    messages = repaired

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
