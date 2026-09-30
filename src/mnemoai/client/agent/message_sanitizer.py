"""Repair orphaned tool-call/result pairs in a message history (pure).

Extracted from ``agent.py``: no agent state. ``LangGraphAgent`` keeps a thin
``_sanitize_tool_pairs`` delegator so its historical surface (used by the unit
tests and by ``AgentConversationManager`` via ``getattr``) is unchanged.
"""

from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


def flatten_tool_blocks(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Render tool calls/results as plain TEXT for an auxiliary, tool-less call.

    The router/decomposer/summarizer call the model WITHOUT binding tools, so a
    replayable ``tool_use``/``tool_result`` block in the history has no matching
    tool schema. Providers reject or mangle that: Bedrock Converse logs
    ``Tool messages (toolUse/toolResult) detected without toolConfig. Converting
    tool blocks to text format…`` + a ``RuntimeWarning`` (leaking into the TUI),
    and a strict provider can 400 outright. These auxiliary calls only need to
    KNOW what ran, not to replay it — so each tool call/result becomes a short
    text line on a plain message and no tool block survives.

    Returns a new list (never mutates the input); non-tool messages pass through
    as the same object.
    """
    out: List[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None) or "tool"
            out.append(
                HumanMessage(content=f"[tool result from {name}]: {_as_text(msg.content)}")
            )
            continue
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            text = _as_text(msg.content)
            calls = "; ".join(
                f"{tc.get('name')}({tc.get('args', {})})" for tc in msg.tool_calls
            )
            joined = f"{text}\n[called tools: {calls}]".strip() if text else (
                f"[called tools: {calls}]"
            )
            out.append(AIMessage(content=joined))
            continue
        # A tool_use/tool_result block can also ride inside block-list content
        # (Bedrock/Anthropic shapes) on a message with no `tool_calls` attribute.
        if isinstance(msg.content, list) and any(
            isinstance(b, dict)
            and b.get("type") in ("tool_use", "tool_result", "toolUse", "toolResult")
            for b in msg.content
        ):
            out.append(msg.model_copy(update={"content": _as_text(msg.content)}))
            continue
        out.append(msg)
    return out


def _as_text(content: object) -> str:
    """Best-effort plain text from string or block-list message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text" or "text" in b:
                    parts.append(str(b.get("text", "")))
            elif isinstance(b, str):
                parts.append(b)
        return "".join(parts)
    return "" if content is None else str(content)


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


def _content_call_id(block: object):
    """Client-executed call id; None means this is not a local tool call."""
    if not isinstance(block, dict):
        return None
    if block.get("type") in ("tool_use", "tool_call"):
        return block.get("id") or ""
    if block.get("type") in ("function_call", "custom_tool_call"):
        # Responses item ids (fc_...) are NOT the tool's call_id.
        return block.get("call_id") or ""
    if isinstance(block.get("toolUse"), dict):
        return block["toolUse"].get("toolUseId") or ""
    return None  # leave provider-executed tools and signed reasoning alone


def _repair_ai_tool_pairs(msg: AIMessage, result_ids: set):
    """Reconcile every replayable representation, not only .tool_calls."""
    kept_ids = set()
    removed = False

    def keep(call_id):
        nonlocal removed
        if isinstance(call_id, str) and call_id and call_id in result_ids:
            kept_ids.add(call_id)
            return True
        removed = True
        return False

    updates = {}
    content = _strip_malformed_thinking(msg.content)
    if isinstance(content, list):
        filtered = [
            block for block in content
            if (call_id := _content_call_id(block)) is None or keep(call_id)
        ]
        if len(filtered) != len(content):
            content = filtered
    if content is not msg.content:
        updates["content"] = content

    # Chunks/raw calls can reconstruct .tool_calls on revalidation or in an
    # adapter; invalid_tool_calls are also serialized by OpenAI.
    for field in ("tool_calls", "invalid_tool_calls", "tool_call_chunks"):
        calls = getattr(msg, field, None)
        if calls:
            good = [call for call in calls if keep(call.get("id"))]
            if len(good) != len(calls):
                updates[field] = good
    raw_calls = msg.additional_kwargs.get("tool_calls")
    if isinstance(raw_calls, list) and raw_calls:
        good = [call for call in raw_calls if keep(call.get("id"))]
        if len(good) != len(raw_calls):
            kwargs = dict(msg.additional_kwargs)
            if good:
                kwargs["tool_calls"] = good
            else:
                kwargs.pop("tool_calls")
            updates["additional_kwargs"] = kwargs

    has_content = bool(content.strip()) if isinstance(content, str) else bool(content)
    if removed and not kept_ids and not has_content:
        return None, kept_ids
    return msg.model_copy(update=updates) if updates else msg, kept_ids


def sanitize_tool_pairs(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Drop orphaned tool calls/results so strict providers don't 400.

    Every assistant ``tool_call`` needs a following ``ToolMessage`` with the same
    id and vice versa; an orphan (from a cut-short turn or a compaction slice)
    makes providers like the OpenAI Responses API reject the request. Keeps only
    calls whose id has a result, including native content blocks, raw calls and
    accumulated stream chunks. Drops results with no surviving call; an assistant
    message left with no calls and no content is dropped. Returns a new list
    (inputs not mutated); a clean history passes through unchanged. This repairs
    the request, not the audit record, and never executes or retries tools.
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
        if isinstance(msg, AIMessage):
            msg, kept = _repair_ai_tool_pairs(msg, result_ids)
            kept_call_ids.update(kept)
            if msg is None:
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
