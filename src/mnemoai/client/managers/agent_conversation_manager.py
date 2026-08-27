"""Conversation manager that uses simple text-based summaries."""

import asyncio
import json
import re
import textwrap
from datetime import date
from typing import Any, Dict, List

from mnemoai.client.agent import stream_policy
from mnemoai.client.agent.subagents import available_subagents_block
from mnemoai.client.memory.skill_store import (
    SkillStore,
    format_available_skills,
)
from mnemoai.utils.config import config
from mnemoai.utils.logger import logger
from mnemoai.utils.tokenization import count_tokens as _count_text_tokens

# Try to import LangChain message types for compatibility
try:
    from langchain_core.messages import AIMessage, BaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# ANSI color codes for green text
GREEN = "\033[92m"
RESET = "\033[0m"


def log_green(message: str, level: str = "info") -> None:
    """Print a user-facing status line in green (e.g. compaction progress).

    These are results/progress the user asked for, not diagnostics, so they go
    to stdout via ``print()`` — clean, no timestamp/level prefix, and always
    visible regardless of ``LOG_LEVEL``. ``level`` is accepted for backward
    compatibility but ignored. Operational diagnostics should use ``logger``.

    Args:
        message: Message to show the user
        level: Ignored (kept for backward compatibility)
    """
    print(f"{GREEN}{message}{RESET}")


def messages_to_dict_list(messages: List[Any]) -> List[Dict]:
    """Convert messages to list of dictionaries.

    Handles both Strands format (dict) and LangChain format (BaseMessage).

    Args:
        messages: List of messages in either format

    Returns:
        List of message dictionaries
    """
    result = []
    for msg in messages:
        if isinstance(msg, dict):
            result.append(msg)
        elif LANGCHAIN_AVAILABLE and isinstance(msg, BaseMessage):
            # Map LangChain message type -> role, preserving tool interactions.
            msg_type = getattr(msg, "type", None)
            entry: Dict[str, Any] = {"content": [{"text": str(msg.content)}]}

            if msg_type == "ai":
                entry["role"] = "assistant"
                # Preserve tool calls so the summary knows what was invoked.
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    entry["tool_calls"] = [
                        {"name": tc.get("name"), "args": tc.get("args", {})}
                        for tc in tool_calls
                    ]
            elif msg_type == "system":
                entry["role"] = "system"
            elif msg_type == "tool":
                # Tool result: keep its own role + the tool name for context.
                entry["role"] = "tool"
                entry["tool_name"] = getattr(msg, "name", None)
            else:
                entry["role"] = "user"

            result.append(entry)
        else:
            # Fallback: try to convert to string
            result.append({
                "role": "user",
                "content": [{"text": str(msg)}]
            })
    return result


# Fraction of the context window one summarization CALL may consume (batch of
# older messages + the rolling summary + the summary prompt + the model's own
# reasoning/output).
_SUMMARY_CALL_FRACTION = 0.15
# The rolling summary carried between batches is capped so late batches (prior
# summary + next batch) can't themselves overflow. In chars (~4 chars/token).
_ROLLING_SUMMARY_MAX_CHARS_FRACTION = 0.10

# Tool-result eviction (the cheapest compaction layer, runs before any LLM
# summary): how many trailing messages to keep verbatim, and the char cap an
# evicted (old) tool-result body is shrunk to. Both config-overridable via
# LLM.TOOL_EVICTION_KEEP_RECENT / LLM.EVICTED_TOOL_RESULT_CHARS.
_TOOL_EVICTION_KEEP_RECENT = 8
_EVICTED_TOOL_RESULT_CHARS = 500
_EVICTION_MARKER = (
    "… [earlier tool output evicted to save context; "
    "re-run the tool if you need the full result] …"
)


class AgentConversationManager:
    def __init__(self, max_tokens: int = 5000) -> None:
        """Initialize conversation manager.

        Args:
            max_tokens: Maximum tokens before summarization
        """
        self.max_tokens = max_tokens
        self.previous_summary = None
        # The latest summary as PLAIN text (``previous_summary`` is the wrapped
        # block). Persisted by /save and the session transcript, so a restore can
        # rebuild the compacted context instead of re-summarizing the raw history.
        self.summary_text = ""
        logger.info(f"Initialized conversation manager with max_tokens={max_tokens}")

    def count_tokens(self, messages: List[Dict]) -> int:
        """Estimate tokens for a list of messages via the shared, provider-aware,
        never-undercount counter (``utils.tokenization``). Adds a small per-message
        overhead for role/formatting wrappers the raw JSON dump misses."""
        text = json.dumps(messages, default=str)
        # ~4 tokens/message for the role + structural wrappers each message costs.
        overhead = 4 * len(messages) if isinstance(messages, list) else 0
        return _count_text_tokens(text) + overhead

    @staticmethod
    def _message_text_for_summary(msg: Dict) -> str:
        """Render one message dict to text for the summary input.

        Includes tool calls (assistant) and tool results so tool interactions
        are captured, not just user/assistant prose.
        """
        parts = []
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
        elif isinstance(content, str):
            parts.append(content)

        text = "".join(parts).strip()
        role = msg.get("role")

        if role == "tool":
            name = msg.get("tool_name") or "tool"
            return f"[tool result from {name}]: {text}" if text else ""

        if role == "assistant" and msg.get("tool_calls"):
            calls = "; ".join(
                f"{tc.get('name')}({tc.get('args', {})})" for tc in msg["tool_calls"]
            )
            tool_line = f"[called tools: {calls}]"
            return f"{text}\n{tool_line}".strip() if text else tool_line

        return text

    # The compaction prompts live in prompts.yaml (SUMMARY_SYSTEM_PROMPT /
    # SUMMARY_TASK_PROMPT) and are mandatory — read via require_prompt (no
    # in-code fallback; a missing one raises PromptError).
    @property
    def _SUMMARY_SYSTEM_PROMPT(self) -> str:
        return config.require_prompt("SUMMARY_SYSTEM_PROMPT")

    @property
    def _SUMMARY_TASK_PROMPT(self) -> str:
        return config.require_prompt("SUMMARY_TASK_PROMPT")

    @staticmethod
    def _strip_analysis(text: str) -> str:
        """Remove the model's <analysis>…</analysis> scratchpad from the output.

        The task prompt asks the model to think in <analysis> tags before the
        structured summary; we keep only the summary. If no tags are present
        (or only an opening tag), return the text unchanged/after the tag.
        """
        # Drop a complete <analysis>...</analysis> block.
        cleaned = re.sub(
            r"<analysis>.*?</analysis>\s*", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        # If only a closing tag remains (unbalanced), keep what follows it.
        if "</analysis>" in cleaned.lower():
            idx = cleaned.lower().rfind("</analysis>")
            cleaned = cleaned[idx + len("</analysis>"):]
        return cleaned

    def _build_summary_prompt(self, focus_instructions: str = "") -> str:
        """Assemble the task prompt, appending any per-call focus instructions.

        The base template tells the model to honor "additional summarization
        instructions provided in the included context", under a
        ``## Compact Instructions`` header. We append the user's focus there.
        """
        prompt = self._SUMMARY_TASK_PROMPT
        if focus_instructions:
            prompt += (
                "\n\n## Compact Instructions\n"
                f"{focus_instructions.strip()}"
            )
        return prompt

    async def generate_summary(
        self, messages: List[Dict], model: Any, focus_instructions: str = ""
    ) -> str:
        """Summarize older messages, batching so the summary CALL never itself
        overflows the model's context window.

        A single-shot summary of very long history 400s ("prompt is too long"),
        which used to silently degrade to a content-free placeholder — losing all
        that history. So we split the messages into batches that each fit a safe
        fraction of the window. **Parallel map + single reduce:** each batch is
        summarized INDEPENDENTLY and CONCURRENTLY (map), then — only if there was
        more than one batch — a single call folds the partial summaries (plus any
        previous compaction's summary) into one coherent whole (reduce). This is
        far faster than the old sequential rolling fold, whose wall-clock was the
        SUM of all batch calls; here it is ~max(one batch) + one reduce.

        Args:
            messages: List of conversation messages
            model: Model instance for generating summary (LangChain or Strands)
            focus_instructions: Optional user guidance on what to emphasize.

        Returns:
            Summary text (never empty; falls back to a bounded excerpt on error).
        """
        # Budget per summary call: a small fraction of the window so the CALL
        # (batch + prompt + reasoning output) fits with wide margin. Relative to
        # max_tokens (never an absolute floor that could exceed a small window).
        budget = max(256, int(self.max_tokens * _SUMMARY_CALL_FRACTION))
        batches = self._batch_messages(messages, budget)

        # MAP: summarize every batch independently and concurrently, bounded by a
        # semaphore (reuse the sub-agent concurrency cap) so we don't hammer the
        # provider. Each batch stands alone (no prior_summary), so order doesn't
        # matter and the calls parallelize.
        max_concurrency = max(
            1, int(config.get("LLM", {}).get("SUBAGENT_MAX_CONCURRENCY", 4))
        )
        sem = asyncio.Semaphore(max_concurrency)

        async def _map_one(idx: int, batch: List[Dict]):
            async with sem:
                try:
                    return idx, await self._with_transient_retry(
                        lambda: self._summarize_batch(
                            batch, model, focus_instructions, prior_summary=None
                        ),
                        f"Summary batch {idx + 1}/{len(batches)}",
                    )
                except Exception as e:
                    logger.warning(
                        "Summary batch %d/%d failed (%s)", idx + 1, len(batches), e
                    )
                    return idx, None

        mapped = await asyncio.gather(
            *(_map_one(i, b) for i, b in enumerate(batches))
        )
        # Keep partials in original batch order (chronological coherence).
        partials = [s for _, s in sorted(mapped, key=lambda t: t[0]) if s]

        if not partials:
            # Every batch failed: keep a bounded excerpt of the raw history rather
            # than a content-free placeholder, so nothing is silently lost.
            log_green("Failed to generate model summary for every batch", "error")
            return self._excerpt_fallback(messages, budget)

        # A single batch (the common case after tool-result eviction) needs no
        # reduce — its map result IS the summary. Only fold when >1 partial, or
        # when a previous compaction's summary must be carried forward.
        if len(partials) == 1 and not self.previous_summary:
            return self._strip_analysis(partials[0]).strip()

        # REDUCE: one call merges the ordered partials (+ any prior summary) into
        # a single coherent summary. If the reduce itself fails, fall back to the
        # concatenated partials so no content is lost.
        try:
            reduced = await self._with_transient_retry(
                lambda: self._reduce_summaries(partials, model, focus_instructions),
                "Summary reduce",
            )
            if reduced:
                return self._strip_analysis(reduced).strip()
        except Exception as e:
            logger.warning("Summary reduce step failed (%s); using joined partials", e)

        joined = "\n\n".join(self._strip_analysis(p).strip() for p in partials)
        return joined or self._excerpt_fallback(messages, budget)

    async def _with_transient_retry(self, call, label: str):
        """Await ``call()``, retrying a transient provider failure (529/overloaded).

        The map fans every batch out at once, so an overloaded provider rejects
        them together — and each batch's fallback drops that slice of history from
        the summary for good. Retrying is what keeps a compaction lossless under
        load; the caller's fallback still runs once the attempts are spent.
        """
        llm = config.get("LLM", {})
        return await stream_policy.acall_with_transient_retry(
            call,
            attempts=stream_policy.aux_attempts(llm.get("MAX_RETRIES", 2)),
            base=float(llm.get("RETRY_DELAY", 1.0)),
            factor=float(llm.get("RETRY_BACKOFF", 2.0)),
            on_retry=lambda e, delay, n, total: logger.debug(
                stream_policy.retry_notice(label, e, delay, n, total)
            ),
        )

    async def _reduce_summaries(
        self, partials: List[str], model: Any, focus_instructions: str
    ) -> str:
        """Fold ordered per-batch partial summaries (and any previous compaction
        summary) into one coherent summary via a single model call.

        The partials are fed as prior-summary context to one more summarization
        pass, so the reduce reuses the same summary prompt/system framing as the
        map — no second template to maintain.
        """
        # Cap each partial so the concatenated reduce input can't itself overflow.
        cap = max(1000, int(self.max_tokens * _ROLLING_SUMMARY_MAX_CHARS_FRACTION * 4))
        pieces = []
        if self.previous_summary:
            pieces.append(f"[Earlier session summary]\n{self.previous_summary[:cap]}")
        for i, p in enumerate(partials, 1):
            pieces.append(f"[Summary of conversation part {i}]\n{p[:cap]}")
        merged_context = "\n\n".join(pieces)

        # Present the partials as the "messages" to summarize; the summary prompt
        # then produces one consolidated summary over them.
        reduce_input = [{"role": "user", "content": [{"text": merged_context}]}]
        return await self._summarize_batch(
            reduce_input, model, focus_instructions, prior_summary=None
        )

    def _batch_messages(self, messages: List[Dict], budget: int) -> List[List[Dict]]:
        """Split messages into ordered batches each under ``budget`` tokens.

        A single message larger than the budget becomes its own batch (the
        per-call summarizer truncates it); this guarantees progress instead of an
        infinite/oversized batch."""
        batches: List[List[Dict]] = []
        current: List[Dict] = []
        used = 0
        for msg in messages:
            mt = self.count_tokens([msg])
            if current and used + mt > budget:
                batches.append(current)
                current, used = [], 0
            current.append(msg)
            used += mt
        if current:
            batches.append(current)
        return batches or [messages]

    def _excerpt_fallback(self, messages: List[Dict], budget: int) -> str:
        """Bounded plain-text excerpt of history when summarization fails outright
        — lossy, but never a content-free placeholder."""
        parts = []
        used = 0
        for msg in messages:
            text = self._message_text_for_summary(msg)
            if not text:
                continue
            t = self.count_tokens([{"content": text}])
            if used + t > budget:
                break
            parts.append(text)
            used += t
        joined = "\n".join(parts).strip()
        if not joined:
            return "Previous conversation covered multiple topics and requests."
        return "Earlier conversation (excerpt, summarization unavailable):\n" + joined

    def _truncate_msg_text(self, text: str) -> str:
        """Cap one message's text so a single oversized message can't overflow its
        own summary batch (head+tail kept with an elision note). ~4 chars/token."""
        cap = max(2000, int(self.max_tokens * _SUMMARY_CALL_FRACTION * 4))
        if len(text) <= cap:
            return text
        head = text[: cap // 2]
        tail = text[-cap // 2:]
        return f"{head}\n…[{len(text) - cap} chars elided]…\n{tail}"

    def _summary_texts(self, messages: List[Dict]):
        """Yield ``(role, text)`` for each summarizable message.

        Shared by both :meth:`_summarize_batch` branches: extracts the text,
        truncates it (head+tail so one giant message can't overflow), and skips
        empties. ``_truncate_msg_text`` returns "" unchanged and non-empty input
        unchanged, so pre- vs post-truncate empty checks select the same messages.
        """
        for msg in messages:
            text = self._truncate_msg_text(self._message_text_for_summary(msg))
            if text:
                yield msg.get("role", "user"), text

    async def _summarize_batch(
        self,
        messages: List[Dict],
        model: Any,
        focus_instructions: str,
        prior_summary: str = None,
    ) -> str:
        """Summarize ONE batch of messages (folding in ``prior_summary``).

        This is the single-call summarization; :meth:`generate_summary` chains it
        across batches so no individual call exceeds the context window. A single
        message larger than the per-call budget is truncated (head+tail) so it
        can't overflow on its own."""
        summary_prompt = self._build_summary_prompt(focus_instructions)
        summary_response = ""

        if LANGCHAIN_AVAILABLE and hasattr(model, "ainvoke"):
            from langchain_core.messages import HumanMessage, SystemMessage

            lc_messages = [SystemMessage(content=self._SUMMARY_SYSTEM_PROMPT)]
            if prior_summary:
                lc_messages.append(SystemMessage(content=prior_summary))

            for role, content in self._summary_texts(messages):
                if role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))

            lc_messages.append(HumanMessage(content=summary_prompt))
            response = await model.ainvoke(lc_messages)
            summary_response = str(response.content)
        else:
            batch = [
                {"role": role, "content": [{"text": text}]}
                for role, text in self._summary_texts(messages)
            ]
            batch.append({"role": "user", "content": [{"text": summary_prompt}]})
            think_param = config.get("LLM", {}).get("SUMMARIZATION_THINK", False)
            system_prompt = self._SUMMARY_SYSTEM_PROMPT
            if prior_summary:
                system_prompt = f"{system_prompt}\n\n{prior_summary}"

            async for event in model.stream(
                batch, system_prompt=system_prompt, think=think_param
            ):
                if (
                    "contentBlockDelta" in event
                    and "delta" in event["contentBlockDelta"]
                    and "text" in event["contentBlockDelta"]["delta"]
                ):
                    summary_response += event["contentBlockDelta"]["delta"]["text"]

        return self._strip_analysis(summary_response).strip()

    def _build_system_with_summary(
        self, clean_summary: str, client: Any = None
    ) -> str:
        """Embed a conversation summary into the configured system prompt.

        The block carries a continuation instruction so the model resumes 
        the work seamlessly instead of re-acknowledging the summary or recapping.

        Args:
            clean_summary: The summary text to embed.
            client: The ``LangGraphClient``, needed to rebuild the session-start
                context blocks (profile / MEMORY.md / playbook). When omitted
                only the client-independent blocks (skills, sub-agents) are
                restored.
        """
        summary_block = textwrap.dedent(
            f"""
            <conversation_summary>
            This summary replaces older messages that were compacted to save
            context. Treat it as the established history of this session.

            {clean_summary}

            Continue the conversation from where it left off without asking the
            user any further questions. Resume directly — do not acknowledge the
            summary, do not recap what was happening, do not preface with "I'll
            continue" or similar. Pick up the last task as if the break never
            happened.
            </conversation_summary>
            """
        ).strip()
        self.previous_summary = summary_block
        self.summary_text = clean_summary

        original_system_prompt = config.system_prompt
        if not original_system_prompt:
            return summary_block

        current_date = date.today().strftime("%Y-%m-%d")
        original_system_prompt = original_system_prompt.format(
            current_date=current_date
        )
        # Re-inject every session-start block: this rebuild re-fetches the base
        # prompt fresh, dropping ALL session-start injections, so without this
        # the profile / MEMORY.md / <available_skills> / <available_subagents> /
        # playbook blocks would silently vanish after the first compaction and
        # the model would lose its learned context for the rest of the session.
        parts = [original_system_prompt]
        parts.extend(self._session_blocks(client))
        parts.append(summary_block)
        return "\n\n".join(parts)

    def apply_restored_summary(self, client: Any, agent: Any, summary: str) -> bool:
        """Re-apply a restored conversation's compaction summary; True if applied.

        The counterpart of the checkpoint written by ``_compact``: a resume, a
        ``/load`` or a ``/branch`` restores the compacted message window, and that
        window only makes sense alongside the summary of what it followed. Rebuilds
        the system prompt through the same path a live compaction uses, so the
        session-start blocks are restored with it and ``previous_summary`` is set —
        without which the NEXT compaction would silently drop this one's history
        from its reduce step.
        """
        if not summary or client is None:
            return False
        try:
            rebuilt = self._build_system_with_summary(str(summary), client=client)
            client.system_prompt = rebuilt
            if agent is not None:
                agent.system_prompt = rebuilt
            return True
        except Exception as e:  # noqa: BLE001 — a restore already succeeded
            logger.warning(f"Could not re-apply the restored summary: {e}")
            return False

    def _session_blocks(self, client: Any = None) -> List[str]:
        """The session-start context blocks to restore after compaction.

        Delegates to the single source in ``context_injection`` so the blocks
        can't drift between this rebuild and the client's session-start path.
        Falls back to the client-independent blocks when no client is available.
        """
        if client is not None:
            from mnemoai.client.context_injection import build_session_blocks

            try:
                return build_session_blocks(client, include_playbook=True)
            except Exception as e:
                # Never let a memory/profile read failure abort a compaction:
                # losing the blocks degrades context, losing the compaction
                # overflows it.
                logger.warning(f"Session-block rebuild failed on compaction: {e}")

        return [
            block for block in (self._skills_block(), self._subagents_block()) if block
        ]

    def _skills_block(self) -> str:
        """Build the tier-1 ``<available_skills>`` block (or "" when disabled/empty).

        Mirrors the client's session-start injection so skills survive compaction.
        """
        if not config.get("ENABLE_SKILLS", True):
            return ""
        return format_available_skills(SkillStore().list_skills())

    def _subagents_block(self) -> str:
        """Build the ``<available_subagents>`` block, so the spawn_agent types
        survive compaction (mirrors the client's session-start injection).

        Delegates to the single source in ``subagents`` so the block prose can't
        drift between this compaction path and the client's session-start path.
        """
        return available_subagents_block()

    async def manage_messages(self, client: Any, model: Any, agent: Any) -> None:
        """Auto-compact: summarize if the conversation exceeds the token limit.

        Only triggers when over ``max_tokens``; otherwise no-op.
        """
        raw_messages = agent.messages.copy() if hasattr(agent, "messages") else []
        messages = messages_to_dict_list(raw_messages)

        if self.count_tokens(messages) <= self.max_tokens:
            return

        keep_recent = config.get("LLM", {}).get("KEEP_RECENT_MESSAGES", 6)
        await self._compact(client, model, agent, keep_recent=keep_recent)

    async def compact(
        self, client: Any, model: Any, agent: Any, focus_instructions: str = ""
    ) -> bool:
        """Manually compact the conversation now (the /compact command).

        A manual compact is an explicit request to shrink context, so it keeps
        a smaller recent window (``MANUAL_COMPACT_KEEP_RECENT``) than the
        automatic threshold path — otherwise a short conversation would have
        nothing older than the keep window and the command would no-op.

        Args:
            focus_instructions: Optional guidance on what to emphasize.

        Returns:
            True if older messages were actually summarized, else False.
        """
        keep_recent = config.get("LLM", {}).get("MANUAL_COMPACT_KEEP_RECENT", 2)
        return await self._compact(
            client,
            model,
            agent,
            keep_recent=keep_recent,
            focus_instructions=focus_instructions,
        )

    def _split_keep_recent(
        self, raw_messages: List[Any], keep_recent: int
    ) -> int:
        """Decide how many trailing messages to keep verbatim.

        Bounded by BOTH a message count (``keep_recent``) and a token budget
        (``KEEP_RECENT_TOKEN_BUDGET``), walking newest -> oldest and stopping as
        soon as either limit would be exceeded. This guarantees that a single
        oversized recent message (e.g. a pasted document that alone fills the
        context window) is NOT kept verbatim — it falls into the 'older' set and
        gets summarized instead.

        Returns:
            The split index: messages[:split] are summarized, messages[split:]
            are kept verbatim.
        """
        n = len(raw_messages)
        if keep_recent <= 0:
            return n  # keep nothing verbatim; summarize everything

        token_budget = config.get("LLM", {}).get(
            "KEEP_RECENT_TOKEN_BUDGET", max(1, int(self.max_tokens * 0.25))
        )

        kept = 0
        used = 0
        # Walk from the newest message backwards.
        for msg in reversed(raw_messages):
            if kept >= keep_recent:
                break
            msg_tokens = self.count_tokens(messages_to_dict_list([msg]))
            # Stop if adding this message would blow the token budget — but
            # always allow it to be summarized (it stays in 'older').
            if used + msg_tokens > token_budget:
                break
            used += msg_tokens
            kept += 1

        split = n - kept
        return self._safe_tool_boundary(raw_messages, split)

    @staticmethod
    def _is_tool_message(msg: Any) -> bool:
        """True if msg is a tool result (LangChain ToolMessage or dict tool role)."""
        if isinstance(msg, dict):
            return msg.get("role") == "tool"
        return getattr(msg, "type", None) == "tool"

    @staticmethod
    def _has_tool_calls(msg: Any) -> bool:
        """True if msg is an assistant turn that issued tool calls."""
        if isinstance(msg, dict):
            return bool(msg.get("tool_calls"))
        return bool(getattr(msg, "tool_calls", None))

    def _safe_tool_boundary(self, raw_messages: List[Any], split: int) -> int:
        """Adjust a split index so it never severs a tool-call/result pair.

        ``messages[:split]`` is summarized; ``messages[split:]`` is kept
        verbatim. A split is UNSAFE when it lands inside a tool exchange:

        * the kept window starts with a tool result (a ToolMessage whose
          originating assistant tool-call turn was summarized away), or
        * the message just before the split is an assistant turn that issued
          tool calls (its results were kept, so the call is now orphaned).

        Either case makes providers like the OpenAI Responses API reject the
        request: "No tool call found for function call output with call_id …".
        We move the split EARLIER (keep a little more verbatim, pulling the whole
        tool exchange into the kept window) until the boundary is clean.

        Returns the adjusted split (0 = keep everything verbatim, also safe).
        """
        n = len(raw_messages)
        while split > 0:
            head_is_tool = split < n and self._is_tool_message(raw_messages[split])
            prev_calls = self._has_tool_calls(raw_messages[split - 1])
            if head_is_tool or prev_calls:
                split -= 1
                continue
            break
        return split

    @staticmethod
    def _tool_result_text(msg: Any) -> str:
        """Return a tool result's textual content (LangChain ToolMessage or dict)."""
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list):
                return "".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict)
                )
            return str(content) if content is not None else ""
        return str(getattr(msg, "content", ""))

    def _shrink_tool_message(self, msg: Any, cap: int) -> Any:
        """Return a copy of ``msg`` with its tool-result body shrunk to ``cap`` chars.

        Only the content is trimmed (head kept + eviction marker); the message,
        its ``tool_call_id`` and ``name`` stay intact, so tool-call/result pairing
        is preserved. Returns ``msg`` unchanged when already short enough.
        """
        text = self._tool_result_text(msg)
        if len(text) <= cap:
            return msg
        shrunk = f"{text[:cap].rstrip()}\n\n{_EVICTION_MARKER}"
        if isinstance(msg, dict):
            new = dict(msg)
            new["content"] = [{"text": shrunk}]
            return new
        # LangChain messages are immutable; model_copy is the sanctioned update
        # path (same pattern the message sanitizer uses).
        if hasattr(msg, "model_copy"):
            return msg.model_copy(update={"content": shrunk})
        return msg

    def evict_old_tool_results(self, agent: Any) -> bool:
        """Cheapest compaction layer: shrink OLD tool-result bodies in place.

        Runs before any LLM summary. Tool results outside the recent window carry
        the bulk of the context (grep/read/web dumps) yet are rarely needed
        verbatim once the model has acted on them. We shrink each old tool result
        to a short head + an eviction marker — keeping recent turns fully verbatim
        and never dropping a message (so tool-call/result pairing is untouched, no
        provider rejects the next turn). No LLM call, so this is near-free.

        Returns True if any tool result was actually shrunk.
        """
        raw_messages = agent.messages if hasattr(agent, "messages") else []
        if not raw_messages:
            return False

        llm_cfg = config.get("LLM", {})
        keep_recent = llm_cfg.get(
            "TOOL_EVICTION_KEEP_RECENT", _TOOL_EVICTION_KEEP_RECENT
        )
        cap = llm_cfg.get(
            "EVICTED_TOOL_RESULT_CHARS", _EVICTED_TOOL_RESULT_CHARS
        )
        if cap <= 0:
            return False

        n = len(raw_messages)
        cutoff = max(0, n - keep_recent)  # messages[:cutoff] are "old"
        changed = False
        new_messages = list(raw_messages)
        for i in range(cutoff):
            msg = new_messages[i]
            if not self._is_tool_message(msg):
                continue
            shrunk = self._shrink_tool_message(msg, cap)
            if shrunk is not msg:
                new_messages[i] = shrunk
                changed = True

        if changed:
            agent.messages = new_messages
            # History shrank: the provider's last exact count is now stale-high
            # and would defeat the high-water check on the next turn.
            if hasattr(agent, "_last_input_tokens"):
                agent._last_input_tokens = None
        return changed

    async def _compact(
        self,
        client: Any,
        model: Any,
        agent: Any,
        keep_recent: int = 6,
        focus_instructions: str = "",
    ) -> bool:
        """Summarize older messages while keeping the most recent turns verbatim.

        The kept window is bounded by both a message count and a token budget
        (see ``_split_keep_recent``), so an oversized recent message is folded
        into the summary rather than kept verbatim. Everything not kept is
        summarized into the system prompt.

        Returns:
            True if there were older messages to summarize, else False.
        """
        # Cheapest layer first, on EVERY compaction path (not just the proactive
        # mid-loop check): shrink OLD tool-result bodies (no LLM call) so the
        # summary that follows has far less bulk to read. In tool-heavy sessions
        # most of the context is old grep/read/web dumps; evicting them first is
        # near-free and cuts the summary's input dramatically.
        self.evict_old_tool_results(agent)

        raw_messages = agent.messages.copy() if hasattr(agent, "messages") else []
        if not raw_messages:
            return False

        split = self._split_keep_recent(raw_messages, keep_recent)
        older = raw_messages[:split]
        recent = raw_messages[split:]
        if not older:
            # Everything is within the keep window; nothing to summarize.
            return False

        # Phased status on the spinner (no fake % bar — a single LLM summary
        # call has no measurable total; we surface the discrete stages instead).
        client.spinner.start(f"Summarizing {len(older)} older messages")
        try:
            summary = await self.generate_summary(
                messages_to_dict_list(older), model, focus_instructions
            )
            client.spinner.set_label("Applying summary")
            clean_summary = "".join(c for c in summary if c.isprintable())

            new_system_content = self._build_system_with_summary(
                clean_summary, client=client
            )

            # Keep recent turns verbatim; drop only the summarized older ones.
            # Sanitize the kept window so a tool call/result pair severed by the
            # split (or an orphan inherited from earlier history) can't break the
            # next turn with "No tool output found for function call …".
            kept = list(recent)
            sanitize = getattr(agent, "_sanitize_tool_pairs", None)
            if callable(sanitize):
                kept = sanitize(kept)
            agent.messages = kept
            client.system_prompt = new_system_content
            agent.system_prompt = new_system_content

            # Checkpoint the new state in the session transcript. The turns
            # summarized away stay on disk in their own records (the log is
            # append-only), so this loses no text — it records what REPLACED them,
            # which is what a later `--resume` must restore. Without it a resume
            # rebuilt the raw pre-compaction history and had to summarize the whole
            # conversation again, discarding the summary just paid for.
            log = getattr(agent, "session_log", None)
            if log is not None:
                try:
                    log.log_compaction(summary=clean_summary, kept=kept)
                except Exception as e:  # noqa: BLE001 — never break a compaction
                    logger.debug(f"Session log compaction marker failed: {e}")

            if hasattr(agent, "_last_input_tokens"):
                agent._last_input_tokens = None
            client.spinner.stop()
            log_green(
                f"Compacted: summarized {len(older)} older messages, "
                f"kept {len(recent)} recent."
            )
            return True
        finally:
            client.spinner.stop()
