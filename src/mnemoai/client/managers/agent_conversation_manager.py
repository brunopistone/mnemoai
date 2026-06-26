"""Conversation manager that uses simple text-based summaries."""

import json
import textwrap
from datetime import date
from typing import Any, Dict, List, Union

import tiktoken

from mnemoai.utils.config import config
from mnemoai.utils.logger import logger

# Try to import LangChain message types for compatibility
try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

MODEL_ID = "gpt-4"  # Default model for token counting

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


class AgentConversationManager:
    def __init__(self, max_tokens: int = 5000) -> None:
        """Initialize conversation manager.

        Args:
            max_tokens: Maximum tokens before summarization
        """
        self.max_tokens = max_tokens
        self.encoder = tiktoken.encoding_for_model(MODEL_ID)
        self.previous_summary = None
        logger.info(f"Initialized conversation manager with max_tokens={max_tokens}")

    def count_tokens(self, messages: List[Dict]) -> int:
        """Count tokens with model-specific approximation.

        For Ollama models, uses character-based approximation.
        For OpenAI/Bedrock models, uses tiktoken encoder.

        Args:
            messages: List of message dictionaries

        Returns:
            Estimated token count
        """
        text = json.dumps(messages, default=str)
        model_type = config.get("MODEL_ID", {}).get("TYPE", "ollama")

        if model_type == "ollama":
            # Ollama approximation: ~1.3 chars per token (configurable)
            multiplier = (
                config.get("LLM", {})
                .get("TOKEN_COUNTING", {})
                .get("OLLAMA_APPROXIMATION", 1.3)
            )
            return int(len(text) / multiplier)
        else:
            # Use tiktoken for OpenAI/Bedrock/SageMaker
            return len(self.encoder.encode(text))

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
        import re

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
        ``## Compact Instructions`` header — exactly how Claude Code injects a
        ``/compact <focus>`` directive. We append the user's focus there.
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
        """Use the model to generate a natural language summary.

        Args:
            messages: List of conversation messages
            model: Model instance for generating summary (LangChain or Strands)
            focus_instructions: Optional user guidance on what to emphasize
                (used by the manual /compact command).

        Returns:
            Summary text
        """
        # Progress is shown via the spinner's phase label (set by the caller in
        # _compact); no separate static line here so it doesn't clutter the
        # spinner line.
        summary_prompt = self._build_summary_prompt(focus_instructions)

        try:
            summary_response = ""

            # Check if this is a LangChain model
            if LANGCHAIN_AVAILABLE and hasattr(model, 'ainvoke'):
                # LangChain model - use ainvoke
                from langchain_core.messages import HumanMessage, SystemMessage

                # Build LangChain messages. The system role frames the task as
                # summarization (mirrors Claude Code's compaction call); any
                # prior summary is carried as additional context.
                lc_messages = [SystemMessage(content=self._SUMMARY_SYSTEM_PROMPT)]
                if self.previous_summary:
                    lc_messages.append(SystemMessage(content=self.previous_summary))

                # Add conversation context. Tool calls and tool results are
                # flattened into text so the summary captures what tools did,
                # not just the user/assistant prose.
                for msg in messages:
                    role = msg.get("role")
                    content = self._message_text_for_summary(msg)
                    if not content:
                        continue
                    if role == "assistant":
                        lc_messages.append(AIMessage(content=content))
                    else:
                        # user, tool results, and anything else become context
                        lc_messages.append(HumanMessage(content=content))

                # Add summary prompt
                lc_messages.append(HumanMessage(content=summary_prompt))

                # Generate summary
                response = await model.ainvoke(lc_messages)
                summary_response = str(response.content)

            else:
                # Strands model - use stream
                messages.append({"role": "user", "content": [{"text": summary_prompt}]})
                think_param = config.get("LLM", {}).get("SUMMARIZATION_THINK", False)
                system_prompt = self._SUMMARY_SYSTEM_PROMPT
                if self.previous_summary:
                    system_prompt = f"{system_prompt}\n\n{self.previous_summary}"

                async for event in model.stream(
                    messages, system_prompt=system_prompt, think=think_param
                ):
                    if (
                        "contentBlockDelta" in event
                        and "delta" in event["contentBlockDelta"]
                        and "text" in event["contentBlockDelta"]["delta"]
                    ):
                        summary_response += event["contentBlockDelta"]["delta"]["text"]

            # Clean the response, dropping the model's <analysis> scratchpad if
            # present (we keep only the structured summary that follows it).
            clean_summary = self._strip_analysis(summary_response).strip()
            return clean_summary

        except Exception as e:
            log_green(f"Failed to generate model summary: {e}", "error")
            return "Previous conversation covered multiple topics and requests."

    def _build_system_with_summary(self, clean_summary: str) -> str:
        """Embed a conversation summary into the configured system prompt.

        The block carries a continuation instruction (mirrors Claude Code) so
        the model resumes the work seamlessly instead of re-acknowledging the
        summary or recapping.
        """
        # Continuation instruction is the verbatim Claude Code text, so the
        # model resumes seamlessly instead of re-acknowledging the summary.
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

        original_system_prompt = config.system_prompt
        if not original_system_prompt:
            return summary_block

        current_date = date.today().strftime("%Y-%m-%d")
        original_system_prompt = original_system_prompt.format(
            current_date=current_date
        )
        # Re-inject the tier-1 skills block: this rebuild re-fetches the base
        # prompt fresh, dropping all session-start injections, so without this
        # the <available_skills> block would silently vanish after compaction
        # and the model would forget which skills it can load.
        parts = [original_system_prompt]
        skills_block = self._skills_block()
        if skills_block:
            parts.append(skills_block)
        parts.append(summary_block)
        return "\n\n".join(parts)

    def _skills_block(self) -> str:
        """Build the tier-1 ``<available_skills>`` block (or "" when disabled/empty).

        Mirrors the client's session-start injection so skills survive compaction.
        """
        if not config.get("ENABLE_SKILLS", True):
            return ""
        from mnemoai.client.memory.skill_store import (
            SkillStore,
            format_available_skills,
        )

        return format_available_skills(SkillStore().list_metadata())

    async def manage_messages(self, client: Any, model: Any, agent: Any) -> None:
        """Auto-compact: summarize if the conversation exceeds the token limit.

        Only triggers when over ``max_tokens``; otherwise no-op.
        """
        raw_messages = agent.messages.copy() if hasattr(agent, "messages") else []
        messages = messages_to_dict_list(raw_messages)

        if self.count_tokens(messages) <= self.max_tokens:
            return

        log_green(
            f"Token limit exceeded ({self.count_tokens(messages)} > "
            f"{self.max_tokens}); compacting conversation"
        )
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
        We move the split EARLIER (summarize a little more, pulling the whole
        tool exchange into the kept window) until the boundary is clean.

        Returns the adjusted split (0 = summarize everything, also safe).
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

            new_system_content = self._build_system_with_summary(clean_summary)

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
            client.spinner.stop()
            log_green(
                f"Compacted: summarized {len(older)} older messages, "
                f"kept {len(recent)} recent."
            )
            return True
        finally:
            client.spinner.stop()
