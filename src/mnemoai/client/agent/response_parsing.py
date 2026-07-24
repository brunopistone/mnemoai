"""Provider-agnostic model-output parsing (pure logic).

Pulls reasoning/thinking and visible answer text out of a completed response or a
streaming chunk across the shapes each provider emits — Bedrock Converse
(``reasoning_content`` / ``thinking`` blocks), OpenAI Responses (``reasoning``
summary blocks), and plain-string providers (Ollama ``<think>`` tags) — and
detects output-token truncation. All functions are pure (no agent state); the
agent keeps thin delegating methods over them, mirroring ``tool_formatting`` /
``message_sanitizer``.
"""

import re
from typing import Optional


def reasoning_content_text(block: dict) -> str:
    """Text from a Bedrock Converse ``reasoning_content`` block.

    Shape: ``{"type":"reasoning_content","reasoning_content":{"text":…}}``.
    """
    rc = block.get("reasoning_content")
    if isinstance(rc, dict):
        return rc.get("text", "")
    return rc if isinstance(rc, str) else ""


def reasoning_summary_text(block: dict) -> str:
    """Concatenate an OpenAI Responses ``reasoning`` summary block's text.

    Shape: ``{"type":"reasoning","summary":[{"type":"summary_text","text":…}]}``.
    """
    summary = block.get("summary")
    if not isinstance(summary, list):
        return ""
    return "".join(
        part.get("text", "")
        for part in summary
        if isinstance(part, dict) and part.get("type") == "summary_text"
    )


def extract_thinking(response) -> Optional[str]:
    """Extract thinking/reasoning from a response, or None.

    Checks additional_kwargs, Bedrock content blocks, OpenAI Responses
    reasoning-summary blocks, and <think>/<thinking> tags.
    """
    # 1. additional_kwargs (Ollama via wrapper, LiteLLM).
    if hasattr(response, "additional_kwargs"):
        thinking = response.additional_kwargs.get("reasoning_content")
        if thinking:
            return thinking

    # 2. Content blocks: Bedrock {"type":"thinking"} and OpenAI Responses
    #    {"type":"reasoning","summary":[…]}.
    if isinstance(response.content, list):
        parts = []
        for block in response.content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "thinking":
                parts.append(block.get("thinking", ""))
            elif block.get("type") == "reasoning_content":
                parts.append(reasoning_content_text(block))
            elif block.get("type") == "reasoning":
                parts.append(reasoning_summary_text(block))
        parts = [p for p in parts if p]
        if parts:
            return "".join(parts)

    # 3. <think>/<thinking> tags in string content (Ollama raw).
    if isinstance(response.content, str):
        match = re.search(
            r"<think(?:ing)?>(.*?)</think(?:ing)?>",
            response.content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

    return None


def was_truncated_by_tokens(response) -> bool:
    """True if the turn was cut short by the output-token limit.

    Responses API reports ``incomplete_details.reason == "max_output_tokens"``;
    Chat/Converse providers signal a ``length`` finish reason.
    """
    meta = getattr(response, "response_metadata", None) or {}
    details = meta.get("incomplete_details") or {}
    if isinstance(details, dict) and details.get("reason") == "max_output_tokens":
        return True
    finish = meta.get("finish_reason") or meta.get("stop_reason")
    return finish in ("length", "max_tokens")


def extract_visible(content) -> str:
    """Extract visible text, stripping <think>/<thinking> tags."""
    if isinstance(content, str):
        return re.sub(
            r"<think(?:ing)?>.*?</think(?:ing)?>",
            "",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def extract_content(chunk) -> tuple[str, str]:
    """Extract ``(content, reasoning_content)`` from a streaming chunk."""
    raw_content = chunk.content if chunk.content else ""
    chunk_content = ""
    reasoning_content = ""

    if isinstance(raw_content, list):
        # Bedrock / Responses content blocks.
        for block in raw_content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "thinking":
                    reasoning_content += block.get("thinking", "")
                elif block_type == "reasoning_content":
                    reasoning_content += reasoning_content_text(block)
                elif block_type == "reasoning":
                    reasoning_content += reasoning_summary_text(block)
                elif block_type == "text":
                    chunk_content += block.get("text", "")
                elif "text" in block:
                    chunk_content += block["text"]
    else:
        chunk_content = str(raw_content) if raw_content else ""

    # Reasoning in additional_kwargs (Ollama, LiteLLM).
    if hasattr(chunk, "additional_kwargs") and chunk.additional_kwargs:
        reasoning = chunk.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            reasoning_content = reasoning

    # Strip a stray </think> tag some models include.
    if "</think>" in chunk_content:
        chunk_content = chunk_content.replace("</think>", "").strip()

    return chunk_content, reasoning_content
