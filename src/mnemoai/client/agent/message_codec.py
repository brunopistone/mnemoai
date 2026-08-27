"""Conversion between Strands and LangChain message formats.

Pure functions with no agent/instance state — extracted from ``agent.py`` so the
codec can be read, tested, and reused independently of the agent loop. The agent
re-exports both names for backward compatibility.

**The round trip must be idempotent.** Every resume of a session decodes the whole
transcript and re-encodes it into the new session file, so any asymmetry compounds
once per resume — see ``_collapse_wrapped_tool_result``.
"""

import ast
import json
import re
from typing import Any, Dict, List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# Head of a tool result whose text is itself a serialized ``[{'text': …}]`` block
# list — matched on the head so a megabyte payload isn't scanned in full.
_WRAPPED_BLOCKS_RE = re.compile(r"^\[\s*\{\s*['\"]text['\"]\s*:")

# Enough to unwrap any real transcript (one layer per resume); a bound, not a
# tuning knob — it only stops a pathological payload from looping.
_MAX_COLLAPSE_DEPTH = 64


def _collapse_wrapped_tool_result(text: str) -> str:
    """Unwrap a tool result an earlier round trip re-serialized into its own text.

    Decoding used to take ``str()`` of the ``toolResult.content`` BLOCK LIST, so a
    result came back as the literal ``"[{'text': …}]"``; re-encoding then wrapped
    that in a new block and re-escaped every quote in it. Each cycle therefore
    roughly DOUBLED the payload — and a cycle is one resume, so the same tool
    results grew without bound across a resumed conversation (measured: a 31-char
    result reached 4,735 chars after ten resumes; real ``fs_read`` results reached
    1.07M chars, blowing past ``MAX_TOOL_RESULT_CHARS``, which is only applied
    once at the source). Transcripts recorded before the fix still hold the nested
    form, so decoding repairs them in place. Only an EXACT single-``text``-block
    literal is collapsed — a genuine payload is vanishingly unlikely to be one.
    """
    for _ in range(_MAX_COLLAPSE_DEPTH):
        stripped = text.strip()
        if not stripped.endswith("]") or not _WRAPPED_BLOCKS_RE.match(stripped):
            break
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            break
        if (
            not isinstance(parsed, list)
            or len(parsed) != 1
            or not isinstance(parsed[0], dict)
            or set(parsed[0]) != {"text"}
            or not isinstance(parsed[0]["text"], str)
        ):
            break
        text = parsed[0]["text"]
    return text


def _tool_result_text(content: Any) -> str:
    """Text of a Strands ``toolResult.content`` — READ, never re-serialized."""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif "json" in block:
                    parts.append(json.dumps(block["json"], default=str))
                else:
                    parts.append(json.dumps(block, default=str))
            else:
                parts.append(str(block))
        text = "".join(parts)
    elif isinstance(content, str):
        text = content
    elif isinstance(content, dict):
        text = json.dumps(content, default=str)
    else:
        text = str(content)
    return _collapse_wrapped_tool_result(text)


def _tool_result_blocks(content: Any) -> List[Dict[str, Any]]:
    """Strands ``toolResult.content`` blocks for a LangChain ToolMessage.

    A list of blocks is mapped block-by-block; ``str()`` of the list would be the
    non-reversible form ``_collapse_wrapped_tool_result`` exists to undo.
    """
    if isinstance(content, list):
        blocks: List[Dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                blocks.append({"text": str(block["text"])})
            elif isinstance(block, dict):
                blocks.append({"text": json.dumps(block, default=str)})
            else:
                blocks.append({"text": str(block)})
        return blocks or [{"text": ""}]
    return [{"text": content if isinstance(content, str) else str(content)}]


def convert_strands_messages_to_langchain(
    messages: List[Dict[str, Any]],
) -> List[BaseMessage]:
    """Convert Strands-format messages to LangChain messages."""
    langchain_messages = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", [])

        text_content = ""
        reasoning_text = ""
        tool_calls = []
        tool_results = []

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if "text" in block:
                        text_content += block["text"]
                    elif "reasoningContent" in block:
                        rc = block["reasoningContent"]
                        reasoning_text += rc.get("reasoningText", {}).get("text", "")
                    elif "toolUse" in block:
                        tool_calls.append(block["toolUse"])
                    elif "toolResult" in block:
                        tool_results.append(block["toolResult"])
                elif isinstance(block, str):
                    text_content += block
        elif isinstance(content, str):
            text_content = content

        if role == "user":
            if tool_results:
                for result in tool_results:
                    langchain_messages.append(
                        ToolMessage(
                            content=_tool_result_text(result.get("content", "")),
                            tool_call_id=result.get("toolUseId", ""),
                        )
                    )
            else:
                langchain_messages.append(HumanMessage(content=text_content))
        elif role == "assistant":
            additional_kwargs = {}
            if reasoning_text:
                additional_kwargs["reasoning_content"] = reasoning_text
            if tool_calls:
                formatted_tool_calls = [
                    {
                        "id": tc.get("toolUseId", ""),
                        "name": tc.get("name", ""),
                        "args": tc.get("input", {}),
                    }
                    for tc in tool_calls
                ]
                langchain_messages.append(
                    AIMessage(
                        content=text_content,
                        tool_calls=formatted_tool_calls,
                        additional_kwargs=additional_kwargs,
                    )
                )
            else:
                langchain_messages.append(
                    AIMessage(
                        content=text_content,
                        additional_kwargs=additional_kwargs,
                    )
                )
        elif role == "system":
            langchain_messages.append(SystemMessage(content=text_content))

    return langchain_messages


def convert_langchain_messages_to_strands(
    messages: List[BaseMessage],
) -> List[Dict[str, Any]]:
    """Convert LangChain messages to Strands format."""
    strands_messages = []

    for msg in messages:
        content_blocks = []

        if isinstance(msg, HumanMessage):
            content_blocks.append({"text": str(msg.content)})
            strands_messages.append({"role": "user", "content": content_blocks})

        elif isinstance(msg, AIMessage):
            # Extract reasoning content from additional_kwargs (Ollama, LiteLLM)
            reasoning_text = ""
            if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                reasoning_text = msg.additional_kwargs.get("reasoning_content", "")

            if msg.content:
                if isinstance(msg.content, list):
                    # Bedrock format: list of content blocks
                    for block in msg.content:
                        if isinstance(block, dict):
                            block_type = block.get("type", "")
                            if block_type == "thinking":
                                # Preserve reasoning as reasoningContent block
                                thinking_text = block.get("thinking", "")
                                if thinking_text:
                                    content_blocks.append(
                                        {
                                            "reasoningContent": {
                                                "reasoningText": {"text": thinking_text}
                                            }
                                        }
                                    )
                            elif block_type == "text":
                                text = block.get("text", "")
                                if text:
                                    content_blocks.append({"text": text})
                            elif "text" in block:
                                content_blocks.append({"text": block["text"]})
                else:
                    content_blocks.append({"text": str(msg.content)})

            # Add reasoning from additional_kwargs if not already added from content blocks
            if reasoning_text and not any(
                "reasoningContent" in b for b in content_blocks
            ):
                content_blocks.insert(
                    0,
                    {"reasoningContent": {"reasoningText": {"text": reasoning_text}}},
                )

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": tc.get("id", ""),
                                "name": tc.get("name", ""),
                                "input": tc.get("args", {}),
                            }
                        }
                    )
            strands_messages.append({"role": "assistant", "content": content_blocks})

        elif isinstance(msg, ToolMessage):
            content_blocks.append(
                {
                    "toolResult": {
                        "toolUseId": msg.tool_call_id,
                        "content": _tool_result_blocks(msg.content),
                    }
                }
            )
            strands_messages.append({"role": "user", "content": content_blocks})

        elif isinstance(msg, SystemMessage):
            content_blocks.append({"text": str(msg.content)})
            strands_messages.append({"role": "system", "content": content_blocks})

    return strands_messages
