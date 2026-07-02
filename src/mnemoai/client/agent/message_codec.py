"""Conversion between Strands and LangChain message formats.

Pure functions with no agent/instance state — extracted from ``agent.py`` so the
codec can be read, tested, and reused independently of the agent loop. The agent
re-exports both names for backward compatibility.
"""

from typing import Any, Dict, List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


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
                            content=str(result.get("content", "")),
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
                        "content": [{"text": str(msg.content)}],
                    }
                }
            )
            strands_messages.append({"role": "user", "content": content_blocks})

        elif isinstance(msg, SystemMessage):
            content_blocks.append({"text": str(msg.content)})
            strands_messages.append({"role": "system", "content": content_blocks})

    return strands_messages
