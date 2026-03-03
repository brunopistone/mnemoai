"""LangGraph-based agent implementation."""

import operator
import re
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END

from utils.formatting.code_formatter import CodeFormatter
from utils.logger import logger


class AgentState(TypedDict):
    """State schema for the LangGraph agent."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    thinking: Optional[str]


class LangGraphAgent:
    """LangGraph-based agent with streaming support."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: List[BaseTool],
        system_prompt: str = "",
        verbose: bool = False,
        callbacks: List[Any] = None,
    ) -> None:
        """Initialize the LangGraph agent.

        Args:
            model: LangChain chat model
            tools: List of LangChain tools
            system_prompt: System prompt for the agent
            verbose: Enable verbose mode for thinking display
            callbacks: Optional list of callback handlers for streaming
        """
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.callbacks = callbacks or []
        self._messages: List[BaseMessage] = []
        self._thinking: Optional[str] = None
        self._code_formatter = CodeFormatter()

        self.model_with_tools = model.bind_tools(tools) if tools else model
        self.graph = self._build_graph()

    @property
    def messages(self) -> List[BaseMessage]:
        """Get the message history.

        Returns:
            List of messages
        """
        return self._messages

    @messages.setter
    def messages(self, value: List[BaseMessage]) -> None:
        """Set the message history.

        Args:
            value: List of messages to set
        """
        self._messages = value

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph.

        Returns:
            Compiled state graph
        """
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self._execute_tools)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END},
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        """Call the model with current state using streaming.

        Args:
            state: Current agent state

        Returns:
            Updated state with model response
        """
        messages = list(state["messages"])

        if self.system_prompt and (
            not messages or not isinstance(messages[0], SystemMessage)
        ):
            messages = [SystemMessage(content=self.system_prompt)] + messages

        config = {"callbacks": self.callbacks} if self.callbacks else {}

        # Reset code formatter for new response
        self._code_formatter = CodeFormatter()

        first_token = True
        had_reasoning = False
        response = None

        for chunk in self.model_with_tools.stream(messages, config=config):
            chunk_content, reasoning_content = self._extract_content(chunk)

            # Stop spinner on first token (thread-safe)
            if first_token and (chunk_content or reasoning_content) and self.callbacks:
                first_token = False
                for cb in self.callbacks:
                    if hasattr(cb, "first_token_received"):
                        cb.first_token_received = True
                    if hasattr(cb, "spinner") and cb.spinner:
                        lock = getattr(cb, "spinner_lock", None)
                        if lock:
                            with lock:
                                cb.spinner.stop()
                        else:
                            cb.spinner.stop()

            # Print reasoning in gray
            if reasoning_content and self.verbose:
                print(f"\033[90m{reasoning_content}\033[0m", end="", flush=True)
                had_reasoning = True

            # Print content with syntax highlighting
            if chunk_content:
                if had_reasoning:
                    # Only add newline if content doesn't start with newlines
                    if not chunk_content.startswith("\n"):
                        print("\n", end="", flush=True)
                    had_reasoning = False
                self._code_formatter.process_chunk(chunk_content)

            response = chunk if response is None else response + chunk

        if response is None:
            response = self.model_with_tools.invoke(messages, config=config)

        # Extract final thinking content from all possible sources
        thinking = None

        # 1. Check additional_kwargs (Ollama via wrapper, LiteLLM)
        if hasattr(response, "additional_kwargs"):
            thinking = response.additional_kwargs.get("reasoning_content")

        # 2. Check Bedrock-style content blocks
        if not thinking and isinstance(response.content, list):
            thinking_parts = []
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    thinking_parts.append(block.get("thinking", ""))
            if thinking_parts:
                thinking = "".join(thinking_parts)

        # 3. Check <think>/<thinking> tags in string content (Ollama raw)
        if not thinking and isinstance(response.content, str):
            think_match = re.search(
                r"<think(?:ing)?>(.*?)</think(?:ing)?>",
                response.content,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if think_match:
                thinking = think_match.group(1).strip()

        # Determine visible content (strip thinking tags from all formats)
        visible = ""
        if isinstance(response.content, str):
            visible = re.sub(
                r"<think(?:ing)?>.*?</think(?:ing)?>",
                "",
                response.content,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
        elif isinstance(response.content, list):
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    visible += block.get("text", "")

        # If model produced only reasoning with no visible content,
        # retry once — user already saw the reasoning (gray text),
        # so only print the visible answer on retry
        if thinking and not visible and not response.tool_calls:
            if not response.additional_kwargs.get("reasoning_content"):
                response.additional_kwargs["reasoning_content"] = thinking

            logger.debug("Model produced only reasoning, retrying for visible answer")

            # Move to a new line so spinner doesn't overwrite reasoning output
            print("", flush=True)

            # Restart spinner so user sees feedback during retry
            for cb in self.callbacks:
                if hasattr(cb, "spinner") and cb.spinner:
                    lock = getattr(cb, "spinner_lock", None)
                    if lock:
                        with lock:
                            cb.spinner.start()
                            if hasattr(cb, "first_token_received"):
                                cb.first_token_received = False
                    else:
                        cb.spinner.start()
                        if hasattr(cb, "first_token_received"):
                            cb.first_token_received = False

            retry_messages = messages + [
                response,
                HumanMessage(
                    content=(
                        "You provided reasoning but no visible response. "
                        "Please provide your answer."
                    )
                ),
            ]

            self._code_formatter = CodeFormatter()
            retry_first_token = True
            retry_response = None
            for chunk in self.model_with_tools.stream(retry_messages, config=config):
                chunk_content, _ = self._extract_content(chunk)

                # Stop spinner on first token
                if retry_first_token and chunk_content and self.callbacks:
                    retry_first_token = False
                    for cb in self.callbacks:
                        if hasattr(cb, "first_token_received"):
                            cb.first_token_received = True
                        if hasattr(cb, "spinner") and cb.spinner:
                            lock = getattr(cb, "spinner_lock", None)
                            if lock:
                                with lock:
                                    cb.spinner.stop()
                            else:
                                cb.spinner.stop()

                # Only print visible content — user already saw reasoning
                if chunk_content:
                    self._code_formatter.process_chunk(chunk_content)
                retry_response = (
                    chunk if retry_response is None else retry_response + chunk
                )

            if retry_response is not None:
                retry_visible = ""
                if isinstance(retry_response.content, str):
                    retry_visible = re.sub(
                        r"<think(?:ing)?>.*?</think(?:ing)?>",
                        "",
                        retry_response.content,
                        flags=re.DOTALL | re.IGNORECASE,
                    ).strip()
                elif isinstance(retry_response.content, list):
                    for block in retry_response.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            retry_visible += block.get("text", "")

                if retry_visible:
                    # Preserve original thinking in the retry response
                    if not retry_response.additional_kwargs.get("reasoning_content"):
                        retry_response.additional_kwargs["reasoning_content"] = thinking
                    return {"messages": [retry_response], "thinking": thinking}

        return {"messages": [response], "thinking": thinking}

    def _extract_content(self, chunk) -> tuple[str, str]:
        """Extract content and reasoning from a chunk.

        Args:
            chunk: Streaming chunk from the model

        Returns:
            Tuple of (content, reasoning_content)
        """
        raw_content = chunk.content if chunk.content else ""
        chunk_content = ""
        reasoning_content = ""

        if isinstance(raw_content, list):
            # Bedrock format
            for block in raw_content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "thinking":
                        reasoning_content += block.get("thinking", "")
                    elif block_type == "text":
                        chunk_content += block.get("text", "")
                    elif "text" in block:
                        chunk_content += block["text"]
        else:
            chunk_content = str(raw_content) if raw_content else ""

        # Check for reasoning in additional_kwargs (Ollama, LiteLLM)
        if hasattr(chunk, "additional_kwargs") and chunk.additional_kwargs:
            reasoning = chunk.additional_kwargs.get("reasoning_content", "")
            if reasoning:
                reasoning_content = reasoning

        # Strip </think> tag from content if present (for models that include it)
        if "</think>" in chunk_content:
            chunk_content = chunk_content.replace("</think>", "").strip()

        return chunk_content, reasoning_content

    def _execute_tools(self, state: AgentState) -> Dict[str, Any]:
        """Execute tools based on the last AI message.

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results
        """
        last_message = state["messages"][-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": []}

        # Stop spinner during tool execution
        for cb in self.callbacks:
            if hasattr(cb, "spinner") and cb.spinner:
                lock = getattr(cb, "spinner_lock", None)
                if lock:
                    with lock:
                        cb.spinner.stop()
                else:
                    cb.spinner.stop()

        tool_results = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            tool = next((t for t in self.tools if t.name == tool_name), None)

            if tool:
                try:
                    logger.debug(f"Executing tool: {tool_name} with args: {tool_args}")
                    result = tool.invoke(tool_args)
                    tool_results.append(
                        ToolMessage(
                            content=str(result), tool_call_id=tool_id, name=tool_name
                        )
                    )
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    tool_results.append(
                        ToolMessage(
                            content=f"Error: {e}", tool_call_id=tool_id, name=tool_name
                        )
                    )
            else:
                logger.warning(f"Tool not found: {tool_name}")
                tool_results.append(
                    ToolMessage(
                        content=f"Tool not found: {tool_name}",
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )

        # Restart spinner for model response after tools
        for cb in self.callbacks:
            if hasattr(cb, "spinner") and cb.spinner:
                lock = getattr(cb, "spinner_lock", None)
                if lock:
                    with lock:
                        cb.spinner.start()
                        if hasattr(cb, "first_token_received"):
                            cb.first_token_received = False
                else:
                    cb.spinner.start()
                    if hasattr(cb, "first_token_received"):
                        cb.first_token_received = False

        # Reset code formatter for new response
        self._code_formatter = CodeFormatter()

        return {"messages": tool_results}

    def _should_continue(self, state: AgentState) -> str:
        """Determine if the agent should continue or end.

        Args:
            state: Current agent state

        Returns:
            "continue" if tools should be executed, "end" otherwise
        """
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        return "end"

    def __call__(self, prompt: str) -> str:
        """Invoke the agent with a prompt.

        Args:
            prompt: User prompt

        Returns:
            Agent response as string
        """
        return self.invoke(prompt)

    def invoke(self, prompt: str) -> str:
        """Invoke the agent with a prompt.

        Args:
            prompt: User prompt

        Returns:
            Agent response as string
        """
        user_message = HumanMessage(content=prompt)
        self._messages.append(user_message)

        initial_state: AgentState = {
            "messages": self._messages.copy(),
            "thinking": None,
        }

        if self.system_prompt:
            initial_state["messages"] = [
                SystemMessage(content=self.system_prompt)
            ] + list(initial_state["messages"])

        result = self.graph.invoke(initial_state)

        final_messages = result["messages"]
        self._thinking = result.get("thinking")

        new_messages = [
            m
            for m in final_messages
            if not isinstance(m, SystemMessage) and m not in self._messages
        ]
        self._messages.extend(new_messages)

        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                content = str(msg.content)
                if content:
                    return content
                # Content is empty but reasoning may exist - return reasoning
                # so the caller knows the model did respond
                if self._thinking:
                    return ""

        return ""

    def get_thinking(self) -> Optional[str]:
        """Get the thinking content from the last response.

        Returns:
            Thinking content or None
        """
        return self._thinking

    def clear_messages(self) -> None:
        """Clear the message history."""
        self._messages.clear()
        self._thinking = None


def convert_strands_messages_to_langchain(
    messages: List[Dict[str, Any]],
) -> List[BaseMessage]:
    """Convert Strands message format to LangChain messages.

    Args:
        messages: List of Strands-format messages

    Returns:
        List of LangChain BaseMessage objects
    """
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
                        reasoning_text += rc.get("reasoningText", {}).get(
                            "text", ""
                        )
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
    """Convert LangChain messages to Strands format.

    Args:
        messages: List of LangChain BaseMessage objects

    Returns:
        List of Strands-format messages
    """
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
                                                "reasoningText": {
                                                    "text": thinking_text
                                                }
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
                    {
                        "reasoningContent": {
                            "reasoningText": {"text": reasoning_text}
                        }
                    },
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
