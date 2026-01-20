"""LangGraph-based agent implementation."""

import json
import operator
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
from langgraph.prebuilt import ToolNode
from utils.logger import logger


class AgentState(TypedDict):
    """State schema for the LangGraph agent."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    thinking: Optional[str]  # For reasoning/thinking content


class LangGraphAgent:
    """LangGraph-based agent that replaces Strands Agent."""

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

        # Bind tools to model
        if tools:
            self.model_with_tools = model.bind_tools(tools)
        else:
            self.model_with_tools = model

        # Build the graph
        self.graph = self._build_graph()

    @property
    def messages(self) -> List[BaseMessage]:
        """Get the message history (compatible with Strands agent.messages)."""
        return self._messages

    @messages.setter
    def messages(self, value: List[BaseMessage]) -> None:
        """Set the message history."""
        self._messages = value

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph.

        Returns:
            Compiled state graph
        """
        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self._execute_tools)

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )

        # Tools always go back to agent
        workflow.add_edge("tools", "agent")

        # Compile
        return workflow.compile()

    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        """Call the model with current state using streaming for real-time output.

        Args:
            state: Current agent state

        Returns:
            Updated state with model response
        """
        import re
        import sys

        messages = list(state["messages"])

        # Add system prompt if not already present
        if self.system_prompt and (
            not messages or not isinstance(messages[0], SystemMessage)
        ):
            messages = [SystemMessage(content=self.system_prompt)] + messages

        # Build config with callbacks for streaming
        config = {}
        if self.callbacks:
            config["callbacks"] = self.callbacks

        # Stream state for handling think tags
        in_thinking = False
        tag_buffer = ""
        first_token = True

        # Stream the model response and process output directly
        response = None
        chunk_count = 0
        for chunk in self.model_with_tools.stream(messages, config=config):
            # Handle different content formats:
            # - Ollama/OpenAI: string content
            # - Bedrock: list of dicts [{'type': 'text', 'text': '...'}]
            raw_content = chunk.content if chunk.content else ""

            if isinstance(raw_content, list):
                # Bedrock format - extract text from list of content blocks
                chunk_content = ""
                for block in raw_content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            chunk_content += block.get("text", "")
                        elif "text" in block:
                            chunk_content += block["text"]
            else:
                # Ollama/OpenAI format - string content
                chunk_content = str(raw_content) if raw_content else ""

            chunk_count += 1

            # Debug first few chunks to understand what's being received
            if chunk_count <= 3:
                import sys

                debug_content = repr(
                    chunk_content[:200] if len(chunk_content) > 200 else chunk_content
                )
                logger.debug(
                    f"[STREAM DEBUG] chunk #{chunk_count}: {debug_content}",
                    file=sys.stderr,
                )
                if hasattr(chunk, "additional_kwargs") and chunk.additional_kwargs:
                    logger.debug(
                        f"[STREAM DEBUG] additional_kwargs: {chunk.additional_kwargs}",
                        file=sys.stderr,
                    )

            # Notify callbacks about first token (for spinner)
            if first_token and chunk_content and self.callbacks:
                first_token = False
                for cb in self.callbacks:
                    if hasattr(cb, "first_token_received"):
                        cb.first_token_received = True
                    if hasattr(cb, "spinner") and cb.spinner:
                        cb.spinner.stop()

            # Process streaming content for thinking tags
            if chunk_content:
                tag_buffer += chunk_content

                # Check for potential partial tags at end
                last_lt = tag_buffer.rfind("<")
                if last_lt >= 0 and last_lt > len(tag_buffer) - 12:
                    to_process = tag_buffer[:last_lt]
                    tag_buffer = tag_buffer[last_lt:]
                else:
                    to_process = tag_buffer
                    tag_buffer = ""

                if to_process:
                    remaining = to_process

                    while remaining:
                        if in_thinking:
                            # Inside thinking - look for closing tag
                            match = re.search(
                                r"</think(?:ing)?>", remaining, re.IGNORECASE
                            )
                            if match:
                                thinking_content = remaining[: match.start()]
                                if thinking_content and self.verbose:
                                    # Print thinking in gray
                                    print(
                                        f"\033[90m{thinking_content}\033[0m",
                                        end="",
                                        flush=True,
                                    )
                                remaining = remaining[match.end() :]
                                in_thinking = False
                                if self.verbose:
                                    print("\n", end="", flush=True)
                            else:
                                if self.verbose:
                                    print(
                                        f"\033[90m{remaining}\033[0m",
                                        end="",
                                        flush=True,
                                    )
                                remaining = ""
                        else:
                            # Outside thinking - look for opening tag
                            match = re.search(
                                r"<think(?:ing)?>", remaining, re.IGNORECASE
                            )
                            if match:
                                regular_content = remaining[: match.start()]
                                if regular_content:
                                    print(regular_content, end="", flush=True)
                                remaining = remaining[match.end() :]
                                in_thinking = True
                            else:
                                # No opening tag - print normally
                                print(remaining, end="", flush=True)
                                remaining = ""

            if response is None:
                response = chunk
            else:
                response = response + chunk

        # Flush any remaining buffer
        if tag_buffer:
            if not in_thinking:
                print(tag_buffer, end="", flush=True)

        # If no chunks received, fall back to invoke
        if response is None:
            response = self.model_with_tools.invoke(messages, config=config)

        # Extract thinking content if present (for Claude extended thinking)
        thinking = None
        if hasattr(response, "additional_kwargs"):
            logger.debug(f"Response additional_kwargs: {response.additional_kwargs}")
            thinking_data = response.additional_kwargs.get("thinking")
            if thinking_data:
                thinking = thinking_data.get("thinking", "")
            # Also check for reasoning_content (LangChain Ollama format)
            reasoning = response.additional_kwargs.get("reasoning_content")
            if reasoning:
                thinking = reasoning
                logger.debug(
                    f"Found reasoning_content: {reasoning[:100] if len(reasoning) > 100 else reasoning}"
                )

        return {
            "messages": [response],
            "thinking": thinking,
        }

    def _execute_tools(self, state: AgentState) -> Dict[str, Any]:
        """Execute tools based on the last AI message.

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results
        """
        messages = state["messages"]
        last_message = messages[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": []}

        tool_results = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            # Find and execute the tool
            tool = next((t for t in self.tools if t.name == tool_name), None)

            if tool:
                try:
                    logger.debug(f"Executing tool: {tool_name} with args: {tool_args}")
                    result = tool.invoke(tool_args)
                    tool_results.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    tool_results.append(
                        ToolMessage(
                            content=f"Error executing tool: {str(e)}",
                            tool_call_id=tool_id,
                            name=tool_name,
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

        return {"messages": tool_results}

    def _should_continue(self, state: AgentState) -> str:
        """Determine if the agent should continue or end.

        Args:
            state: Current agent state

        Returns:
            "continue" if tools should be executed, "end" otherwise
        """
        messages = state["messages"]
        last_message = messages[-1]

        # If the last message has tool calls, continue to execute them
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"

        # Otherwise, end the conversation turn
        return "end"

    def __call__(self, prompt: str) -> str:
        """Invoke the agent with a prompt (Strands-compatible interface).

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
        # Add the user message to history
        user_message = HumanMessage(content=prompt)
        self._messages.append(user_message)

        # Build initial state
        initial_state: AgentState = {
            "messages": self._messages.copy(),
            "thinking": None,
        }

        # Add system message if present
        if self.system_prompt:
            initial_state["messages"] = [
                SystemMessage(content=self.system_prompt)
            ] + list(initial_state["messages"])

        # Run the graph
        result = self.graph.invoke(initial_state)

        # Extract the final response
        final_messages = result["messages"]
        self._thinking = result.get("thinking")

        # Update internal message history with new messages (excluding system)
        new_messages = [
            m
            for m in final_messages
            if not isinstance(m, SystemMessage) and m not in self._messages
        ]
        self._messages.extend(new_messages)

        # Find the last AI message for the response
        last_ai_message = None
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                last_ai_message = msg
                break

        if last_ai_message:
            return str(last_ai_message.content)

        return ""

    def stream(self, prompt: str):
        """Stream the agent response.

        Args:
            prompt: User prompt

        Yields:
            Stream events from the graph
        """
        # Add the user message to history
        user_message = HumanMessage(content=prompt)
        self._messages.append(user_message)

        # Build initial state
        initial_state: AgentState = {
            "messages": self._messages.copy(),
            "thinking": None,
        }

        # Add system message if present
        if self.system_prompt:
            initial_state["messages"] = [
                SystemMessage(content=self.system_prompt)
            ] + list(initial_state["messages"])

        # Stream the graph execution
        for event in self.graph.stream(initial_state, stream_mode="values"):
            yield event

        # Update message history from final state
        if event:
            final_messages = event.get("messages", [])
            new_messages = [
                m
                for m in final_messages
                if not isinstance(m, SystemMessage) and m not in self._messages
            ]
            self._messages.extend(new_messages)

    def astream(self, prompt: str):
        """Async stream the agent response.

        Args:
            prompt: User prompt

        Yields:
            Stream events from the graph
        """
        # Add the user message to history
        user_message = HumanMessage(content=prompt)
        self._messages.append(user_message)

        # Build initial state
        initial_state: AgentState = {
            "messages": self._messages.copy(),
            "thinking": None,
        }

        # Add system message if present
        if self.system_prompt:
            initial_state["messages"] = [
                SystemMessage(content=self.system_prompt)
            ] + list(initial_state["messages"])

        # Return async generator
        return self.graph.astream(initial_state, stream_mode="values")

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

        # Extract text content
        text_content = ""
        tool_calls = []
        tool_results = []

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if "text" in block:
                        text_content += block["text"]
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
                # This is a tool result message
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
            if tool_calls:
                # AI message with tool calls
                formatted_tool_calls = [
                    {
                        "id": tc.get("toolUseId", ""),
                        "name": tc.get("name", ""),
                        "args": tc.get("input", {}),
                    }
                    for tc in tool_calls
                ]
                langchain_messages.append(
                    AIMessage(content=text_content, tool_calls=formatted_tool_calls)
                )
            else:
                langchain_messages.append(AIMessage(content=text_content))
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
            if msg.content:
                content_blocks.append({"text": str(msg.content)})

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
