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
    route: Optional[str]


class LangGraphAgent:
    """LangGraph-based agent with streaming support."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: List[BaseTool],
        system_prompt: str = "",
        verbose: bool = False,
        callbacks: List[Any] = None,
        router=None,
        tool_routes: Optional[Dict[str, Optional[List[str]]]] = None,
        orchestrator_enabled: bool = False,
    ) -> None:
        """Initialize the LangGraph agent.

        Args:
            model: LangChain chat model
            tools: List of LangChain tools
            system_prompt: System prompt for the agent
            verbose: Enable verbose mode for thinking display
            callbacks: Optional list of callback handlers for streaming
            router: Optional QueryRouter for query classification
            tool_routes: Optional dict mapping route names to tool name lists
            orchestrator_enabled: Enable orchestrator for 'full' route
        """
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.callbacks = callbacks or []
        self._messages: List[BaseMessage] = []
        self._thinking: Optional[str] = None
        self._code_formatter = CodeFormatter()
        self.router = router
        self.orchestrator_enabled = orchestrator_enabled and router is not None

        self.model_with_tools = model.bind_tools(tools) if tools else model

        # Build per-route tool subsets and model bindings
        self.tools_by_route: Optional[Dict[str, List[BaseTool]]] = None
        self.models_by_route: Optional[Dict[str, BaseChatModel]] = None
        if router and tool_routes:
            self.tools_by_route = {}
            self.models_by_route = {}
            for route_name, tool_names in tool_routes.items():
                if tool_names is None:
                    route_tools = tools
                elif not tool_names:
                    route_tools = []
                else:
                    route_tools = [t for t in tools if t.name in tool_names]
                self.tools_by_route[route_name] = route_tools
                self.models_by_route[route_name] = (
                    model.bind_tools(route_tools) if route_tools else model
                )

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

        if self.router:
            workflow.add_node("classifier", self._classify)
            workflow.set_entry_point("classifier")

            if self.orchestrator_enabled:
                workflow.add_node("orchestrator", self._orchestrate)
                workflow.add_conditional_edges(
                    "classifier",
                    self._route_after_classify,
                    {"agent": "agent", "orchestrator": "orchestrator"},
                )
                workflow.add_edge("orchestrator", END)
            else:
                workflow.add_edge("classifier", "agent")
        else:
            workflow.set_entry_point("agent")

        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self._execute_tools)
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END},
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    def _route_after_classify(self, state: AgentState) -> str:
        """Route to orchestrator for 'full' tasks, agent otherwise.

        Args:
            state: Current agent state

        Returns:
            "orchestrator" or "agent"
        """
        if state.get("route") == "full":
            return "orchestrator"
        return "agent"

    def _classify(self, state: AgentState) -> Dict[str, Any]:
        """Classify the query and set the route in state.

        Args:
            state: Current agent state

        Returns:
            Updated state with route
        """
        messages = state["messages"]
        if not messages:
            return {"route": "full"}

        # Build conversation context from recent messages (excluding the last)
        context = ""
        if len(messages) > 1:
            recent = messages[-min(4, len(messages)) : -1]
            context = "\n".join(
                str(m.content)[:200]
                for m in recent
                if hasattr(m, "content") and m.content
            )

        query = str(messages[-1].content) if messages else ""
        route = self.router.classify(query, context)
        logger.debug(f"Query routed to: {route}")
        return {"route": route}

    def _orchestrate(self, state: AgentState) -> Dict[str, Any]:
        """Decompose a complex task into subtasks, execute workers, aggregate.

        This node handles the full orchestration pipeline:
        1. Decompose the query into subtasks via LLM
        2. Execute each subtask with a route-specific worker loop
        3. Aggregate results into a final response

        Args:
            state: Current agent state

        Returns:
            Updated state with the final aggregated response
        """
        from client.orchestrator import (
            get_orchestrator_prompt,
            get_aggregator_prompt,
            parse_subtasks,
        )
        from client.router import ROUTE_TOOLS

        messages = state["messages"]
        # Extract user query (skip system prompt)
        query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = str(msg.content)
                break

        if not query:
            return {"messages": [AIMessage(content="No query found.")]}

        # Step 1: Decompose task into subtasks
        logger.debug("Orchestrator: decomposing task")
        subtasks = self._decompose_task(
            query, get_orchestrator_prompt(), set(ROUTE_TOOLS.keys())
        )
        logger.debug(f"Orchestrator: {len(subtasks)} subtasks")

        # Step 2: Execute each subtask with a worker
        worker_results = []
        for i, subtask in enumerate(subtasks):
            desc = subtask["description"]
            category = subtask["category"]

            # Print subtask header
            total = len(subtasks)
            short_desc = desc[:70] + ("..." if len(desc) > 70 else "")
            print(
                f"\n\033[90m[Step {i + 1}/{total}: {short_desc}]\033[0m",
                flush=True,
            )

            # Get route-specific model and tools
            if category == "full" or not self.tools_by_route:
                worker_tools = self.tools
                worker_model = self.model_with_tools
            elif category == "simple_qa":
                worker_tools = []
                worker_model = self.model
            else:
                worker_tools = self.tools_by_route.get(category, self.tools)
                worker_model = self.models_by_route.get(category, self.model_with_tools)

            # Build worker prompt with context from previous results
            worker_prompt = desc
            if worker_results:
                context_parts = []
                for r in worker_results:
                    context_parts.append(f"[Completed: {r['task']}]\n{r['result']}")
                context_text = "\n\n".join(context_parts)
                worker_prompt = (
                    f"Context from completed steps:\n{context_text}"
                    f"\n\nCurrent task: {desc}"
                )

            # Execute worker
            result = self._run_worker_loop(worker_model, worker_tools, worker_prompt)
            worker_results.append(
                {
                    "task": desc,
                    "category": category,
                    "result": result,
                }
            )

        # Step 3: Aggregate results
        if len(subtasks) == 1:
            final_content = worker_results[0]["result"]
        else:
            print(
                "\n\033[90m[Synthesizing results...]\033[0m",
                flush=True,
            )
            final_content = self._aggregate_results(
                query, worker_results, get_aggregator_prompt()
            )

        return {"messages": [AIMessage(content=final_content)]}

    def _decompose_task(
        self, query: str, orchestrator_prompt: str, valid_categories: set
    ) -> List[Dict[str, Any]]:
        """Call the LLM to decompose a task into subtasks.

        Args:
            query: The user's original query
            orchestrator_prompt: System prompt for decomposition
            valid_categories: Set of valid category names

        Returns:
            List of subtask dicts
        """
        from client.orchestrator import parse_subtasks

        messages = [
            SystemMessage(content=orchestrator_prompt),
            HumanMessage(content=query),
        ]

        # Suppress callbacks to keep spinner running
        saved_callbacks = getattr(self.model, "callbacks", None)
        self.model.callbacks = None
        try:
            response = self.model.invoke(messages, config={"callbacks": []})
        finally:
            self.model.callbacks = saved_callbacks

        return parse_subtasks(response.content, query, valid_categories)

    def _run_worker_loop(
        self,
        worker_model,
        worker_tools: List[BaseTool],
        prompt: str,
        max_iterations: int = 10,
    ) -> str:
        """Execute a worker agent loop with streaming until completion.

        Args:
            worker_model: Model with route-specific tools bound
            worker_tools: Tools available to this worker
            prompt: The worker's task prompt
            max_iterations: Safety limit for agent loop

        Returns:
            Worker's final text response
        """
        worker_messages: List[BaseMessage] = []
        if self.system_prompt:
            worker_messages.append(SystemMessage(content=self.system_prompt))
        worker_messages.append(HumanMessage(content=prompt))

        config = {"callbacks": self.callbacks} if self.callbacks else {}

        for _ in range(max_iterations):
            self._start_spinner()

            response, _ = self._stream_response(
                worker_messages, config, model=worker_model
            )

            if response is None:
                response = worker_model.invoke(worker_messages, config=config)

            worker_messages.append(response)

            # If no tool calls, worker is done
            if not isinstance(response, AIMessage) or not response.tool_calls:
                self._stop_spinner()
                visible = self._extract_visible(response.content)
                return visible or str(response.content)

            # Execute tools
            self._stop_spinner()
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_id = tc["id"]
                tool_args = tc["args"]

                tool = next((t for t in worker_tools if t.name == tool_name), None)
                # Fall back to all tools if not found in worker subset
                if not tool:
                    tool = next((t for t in self.tools if t.name == tool_name), None)

                if tool:
                    try:
                        logger.debug(f"Worker tool: {tool_name} args: {tool_args}")
                        result = tool.invoke(tool_args)
                        worker_messages.append(
                            ToolMessage(
                                content=str(result),
                                tool_call_id=tool_id,
                                name=tool_name,
                            )
                        )
                    except Exception as e:
                        logger.error(f"Worker tool error: {e}")
                        worker_messages.append(
                            ToolMessage(
                                content=f"Error: {e}",
                                tool_call_id=tool_id,
                                name=tool_name,
                            )
                        )
                else:
                    worker_messages.append(
                        ToolMessage(
                            content=f"Tool not found: {tool_name}",
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )

        self._stop_spinner()
        return "Task completed (max iterations reached)"

    def _aggregate_results(
        self,
        original_query: str,
        worker_results: List[Dict[str, Any]],
        aggregator_prompt: str,
    ) -> str:
        """Aggregate worker results into a final response via LLM.

        Args:
            original_query: The user's original query
            worker_results: List of worker result dicts
            aggregator_prompt: System prompt for aggregation

        Returns:
            Aggregated response text
        """
        results_text = "\n\n".join(
            f"## Subtask: {r['task']}\n{r['result']}" for r in worker_results
        )

        messages = [
            SystemMessage(content=aggregator_prompt),
            HumanMessage(
                content=(
                    f"Original request: {original_query}\n\n"
                    f"Completed subtask results:\n\n{results_text}"
                )
            ),
        ]

        config = {"callbacks": self.callbacks} if self.callbacks else {}

        self._start_spinner()
        response, _ = self._stream_response(messages, config)

        if response is None:
            response = self.model.invoke(messages, config=config)

        self._stop_spinner()
        return self._extract_visible(response.content) or str(response.content)

    def _get_route_model(self, state: AgentState):
        """Get the model binding for the current route.

        Args:
            state: Current agent state

        Returns:
            Model with appropriate tools bound
        """
        route = state.get("route")
        if route and self.models_by_route:
            return self.models_by_route.get(route, self.model_with_tools)
        return self.model_with_tools

    def _get_route_tools(self, state: AgentState) -> List[BaseTool]:
        """Get the tool list for the current route.

        Args:
            state: Current agent state

        Returns:
            List of tools available for the route
        """
        route = state.get("route")
        if route and self.tools_by_route:
            return self.tools_by_route.get(route, self.tools)
        return self.tools

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

        active_model = self._get_route_model(state)
        response, had_reasoning = self._stream_response(
            messages, config, model=active_model
        )

        if response is None:
            response = active_model.invoke(messages, config=config)

        thinking = self._extract_thinking(response)
        visible = self._extract_visible(response.content)

        # If model produced only reasoning with no visible content,
        # retry once with reasoning disabled
        if thinking and not visible and not response.tool_calls:
            if not response.additional_kwargs.get("reasoning_content"):
                response.additional_kwargs["reasoning_content"] = thinking

            logger.debug("Model produced only reasoning, retrying without thinking")

            if had_reasoning:
                print("", flush=True)
            self._start_spinner()

            retry_messages = messages + [
                response,
                HumanMessage(
                    content=(
                        "You provided reasoning but no visible response. "
                        "Please provide your answer."
                    )
                ),
            ]

            saved = self._disable_reasoning()
            try:
                retry_response, _ = self._stream_response(
                    retry_messages,
                    config,
                    print_reasoning=False,
                    model=active_model,
                )
            finally:
                self._restore_reasoning(saved)

            if retry_response is not None:
                retry_visible = self._extract_visible(retry_response.content)
                if retry_visible:
                    if not retry_response.additional_kwargs.get("reasoning_content"):
                        retry_response.additional_kwargs["reasoning_content"] = thinking
                    return {"messages": [retry_response], "thinking": thinking}

        return {"messages": [response], "thinking": thinking}

    def _stream_response(
        self,
        messages: list,
        config: dict,
        print_reasoning: bool = True,
        model=None,
    ) -> tuple:
        """Stream model response, handling spinner and output.

        Args:
            messages: Messages to send to the model
            config: LangChain config dict
            print_reasoning: Whether to print reasoning in gray
            model: Optional model override (defaults to self.model_with_tools)

        Returns:
            Tuple of (response, had_reasoning)
        """
        active_model = model or self.model_with_tools
        self._code_formatter = CodeFormatter()
        first_token = True
        had_reasoning = False
        response = None

        for chunk in active_model.stream(messages, config=config):
            chunk_content, reasoning_content = self._extract_content(chunk)

            if first_token and (chunk_content or reasoning_content) and self.callbacks:
                first_token = False
                self._stop_spinner()

            if reasoning_content and self.verbose and print_reasoning:
                print(f"\033[90m{reasoning_content}\033[0m", end="", flush=True)
                had_reasoning = True

            if chunk_content:
                if had_reasoning:
                    if not chunk_content.startswith("\n"):
                        print("\n", end="", flush=True)
                    had_reasoning = False
                self._code_formatter.process_chunk(chunk_content)

            response = chunk if response is None else response + chunk

        return response, had_reasoning

    def _stop_spinner(self) -> None:
        """Stop the spinner and mark first token received."""
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

    def _start_spinner(self) -> None:
        """Restart the spinner and reset first token flag."""
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

    def _extract_thinking(self, response) -> Optional[str]:
        """Extract thinking/reasoning content from a response.

        Checks all possible sources: additional_kwargs, Bedrock content blocks,
        and <think>/<thinking> tags in string content.

        Args:
            response: AIMessage response

        Returns:
            Thinking text or None
        """
        # 1. Check additional_kwargs (Ollama via wrapper, LiteLLM)
        if hasattr(response, "additional_kwargs"):
            thinking = response.additional_kwargs.get("reasoning_content")
            if thinking:
                return thinking

        # 2. Check Bedrock-style content blocks
        if isinstance(response.content, list):
            parts = [
                block.get("thinking", "")
                for block in response.content
                if isinstance(block, dict) and block.get("type") == "thinking"
            ]
            if parts:
                return "".join(parts)

        # 3. Check <think>/<thinking> tags in string content (Ollama raw)
        if isinstance(response.content, str):
            match = re.search(
                r"<think(?:ing)?>(.*?)</think(?:ing)?>",
                response.content,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if match:
                return match.group(1).strip()

        return None

    def _extract_visible(self, content) -> str:
        """Extract visible content, stripping thinking tags.

        Args:
            content: Response content (str or list of blocks)

        Returns:
            Visible text content
        """
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

    def _disable_reasoning(self) -> dict:
        """Temporarily disable reasoning/thinking on the model.

        Returns:
            Saved state to pass to _restore_reasoning()
        """
        saved = {}
        reasoning = getattr(self.model, "reasoning", None)
        if reasoning is not None:
            saved["reasoning"] = reasoning
            self.model.reasoning = False
        if (
            hasattr(self.model, "model_kwargs")
            and "thinking" in self.model.model_kwargs
        ):
            saved["thinking"] = self.model.model_kwargs.pop("thinking")
            saved["temperature"] = self.model.model_kwargs.get("temperature")
            self.model.model_kwargs["temperature"] = 0.1
        return saved

    def _restore_reasoning(self, saved: dict) -> None:
        """Restore reasoning/thinking settings on the model.

        Args:
            saved: State from _disable_reasoning()
        """
        if "reasoning" in saved:
            self.model.reasoning = saved["reasoning"]
        if "thinking" in saved:
            self.model.model_kwargs["thinking"] = saved["thinking"]
        if "temperature" in saved:
            self.model.model_kwargs["temperature"] = saved["temperature"]

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

        self._stop_spinner()

        route_tools = self._get_route_tools(state)

        tool_results = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            # Look up in route tools first, fall back to all tools
            tool = next((t for t in route_tools if t.name == tool_name), None)
            if not tool:
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

        self._start_spinner()

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
            "route": None,
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
