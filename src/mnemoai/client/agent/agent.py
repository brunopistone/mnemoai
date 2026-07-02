"""LangGraph-based agent implementation."""

import operator
import re
import sys
import time
from typing import Annotated, Any, Callable, Dict, List, Optional, Sequence, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from mnemoai.client.agent import message_sanitizer, plan_policy, tool_formatting
from mnemoai.client.agent.orchestrator import (
    get_aggregator_prompt,
    get_orchestrator_prompt,
    parse_subtasks,
)
from mnemoai.client.agent.reasoning_utils import disable_reasoning, restore_reasoning
from mnemoai.client.agent.router import ROUTE_TOOLS, is_trivial_query
from mnemoai.client.ui import turn_view
from mnemoai.utils.config import config
from mnemoai.utils.formatting.code_formatter import CodeFormatter
from mnemoai.utils.logger import logger


class AgentState(TypedDict):
    """State schema for the LangGraph agent."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    thinking: Optional[str]
    route: Optional[str]


class LangGraphAgent:
    """LangGraph-based agent with streaming support."""

    # Task-agnostic "meta" tools bound on EVERY route (incl. no-tools simple_qa),
    # since a matching query can classify onto any route: memory ("remember
    # this"), describe_image (any query may reference an image), fs_read
    # (universal read), use_skill (a skill-matching query may be simple_qa).
    _ALWAYS_AVAILABLE_TOOLS = {"memory", "describe_image", "fs_read", "use_skill"}

    # Aliases keeping the historical class-attribute surface (used by unit tests
    # and the delegating methods) pointing at the single source in plan_policy.
    _PLAN_BLOCKED_TOOLS = plan_policy.PLAN_BLOCKED_TOOLS
    _PLAN_FILE_SUFFIX = plan_policy.PLAN_FILE_SUFFIX
    _READONLY_BASH_CMDS = plan_policy.READONLY_BASH_CMDS
    _BASH_MUTATION_OPS = plan_policy.BASH_MUTATION_OPS
    _READONLY_GIT_SUBCMDS = plan_policy.READONLY_GIT_SUBCMDS
    _BASH_MUTATING_FLAGS = plan_policy.BASH_MUTATING_FLAGS

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
        plan_mode_provider: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Initialize the LangGraph agent.

        Args:
            model: LangChain chat model
            tools: List of LangChain tools
            system_prompt: System prompt for the agent
            verbose: Enable verbose mode for thinking display
            callbacks: Optional streaming callback handlers
            router: Optional QueryRouter for classification
            tool_routes: Optional route-name → tool-name-list mapping
            orchestrator_enabled: Enable the orchestrator for the 'full' route
            plan_mode_provider: Callable returning True while plan mode is active
                (gates the mutating tools client-side).
        """
        self._plan_mode_provider = plan_mode_provider or (lambda: False)
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.callbacks = callbacks or []
        # When True (pinned-input UI), reasoning is buffered into a collapsed
        # "Thought for Ns…" block and tool calls render as a styled name+↳arg
        # block; default False keeps the inline gray reasoning + [⚙ …] marker.
        self.styled_turn_view = False
        self._messages: List[BaseMessage] = []
        self._thinking: Optional[str] = None
        self._code_formatter = CodeFormatter()
        # Categories the user chose to trust this session (the "a = allow" option),
        # skipping re-prompts until restart.
        self._trusted_confirm_categories: set = set()
        self.router = router
        self.orchestrator_enabled = orchestrator_enabled and router is not None
        # Runaway guard on the model<->tool loop, set high (default 200) so real
        # long tasks never hit it; compaction is the actual context limiter.
        self.recursion_limit = config.get("LLM", {}).get("RECURSION_LIMIT", 200)
        # Some endpoints (Bedrock Mantle reasoning models on the Responses API)
        # intermittently return a fully empty response; a retry recovers it.
        self._empty_response_retries = max(
            0, int(config.get("LLM", {}).get("MAX_RETRIES", 2))
        )

        self.model_with_tools = model.bind_tools(tools) if tools else model

        # External (mcp.json) tools aren't in any route allowlist; tracked so the
        # orchestrator can describe them and they can be bound on every route.
        self.external_tools: List[BaseTool] = []

        # Build per-route tool subsets and model bindings.
        self.tools_by_route: Optional[Dict[str, List[BaseTool]]] = None
        self.models_by_route: Optional[Dict[str, BaseChatModel]] = None
        if router and tool_routes:
            self.tools_by_route = {}
            self.models_by_route = {}
            # Meta tools reach every route (incl. simple_qa); excluded from
            # external_tools so the orchestrator doesn't re-describe them.
            always_tools = [t for t in tools if t.name in self._ALWAYS_AVAILABLE_TOOLS]
            # Any tool not in a route allowlist and not a meta tool is external.
            known_names = {
                n for names in tool_routes.values() if names for n in names
            }
            external_tools = [
                t for t in tools
                if t.name not in known_names
                and t.name not in self._ALWAYS_AVAILABLE_TOOLS
            ]
            self.external_tools = external_tools
            for route_name, tool_names in tool_routes.items():
                if tool_names is None:
                    route_tools = tools  # 'full' already binds everything
                elif not tool_names:
                    # simple_qa: meta tools only, plus external tools (which must
                    # stay reachable even here — a factual question lands here).
                    route_tools = list(always_tools) + external_tools
                else:
                    matched = [t for t in tools if t.name in tool_names]
                    route_tools = matched + external_tools + always_tools
                self.tools_by_route[route_name] = route_tools
                self.models_by_route[route_name] = (
                    model.bind_tools(route_tools) if route_tools else model
                )

        self.graph = self._build_graph()

    @property
    def messages(self) -> List[BaseMessage]:
        """The message history."""
        return self._messages

    @messages.setter
    def messages(self, value: List[BaseMessage]) -> None:
        self._messages = value

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph state graph."""
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
        """Route 'full' tasks to the orchestrator, others to the agent.

        A trivial 'full' query (short/signal-free) still goes to ``agent`` —
        decomposing it adds overhead for no gain. Only substantive 'full' tasks
        are decomposed.
        """
        if state.get("route") == "full":
            query = ""
            for msg in reversed(state.get("messages", [])):
                if isinstance(msg, HumanMessage):
                    query = str(msg.content)
                    break
            if not is_trivial_query(query):
                return "orchestrator"
        return "agent"

    def _classify(self, state: AgentState) -> Dict[str, Any]:
        """Classify the query and set the route in state."""
        messages = state["messages"]
        if not messages:
            return {"route": "full"}

        # Conversation context from recent messages (excluding the last).
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
        """Decompose the task into subtasks, run a worker per subtask, aggregate."""
        messages = state["messages"]
        # Extract user query (skip system prompt).
        query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = str(msg.content)
                break

        if not query:
            return {"messages": [AIMessage(content="No query found.")]}

        # Step 1: decompose into subtasks. External tools are described to the
        # decomposer so it can route subtasks needing them to 'full'.
        orchestrator_prompt = get_orchestrator_prompt()
        orchestrator_prompt += self._external_tools_prompt_block()
        logger.debug("Orchestrator: decomposing task")
        subtasks = self._decompose_task(
            query, orchestrator_prompt, set(ROUTE_TOOLS.keys())
        )
        logger.debug(f"Orchestrator: {len(subtasks)} subtasks")

        # Step 2: Execute each subtask with a worker
        worker_results = []
        for i, subtask in enumerate(subtasks):
            desc = subtask["description"]
            category = subtask["category"]

            total = len(subtasks)
            short_desc = desc[:70] + ("..." if len(desc) > 70 else "")
            print(
                f"\n\033[90m[Step {i + 1}/{total}: {short_desc}]\033[0m",
                flush=True,
            )

            # Route-specific model and tools.
            if category == "full" or not self.tools_by_route:
                worker_tools = self.tools
                worker_model = self.model_with_tools
            elif category == "simple_qa":
                worker_tools = []
                worker_model = self.model
            else:
                worker_tools = self.tools_by_route.get(category, self.tools)
                worker_model = self.models_by_route.get(category, self.model_with_tools)

            # Prepend context from previously-completed steps.
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

            # Execute worker (a single worker failing shouldn't abort the
            # whole orchestration — record the error and continue).
            try:
                result, worker_msgs = self._run_worker_loop(
                    worker_model, worker_tools, worker_prompt
                )
            except Exception as e:
                logger.error(f"Worker for subtask {i + 1} failed: {e}")
                self._stop_spinner()
                result = f"(This step could not be completed: {e})"
                worker_msgs = []
            worker_results.append(
                {
                    "task": desc,
                    "category": category,
                    "result": result,
                    "messages": worker_msgs,
                }
            )

        # Collect all intermediate worker messages for conversation saving.
        all_worker_messages: List[BaseMessage] = []
        for wr in worker_results:
            all_worker_messages.extend(wr.get("messages", []))

        # Step 3: aggregate.
        if len(subtasks) == 1:
            final_content = worker_results[0]["result"]
        else:
            print(
                "\n\033[90m[Synthesizing results...]\033[0m",
                flush=True,
            )
            try:
                final_content = self._aggregate_results(
                    query, worker_results, get_aggregator_prompt()
                )
            except Exception as e:
                # If synthesis fails, fall back to concatenating the per-step
                # results so the user still gets the work that was done.
                logger.error(f"Aggregation failed: {e}; concatenating results")
                self._stop_spinner()
                final_content = "\n\n".join(
                    f"### {r['task']}\n{r['result']}" for r in worker_results
                )

        return {"messages": all_worker_messages + [AIMessage(content=final_content)]}

    def _external_tools_prompt_block(self) -> str:
        """Prompt fragment listing external MCP tools for the decomposer.

        The orchestrator prompt only knows built-in categories, so external tools
        are listed here with an instruction to route subtasks needing them to
        'full'. Empty string when there are none.
        """
        if not self.external_tools:
            return ""

        lines = []
        for t in self.external_tools:
            desc = (t.description or "").strip().replace("\n", " ")
            if len(desc) > 120:
                desc = desc[:117] + "..."
            lines.append(f'  - "{t.name}": {desc}' if desc else f'  - "{t.name}"')
        tools_list = "\n".join(lines)
        return (
            "\n\n  <external_tools>\n"
            "  These additional tools are available via category \"full\" "
            "(which binds every tool):\n"
            f"{tools_list}\n"
            "  For any subtask that needs one of these tools, set its category "
            'to "full".\n'
            "  </external_tools>"
        )

    def _decompose_task(
        self, query: str, orchestrator_prompt: str, valid_categories: set
    ) -> List[Dict[str, Any]]:
        """Call the LLM to decompose a task into subtask dicts."""
        messages = [
            SystemMessage(content=orchestrator_prompt),
            HumanMessage(content=query),
        ]

        # Suppress callbacks (keeps the spinner up) and disable reasoning so the
        # JSON subtask list lands in response.content (reasoning models else
        # leave content empty and parsing fails).
        saved_callbacks = getattr(self.model, "callbacks", None)
        saved_reasoning = None
        try:
            self.model.callbacks = None
            saved_reasoning = self._disable_reasoning()
            response = self.model.invoke(messages, config={"callbacks": []})
            return parse_subtasks(response.content, query, valid_categories)
        except Exception as e:
            # A failed decomposition shouldn't crash the turn: fall back to
            # treating the whole query as a single 'full' subtask.
            logger.warning(f"Task decomposition failed: {e}; using single subtask")
            return [{"description": query, "category": "full"}]
        finally:
            if saved_reasoning is not None:
                self._restore_reasoning(saved_reasoning)
            self.model.callbacks = saved_callbacks

    def _run_worker_loop(
        self,
        worker_model,
        worker_tools: List[BaseTool],
        prompt: str,
        max_iterations: int = 10,
    ) -> tuple:
        """Run a worker agent loop (streaming) until completion.

        Returns ``(final_text, worker_messages)`` where ``worker_messages`` holds
        the intermediate AI/Tool messages for saving.
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

            # No tool calls → worker is done.
            if not isinstance(response, AIMessage) or not response.tool_calls:
                visible = self._extract_visible(response.content)
                # Reasoning-only/empty turn: salvage a visible reply so the
                # orchestrator doesn't surface a blank answer.
                if not visible:
                    visible = self._salvage_empty_worker_turn(
                        worker_messages, config, worker_model
                    )
                self._stop_spinner()
                saveable = [
                    m for m in worker_messages if not isinstance(m, SystemMessage)
                ]
                return visible or str(response.content), saveable

            self._stop_spinner()
            if self.verbose:
                for tc in response.tool_calls:
                    print(
                        f"\n\033[90m[⚙ {self._format_tool_call(tc)}]\033[0m\n",
                        flush=True,
                    )
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_id = tc["id"]
                tool_args = self._normalize_tool_args(tc["args"])

                tool = next((t for t in worker_tools if t.name == tool_name), None)
                # Fall back to all tools if not in the worker subset.
                if not tool:
                    tool = next((t for t in self.tools if t.name == tool_name), None)

                if tool and self._is_blocked_by_plan_mode(tool_name, tool_args):
                    worker_messages.append(
                        ToolMessage(
                            content=self._plan_mode_block_message(tool_name),
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )
                elif tool and not self._confirm_tool(tool_name, tool_args):
                    worker_messages.append(
                        ToolMessage(
                            content="User declined to run this command.",
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )
                elif tool:
                    try:
                        logger.debug(f"Worker tool: {tool_name} args: {tool_args}")
                        result = self._invoke_tool(tool, tool_name, tool_args)
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
                                content=self._tool_error_message(tool_name, e),
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
        saveable = [m for m in worker_messages if not isinstance(m, SystemMessage)]
        # Salvage the last visible output and flag the step as truncated.
        partial = self._last_visible_from(worker_messages)
        truncated_note = (
            f"(Step stopped after {max_iterations} tool iterations without "
            "finishing.)"
        )
        result = f"{partial}\n\n{truncated_note}" if partial else truncated_note
        return result, saveable

    def _salvage_empty_worker_turn(
        self, worker_messages: List[BaseMessage], config: dict, worker_model
    ) -> str:
        """Recover a visible answer when a worker turn produced none.

        Mirrors :meth:`_call_model`'s guarantee: retry once with reasoning
        disabled (streamed), else print and return a visible fallback. The
        recovered message is appended to ``worker_messages``. Never returns empty.
        """
        logger.debug("Worker produced no visible content; salvaging.")
        retry_messages = list(worker_messages) + [
            HumanMessage(
                content=(
                    "You provided reasoning but no visible response. "
                    "Please provide your answer."
                )
            )
        ]
        self._start_spinner()
        saved = self._disable_reasoning()
        try:
            retry_response, _ = self._stream_response(
                retry_messages,
                config,
                print_reasoning=False,
                model=worker_model,
                mark_answer=True,
            )
        finally:
            self._restore_reasoning(saved)

        if retry_response is not None:
            visible = self._extract_visible(retry_response.content)
            if visible:
                worker_messages.append(retry_response)
                return visible

        # Still nothing usable: surface a fallback (never a silent turn).
        fallback = (
            "I wasn't able to produce a response for that. "
            "Could you rephrase or give me a bit more detail?"
        )
        self._stop_spinner()
        print(f"\n{fallback}", flush=True)
        worker_messages.append(AIMessage(content=fallback))
        return fallback

    def _aggregate_results(
        self,
        original_query: str,
        worker_results: List[Dict[str, Any]],
        aggregator_prompt: str,
    ) -> str:
        """Aggregate worker results into a final response via the LLM."""
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
        response, _ = self._stream_response(messages, config, mark_answer=True)

        if response is None:
            response = self.model.invoke(messages, config=config)

        self._stop_spinner()
        return self._extract_visible(response.content) or str(response.content)

    def _get_route_model(self, state: AgentState):
        """The model binding for the current route (falls back to all tools)."""
        route = state.get("route")
        if route and self.models_by_route:
            return self.models_by_route.get(route, self.model_with_tools)
        return self.model_with_tools

    def _get_route_tools(self, state: AgentState) -> List[BaseTool]:
        """The tool list for the current route (falls back to all tools)."""
        route = state.get("route")
        if route and self.tools_by_route:
            return self.tools_by_route.get(route, self.tools)
        return self.tools

    @staticmethod
    def _sanitize_tool_pairs(messages: List[BaseMessage]) -> List[BaseMessage]:
        """Delegates to :func:`message_sanitizer.sanitize_tool_pairs`."""
        return message_sanitizer.sanitize_tool_pairs(messages)

    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        """Call the model with the current state, streaming the response."""
        messages = list(state["messages"])

        # Repair orphaned tool call/result pairs — an orphan from earlier history
        # would make the provider reject every subsequent turn.
        messages = self._sanitize_tool_pairs(messages)

        if self.system_prompt and (
            not messages or not isinstance(messages[0], SystemMessage)
        ):
            messages = [SystemMessage(content=self.system_prompt)] + messages

        config = {"callbacks": self.callbacks} if self.callbacks else {}

        # Spin while awaiting the first token (stopped once text/reasoning streams
        # or tools start). Started here so the gap after the last tool result
        # never shows a frozen terminal. Idempotent.
        self._start_spinner()

        active_model = self._get_route_model(state)
        response, had_reasoning = self._stream_response(
            messages, config, model=active_model, mark_answer=True
        )

        if response is None:
            response = active_model.invoke(messages, config=config)

        thinking = self._extract_thinking(response)
        visible = self._extract_visible(response.content)

        # Streaming chunks carry no response_metadata, so a token-truncated turn
        # is invisible while streaming. On an empty streamed turn, do one
        # authoritative non-streaming invoke to get definitive content + metadata.
        if not visible and not response.tool_calls:
            authoritative = active_model.invoke(messages, config=config)
            if authoritative is not None:
                response = authoritative
                thinking = self._extract_thinking(response)
                visible = self._extract_visible(response.content)

        # Turn cut short by the output-token limit before any answer (reasoning
        # ate the whole budget). Checked BEFORE the reasoning-retry — retrying
        # can't help when the budget is the limit. Surface an actionable message.
        if (
            not visible
            and not response.tool_calls
            and self._was_truncated_by_tokens(response)
        ):
            logger.warning(
                "Model response truncated by the output-token limit before any "
                "answer was produced — increase MODEL_ID.MAX_TOKENS (reasoning "
                "models need headroom to reason and answer)."
            )
            truncated = AIMessage(
                content=(
                    "My response was cut off by the output-token limit before I "
                    "could answer. This model reasons before replying, so it "
                    "needs more room — increase `MAX_TOKENS` (e.g. via /params or "
                    "in config.yaml) and try again."
                )
            )
            if thinking:
                truncated.additional_kwargs["reasoning_content"] = thinking
            print("\n", end="", flush=True)
            self._stop_spinner()
            print(truncated.content, flush=True)
            return {"messages": [truncated], "thinking": thinking}

        # Only reasoning, no visible content: retry once with reasoning disabled.
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
                    mark_answer=True,
                )
            finally:
                self._restore_reasoning(saved)

            if retry_response is not None:
                retry_visible = self._extract_visible(retry_response.content)
                if retry_visible or retry_response.tool_calls:
                    if not retry_response.additional_kwargs.get("reasoning_content"):
                        retry_response.additional_kwargs["reasoning_content"] = thinking
                    return {"messages": [retry_response], "thinking": thinking}

            # Both attempts yielded nothing usable: surface a fallback (never silent).
            fallback = AIMessage(
                content=(
                    "I wasn't able to produce a response for that. "
                    "Could you rephrase or give me a bit more detail?"
                )
            )
            fallback.additional_kwargs["reasoning_content"] = thinking
            print("\n", end="", flush=True)
            self._stop_spinner()
            print(fallback.content, flush=True)
            return {"messages": [fallback], "thinking": thinking}

        return {"messages": [response], "thinking": thinking}

    def _stream_response(
        self,
        messages: list,
        config: dict,
        print_reasoning: bool = True,
        model=None,
        mark_answer: bool = False,
    ) -> tuple:
        """Stream a model response, handling spinner and output.

        Retries a completely empty turn (transient endpoint hiccup) up to
        ``_empty_response_retries`` times. ``mark_answer`` prints a marker before
        the answer on user-facing turns. Returns ``(response, had_reasoning)``.
        """
        active_model = model or self.model_with_tools
        attempts = getattr(self, "_empty_response_retries", 0) + 1
        for attempt in range(attempts):
            response, had_reasoning = self._stream_once(
                active_model, messages, config, print_reasoning, mark_answer
            )
            # Retry only a completely empty turn; the reasoning-only case is the
            # caller's responsibility.
            if not self._is_empty_response(response) or attempt == attempts - 1:
                return response, had_reasoning
            logger.debug(
                "Empty model response (attempt %d/%d); retrying",
                attempt + 1,
                attempts,
            )
            self._start_spinner()
        return response, had_reasoning

    def _is_empty_response(self, response) -> bool:
        """True if a response carries no content, no reasoning, no tool calls."""
        if response is None:
            return True
        if getattr(response, "tool_calls", None):
            return False
        if self._extract_visible(response.content):
            return False
        if self._extract_thinking(response):
            return False
        return True

    def _stream_once(
        self,
        active_model,
        messages: list,
        config: dict,
        print_reasoning: bool = True,
        mark_answer: bool = False,
    ) -> tuple:
        """Single streaming attempt (see _stream_response for the retry wrapper)."""
        self._code_formatter = CodeFormatter()
        first_token = True
        had_reasoning = False
        answer_marker_printed = False
        building_tool_args = False
        response = None
        # Styled-turn-view (pinned UI): buffer reasoning; also feed a live sink
        # (if wired) so it streams in the transient region, then commits.
        styled = getattr(self, "styled_turn_view", False)
        sink = getattr(self, "reasoning_sink", None) if styled else None
        reasoning_buf: List[str] = []
        reasoning_started = None

        try:
            for chunk in active_model.stream(messages, config=config):
                chunk_content, reasoning_content = self._extract_content(chunk)

                # Stop the spinner only for what's displayed NOW. Styled mode
                # buffers reasoning (nothing shows until the answer), so reasoning
                # there must keep the spinner running — else a dead pause.
                will_show_reasoning = bool(
                    reasoning_content and self.verbose and print_reasoning
                    and not styled
                )
                if (
                    first_token
                    and (chunk_content or will_show_reasoning)
                    and self.callbacks
                ):
                    first_token = False
                    self._stop_spinner()

                if reasoning_content and self.verbose and print_reasoning:
                    if styled:
                        # Buffer for the final block; also feed the live sink.
                        if reasoning_started is None:
                            reasoning_started = time.time()
                            if sink is not None:
                                sink.start(time.monotonic())
                        reasoning_buf.append(reasoning_content)
                        if sink is not None:
                            sink.append(reasoning_content)
                    else:
                        print(
                            f"\033[90m{reasoning_content}\033[0m", end="", flush=True
                        )
                    had_reasoning = True

                # Tool-arg chunks carry no visible content — a long silent stretch
                # for a big arg. Re-raise a "Preparing …" spinner so it doesn't
                # look frozen until the tool marker prints.
                if (
                    not chunk_content
                    and self.callbacks
                    and getattr(chunk, "tool_call_chunks", None)
                    and not building_tool_args
                ):
                    building_tool_args = True
                    self._start_spinner("Preparing tool call")

                if chunk_content:
                    # Real answer text after tool-arg chunks (interleaved) — drop
                    # the preparing spinner first.
                    if building_tool_args:
                        building_tool_args = False
                        self._stop_spinner()
                    if not answer_marker_printed:
                        if styled and reasoning_buf:
                            self._flush_reasoning_block(
                                reasoning_buf, reasoning_started
                            )
                            reasoning_buf = []
                            had_reasoning = False
                            print("", flush=True)
                            chunk_content = chunk_content.lstrip("\n")
                        elif had_reasoning:
                            # Separate gray reasoning above from the answer.
                            print("\n\n", end="", flush=True)
                            had_reasoning = False
                            chunk_content = chunk_content.lstrip("\n")
                        if mark_answer:
                            # Prepend the marker to the first answer chunk so it's
                            # part of the SAME committed line — a separate flushed
                            # marker write is a lone partial line the pinned UI
                            # repaint erases before the answer joins it.
                            chunk_content = self._answer_marker() + chunk_content
                        answer_marker_printed = True
                    self._code_formatter.process_chunk(chunk_content)

                response = chunk if response is None else response + chunk

            # Flush the formatter so a trailing backtick / unclosed code fence is
            # still emitted and the terminal color is reset.
            if answer_marker_printed:
                self._code_formatter.flush()
                # Terminate the answer line NOW. patch_stdout only commits a line
                # on a newline; without this the streamed answer sits uncommitted
                # in the buffer and a pinned-UI repaint (during the post-turn work
                # before [Context]) erases it. Styled/pinned mode only.
                if styled:
                    print(flush=True)
            # Reasoning with no following content (model went straight to a tool
            # call): still show the block so the thinking isn't lost.
            if styled and reasoning_buf:
                self._flush_reasoning_block(reasoning_buf, reasoning_started)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # Don't lose the turn on a stream error: retry once non-streaming and
            # prefer its complete result; keep the partial only if that yields none.
            logger.warning(f"Streaming error: {e}; falling back to non-streaming")
            self._stop_spinner()
            try:
                full = active_model.invoke(messages, config=config)
                if full is not None:
                    response = full
            except Exception as e2:
                logger.error(f"Non-streaming fallback also failed: {e2}")
        finally:
            # Never leave a lingering transient block (cancel / error / no flush).
            if sink is not None:
                sink.stop()

        return response, had_reasoning

    @staticmethod
    def _answer_marker() -> str:
        """The cyan ● prefix for a streamed answer (prepended to the first chunk)."""
        return "\033[36m●\033[0m "

    def _flush_reasoning_block(self, parts: list, started: Optional[float]) -> None:
        """Commit the collapsed 'Thought for Ns…' block to scrollback and clear
        the live sink; no-op if empty."""
        sink = getattr(self, "reasoning_sink", None)
        if sink is not None:
            sink.stop()  # clear the transient view before the block commits
        text = "".join(parts).strip()
        if not text:
            return
        seconds = (time.time() - started) if started is not None else 0.0
        block = turn_view.render_reasoning_block(text, seconds)
        if block:
            print(block, flush=True)

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

    def _start_spinner(self, label: str = "Thinking") -> None:
        """Restart the spinner (with ``label``) and reset the first-token flag."""
        for cb in self.callbacks:
            if hasattr(cb, "spinner") and cb.spinner:
                lock = getattr(cb, "spinner_lock", None)
                if lock:
                    with lock:
                        cb.spinner.start(label)
                        if hasattr(cb, "first_token_received"):
                            cb.first_token_received = False
                else:
                    cb.spinner.start(label)
                    if hasattr(cb, "first_token_received"):
                        cb.first_token_received = False

    def _tool_progress_label(self, tool_name: str, tool_args: dict) -> str:
        """A short 'still working' label shown while a slow tool runs."""
        if tool_name == "execute_bash":
            cmd = str(tool_args.get("command", "")).strip().replace("\n", " ")
            if len(cmd) > 50:
                cmd = cmd[:47] + "…"
            return f"Running: {cmd}" if cmd else "Running command"
        if tool_name in ("fs_write", "file_edit"):
            path = str(tool_args.get("path", "")).strip()
            return f"Writing {path}" if path else "Writing file"
        labels = {
            "web_search": "Searching the web",
            "describe_image": "Analyzing image",
            "start_background_task": "Starting background task",
        }
        return labels.get(tool_name, f"Running {tool_name}")

    # Tools that print their OWN live progress to the terminal (e.g. crawl4ai's
    # [INIT]/[FETCH]/[SCRAPE] lines, emitted on stderr by the web_crawler
    # subprocess). Animating our spinner over them collides on the same lines —
    # so for these we keep the spinner stopped and let the tool's output show.
    _SELF_REPORTING_TOOLS = {"web_crawler"}

    def _invoke_tool(self, tool, tool_name: str, tool_args: dict):
        """Invoke a tool with a progress spinner, unless it's self-reporting
        (``_SELF_REPORTING_TOOLS`` emit their own progress → spinner left off)."""
        if tool_name in self._SELF_REPORTING_TOOLS:
            self._stop_spinner()
            return tool.invoke(tool_args)

        self._start_spinner(self._tool_progress_label(tool_name, tool_args))
        try:
            return tool.invoke(tool_args)
        finally:
            self._stop_spinner()

    @staticmethod
    def _reasoning_summary_text(block: dict) -> str:
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

    def _extract_thinking(self, response) -> Optional[str]:
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
                elif block.get("type") == "reasoning":
                    parts.append(self._reasoning_summary_text(block))
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

    @staticmethod
    def _was_truncated_by_tokens(response) -> bool:
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

    def _extract_visible(self, content) -> str:
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

    def _disable_reasoning(self) -> dict:
        """Temporarily disable reasoning; returns saved state for _restore."""
        return disable_reasoning(self.model)

    def _restore_reasoning(self, saved: dict) -> None:
        """Restore reasoning settings saved by _disable_reasoning()."""
        restore_reasoning(self.model, saved)

    def _extract_content(self, chunk) -> tuple[str, str]:
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
                    elif block_type == "reasoning":
                        reasoning_content += self._reasoning_summary_text(block)
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

    @staticmethod
    def _elide_middle(text: str, limit: int = 72) -> str:
        """Delegates to :func:`tool_formatting.elide_middle`."""
        return tool_formatting.elide_middle(text, limit)

    @staticmethod
    def _format_tool_call(tool_call: dict) -> str:
        """Delegates to :func:`tool_formatting.format_tool_call`."""
        return tool_formatting.format_tool_call(tool_call)

    @staticmethod
    def _normalize_tool_args(args: Any) -> Any:
        """Delegates to :func:`tool_formatting.normalize_tool_args`."""
        return tool_formatting.normalize_tool_args(args)

    def _is_blocked_by_plan_mode(self, tool_name: str, tool_args: dict = None) -> bool:
        """True when plan mode is active and this tool/call would mutate.

        Thin wrapper over :func:`plan_policy.is_blocked_by_plan_mode` that reads
        the live plan-mode flag from ``self._plan_mode_provider`` (the MCP server
        can't see client state, so this is enforced client-side at the tool
        chokepoints).
        """
        return plan_policy.is_blocked_by_plan_mode(
            tool_name, tool_args, plan_active=self._plan_mode_provider()
        )

    @staticmethod
    def _is_readonly_bash(command: str) -> bool:
        """Delegates to :func:`plan_policy.is_readonly_bash`."""
        return plan_policy.is_readonly_bash(command)

    @staticmethod
    def _is_plan_file(path: str) -> bool:
        """Delegates to :func:`plan_policy.is_plan_file`."""
        return plan_policy.is_plan_file(path)

    @staticmethod
    def _plan_mode_block_message(tool_name: str) -> str:
        """Delegates to :func:`plan_policy.plan_mode_block_message`."""
        return plan_policy.plan_mode_block_message(tool_name)

    @staticmethod
    def _tool_error_message(tool_name: str, exc: Exception) -> str:
        """Delegates to :func:`tool_formatting.tool_error_message`."""
        return tool_formatting.tool_error_message(tool_name, exc)

    # Tools gated by a hard confirmation prompt, keyed by category.
    _CONFIRM_BASH_TOOLS = {"execute_bash"}
    _CONFIRM_WRITE_TOOLS = {"fs_write", "file_edit"}
    _CONFIRM_MEMORY_TOOLS = {"memory"}

    def _confirm_tool(self, tool_name: str, tool_args: dict) -> bool:
        """Ask the user to approve a destructive tool before it runs.

        Returns True to proceed. Gates shell (``execute_bash``), file writes
        (``fs_write``/``file_edit``), and memory writes, each behind its
        ``REQUIRE_*`` toggle; every other tool proceeds. Enforced client-side (the
        MCP subprocess can't prompt); non-TTY runs auto-proceed.
        """
        if tool_name in self._CONFIRM_BASH_TOOLS:
            category, toggle, toggle_default, header, detail = (
                "bash",
                "REQUIRE_BASH_CONFIRMATION",
                True,
                "▶ Run shell command?",
                tool_args.get("command", ""),
            )
        elif tool_name in self._CONFIRM_WRITE_TOOLS:
            # fs_write previews with dry_run=True (no actual write) before the
            # real call — only gate the write itself, not the harmless preview.
            if tool_args.get("dry_run") is True:
                return True
            path = tool_args.get("path", "")
            op = tool_args.get("command", "edit")  # fs_write: create/str_replace/…
            category, toggle, toggle_default, header, detail = (
                "write",
                "REQUIRE_WRITE_CONFIRMATION",
                True,
                "▶ Write to file?",
                f"{op} {path}".strip(),
            )
        elif tool_name in self._CONFIRM_MEMORY_TOOLS:
            # Only the write actions touch the file; a bad/read action proceeds.
            action = (tool_args.get("action") or "").strip().lower()
            if action not in ("add", "replace", "remove"):
                return True
            text = tool_args.get("text") or tool_args.get("old_text") or ""
            category, toggle, toggle_default, header, detail = (
                "memory",
                "REQUIRE_MEMORY_CONFIRMATION",
                False,
                "▶ Update memory?",
                f"{action}: {text[:60]}",
            )
        else:
            return True

        if not config.get(toggle, toggle_default):
            return True
        # Already trusted this session (user answered "a" earlier).
        trusted = getattr(self, "_trusted_confirm_categories", None)
        if trusted is not None and category in trusted:
            return True
        if not sys.stdin.isatty():
            return True  # non-interactive: can't prompt, don't block

        self._stop_spinner()

        # The pinned-input UI installs a `_confirm_ui` hook (in-app y/N/a keypress
        # → yes|no|all) since a plain input() would fight the live app for stdin.
        # Absent (plain loop / unit-test bare object) → legacy print()+input().
        confirm_ui = getattr(self, "_confirm_ui", None)
        if confirm_ui is not None:
            answer = confirm_ui(header, detail, category)
            if answer == "all":
                self._trusted_confirm_categories.add(category)
                return True
            return answer == "yes"

        # "a" = allow this whole category for the rest of the session.
        print(f"\n\033[93m{header}\033[0m\n  \033[1m{detail}\033[0m")
        try:
            answer = input("  Proceed? (y/N/a=allow all this session): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            answer = ""
        if answer in ("a", "all", "always"):
            if not hasattr(self, "_trusted_confirm_categories"):
                self._trusted_confirm_categories = set()
            self._trusted_confirm_categories.add(category)
            return True
        return answer in ("y", "yes")

    def _execute_tools(self, state: AgentState) -> Dict[str, Any]:
        """Execute the tool calls on the last AI message."""
        last_message = state["messages"][-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": []}

        self._stop_spinner()

        route_tools = self._get_route_tools(state)

        # Visible tool marker so reasoning before/after a call is separated. In
        # styled mode (pinned UI) render the name + ↳arg block, else the [⚙ …] one.
        if self.verbose:
            for tool_call in last_message.tool_calls:
                if getattr(self, "styled_turn_view", False):
                    print(
                        "\n"
                        + turn_view.render_tool_call(
                            tool_call.get("name", "tool"),
                            self._normalize_tool_args(tool_call.get("args") or {}),
                        )
                        + "\n",
                        flush=True,
                    )
                else:
                    print(
                        f"\n\033[90m[⚙ {self._format_tool_call(tool_call)}]\033[0m\n",
                        flush=True,
                    )

        tool_results = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = self._normalize_tool_args(tool_call["args"])
            tool_id = tool_call["id"]

            # Route tools first, then fall back to all tools.
            tool = next((t for t in route_tools if t.name == tool_name), None)
            if not tool:
                tool = next((t for t in self.tools if t.name == tool_name), None)

            if tool:
                # Plan mode: hard-block mutating/exec tools (read-only planning).
                if self._is_blocked_by_plan_mode(tool_name, tool_args):
                    tool_results.append(
                        ToolMessage(
                            content=self._plan_mode_block_message(tool_name),
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )
                    continue
                # Hard gate: confirm destructive tools before running.
                if not self._confirm_tool(tool_name, tool_args):
                    tool_results.append(
                        ToolMessage(
                            content="User declined to run this command.",
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )
                    continue
                try:
                    logger.debug(f"Executing tool: {tool_name} with args: {tool_args}")
                    result = self._invoke_tool(tool, tool_name, tool_args)
                    tool_results.append(
                        ToolMessage(
                            content=str(result), tool_call_id=tool_id, name=tool_name
                        )
                    )
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    tool_results.append(
                        ToolMessage(
                            content=self._tool_error_message(tool_name, e),
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

        self._start_spinner()

        return {"messages": tool_results}

    def _should_continue(self, state: AgentState) -> str:
        """"continue" if the last AI message has tool calls, else "end"."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        return "end"

    def __call__(self, prompt: str) -> str:
        """Invoke the agent with a prompt."""
        return self.invoke(prompt)

    # Ephemeral per-turn reminder blocks (e.g. the plan-mode banner) the client
    # prepends: sent to the model this turn but stripped before storage, so a
    # reloaded conversation never carries a stale "plan mode is active" banner.
    _EPHEMERAL_BLOCK_RE = re.compile(
        r"<plan-mode-active>.*?</plan-mode-active>\s*", re.DOTALL
    )

    @classmethod
    def _strip_ephemeral(cls, text: str) -> str:
        """Remove ephemeral per-turn reminder blocks from a prompt for storage."""
        return cls._EPHEMERAL_BLOCK_RE.sub("", text)

    def invoke(self, prompt: str) -> str:
        """Invoke the agent with a prompt; returns the response string."""
        # Model sees the full prompt this turn; only the clean prompt is stored.
        stored_prompt = self._strip_ephemeral(prompt)
        self._messages.append(HumanMessage(content=stored_prompt))

        # The turn the model runs on: clean history + the full current prompt.
        model_messages = self._messages[:-1] + [HumanMessage(content=prompt)]
        initial_state: AgentState = {
            "messages": model_messages,
            "thinking": None,
            "route": None,
        }

        if self.system_prompt:
            initial_state["messages"] = [
                SystemMessage(content=self.system_prompt)
            ] + list(initial_state["messages"])

        # recursion_limit is a runaway guard; hitting it means a likely stuck loop.
        try:
            result = self.graph.invoke(
                initial_state, config={"recursion_limit": self.recursion_limit}
            )
        except GraphRecursionError:
            logger.warning(
                "Agent stopped after the safety step limit (%d); the task may be "
                "looping. Returning the work so far — raise LLM.RECURSION_LIMIT "
                "if a legitimate task needs more steps.",
                self.recursion_limit,
            )
            self._stop_spinner()
            partial = self._last_visible_from(self._messages)
            return partial or (
                "I reached my safety step limit while working on that and "
                "couldn't finish. Try narrowing the request, or raise "
                "LLM.RECURSION_LIMIT in config if the task legitimately needs "
                "more steps."
            )

        final_messages = result["messages"]
        self._thinking = result.get("thinking")

        # Keep only the NEW assistant/tool messages. Skip System/Human — the user
        # turn was already stored as the clean prompt, so the reminder-bearing
        # HumanMessage the model ran on must not be re-added.
        new_messages = [
            m
            for m in final_messages
            if not isinstance(m, (SystemMessage, HumanMessage))
            and m not in self._messages
        ]
        self._messages.extend(new_messages)

        # Prefer the most recent AI turn with visible text.
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                visible = self._extract_visible(msg.content)
                if visible:
                    return visible

        # Empty final turn: salvage the last tool result rather than return "".
        last_tool = self._last_tool_result(final_messages)
        if last_tool:
            return f"The last tool reported:\n{last_tool}"
        return (
            "I wasn't able to produce a response for that. Could you rephrase "
            "or give me a bit more detail?"
        )

    def _last_visible_from(self, messages: List[BaseMessage]) -> str:
        """Most recent visible AI text (to salvage a cut-short answer), or ""."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                visible = self._extract_visible(msg.content)
                if visible:
                    return visible
        return ""

    def _last_tool_result(self, messages: List[BaseMessage]) -> str:
        """Most recent ToolMessage content (trimmed to 500 chars), or "" — to
        salvage something useful when the model ends on an empty post-tool turn."""
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                text = str(msg.content).strip()
                if text:
                    return text[:500]
        return ""

    def get_thinking(self) -> Optional[str]:
        """The thinking content from the last response, or None."""
        return self._thinking

    def clear_messages(self) -> None:
        """Clear the message history."""
        self._messages.clear()
        self._thinking = None

