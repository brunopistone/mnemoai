"""LangGraph-based agent implementation."""

import operator
import os
import re
import sys
from pathlib import Path
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

from mnemoai.client.agent.orchestrator import (
    get_aggregator_prompt,
    get_orchestrator_prompt,
    parse_subtasks,
)
from mnemoai.client.agent.reasoning_utils import disable_reasoning, restore_reasoning
from mnemoai.client.agent.router import ROUTE_TOOLS, is_trivial_query
from mnemoai.utils.config import config
from mnemoai.utils.formatting.code_formatter import CodeFormatter
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import plans_dir


class AgentState(TypedDict):
    """State schema for the LangGraph agent."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    thinking: Optional[str]
    route: Optional[str]


class LangGraphAgent:
    """LangGraph-based agent with streaming support."""

    # Task-agnostic "meta" tools bound on EVERY route (incl. the no-tools
    # simple_qa route), regardless of how the query was classified:
    #   - memory: a "remember this" request classifies as simple_qa.
    #   - describe_image: an image can be referenced in any kind of query
    #     ("what's in this image?" classifies as simple_qa/knowledge), and the
    #     vision tool must be reachable there — otherwise the model falls back to
    #     reading a binary file as text.
    #   - fs_read: read-only and universal (handles every format via its mode).
    #     A user can reference a file in ANY query ("what's in config.yaml?"
    #     classifies as simple_qa/knowledge), so reading must never be gated out.
    #   - use_skill: loads an authored skill's instructions on demand. A request
    #     matching a skill can classify as simple_qa (e.g. "write a commit
    #     message"), so the loader must reach every route or the skill could
    #     never trigger (same reasoning as external MCP tools, see 0.8.21).
    _ALWAYS_AVAILABLE_TOOLS = {"memory", "describe_image", "fs_read", "use_skill"}

    # Mutating / shell-executing tools hard-blocked while plan mode is active
    # Read-only tools (fs_read, glob/grep search, web, document readers) and
    # the `memory` notebook are allowed.
    # NOTE: execute_bash, fs_write, and file_edit are blocked CONDITIONALLY (see
    # _is_blocked_by_plan_mode): read-only bash and writes to the plan file are
    # permitted; everything else here is an unconditional block.
    _PLAN_BLOCKED_TOOLS = {
        "execute_bash",
        "fs_write",
        "file_edit",
        "git_safe",
        "git_commit_safe",
        "start_background_task",
    }

    # In plan mode, fs_write/file_edit are allowed ONLY for the plan file (a
    # single writable path under the plans dir. Everything else writing is blocked.
    _PLAN_FILE_SUFFIX = ".md"

    # Read-only shell commands permitted in plan mode (the leading program must
    # be one of these AND the command must contain no shell operator that could
    # chain a mutation — see _is_readonly_bash). Conservative on purpose: when in
    # doubt, the command is treated as NOT read-only and blocked.
    _READONLY_BASH_CMDS = {
        "ls", "cat", "head", "tail", "less", "more", "pwd", "echo", "find",
        "grep", "rg", "egrep", "fgrep", "wc", "stat", "file", "tree", "du",
        "df", "which", "type", "whoami", "hostname", "date", "env", "printenv",
        "ps", "uname", "id", "diff", "sort", "uniq", "cut", "awk", "sed",
        "realpath", "readlink", "basename", "dirname", "git",
    }
    # Shell operators that could append/redirect a mutation onto a read-only
    # command. Their presence forces the command to be treated as non-read-only.
    _BASH_MUTATION_OPS = (">", ">>", "|", ";", "&&", "||", "`", "$(", "&")
    # git subcommands that are unambiguously read-only (others — commit/push/
    # checkout/tag/stash/config-set/branch -d… — can mutate, so are blocked).
    _READONLY_GIT_SUBCMDS = {
        "status", "log", "diff", "show", "rev-parse", "describe", "blame",
        "ls-files", "ls-tree", "cat-file", "shortlog",
    }
    # Per-program flags that turn an otherwise-read-only allowlisted command into
    # a mutating one. `-i` is program-specific (sed: in-place edit; but grep/ls
    # `-i` are read-only), so it's keyed by program rather than global.
    _BASH_MUTATING_FLAGS = {
        "sed": ("-i", "--in-place"),  # also matches `-i.bak` (prefix check)
        "find": ("-delete", "-exec", "-execdir", "-fprint", "-fprintf", "-fls"),
        "awk": ("-i",),  # gawk -i inplace
    }

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
            callbacks: Optional list of callback handlers for streaming
            router: Optional QueryRouter for query classification
            tool_routes: Optional dict mapping route names to tool name lists
            orchestrator_enabled: Enable orchestrator for 'full' route
            plan_mode_provider: Optional callable returning True while the user
                has plan mode active; gates the mutating tools client-side.
        """
        self._plan_mode_provider = plan_mode_provider or (lambda: False)
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
        # Bound on the model<->tool loop. Claude Code has no hard step cap —
        # context compaction is the real limiter and the loop runs until the
        # model stops calling tools. LangGraph requires a finite recursion_limit
        # (its runaway guard), so we set it high enough that legitimate long
        # tasks never hit it; compaction keeps context in check. Configurable.
        self.recursion_limit = config.get("LLM", {}).get("RECURSION_LIMIT", 200)
        # Some endpoints (notably Bedrock Mantle reasoning models on the
        # Responses API) intermittently return a completely empty response
        # — no content, no reasoning, no tool calls. A retry reliably recovers
        # it. Bounded by LLM.MAX_RETRIES (default 2 extra attempts).
        self._empty_response_retries = max(
            0, int(config.get("LLM", {}).get("MAX_RETRIES", 2))
        )

        self.model_with_tools = model.bind_tools(tools) if tools else model

        # External (mcp.json) tools aren't named in any route allowlist, so
        # they'd be filtered out of every specific route. Treat any tool not
        # referenced by a route as external. Kept as an attribute so the
        # orchestrator can describe them when decomposing tasks.
        self.external_tools: List[BaseTool] = []

        # Build per-route tool subsets and model bindings
        self.tools_by_route: Optional[Dict[str, List[BaseTool]]] = None
        self.models_by_route: Optional[Dict[str, BaseChatModel]] = None
        if router and tool_routes:
            self.tools_by_route = {}
            self.models_by_route = {}
            # Meta tools are task-agnostic and must be reachable on EVERY route,
            # including the no-tools 'simple_qa' route. The 'memory' tool is one:
            # a "remember this" request classifies as simple_qa, so without this
            # it could never be called. These are excluded from external_tools so
            # the orchestrator doesn't redundantly describe them.
            always_tools = [t for t in tools if t.name in self._ALWAYS_AVAILABLE_TOOLS]
            # Any tool not named in a route allowlist AND not a meta tool is
            # external (mcp.json). Append it to non-empty specific routes so
            # configured MCP tools are reachable, not just via 'full'.
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
                    # e.g. simple_qa: built-in meta tools only — BUT external
                    # (mcp.json) tools are user-configured capabilities and must
                    # stay reachable even here. A short factual question ("what
                    # time is it in Singapore?") classifies as simple_qa, so
                    # without this an external server like `time` would be
                    # invisible on the very route such questions land in.
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

        A 'full' classification still goes to the normal call_model ``agent``
        path when the query is trivial (short, signal-free chit-chat or a
        clarification): orchestrating it into a single subtask adds overhead and
        previously could surface a blank answer, while ``agent`` binds the same
        tools and has the empty-turn safety net. Only substantive 'full' tasks
        are actually decomposed.

        Args:
            state: Current agent state

        Returns:
            "orchestrator" or "agent"
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
        messages = state["messages"]
        # Extract user query (skip system prompt)
        query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = str(msg.content)
                break

        if not query:
            return {"messages": [AIMessage(content="No query found.")]}

        # Step 1: Decompose task into subtasks. If external (mcp.json) tools are
        # configured, tell the decomposer about them so it can deliberately
        # route subtasks needing them to the 'full' category (which binds every
        # tool) — otherwise the decomposer, unaware they exist, can't target them.
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

        # Collect all intermediate worker messages for conversation saving
        all_worker_messages: List[BaseMessage] = []
        for wr in worker_results:
            all_worker_messages.extend(wr.get("messages", []))

        # Step 3: Aggregate results
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

        The orchestrator prompt only knows the built-in categories. External
        (mcp.json) tools don't belong to any of them, so without this the
        decomposer can't deliberately route a subtask toward one. We list the
        external tool names + descriptions and instruct it to use the 'full'
        category for subtasks that need them ('full' binds every tool). Empty
        string when there are no external tools, so the prompt is unchanged.
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
        """Call the LLM to decompose a task into subtasks.

        Args:
            query: The user's original query
            orchestrator_prompt: System prompt for decomposition
            valid_categories: Set of valid category names

        Returns:
            List of subtask dicts
        """
        messages = [
            SystemMessage(content=orchestrator_prompt),
            HumanMessage(content=query),
        ]

        # Suppress callbacks to keep spinner running, and disable reasoning
        # so the JSON subtask list lands in response.content (reasoning
        # models otherwise leave content empty and parsing fails).
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
        """Execute a worker agent loop with streaming until completion.

        Args:
            worker_model: Model with route-specific tools bound
            worker_tools: Tools available to this worker
            prompt: The worker's task prompt
            max_iterations: Safety limit for agent loop

        Returns:
            Tuple of (final_text, worker_messages) where worker_messages
            contains all intermediate AI and Tool messages for saving
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
                visible = self._extract_visible(response.content)
                # Reasoning-only / empty turn: nothing was streamed to the
                # screen, so the orchestrator would surface a blank answer.
                # Salvage a visible reply (same guarantee as _call_model).
                if not visible:
                    visible = self._salvage_empty_worker_turn(
                        worker_messages, config, worker_model
                    )
                self._stop_spinner()
                # Return non-system messages for conversation saving
                saveable = [
                    m for m in worker_messages if not isinstance(m, SystemMessage)
                ]
                return visible or str(response.content), saveable

            # Execute tools
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
                # Fall back to all tools if not found in worker subset
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
        saveable = [m for m in worker_messages if not isinstance(m, SystemMessage)]
        # Salvage the last visible output rather than discarding it behind a
        # generic "completed" string, and flag that the step was truncated.
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

        A worker can finish with no tool calls and no visible text — typically a
        reasoning model that streamed only hidden thinking on a trivial prompt.
        Nothing was streamed to the screen, so the orchestrator would otherwise
        surface a blank answer. Mirror :meth:`_call_model`'s guarantee: retry
        once with reasoning disabled (streamed, so the answer prints), and if
        that still yields nothing, print and return a visible fallback. The
        recovered message is appended to ``worker_messages`` so it's saved.

        Returns the visible answer text (never empty).
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

        # Still nothing usable: surface a fallback so the turn is never silent.
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
        response, _ = self._stream_response(messages, config, mark_answer=True)

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

        # Spin while we wait for the model's first token (it's stopped as soon as
        # visible text/reasoning streams, or when tools start). Started here —
        # not left to the predecessor node — so the gap between the last tool
        # result and the final answer never shows a frozen terminal. Idempotent:
        # Spinner.start() no-ops if it's already running. Mirrors _aggregate().
        self._start_spinner()

        active_model = self._get_route_model(state)
        response, had_reasoning = self._stream_response(
            messages, config, model=active_model, mark_answer=True
        )

        if response is None:
            response = active_model.invoke(messages, config=config)

        thinking = self._extract_thinking(response)
        visible = self._extract_visible(response.content)

        # The streaming path yields chunks with no response_metadata, so a
        # token-truncated turn (status=incomplete / finish_reason=length) is
        # invisible while streaming. When the streamed turn comes back empty
        # (no text, no tool call), do ONE authoritative non-streaming invoke to
        # get definitive content + metadata before deciding what went wrong.
        if not visible and not response.tool_calls:
            authoritative = active_model.invoke(messages, config=config)
            if authoritative is not None:
                response = authoritative
                thinking = self._extract_thinking(response)
                visible = self._extract_visible(response.content)

        # Turn cut short by the output-token limit before any answer (common
        # with reasoning models on a low MAX_TOKENS: reasoning consumes the
        # whole budget). Checked BEFORE the reasoning-retry below — retrying
        # can't help when the budget itself is the limit, and would just
        # truncate again. Surface an actionable message, not a silent turn.
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

            # Both attempts yielded no usable output: surface a fallback so the
            # user never sees a silent turn.
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
        """Stream model response, handling spinner and output.

        Args:
            messages: Messages to send to the model
            config: LangChain config dict
            print_reasoning: Whether to print reasoning in gray
            model: Optional model override (defaults to self.model_with_tools)
            mark_answer: Print a marker before the answer when no reasoning is
                shown, so it's visually distinct from the user's prompt. Set on
                user-facing answer turns; left off for worker streams (which
                already carry a `[Step N/N]` header).

        Returns:
            Tuple of (response, had_reasoning)
        """
        active_model = model or self.model_with_tools
        attempts = getattr(self, "_empty_response_retries", 0) + 1
        for attempt in range(attempts):
            response, had_reasoning = self._stream_once(
                active_model, messages, config, print_reasoning, mark_answer
            )
            # Retry only a *completely* empty turn (no content, no reasoning,
            # no tool calls) — a transient endpoint hiccup, not a real answer.
            # The reasoning-only case is handled separately by the caller.
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
        response = None

        try:
            for chunk in active_model.stream(messages, config=config):
                chunk_content, reasoning_content = self._extract_content(chunk)

                # Stop the spinner only when something is about to be DISPLAYED:
                # visible answer text, or reasoning we'll actually print. Some
                # models reasoning chunks whose text we never show — stopping
                # on those leaves a dead pause (spinner gone, nothing printed)
                # until the answer arrives. Keep spinning through hidden reasoning.
                will_show_reasoning = bool(
                    reasoning_content and self.verbose and print_reasoning
                )
                if (
                    first_token
                    and (chunk_content or will_show_reasoning)
                    and self.callbacks
                ):
                    first_token = False
                    self._stop_spinner()

                if reasoning_content and self.verbose and print_reasoning:
                    print(f"\033[90m{reasoning_content}\033[0m", end="", flush=True)
                    had_reasoning = True

                if chunk_content:
                    if not answer_marker_printed:
                        if had_reasoning:
                            # Reasoning already printed (gray) above — separate
                            # it from the answer with a blank line.
                            print("\n\n", end="", flush=True)
                            had_reasoning = False
                            chunk_content = chunk_content.lstrip("\n")
                        if mark_answer:
                            # Marker before the first answer chunk so the reply
                            # is visually distinct from the prompt (and from the
                            # gray reasoning). The answer continues on the same
                            # line, after the marker.
                            self._print_answer_marker()
                        answer_marker_printed = True
                    self._code_formatter.process_chunk(chunk_content)

                response = chunk if response is None else response + chunk

            # Stream finished cleanly: flush the formatter so a trailing
            # backtick or a response that ended inside an unclosed code fence is
            # still emitted (not silently dropped) and the terminal color reset.
            if answer_marker_printed:
                self._code_formatter.flush()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # A streaming error shouldn't lose the whole turn. Any partial
            # `response` accumulated before the error is, by definition,
            # truncated — it may be missing content or tool_calls. So always
            # retry once with a single non-streaming call and prefer its
            # complete, authoritative result; only keep the partial if the
            # non-streaming retry yields nothing. (Without this, a mid-stream
            # parse failure could surface as an empty/incomplete turn.)
            logger.warning(f"Streaming error: {e}; falling back to non-streaming")
            self._stop_spinner()
            try:
                full = active_model.invoke(messages, config=config)
                if full is not None:
                    response = full
            except Exception as e2:
                logger.error(f"Non-streaming fallback also failed: {e2}")

        return response, had_reasoning

    def _print_answer_marker(self) -> None:
        """Print a subtle marker before a streamed answer.

        A small cyan bullet makes the assistant's reply visually distinct from
        the user's prompt (and from any gray reasoning printed above it). The
        answer continues on the same line, right after the marker.
        """
        # Cyan ● bullet + a trailing space; no newline so the answer follows it.
        print("\033[36m●\033[0m ", end="", flush=True)

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
        """Restart the spinner and reset first token flag.

        Args:
            label: Text shown next to the animated glyph (e.g. "Running command"
                while a tool executes), so a slow tool never looks stuck.
        """
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
        """A short 'still working' label shown while a tool runs.

        Keeps the user informed during a slow ``tool.invoke()`` (e.g. executing
        Python, a long shell command, a web fetch) so it never looks stuck while
        it's just waiting for completion.
        """
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
            "web_crawl": "Fetching web page",
            "describe_image": "Analyzing image",
            "start_background_task": "Starting background task",
        }
        return labels.get(tool_name, f"Running {tool_name}")

    def _invoke_tool(self, tool, tool_name: str, tool_args: dict):
        """Invoke a tool while showing a progress spinner, then stop it.

        The spinner animates with a per-tool label for the duration of the call
        so a long-running tool never presents a frozen, blank terminal.
        """
        self._start_spinner(self._tool_progress_label(tool_name, tool_args))
        try:
            return tool.invoke(tool_args)
        finally:
            self._stop_spinner()

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

    @staticmethod
    def _was_truncated_by_tokens(response) -> bool:
        """Detect a turn cut short by the output-token limit.

        Reasoning models on the OpenAI Responses API (e.g. Mantle Grok / GPT-5)
        spend output tokens reasoning before they answer. With a small
        ``MAX_TOKENS`` the budget is consumed mid-reasoning and the answer is
        never emitted: ``response_metadata`` reports ``status: "incomplete"``
        with ``incomplete_details.reason == "max_output_tokens"``. Chat /
        Converse providers signal the same via a ``length`` finish reason.

        Args:
            response: The model response (AIMessage).

        Returns:
            True if the turn was truncated by the token limit.
        """
        meta = getattr(response, "response_metadata", None) or {}
        details = meta.get("incomplete_details") or {}
        if isinstance(details, dict) and details.get("reason") == "max_output_tokens":
            return True
        finish = meta.get("finish_reason") or meta.get("stop_reason")
        return finish in ("length", "max_tokens")

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
        return disable_reasoning(self.model)

    def _restore_reasoning(self, saved: dict) -> None:
        """Restore reasoning/thinking settings on the model.

        Args:
            saved: State from _disable_reasoning()
        """
        restore_reasoning(self.model, saved)

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

    @staticmethod
    def _format_tool_call(tool_call: dict) -> str:
        """Compact one-line ``name(arg=value, …)`` rendering of a tool call.

        Argument values are stringified and truncated so a large payload (e.g.
        file content for a write) doesn't flood the marker line.
        """
        name = tool_call.get("name", "tool")
        args = tool_call.get("args") or {}
        parts = []
        for key, value in args.items():
            text = str(value).replace("\n", " ")
            if len(text) > 60:
                text = text[:57] + "..."
            parts.append(f"{key}={text}")
        return f"{name}({', '.join(parts)})"

    # Matches a key that is really a ``field="value"`` (or field='value', or
    # bare field=value) expression — a common small-model malformation where
    # the model emits Python-call syntax as a single JSON arg key.
    _ARG_KEY_EXPR = re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*$', re.DOTALL)

    @classmethod
    def _normalize_tool_args(cls, args: Any) -> Any:
        """Repair a common malformed tool-args shape from smaller models.

        Models sometimes emit ``{'query="USPTO fees"': ''}`` instead of
        ``{'query': 'USPTO fees'}`` — i.e. they pack a ``field=value`` expression
        into a single dict KEY with an empty value. Detect that exact shape and
        rebuild it into the intended ``{field: value}``, stripping surrounding
        quotes from the value. Anything that doesn't match is returned unchanged,
        so well-formed args are never touched.
        """
        if not isinstance(args, dict) or len(args) != 1:
            return args
        (key, value), = args.items()
        # Only attempt a repair when the value is empty (the tell-tale sign);
        # a real single-arg call has its value populated, not in the key.
        if value not in ("", None):
            return args
        m = cls._ARG_KEY_EXPR.match(str(key))
        if not m:
            return args
        field, raw = m.group(1), m.group(2)
        if (raw.startswith('"') and raw.endswith('"')) or (
            raw.startswith("'") and raw.endswith("'")
        ):
            raw = raw[1:-1]
        return {field: raw}

    def _is_blocked_by_plan_mode(self, tool_name: str, tool_args: dict = None) -> bool:
        """True when plan mode is active and this tool/call would mutate.

        Enforced client-side at the tool chokepoints (the MCP server can't see
        client state). Read-only tools and the memory notebook always pass.
        Three tools are CONDITIONAL:

        * ``execute_bash`` — allowed if the command is read-only (see
          :meth:`_is_readonly_bash`), else blocked.
        * ``fs_write`` / ``file_edit`` — allowed only when writing the plan file
          (a single Markdown file under the plans dir), else blocked.

        Everything else in ``_PLAN_BLOCKED_TOOLS`` is unconditionally blocked.
        """
        if not self._plan_mode_provider():
            return False
        if tool_name not in self._PLAN_BLOCKED_TOOLS:
            return False

        args = tool_args or {}
        if tool_name == "execute_bash":
            return not self._is_readonly_bash(str(args.get("command", "")))
        if tool_name in ("fs_write", "file_edit"):
            return not self._is_plan_file(str(args.get("path", "")))
        return True

    @classmethod
    def _is_readonly_bash(cls, command: str) -> bool:
        """Heuristically decide if a shell command is read-only (plan-mode safe).

        Conservative: the leading program must be in the read-only allowlist,
        the command must contain no operator that could chain/redirect a
        mutation, and a ``git`` command must use a read-only subcommand. Anything
        uncertain returns False (so it stays blocked).
        """
        cmd = (command or "").strip()
        if not cmd:
            return False
        if any(op in cmd for op in cls._BASH_MUTATION_OPS):
            return False
        tokens = cmd.split()
        prog = tokens[0]
        if prog not in cls._READONLY_BASH_CMDS:
            return False
        # Reject program-specific mutating flags (e.g. `sed -i`, `sed -i.bak`,
        # `find … -delete`/`-exec`) even though the program is allowlisted.
        bad_flags = cls._BASH_MUTATING_FLAGS.get(prog, ())
        for tok in tokens[1:]:
            if any(tok == f or tok.startswith(f) for f in bad_flags):
                return False
        if prog == "git":
            sub = tokens[1] if len(tokens) > 1 else ""
            return sub in cls._READONLY_GIT_SUBCMDS
        return True

    def _is_plan_file(self, path: str) -> bool:
        """True if ``path`` is the writable plan file (under the plans dir)."""
        if not path:
            return False
        try:
            target = Path(os.path.expanduser(path)).resolve()
            base = plans_dir().resolve()
            return (
                target.suffix == self._PLAN_FILE_SUFFIX
                and base in target.parents
            )
        except Exception:
            return False

    def _plan_mode_block_message(self, tool_name: str) -> str:
        """The ToolMessage returned when a tool is blocked by plan mode.

        Tailored per tool so the model knows the read-only escape hatch (run a
        read-only shell command, or write the plan file) rather than just
        erroring out.
        """
        if tool_name == "execute_bash":
            return (
                "Blocked: plan mode is active (read-only). Only read-only shell "
                "commands (e.g. ls, cat, grep, git status/log/diff) are allowed "
                "while planning. Investigate with read-only tools and present a "
                "plan; the user must exit plan mode (/plan) before mutating "
                "commands can run."
            )
        if tool_name in ("fs_write", "file_edit"):
            try:
                plan_hint = str(plans_dir())
            except Exception:
                plan_hint = "the plans directory"
            return (
                "Blocked: plan mode is active (read-only). You may only write your "
                f"plan as a Markdown file under {plan_hint}. Editing other files is "
                "blocked — present a plan for the user to review; the user must "
                "exit plan mode (/plan) before other changes can be made."
            )
        return (
            "Blocked: plan mode is active (read-only). Present a plan for the user "
            "to review; the user must exit plan mode (/plan) before this tool can "
            "run."
        )

    # Tools gated by a hard confirmation prompt, keyed by category.
    _CONFIRM_BASH_TOOLS = {"execute_bash"}
    _CONFIRM_WRITE_TOOLS = {"fs_write", "file_edit"}
    _CONFIRM_MEMORY_TOOLS = {"memory"}

    def _confirm_tool(self, tool_name: str, tool_args: dict) -> bool:
        """Ask the user to approve a destructive tool before it runs (Claude Code-style).

        Returns True to proceed, False if the user declines. Gates shell commands
        (``execute_bash``, toggle ``REQUIRE_BASH_CONFIRMATION``, default True),
        file writes (``fs_write``/``file_edit``, ``REQUIRE_WRITE_CONFIRMATION``,
        default True), and memory writes (``memory``, ``REQUIRE_MEMORY_CONFIRMATION``,
        default **False** — auto-save like Hermes); every other tool proceeds.
        The prompt is a hard gate enforced here in the client (which owns the
        terminal) — the MCP server is a piped subprocess and can't prompt.
        Non-interactive runs (no TTY, e.g. tests/CI) auto-proceed so they
        don't hang.
        """
        if tool_name in self._CONFIRM_BASH_TOOLS:
            toggle, toggle_default, header, detail = (
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
            toggle, toggle_default, header, detail = (
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
            toggle, toggle_default, header, detail = (
                "REQUIRE_MEMORY_CONFIRMATION",
                False,
                "▶ Update memory?",
                f"{action}: {text[:60]}",
            )
        else:
            return True

        if not config.get(toggle, toggle_default):
            return True
        if not sys.stdin.isatty():
            return True  # non-interactive: can't prompt, don't block

        self._stop_spinner()
        # Yellow header + the action detail, then a strict y/N prompt.
        print(f"\n\033[93m{header}\033[0m\n  \033[1m{detail}\033[0m")
        try:
            answer = input("  Proceed? (y/N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            answer = ""
        return answer in ("y", "yes")

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

        # Print a visible tool-invocation marker so reasoning before a tool call
        # is separated from reasoning after it (otherwise the two run together).
        # Mirrors the gray "[Step …]" markers the orchestrator prints.
        if self.verbose:
            for tool_call in last_message.tool_calls:
                print(
                    f"\n\033[90m[⚙ {self._format_tool_call(tool_call)}]\033[0m\n",
                    flush=True,
                )

        tool_results = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = self._normalize_tool_args(tool_call["args"])
            tool_id = tool_call["id"]

            # Look up in route tools first, fall back to all tools
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
                # Hard gate: ask the user before running a shell command.
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

        # Bound the model<->tools loop. This is a runaway guard, not a normal
        # stopping point — set high (default 200, configurable) so legitimate
        # long tasks run to completion, with context compaction as the real
        # limiter (like Claude Code). Hitting it means a likely stuck loop.
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

        new_messages = [
            m
            for m in final_messages
            if not isinstance(m, SystemMessage) and m not in self._messages
        ]
        self._messages.extend(new_messages)

        # Prefer the most recent AI turn that actually has visible text.
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                visible = self._extract_visible(msg.content)
                if visible:
                    return visible

        # No visible final answer (the model ended on an empty turn — e.g. it
        # called a tool, got an error/timeout result, then said nothing). Never
        # return a silent empty string: salvage the last tool result so the user
        # learns what happened, else fall back to a generic message.
        last_tool = self._last_tool_result(final_messages)
        if last_tool:
            return f"The last tool reported:\n{last_tool}"
        return (
            "I wasn't able to produce a response for that. Could you rephrase "
            "or give me a bit more detail?"
        )

    def _last_visible_from(self, messages: List[BaseMessage]) -> str:
        """Return the most recent visible AI text from a message list.

        Used to salvage a partial answer when the agent loop is cut short.

        Args:
            messages: Message history to scan (most recent last)

        Returns:
            Visible text of the last AIMessage with content, or empty string
        """
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                visible = self._extract_visible(msg.content)
                if visible:
                    return visible
        return ""

    def _last_tool_result(self, messages: List[BaseMessage]) -> str:
        """Return the most recent ToolMessage content (trimmed), or "".

        Used to salvage a useful answer when the model ends on an empty turn
        right after a tool ran (e.g. a bash timeout): the tool's result is the
        most informative thing we can still show the user.
        """
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                text = str(msg.content).strip()
                if text:
                    return text[:500]
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
