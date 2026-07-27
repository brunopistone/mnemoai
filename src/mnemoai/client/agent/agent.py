"""LangGraph-based agent implementation."""

import operator
import queue
import re
import sys  # noqa: F401  — tests patch agent_mod.sys.stdin.isatty (the confirm gate reads the same sys.stdin)
import threading
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

from mnemoai.client.agent import (
    confirmation_gate,
    message_sanitizer,
    plan_policy,
    response_parsing,
    steering,
    stream_policy,
    subagent_runner,
    tool_formatting,
)
from mnemoai.client.agent.agent_activity import ActivitySink, AgentActivityStore
from mnemoai.client.agent.background_agents import BackgroundAgentRegistry
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

INTERRUPTED_MARKER = "[Turn interrupted by the user before it completed.]"


class _ContextOverflow(Exception):
    """Raised when a model call exceeds the context window, so the caller can
    compact history and re-invoke on the shrunken prompt (continue the task)
    rather than retry the same oversized prompt in a loop."""


class _StreamIdleTimeout(Exception):
    """Raised when a streaming response goes silent for longer than the idle
    timeout — no chunk arrived within the window. The socket is (or looks) dead
    (e.g. the laptop slept and the TCP connection died); the streaming read would
    otherwise block the worker thread forever. The retry wrapper discards the
    partial and re-runs the turn on a fresh connection."""


class AgentState(TypedDict):
    """State schema for the LangGraph agent."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    thinking: Optional[str]
    route: Optional[str]
    # Subtasks the `decompose` node produced, read by `_route_after_decompose`
    # (route on count) and `_orchestrate` (execute). Plain LastValue channel — no
    # reducer, so it never touches the `messages` operator.add. Absent for
    # trivial/plan turns and for direct `_orchestrate` callers (which decompose
    # internally); readers use `state.get("subtasks")` so a missing key is safe.
    subtasks: Optional[List[Dict[str, Any]]]


class LangGraphAgent:
    """LangGraph-based agent with streaming support."""

    # Task-agnostic "meta" tools bound on EVERY route (incl. no-tools simple_qa),
    # since a matching query can classify onto any route: memory ("remember
    # this"), describe_image (any query may reference an image), fs_read
    # (universal read), use_skill (a skill-matching query may be simple_qa).
    _ALWAYS_AVAILABLE_TOOLS = {
        "memory", "describe_image", "fs_read", "use_skill", "exit_plan_mode",
        "spawn_agent", "resume_agent",
    }

    # Aliases keeping the historical class-attribute surface (used by unit tests
    # and the delegating methods) pointing at the single source in plan_policy.
    _PLAN_BLOCKED_TOOLS = plan_policy.PLAN_BLOCKED_TOOLS
    _PLAN_FILE_SUFFIX = plan_policy.PLAN_FILE_SUFFIX
    _READONLY_BASH_CMDS = plan_policy.READONLY_BASH_CMDS
    _BASH_MUTATION_OPS = plan_policy.BASH_MUTATION_OPS
    _READONLY_GIT_SUBCMDS = plan_policy.READONLY_GIT_SUBCMDS
    _BASH_MUTATING_FLAGS = plan_policy.BASH_MUTATING_FLAGS

    # --- Destructive-tool confirmation categories (gated by _confirm_tool) ---
    # Aliases keeping the class-attribute surface pointing at the single source
    # in confirmation_gate (the plan_policy alias pattern).
    _CONFIRM_BASH_TOOLS = confirmation_gate.CONFIRM_BASH_TOOLS
    _CONFIRM_WRITE_TOOLS = confirmation_gate.CONFIRM_WRITE_TOOLS
    _CONFIRM_MEMORY_TOOLS = confirmation_gate.CONFIRM_MEMORY_TOOLS

    # Tools that print their OWN live progress to the terminal, so animating our
    # spinner over them would collide on the same lines — for these we keep the
    # spinner stopped. Empty now: the web_crawler subprocess runs its browser
    # quietly (verbose=False) and any stderr goes to the MCP log (since 1.1.2), not
    # the terminal — so it shows a spinner like any other slow tool.
    _SELF_REPORTING_TOOLS: set = set()

    # --- Streaming error classification (used by _stream_response's retry) ---
    # Aliases keeping the historical class-attribute surface pointing at the
    # single source in stream_policy (the plan_policy alias pattern).
    _CONTEXT_OVERFLOW_MARKERS = stream_policy.CONTEXT_OVERFLOW_MARKERS
    _TRANSIENT_NETWORK_MARKERS = stream_policy.TRANSIENT_NETWORK_MARKERS
    # Sentinel the stream reader thread enqueues to signal a clean end of stream.
    _STREAM_DONE = object()
    # How often the idle-timeout stream wait re-checks the cancel event (seconds),
    # so a stalled-stream cancel is noticed promptly instead of after the full
    # idle window. Small enough to feel instant, large enough to be cheap.
    _CANCEL_POLL_SECONDS = 0.25

    # Ephemeral per-turn reminder blocks (the plan-mode banner, the STEERING.md
    # block) the client prepends: sent to the model this turn but stripped before
    # storage, so a reloaded conversation never carries a stale banner and
    # compaction never summarizes always-on instructions into a lossy paraphrase
    # (they're re-injected verbatim from disk each turn instead).
    _EPHEMERAL_BLOCK_RE = re.compile(
        r"<(plan-mode-active|steering)>.*?</\1>\s*", re.DOTALL
    )

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
        # Plan-mode approval hooks, set by the client (like _confirm_ui). None on
        # bare test objects / non-TTY → exit_plan_mode auto-approves so scripted
        # runs never block. `_plan_approval_ui(plan) -> "approve"|"edit"|
        # "keep_planning"`; `_exit_plan_mode_provider(plan)` flips plan mode off +
        # persists the plan.
        self._plan_approval_ui: Optional[Callable[[str], str]] = None
        self._exit_plan_mode_provider: Optional[Callable[[str], None]] = None
        # Commands pre-approved by an approved plan (exit_plan_mode allowed_bash):
        # they skip the per-command confirmation prompt during execution.
        self._preapproved_bash: List[str] = []
        # Set when a plan is approved: execution may need any tool, so per-turn
        # routing must not narrow the toolset (an approved implementation plan
        # re-classified as a read-only route would find its write/exec tools
        # unbound). Forces the full toolset until /clear or plan re-entry.
        self._execute_plan_route: bool = False
        # Set while THIS loop is itself a spawned sub-agent, to block nested
        # spawning (a sub-agent must not spawn its own sub-agents). Thread-local
        # (see the _spawn_depth property): parallel sub-agents run on separate
        # pool threads, and a shared counter would make one top-level spawn's
        # increment trip another's nested-guard check. Serializes confirm prompts
        # across concurrent sub-agents so two never prompt at once.
        self._spawn_depth_tl = threading.local()
        self._confirm_lock = threading.Lock()
        # Thread-local "this thread is a headless (background) sub-agent" flag.
        # Set on a background daemon thread so _confirm_tool auto-DENIES an
        # untrusted destructive tool there (it has no TTY to prompt on), while the
        # foreground turn on the main worker thread still prompts normally.
        self._headless_tl = threading.local()
        # Registry of background (run_in_background) sub-agents: they run on daemon
        # threads while the parent turn continues; their completions are drained
        # and injected into a later turn via the steering path.
        self._bg_agents = BackgroundAgentRegistry()
        # Live activity feed for ALL hidden sub-agent runs (foreground spawns,
        # background spawns, orchestrator subtask workers) — the pinned-TUI
        # "agents" panel + detail view read it. Always created (writers open a run
        # regardless of TTY); the on_change repaint hook is wired only on a TTY.
        self._activity = AgentActivityStore()
        # Mid-turn steering: messages the user types WHILE a turn is running are
        # pushed here by the UI and drained at tool-round boundaries (the top of
        # _execute_tools / between orchestrator waves), injected as user messages
        # so the model addresses them without ending the turn
        self._steer_queue: List[str] = []
        self._steer_lock = threading.Lock()
        # Cooperative cancel: Esc/Ctrl+C sets this (via request_cancel) so the
        # blocking stream waits (idle-timeout queue get + network-retry backoff)
        # abort IMMEDIATELY, instead of waiting on an async KeyboardInterrupt that
        # can't preempt a thread parked in a C-level socket read / time.sleep.
        self._cancel_event = threading.Event()
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
        # Exact prompt-token count from the provider's own usage_metadata on the
        # last model turn — ground truth for "how big is my context"
        self._last_input_tokens: Optional[int] = None
        self._code_formatter = CodeFormatter()
        # Set True once a turn has STREAMED visible answer text to the terminal.
        # The display contract is "streaming prints the answer", so any path that
        # produces answer text WITHOUT a visible stream (orchestrator single
        # subtask, aggregation fallback, context-overflow / stream-error / recursion
        # terminal messages, empty-final salvage) would otherwise be a silent turn.
        # invoke() checks this flag and emits the answer if nothing was shown.
        self._answer_displayed = False
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
        # A turn cut off by the output-token limit (reasoning + answer exceeded
        # MAX_TOKENS) is auto-continued: the partial turn is fed back and the model
        # resumes, so the user never has to type "continue". Capped like Claude
        # Code's max_output_tokens recovery.
        self._max_continue_retries = max(
            0, int(config.get("LLM", {}).get("MAX_OUTPUT_CONTINUE_RETRIES", 3))
        )
        # Per-chunk idle timeout on a streaming read (seconds; 0 disables). A
        # streaming body that goes silent this long — e.g. the socket died when
        # the laptop slept — is abandoned so it can't block the worker thread
        # forever; the retry wrapper then re-runs the turn on a fresh connection.
        # The provider's own request timeout only covers getting response HEADERS,
        # not a stalled body, so this watchdog is what unwedges a dead stream.
        self._stream_idle_timeout = float(
            config.get("LLM", {}).get("STREAM_IDLE_TIMEOUT", 120)
        )
        # Cap one tool result so a runaway (e.g. grep_search max_results=4000)
        # can't alone overflow the context window. Defaults to 10% of the context
        # window (converted tokens->chars at ~4 chars/token), so it scales with
        # the model instead of a fixed number; explicit config wins, 0 disables.
        _max_conv = int(config.get("MAX_CONVERSATION_TOKENS", 1024 * 8))
        self._max_tool_result_chars = int(
            config.get("LLM", {}).get(
                "MAX_TOOL_RESULT_CHARS", int(_max_conv * 0.10 * 4)
            )
        )
        # Set by the client to a sync compaction callable `(force=False) -> bool`;
        # the client owns the token-budget check. None on bare test objects
        # (guarded at every call site).
        self._compact_provider: Optional[Callable[..., bool]] = None
        # Upper bound on sub-agents run in parallel from one turn's tool calls
        # (a bounded pool; the rest queue). 1 = force sequential.
        self._max_subagent_concurrency = max(
            1, int(config.get("LLM", {}).get("SUBAGENT_MAX_CONCURRENCY", 4))
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

    def steer(self, text: str) -> None:
        """Queue a mid-turn user message to inject into the RUNNING turn.

        Called by the UI (event-loop thread) when the user submits a line while a
        turn is in flight. Thread-safe; drained at the next tool-round boundary by
        :meth:`_drain_steering`. Never aborts the turn — the current tool batch
        finishes, then the message is folded in. Delegates to
        :func:`steering.enqueue`."""
        steering.enqueue(self, text)

    def _drain_steering(self) -> List[BaseMessage]:
        """Delegates to :func:`steering.drain`."""
        return steering.drain(self)

    def _has_steering(self) -> bool:
        """Delegates to :func:`steering.has_pending`."""
        return steering.has_pending(self)

    def clear_steering(self) -> None:
        """Delegates to :func:`steering.clear`."""
        steering.clear(self)

    def request_cancel(self) -> None:
        """Signal a cooperative cancel of the running turn (called from the UI
        thread on Esc/Ctrl+C, alongside the async-exc injection).

        The async ``KeyboardInterrupt`` can't preempt a worker parked in a C-level
        blocking wait — a stalled-stream ``queue.get(timeout=…)`` or a network-retry
        ``time.sleep`` backoff — so it only fires when the wait finally returns
        (up to `STREAM_IDLE_TIMEOUT`/30s later), which is the "cancel takes ages"
        bug. This event is the mnemoai analog of an ``AbortSignal``: the blocking
        waits ``.wait()`` on it (waking instantly) and check it at each retry, so
        the turn tears down immediately. Delegates to :func:`steering.request_cancel`."""
        steering.request_cancel(self)

    def _cancelled(self) -> bool:
        """Delegates to :func:`steering.is_cancelled`."""
        return steering.is_cancelled(self)

    def _is_headless(self) -> bool:
        """True on a background sub-agent's thread (no TTY → can't prompt)."""
        tl = getattr(self, "_headless_tl", None)
        return bool(getattr(tl, "value", False)) if tl is not None else False

    def _set_headless(self, value: bool) -> None:
        """Mark the CURRENT thread headless (or not). Thread-local so it only
        affects the background daemon thread, never the foreground turn."""
        tl = getattr(self, "_headless_tl", None)
        if tl is not None:
            tl.value = value

    @property
    def _spawn_depth(self) -> int:
        """Per-thread nested-spawn depth (0 = not inside a sub-agent).

        Thread-local so parallel sub-agents on separate pool threads don't share
        a counter: each concurrent top-level spawn sees depth 0 on its own thread,
        while a nested spawn (same thread) sees >0 and is refused. Falls back to a
        plain attribute if the thread-local isn't set up (bare test objects built
        via ``__new__``), and tests can still assign ``a._spawn_depth = 1``.
        """
        tl = getattr(self, "_spawn_depth_tl", None)
        if tl is None:
            return getattr(self, "_spawn_depth_plain", 0)
        return getattr(tl, "value", 0)

    @_spawn_depth.setter
    def _spawn_depth(self, value: int) -> None:
        tl = getattr(self, "_spawn_depth_tl", None)
        if tl is None:
            self._spawn_depth_plain = value
        else:
            tl.value = value

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph state graph."""
        workflow = StateGraph(AgentState)

        if self.router:
            workflow.add_node("classifier", self._classify)
            workflow.set_entry_point("classifier")

            if self.orchestrator_enabled:
                workflow.add_node("decompose", self._decompose)
                workflow.add_node("orchestrator", self._orchestrate)
                # The classifier decides WHETHER to decompose (a non-trivial
                # 'full' task); trivial/plan turns skip straight to the streaming
                # agent and never pay the decompose LLM call.
                workflow.add_conditional_edges(
                    "classifier",
                    self._route_after_classify,
                    {"agent": "agent", "decompose": "decompose"},
                )
                # decompose then routes on the ACTUAL subtask count: an atomic
                # task (<=1 subtask) runs through the normal streaming agent (full
                # tools, native history), a genuine multi-step plan (>=2) is owned
                # by the orchestrator. No more degenerate "step 1/1" worker.
                workflow.add_conditional_edges(
                    "decompose",
                    self._route_after_decompose,
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

    @staticmethod
    def _last_human_query(messages: Sequence[BaseMessage]) -> str:
        """The content of the most recent HumanMessage, or "" if there is none."""
        for msg in reversed(messages or []):
            if isinstance(msg, HumanMessage):
                return str(msg.content)
        return ""

    def _route_after_classify(self, state: AgentState) -> str:
        """Send a non-trivial 'full' task to the ``decompose`` node; everything
        else straight to the streaming agent.

        Decomposition runs BEFORE the agent-vs-orchestrator choice, which is then
        made on the ACTUAL subtask count (``_route_after_decompose``): an atomic
        task falls back to the normal streaming agent rather than a degenerate
        single-subtask worker. A trivial 'full' query (short/signal-free) skips
        decomposition — it goes to ``agent`` directly, paying no decompose LLM
        call. During plan execution we also skip decomposition entirely (the
        approved plan IS the decomposition — re-decomposing would spawn read-only
        workers that can't apply the plan's edits).
        """
        if getattr(self, "_execute_plan_route", False):
            return "agent"
        if state.get("route") == "full":
            query = self._last_human_query(state.get("messages", []))
            if not is_trivial_query(query):
                return "decompose"
        return "agent"

    def _route_after_decompose(self, state: AgentState) -> str:
        """After decomposition, own the turn only for a genuine multi-step plan.

        >=2 subtasks → the orchestrator executes + aggregates. <=1 (atomic task,
        a parse/decompose fallback, or no query) → the normal streaming agent,
        which has the full toolset and native conversation history — so an atomic
        'full' task streams normally instead of running a hidden quiet worker.
        """
        subtasks = state.get("subtasks") or []
        return "orchestrator" if len(subtasks) >= 2 else "agent"

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

    def _prior_history(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """The REAL prior-conversation messages for the decomposer + every worker.

        The orchestrator route otherwise sees ONLY the current query, so a
        context-dependent follow-up ("write the issue to a file", "fix it") is
        decomposed and executed with no idea what "the issue"/"it" refers to and
        the worker fabricates content. We hand the decomposer and each worker the
        SAME messages the main agent sees — the actual Human/AI/Tool turns, not a
        re-rendered text block. ``state["messages"]`` is ALREADY bounded to the
        model window by the compaction layer, so NO extra count/token cap is
        applied here (that would be redundant and could truncate the very turn a
        follow-up refers to). Drops the leading system prompt(s) (the
        worker/decomposer carry their own) and the trailing current-query
        HumanMessage (passed separately as the subtask/decomposition prompt), then
        repairs orphaned tool-call/result pairs so strict providers don't 400.
        Returns [] on a first turn — every single-turn path is unchanged.
        """
        if not messages:
            return []
        history = list(messages)
        # Drop the trailing current-query HumanMessage (passed separately).
        for i in range(len(history) - 1, -1, -1):
            if isinstance(history[i], HumanMessage):
                del history[i]
                break
        # Drop the leading system prompt(s) / any stray SystemMessage.
        history = [m for m in history if not isinstance(m, SystemMessage)]
        return self._sanitize_tool_pairs(history)

    def _decompose(self, state: AgentState) -> Dict[str, Any]:
        """Graph node: decompose a non-trivial 'full' task into subtasks.

        Runs between the classifier and the agent-vs-orchestrator branch so the
        route decision (``_route_after_decompose``) can be made on the ACTUAL
        subtask count — an atomic task (<=1 subtask) then falls back to the
        normal streaming ``agent`` instead of a degenerate single-subtask worker.
        Writes ONLY the ``subtasks`` channel (no ``messages`` key) so the agent's
        conversation history stays pristine if the turn falls back to ``agent``.
        """
        messages = state["messages"]
        query = self._last_human_query(messages)
        if not query:
            return {"subtasks": []}
        # Same real prior history the workers get, so references resolve.
        history = self._prior_history(messages)
        orchestrator_prompt = get_orchestrator_prompt()
        orchestrator_prompt += self._external_tools_prompt_block()
        logger.debug("Decompose: decomposing task")
        subtasks = self._decompose_task(
            query, orchestrator_prompt, set(ROUTE_TOOLS.keys()), history=history
        )
        logger.debug(f"Decompose: {len(subtasks)} subtasks")
        return {"subtasks": subtasks}

    def _orchestrate(self, state: AgentState) -> Dict[str, Any]:
        """Decompose the task into subtasks, run a worker per subtask, aggregate."""
        messages = state["messages"]
        # Extract user query (skip system prompt).
        query = self._last_human_query(messages)
        if not query:
            return {"messages": [AIMessage(content="No query found.")]}

        # The REAL prior-conversation messages (empty on a first turn), threaded
        # into BOTH the decomposer and every worker so a context-dependent
        # follow-up ("write the issue to a file") resolves its references against
        # the same conversation the main agent sees, instead of fabricating.
        history = self._prior_history(messages)

        # Step 1: reuse the subtasks the ``decompose`` node already produced. A
        # direct caller (tests, or any path that reaches _orchestrate without the
        # graph's decompose node) leaves the channel unset — decompose here so
        # those callers stay backward-compatible.
        subtasks = state.get("subtasks")
        if not subtasks:
            orchestrator_prompt = get_orchestrator_prompt()
            orchestrator_prompt += self._external_tools_prompt_block()
            logger.debug("Orchestrator: decomposing task")
            subtasks = self._decompose_task(
                query, orchestrator_prompt, set(ROUTE_TOOLS.keys()), history=history
            )
        logger.debug(f"Orchestrator: {len(subtasks)} subtasks")

        # Step 2: execute subtasks, scheduling by their ``depends_on`` graph —
        # independent subtasks run concurrently (bounded pool), dependents wait.
        results_by_index = self._run_subtasks_scheduled(subtasks, history)
        worker_results = [results_by_index[i] for i in range(len(subtasks))]

        # Collect all intermediate worker messages for conversation saving.
        all_worker_messages: List[BaseMessage] = []
        for wr in worker_results:
            all_worker_messages.extend(wr.get("messages", []))

        # Mid-turn steering: any messages the user typed during the orchestration
        # are folded into the aggregator (the orchestration turn's final model
        # call) so they're addressed in THIS turn. A lone single-subtask orchestration 
        # has no aggregator, so steering there falls through to the next turn 
        # (it still lands in history below).
        steering = self._drain_steering()

        # Step 3: aggregate.
        if len(subtasks) == 1:
            # Single subtask: its result IS the answer — no aggregator call. There
            # is no aggregator to fold steering into, so keep any steering message
            # in history so the NEXT turn still addresses it (not dropped).
            final_content = worker_results[0]["result"]
            if steering:
                all_worker_messages.extend(steering)
        else:
            print(
                "\n\033[90m[Synthesizing results...]\033[0m",
                flush=True,
            )
            try:
                final_content = self._aggregate_results(
                    query, worker_results, get_aggregator_prompt(), steering=steering
                )
            except Exception as e:
                # If synthesis fails, fall back to concatenating the per-step
                # results so the user still gets the work that was done.
                logger.error(f"Aggregation failed: {e}; concatenating results")
                self._stop_spinner()
                final_content = "\n\n".join(
                    f"### {r['task']}\n{r['result']}" for r in worker_results
                )
                # Aggregation didn't consume the steering messages; keep them in
                # history so the next turn addresses them.
                if steering:
                    all_worker_messages.extend(steering)

        return {"messages": all_worker_messages + [AIMessage(content=final_content)]}

    def _run_subtasks_scheduled(
        self,
        subtasks: List[Dict[str, Any]],
        history: Optional[List[BaseMessage]] = None,
    ) -> Dict[int, dict]:
        """Run subtasks respecting their ``depends_on`` graph.

        Repeatedly runs every not-yet-done subtask whose dependencies are all
        complete as one **concurrent wave** (bounded by ``SUBAGENT_MAX_CONCURRENCY``
        — the same pool the parallel sub-agents use), then feeds completed results
        into the dependents. A subtask with no deps runs in the first wave; a chain
        ``0 -> 1 -> 2`` runs strictly sequentially; two independent subtasks run
        together. Deadlock guard: if a wave would be empty while work remains (a
        dependency cycle survived sanitization, or all remaining deps failed), the
        remaining subtasks are forced to run so orchestration always completes.
        Returns ``{index: result_dict}``."""
        from concurrent.futures import ThreadPoolExecutor

        total = len(subtasks)
        results: Dict[int, dict] = {}
        max_workers = getattr(self, "_max_subagent_concurrency", 1)
        remaining = set(range(total))

        while remaining:
            ready = [
                i for i in sorted(remaining)
                if all(d in results for d in subtasks[i].get("depends_on", []))
            ]
            if not ready:  # broken/cyclic deps — force the rest so we never hang
                ready = sorted(remaining)

            if len(ready) == 1 or max_workers <= 1:
                # A lone/sequential worker runs on THIS thread and CAN prompt for
                # destructive-tool confirmation (only one prompt at a time). Keep a
                # spinner up while it works — the worker runs quiet (no trace), so
                # without this the UI looks finished while the step is still going.
                for i in ready:
                    desc = subtasks[i].get("description", "")
                    label = desc[:40] + ("…" if len(desc) > 40 else "")
                    self._start_spinner(f"step {i + 1}/{total}: {label}")
                    try:
                        results[i] = self._run_subtask(
                            i, subtasks, results, history
                        )
                    finally:
                        self._stop_spinner()
            else:
                if self.verbose:
                    print(
                        f"\n\033[90m[Running {len(ready)} steps in parallel]\033[0m",
                        flush=True,
                    )
                self._start_spinner(f"{len(ready)} steps running…")

                # Parallel-wave workers run HEADLESS: several run at once on pool
                # threads, and stacking interactive confirmation prompts on one
                # terminal is unworkable — so an untrusted destructive tool
                # auto-denies (same safety rule as background sub-agents), while a
                # category the user already trusted this session still proceeds.
                def _run_headless(idx):
                    self._set_headless(True)
                    try:
                        return idx, self._run_subtask(
                            idx, subtasks, results, history
                        )
                    finally:
                        self._set_headless(False)

                try:
                    with ThreadPoolExecutor(
                        max_workers=min(max_workers, len(ready))
                    ) as pool:
                        for i, res in pool.map(_run_headless, ready):
                            results[i] = res
                finally:
                    self._stop_spinner()
            remaining -= set(ready)
        return results

    def _run_subtask(
        self,
        i: int,
        subtasks: List[Dict[str, Any]],
        done: Dict[int, dict],
        history: Optional[List[BaseMessage]] = None,
    ) -> dict:
        """Run one orchestrator subtask and return its result dict.

        Binds the subtask's category tools, prepends the results of the subtasks
        it ``depends_on`` (only those — not every prior step), passes the real
        prior-conversation ``history`` to the worker loop (so a context-dependent
        subtask like "write the issue to a file" resolves its references), and
        runs the loop ``quiet`` (silent, concurrency-safe) so a parallel wave
        doesn't interleave output. A failure
        is captured, never aborting the wave."""
        desc = subtasks[i]["description"]
        category = subtasks[i]["category"]
        deps = subtasks[i].get("depends_on", [])

        total = len(subtasks)
        short_desc = desc[:70] + ("..." if len(desc) > 70 else "")
        if self.verbose:
            print(
                f"\n\033[90m[Step {i + 1}/{total}: {short_desc}]\033[0m",
                flush=True,
            )

        # Route-specific tools, bound onto a CALLBACK-FREE model copy: workers run
        # quiet, so (like sub-agents) they must not fire the shared spinner's
        # callbacks — else a parallel wave's workers tear down the "N steps
        # running…" spinner and race each other. Rebind per-subtask off the
        # stripped base rather than reusing the callback-carrying route models.
        base = self._callback_free_model()
        if category == "full" or not self.tools_by_route:
            worker_tools = self.tools
        elif category == "simple_qa":
            worker_tools = []
        else:
            worker_tools = self.tools_by_route.get(category, self.tools)
        worker_model = base.bind_tools(worker_tools) if worker_tools else base

        # Prepend ONLY the results this subtask declared a dependency on (falls
        # back to all completed steps when it declared none but some ran first,
        # preserving the old "context from prior steps" behavior for linear plans).
        context_sources = deps if deps else sorted(done.keys())
        worker_prompt = desc
        if context_sources:
            context_parts = [
                f"[Completed: {done[d]['task']}]\n{done[d]['result']}"
                for d in context_sources
                if d in done
            ]
            if context_parts:
                worker_prompt = (
                    "Context from completed steps:\n"
                    + "\n\n".join(context_parts)
                    + f"\n\nCurrent task: {desc}"
                )

        # Live activity run for the agents panel (orchestrator path bypasses
        # _run_one_subagent, so open/close the run here).
        sink = self._activity.open_run(category, short_desc, "orchestrator")
        try:
            # The real prior conversation (uncapped; only via this orchestrator
            # path) is inserted between the worker's system prompt and this
            # subtask, so a context-dependent subtask resolves its references.
            result, worker_msgs = self._run_worker_loop(
                worker_model, worker_tools, worker_prompt, quiet=True,
                history=history, activity=sink,
            )
            sink.finish_ok(result)
        except Exception as e:
            logger.error(f"Worker for subtask {i + 1} failed: {e}")
            result = f"(This step could not be completed: {e})"
            worker_msgs = []
            sink.finish("failed")
        finally:
            # Backstop for a cancel (KeyboardInterrupt is BaseException, not
            # caught above) so a stopped subtask isn't shown "running" forever.
            sink.finish_if_running("failed")
        return {
            "task": desc,
            "category": category,
            "result": result,
            "messages": worker_msgs,
        }

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
        self,
        query: str,
        orchestrator_prompt: str,
        valid_categories: set,
        history: Optional[List[BaseMessage]] = None,
    ) -> List[Dict[str, Any]]:
        """Call the LLM to decompose a task into subtask dicts.

        ``history`` (the real prior conversation) is inserted between the
        orchestrator system prompt and the query so the decomposer can split a
        context-dependent request against the same conversation the main agent
        sees. None (the default) keeps the old system-prompt+query shape.

        This call binds NO tools, so any tool_use/tool_result block in ``history``
        has no matching schema — Bedrock Converse then warns and rewrites them
        (leaking a RuntimeWarning into the TUI) and a strict provider can 400. The
        decomposer only needs to know what ran, so tool blocks are flattened to
        text first.
        """
        messages = [SystemMessage(content=orchestrator_prompt)]
        if history:
            messages.extend(message_sanitizer.flatten_tool_blocks(history))
        messages.append(HumanMessage(content=query))

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
        system_prompt: Optional[str] = None,
        quiet: bool = False,
        progress: Optional[Callable[[str], None]] = None,
        history: Optional[List[BaseMessage]] = None,
        activity: Optional[ActivitySink] = None,
    ) -> tuple:
        """Run a worker agent loop until completion.

        Runs a self-contained model↔tool loop on its OWN local message list (never
        ``self._messages``), so its context is isolated from the parent turn.
        ``system_prompt`` overrides the parent's (used by spawned sub-agents, which
        carry their type's own prompt); defaults to ``self.system_prompt``.

        ``quiet=True`` (spawned sub-agents) still **streams** the model call — so
        it keeps the stalled-stream idle-timeout + network retry (a sub-agent whose
        socket dies on sleep must not hang) — but **suppresses all display** and
        touches no shared display state (via ``_stream_once_quiet``): no reasoning/
        tool markers, no ``print``, no shared ``_code_formatter``/spinner. So the
        sub-agent's trace stays out of the terminal (display isolation) AND it is
        safe to run several concurrently. Which streams sub-agent calls under 
        the hood and merely drops the delta events. ``progress(label)`` is an optional
        callback invoked per tool call so a caller can drive a live "N tool calls…" line.

        Returns ``(final_text, worker_messages)`` where ``worker_messages`` holds
        the intermediate AI/Tool messages for saving.
        """
        worker_messages: List[BaseMessage] = []
        sys_prompt = system_prompt if system_prompt is not None else self.system_prompt
        if sys_prompt:
            worker_messages.append(SystemMessage(content=sys_prompt))
        # Real prior-conversation messages (uncapped; passed ONLY by the
        # orchestrator via _run_subtask — spawn/resume sub-agents pass None, so
        # their context isolation is preserved) sit BETWEEN the system prompt and
        # this worker's own turn, so a context-dependent subtask resolves its
        # references. Everything up to here is injected CONTEXT, not this worker's
        # output: `_seed_len` marks the boundary so it's excluded from `saveable`
        # (else the whole prior conversation would be re-appended, once per
        # worker, into the saved history). When history is None/empty and a
        # system prompt is present this is byte-equivalent to the old behavior.
        if history:
            worker_messages.extend(history)
        _seed_len = len(worker_messages)
        worker_messages.append(HumanMessage(content=prompt))

        # A quiet sub-agent must NOT pass the parent's callbacks: those drive the
        # shared spinner (on_llm_new_token/on_tool_start → spinner.stop), so a
        # quiet stream would tear down the batch's "N running…" toolbar and race
        # sibling sub-agents. Empty config keeps the sub-agent's stream silent.
        config = {} if quiet else ({"callbacks": self.callbacks} if self.callbacks else {})
        tool_calls_made = 0

        for _ in range(max_iterations):
            # Cooperative cancel: either the whole turn was cancelled (Esc/Ctrl+C
            # → global _cancel_event) OR this specific agent was stopped from the
            # panel (x / stop-all → its per-run cancel Event). Sub-agents run on
            # pool/daemon threads the injected KeyboardInterrupt can't reach, so
            # they must poll — else a stopped agent keeps looping (tools climbing,
            # timer ticking) after the UI said it stopped.
            if self._cancelled() or (activity is not None and activity.is_cancelled()):
                if not quiet:
                    self._stop_spinner()
                saveable = worker_messages[_seed_len:]
                partial = self._last_visible_from(worker_messages)
                return (partial or "(cancelled)", saveable)

            if not quiet:
                self._start_spinner()

            try:
                # Both stream (so quiet sub-agents keep the idle-timeout + network
                # retry); quiet just suppresses display + touches no shared state
                # (streams, drops deltas), making it concurrency-safe.
                response, _ = self._stream_response(
                    worker_messages, config, model=worker_model, quiet=quiet
                )
            except _ContextOverflow as overflow:
                # A worker's own history overflowed. Workers run on a local
                # message list (not self._messages), so end this worker with
                # what it has rather than crash the orchestration.
                logger.warning(f"Worker context overflow: {overflow}; ending worker")
                if not quiet:
                    self._stop_spinner()
                saveable = worker_messages[_seed_len:]  # exclude system + history
                partial = self._last_visible_from(worker_messages)
                return (
                    partial
                    or "This subtask exceeded the context window before completing.",
                    saveable,
                )

            if response is None:
                # Fallback invoke bypasses _stream_response, so scrub here too.
                response = worker_model.invoke(
                    self._strip_malformed_reasoning(worker_messages), config=config
                )

            worker_messages.append(response)

            # No tool calls → worker is done.
            if not isinstance(response, AIMessage) or not response.tool_calls:
                visible = self._extract_visible(response.content)
                # Reasoning-only/empty turn: salvage a visible reply so the
                # orchestrator doesn't surface a blank answer. (Quiet sub-agents
                # use a silent — streamed, no-display — re-ask.)
                if not visible:
                    if quiet:
                        visible = self._salvage_empty_worker_quiet(
                            worker_messages, worker_model
                        )
                    else:
                        visible = self._salvage_empty_worker_turn(
                            worker_messages, config, worker_model
                        )
                if not quiet:
                    self._stop_spinner()
                saveable = worker_messages[_seed_len:]  # exclude system + history
                return visible or str(response.content), saveable

            if not quiet:
                self._stop_spinner()
                if self.verbose:
                    for tc in response.tool_calls:
                        self._print_tool_marker(tc)
            else:
                tool_calls_made += len(response.tool_calls)
                if progress is not None:
                    progress(
                        f"{tool_calls_made} tool call"
                        f"{'s' if tool_calls_made != 1 else ''}…"
                    )
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_id = tc["id"]
                tool_args = self._normalize_tool_args(tc["args"])

                # exit_plan_mode / spawn_agent / resume_agent are client-side stubs
                # (no batch here — a worker runs any spawn inline).
                client_msg = self._client_side_tool_message(
                    tool_name, tool_args, tool_id
                )
                if client_msg is not None:
                    worker_messages.append(client_msg)
                    continue

                # Feed the live activity panel (no-op when activity is None).
                if activity is not None:
                    activity.tool_call(tool_name, tool_args)

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
                        result = self._invoke_tool(
                            tool, tool_name, tool_args, quiet=quiet
                        )
                        worker_messages.append(
                            ToolMessage(
                                content=self._truncate_tool_result(str(result)),
                                tool_call_id=tool_id,
                                name=tool_name,
                            )
                        )
                        if activity is not None:
                            activity.tool_result(tool_name, str(result))
                    except Exception as e:
                        logger.error(f"Worker tool error: {e}")
                        if activity is not None:
                            activity.tool_error(tool_name, str(e))
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

        if not quiet:
            self._stop_spinner()
        saveable = worker_messages[_seed_len:]  # exclude system + injected history
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
        except _ContextOverflow:
            retry_response = None  # fall through to the fallback message below
        finally:
            self._restore_reasoning(saved)

        if retry_response is not None:
            visible = self._extract_visible(retry_response.content)
            if visible:
                worker_messages.append(retry_response)
                return visible

        # Still nothing usable: surface a fallback (never a silent turn). No
        # ad-hoc print — the fallback becomes the worker's result and, via the
        # orchestrator, the turn's AIMessage, which the central net (invoke() →
        # _emit_answer) renders exactly once; printing here too would double it.
        fallback = (
            "I wasn't able to produce a response for that. "
            "Could you rephrase or give me a bit more detail?"
        )
        self._stop_spinner()
        worker_messages.append(AIMessage(content=fallback))
        return fallback

    def _salvage_empty_worker_quiet(
        self, worker_messages: List[BaseMessage], worker_model
    ) -> str:
        """Recover a visible answer for a quiet sub-agent whose turn was reasoning-
        only — a single silent (streamed, no-display) re-ask. Never returns empty."""
        retry_messages = list(worker_messages) + [
            HumanMessage(
                content=(
                    "You provided reasoning but no visible response. "
                    "Please provide your answer."
                )
            )
        ]
        saved = self._disable_reasoning()
        try:
            retry_response, _ = self._stream_response(
                retry_messages, {}, model=worker_model, quiet=True
            )
        except Exception:
            retry_response = None
        finally:
            self._restore_reasoning(saved)

        if retry_response is not None:
            visible = self._extract_visible(retry_response.content)
            if visible:
                worker_messages.append(retry_response)
                return visible
        fallback = "(The sub-agent did not produce a usable response.)"
        worker_messages.append(AIMessage(content=fallback))
        return fallback

    def _aggregate_results(
        self,
        original_query: str,
        worker_results: List[Dict[str, Any]],
        aggregator_prompt: str,
        steering: Optional[List[BaseMessage]] = None,
    ) -> str:
        """Aggregate worker results into a final response via the LLM.

        ``steering`` (mid-turn messages the user typed during orchestration) is
        appended after the results so the synthesis addresses them in this turn."""
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
        if steering:
            messages.extend(steering)

        config = {"callbacks": self.callbacks} if self.callbacks else {}

        self._start_spinner()
        try:
            response, _ = self._stream_response(messages, config, mark_answer=True)
        except _ContextOverflow:
            # Aggregation prompt overflowed; force-compact and let the retry (or
            # the None-fallback invoke below) run on the shrunken prompt.
            rebuilt = self._compact_and_rebuild(messages)
            response = None
            if rebuilt is not None:
                messages = rebuilt
                try:
                    response, _ = self._stream_response(messages, config, mark_answer=True)
                except _ContextOverflow:
                    response = None

        if response is None:
            # Fallback invoke bypasses _stream_response's reasoning scrub.
            response = self.model.invoke(
                self._strip_malformed_reasoning(messages), config=config
            )

        self._stop_spinner()
        return self._extract_visible(response.content) or str(response.content)

    def _effective_route(self, state: AgentState) -> Optional[str]:
        """The route to bind tools for, honoring plan-execution override.

        After a plan is approved, execution may call any tool, so we ignore the
        per-turn classifier and bind the full toolset — otherwise an approved
        implementation plan re-classified as a read-only route (e.g. a docs
        question) would find `file_edit`/`fs_write`/`execute_bash` unbound.
        """
        if getattr(self, "_execute_plan_route", False):
            return "full"
        return state.get("route")

    def _get_route_model(self, state: AgentState):
        """The model binding for the current route (falls back to all tools)."""
        route = self._effective_route(state)
        if route and self.models_by_route:
            return self.models_by_route.get(route, self.model_with_tools)
        return self.model_with_tools

    def _get_route_tools(self, state: AgentState) -> List[BaseTool]:
        """The tool list for the current route (falls back to all tools)."""
        route = self._effective_route(state)
        if route and self.tools_by_route:
            return self.tools_by_route.get(route, self.tools)
        return self.tools

    @staticmethod
    def _sanitize_tool_pairs(messages: List[BaseMessage]) -> List[BaseMessage]:
        """Delegates to :func:`message_sanitizer.sanitize_tool_pairs`."""
        return message_sanitizer.sanitize_tool_pairs(messages)

    @staticmethod
    def _strip_malformed_reasoning(messages: List[BaseMessage]) -> List[BaseMessage]:
        """Delegates to :func:`message_sanitizer.strip_malformed_reasoning` — the
        provider-agnostic egress guard against re-feeding an invalid reasoning
        block (thinking-only scrub, no tool-pair repair)."""
        return message_sanitizer.strip_malformed_reasoning(messages)

    def _capture_input_tokens(self, response: Any) -> None:
        """Record the provider's exact prompt-token count from a response's
        ``usage_metadata`` (LangChain normalizes it across Anthropic/OpenAI/
        Bedrock). This is ground truth for the current context size — far more
        accurate than re-estimating — and drives the compaction decision."""
        try:
            um = getattr(response, "usage_metadata", None) or {}
            it = um.get("input_tokens")
            if it:
                self._last_input_tokens = int(it)
        except Exception:
            pass

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
        try:
            response, had_reasoning = self._stream_response(
                messages, config, model=active_model, mark_answer=True
            )
        except _ContextOverflow as overflow:
            # The prompt exceeded the context window. Compact history and retry
            # ONCE on the shrunken prompt so the in-flight task continues instead
            # of dead-ending. If it still overflows (or nothing could be
            # compacted), surface a terminal message rather than loop.
            rebuilt = self._compact_and_rebuild(messages)
            if rebuilt is None:
                logger.warning(f"Context overflow: {overflow}; could not compact")
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "The conversation grew past the model's context "
                                "window and I couldn't compact it further. Use "
                                "/clear to start fresh, or /compact <focus>."
                            )
                        )
                    ],
                    "thinking": None,
                }
            logger.warning(f"Context overflow: {overflow}; compacted and retrying")
            messages = rebuilt
            self._start_spinner()
            try:
                response, had_reasoning = self._stream_response(
                    messages, config, model=active_model, mark_answer=True
                )
            except _ContextOverflow as overflow2:
                logger.error(f"Context overflow persists after compaction: {overflow2}")
                self._stop_spinner()
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "The conversation is still too large after "
                                "compaction. Please use /clear to start fresh."
                            )
                        )
                    ],
                    "thinking": None,
                }
        except _ContextOverflow:
            raise  # handled by the dedicated branch above; never reach here as a generic error
        except KeyboardInterrupt:
            raise  # user cancel — propagate so the turn rolls back cleanly
        except Exception as e:
            # A stream error that survived the retry wrapper. Two families:
            #  - transient (dead/stalled connection, 5xx/overloaded) exhausted its
            #    abortable retries — e.g. the network is still down after a sleep;
            #  - a non-transient API error (the streaming call raised something the
            #    retry wrapper won't retry) — previously this was swallowed by a
            #    BLOCKING non-streaming invoke here, which couldn't be cancelled and
            #    wedged the turn (the reported bug). We no longer do that.
            # Either way, END the turn with a clear, non-crashing message: the
            # history is intact, so the user can just send again (a transient issue
            # then re-runs on a fresh connection). Tailor the wording per family.
            self._stop_spinner()
            if isinstance(e, _StreamIdleTimeout) or self._is_transient_network_error(e):
                logger.error(f"Stream failed after retries (connection issue): {e}")
                msg = (
                    "I lost the connection to the model and couldn't reconnect after "
                    "several retries (this can happen after the machine sleeps or the "
                    "network drops). Your conversation is intact — just send your "
                    "message again and I'll continue."
                )
                # A read timeout on a LARGE context is not a flaky network: the
                # first response byte simply takes longer than the per-request
                # ceiling (prefill + reasoning scale with prompt size), so every
                # retry re-pays the same doomed wait. Name the knob instead of
                # leaving the user to guess.
                if "read timed out" in str(e).lower():
                    msg += (
                        "\n\nIf this keeps happening on a long conversation, the "
                        "request is timing out before the model's first token "
                        "rather than dropping — raise LLM.REQUEST_TIMEOUT (or "
                        "/compact to shrink the prompt)."
                    )
            else:
                logger.error(f"Model request failed: {e}")
                msg = (
                    "The model request failed with an error I can't recover from "
                    "automatically. Your conversation is intact — please try again."
                )
            return {"messages": [AIMessage(content=msg)], "thinking": None}

        if response is None:
            response = active_model.invoke(messages, config=config)

        self._capture_input_tokens(response)
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

        # Turn cut short by the output-token limit (reasoning + a partial answer
        # or tool call exceeded MAX_TOKENS), leaving no completed tool call. Checked
        # BEFORE the reasoning-retry. Instead of dead-ending (which forced the user
        # to type "continue"), AUTO-CONTINUE: feed the partial turn back and let the
        # model resume until it finishes or we hit the retry cap. Only when the turn
        # didn't already produce a tool call — a truncated-but-valid tool call flows
        # to the graph, which just executes it.
        if not response.tool_calls and self._was_truncated_by_tokens(response):
            if self._max_continue_retries > 0:
                continued = self._continue_truncated_turn(
                    messages, response, visible, thinking, active_model, config
                )
                if continued is not None:
                    return continued
            # Continuation disabled or it produced nothing usable. If we at least
            # have partial visible text, keep it; else surface an actionable hint.
            if not visible:
                logger.warning(
                    "Model response truncated by the output-token limit and could "
                    "not be auto-continued — increase MODEL_ID.MAX_TOKENS (reasoning "
                    "models need headroom to reason and answer)."
                )
                truncated = AIMessage(
                    content=(
                        "My response was cut off by the output-token limit before I "
                        "could finish, and I couldn't auto-continue. This model "
                        "reasons before replying, so it needs more room — increase "
                        "`MAX_TOKENS` (e.g. via /params or in config.yaml) and try "
                        "again."
                    )
                )
                if thinking:
                    truncated.additional_kwargs["reasoning_content"] = thinking
                # No ad-hoc print — the central net (invoke() → _emit_answer) renders
                # this returned AIMessage exactly once; printing here too would
                # double it (nothing streamed, so _answer_displayed is still False).
                self._stop_spinner()
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
            except _ContextOverflow:
                retry_response = None  # degrade: keep the reasoning-only response
            finally:
                self._restore_reasoning(saved)

            if retry_response is not None:
                retry_visible = self._extract_visible(retry_response.content)
                if retry_visible or retry_response.tool_calls:
                    if not retry_response.additional_kwargs.get("reasoning_content"):
                        retry_response.additional_kwargs["reasoning_content"] = thinking
                    return {"messages": [retry_response], "thinking": thinking}

            # Both attempts yielded nothing usable: surface a fallback (never silent).
            # No ad-hoc print here — the fallback flows back as the turn's AIMessage
            # and the central net (invoke() → _emit_answer) renders it exactly once,
            # with the ● marker; printing here too would double it (nothing streamed,
            # so _answer_displayed is still False).
            fallback = AIMessage(
                content=(
                    "I wasn't able to produce a response for that. "
                    "Could you rephrase or give me a bit more detail?"
                )
            )
            fallback.additional_kwargs["reasoning_content"] = thinking
            self._stop_spinner()
            return {"messages": [fallback], "thinking": thinking}

        return {"messages": [response], "thinking": thinking}

    def _continue_truncated_turn(
        self, messages, response, visible, thinking, active_model, config
    ) -> Optional[Dict[str, Any]]:
        """Auto-continue a turn cut off by the output-token limit.

        The model ran out of output budget mid-turn (typically reasoning + a
        partial answer). Feed the partial turn back with a short continuation
        instruction and re-stream, accumulating visible text, up to
        ``_max_continue_retries`` times — so the user never has to type
        "continue". Stops early when a turn finishes cleanly (not truncated) or
        emits a tool call (returned so the graph executes it).

        Returns the state dict to return from ``_call_model``, or None if nothing
        usable was produced (the caller then falls back to its message).
        """
        accumulated = visible or ""
        convo = list(messages)
        last_thinking = thinking

        for attempt in range(self._max_continue_retries):
            logger.debug(
                "Output-token truncation: auto-continuing (attempt %d/%d)",
                attempt + 1,
                self._max_continue_retries,
            )
            # Hand the partial assistant turn back, then ask it to keep going. The
            # nudge is phrased so the model resumes rather than restarts.
            convo = convo + [
                response,
                HumanMessage(
                    content=(
                        "Your previous response was cut off by the output length "
                        "limit. Continue exactly where you left off — do not repeat "
                        "what you already wrote."
                    )
                ),
            ]
            self._start_spinner()
            try:
                response, _ = self._stream_response(
                    convo, config, model=active_model, mark_answer=True
                )
            except _ContextOverflow:
                break  # the continuation prompt overflowed; give up gracefully

            if response is None:
                break

            self._capture_input_tokens(response)
            last_thinking = self._extract_thinking(response) or last_thinking
            part = self._extract_visible(response.content)
            if part:
                accumulated = (accumulated + part) if accumulated else part

            # A tool call means the model is ready to act — let the graph run it.
            if response.tool_calls:
                if accumulated and not self._extract_visible(response.content):
                    # Preserve the text gathered so far alongside the tool call.
                    response.content = accumulated
                if last_thinking and not response.additional_kwargs.get(
                    "reasoning_content"
                ):
                    response.additional_kwargs["reasoning_content"] = last_thinking
                return {"messages": [response], "thinking": last_thinking}

            # Finished cleanly (not truncated again): return the assembled answer.
            if not self._was_truncated_by_tokens(response):
                if accumulated:
                    final = AIMessage(content=accumulated)
                    if last_thinking:
                        final.additional_kwargs["reasoning_content"] = last_thinking
                    return {"messages": [final], "thinking": last_thinking}
                return None
            # Still truncated → loop and continue again (budget permitting).

        # Retries exhausted while still truncated. Keep whatever we assembled.
        if accumulated:
            final = AIMessage(content=accumulated)
            if last_thinking:
                final.additional_kwargs["reasoning_content"] = last_thinking
            return {"messages": [final], "thinking": last_thinking}
        return None

    def _compact_and_rebuild(self, current_messages: list) -> Optional[list]:
        """Force-compact history, then rebuild the model's message list from the
        compacted state so the current turn can be retried on a smaller prompt.

        Returns the new message list, or None if nothing could be compacted (so
        the caller stops instead of looping on an unshrinkable prompt).
        """
        compact = getattr(self, "_compact_provider", None)
        if compact is None:
            return None
        # Token count before/after tells us whether compaction actually helped;
        # if it couldn't shrink anything, retrying is pointless.
        before = len(self._messages)
        try:
            compact(force=True)
        except Exception as ce:
            logger.error(f"Forced compaction failed: {ce}")
            return None
        if len(self._messages) >= before:
            return None  # nothing summarized away — don't loop

        # Rebuild: fresh system prompt (now carrying the summary) + the compacted
        # history. sanitize_tool_pairs repairs any tool call/result orphaned by
        # the split so strict providers accept the retry.
        rebuilt = self._sanitize_tool_pairs(list(self._messages))
        if self.system_prompt:
            rebuilt = [SystemMessage(content=self.system_prompt)] + rebuilt
        return rebuilt

    def _stream_response(
        self,
        messages: list,
        config: dict,
        print_reasoning: bool = True,
        model=None,
        mark_answer: bool = False,
        quiet: bool = False,
    ) -> tuple:
        """Stream a model response, handling spinner and output.

        Retries a completely empty turn (transient endpoint hiccup) up to
        ``_empty_response_retries`` times, AND re-runs the turn on a dead/stalled
        stream — an idle-timeout or transient network error (e.g. the socket died
        when the laptop slept) — with exponential backoff, so a wedged connection
        recovers on wake instead of hanging the worker. ``mark_answer`` prints a
        marker before the answer on user-facing turns. ``quiet`` streams silently
        (spawned sub-agents) — still idle-timeout/retry-protected, no display.
        Returns ``(response, had_reasoning)``.
        """
        active_model = model or self.model_with_tools
        # Provider-agnostic egress guard: repair any invalid reasoning block before
        # it's re-fed (see message_sanitizer.strip_malformed_reasoning). This is
        # the single chokepoint for the streamed paths — main, quiet sub-agent,
        # salvage, continue-truncated, aggregate — so no path replays a block the
        # provider would reject (e.g. Anthropic's thinking.thinking: Field
        # required). Thinking-only: unlike sanitize_tool_pairs it does NOT drop
        # orphan tool pairs, which mid-continuation are legitimately in flight.
        messages = self._strip_malformed_reasoning(messages)
        attempts = getattr(self, "_empty_response_retries", 0) + 1
        for attempt in range(attempts):
            try:
                response, had_reasoning = self._stream_once(
                    active_model, messages, config, print_reasoning, mark_answer,
                    quiet=quiet,
                )
            except Exception as e:
                # Context overflow is the caller's to handle — never retry it here.
                if isinstance(e, _ContextOverflow):
                    raise
                retriable = isinstance(e, _StreamIdleTimeout) or (
                    self._is_transient_network_error(e)
                )
                if not retriable or attempt == attempts - 1:
                    raise
                delay = self._network_retry_delay(attempt)
                logger.warning(
                    "Stream connection failed (%s); retrying turn on a fresh "
                    "connection in %.1fs (attempt %d/%d)",
                    e, delay, attempt + 1, attempts,
                )
                # Abortable backoff: wait ON the cancel event, not time.sleep — so
                # Esc/Ctrl+C wakes it INSTANTLY (an async KeyboardInterrupt can't
                # preempt a C-level time.sleep, which is why cancel felt stuck for
                # the whole backoff). If cancelled mid-wait, abort the turn now.
                if self._sleep_or_cancel(delay):
                    raise KeyboardInterrupt("cancelled during retry backoff")
                if not quiet:  # quiet runs may be concurrent — don't touch spinner
                    self._start_spinner()
                continue
            # Retry only a completely empty turn; the reasoning-only case is the
            # caller's responsibility.
            if not self._is_empty_response(response) or attempt == attempts - 1:
                return response, had_reasoning
            logger.debug(
                "Empty model response (attempt %d/%d); retrying",
                attempt + 1,
                attempts,
            )
            if not quiet:
                self._start_spinner()
        return response, had_reasoning

    def _network_retry_delay(self, attempt: int) -> float:
        """Exponential backoff (seconds) for a network-error stream retry, using
        the same LLM.RETRY_DELAY / RETRY_BACKOFF knobs as the rest of the app.

        The config read stays here (tests patch this module's ``config``); the
        capped-exponential math delegates to :func:`stream_policy.network_retry_delay`.
        """
        llm = config.get("LLM", {})
        base = float(llm.get("RETRY_DELAY", 1.0))
        factor = float(llm.get("RETRY_BACKOFF", 2.0))
        return stream_policy.network_retry_delay(attempt, base, factor)

    def _sleep_or_cancel(self, delay: float) -> bool:
        """Delegates to :func:`stream_policy.sleep_or_cancel` with this agent's
        cancel event (``None`` on a bare object → a plain short sleep)."""
        return stream_policy.sleep_or_cancel(getattr(self, "_cancel_event", None), delay)

    @staticmethod
    def _is_context_overflow_error(exc: Exception) -> bool:
        """Delegates to :func:`stream_policy.is_context_overflow_error`."""
        return stream_policy.is_context_overflow_error(exc)

    @staticmethod
    def _is_transient_network_error(exc: Exception) -> bool:
        """Delegates to :func:`stream_policy.is_transient_network_error`."""
        return stream_policy.is_transient_network_error(exc)

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

    def _iter_stream_with_idle_timeout(self, active_model, messages, config):
        """Yield stream chunks, but raise :class:`_StreamIdleTimeout` if no chunk
        arrives within ``self._stream_idle_timeout`` seconds.

        ``model.stream()`` is a blocking C-level socket read; if the connection
        dies silently (laptop sleep), it can park the worker thread forever — which
        also freezes the (FIFO, single-worker) UI. We run the stream on a daemon
        reader thread feeding a queue and time each ``get``. On timeout we raise and
        ABANDON the reader thread (it's a daemon; it unwinds when its socket finally
        errors) rather than blocking on it — the whole point is not to wait. When
        the timeout is 0/disabled, we iterate directly with no extra thread."""
        idle = getattr(self, "_stream_idle_timeout", 0) or 0
        if idle <= 0:
            yield from active_model.stream(messages, config=config)
            return

        q: "queue.Queue" = queue.Queue()

        def _reader():
            try:
                for chunk in active_model.stream(messages, config=config):
                    q.put((None, chunk))
            except BaseException as e:  # propagate the stream's own error verbatim
                q.put((e, None))
            else:
                q.put((self._STREAM_DONE, None))

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        # Poll in short slices so a cancel (Esc/Ctrl+C) is noticed within
        # ~POLL seconds instead of after the full `idle` window — a blocking
        # queue.get(timeout=120) can't be preempted by the async KeyboardInterrupt,
        # which is why a stalled-stream cancel felt stuck. We still only declare an
        # idle-timeout once `idle` seconds have truly elapsed with no chunk.
        poll = min(idle, self._CANCEL_POLL_SECONDS)
        waited = 0.0
        while True:
            if self._cancelled():
                raise KeyboardInterrupt("cancelled while waiting for stream")
            try:
                item, chunk = q.get(timeout=poll)
            except queue.Empty:
                waited += poll
                if waited >= idle:
                    # No chunk for `idle` seconds — the stream is wedged (likely a
                    # dead socket after sleep). Abandon the reader; let retry re-run.
                    raise _StreamIdleTimeout(
                        f"No stream data for {idle:.0f}s (connection likely dropped)"
                    )
                continue
            waited = 0.0  # a chunk (or terminal item) arrived — reset the idle clock
            if item is self._STREAM_DONE:
                return
            if item is not None:  # the reader captured an exception — re-raise it
                raise item
            yield chunk

    def _stream_once_quiet(self, active_model, messages: list, config: dict) -> tuple:
        """Stream a turn SILENTLY, accumulating the response with no display.

        Used by quiet (spawned) sub-agents: it drives the same idle-timeout stream
        iterator as the visible path — so a dead/stalled socket still raises
        ``_StreamIdleTimeout`` and gets retried by ``_stream_response`` — but emits
        nothing and touches no shared display state, so N of these can run on
        concurrent threads without corrupting the terminal. Returns
        ``(response, had_reasoning=False)`` to match ``_stream_once``'s shape."""
        response = None
        try:
            for chunk in self._iter_stream_with_idle_timeout(
                active_model, messages, config
            ):
                response = chunk if response is None else response + chunk
        except Exception as e:
            if self._is_context_overflow_error(e):
                raise _ContextOverflow(e) from e
            raise
        return response, False

    def _stream_once(
        self,
        active_model,
        messages: list,
        config: dict,
        print_reasoning: bool = True,
        mark_answer: bool = False,
        quiet: bool = False,
    ) -> tuple:
        """Single streaming attempt (see _stream_response for the retry wrapper).

        ``quiet=True`` (spawned sub-agents) STILL streams — so it inherits the
        stalled-stream idle-timeout + network retry (a sub-agent whose socket dies
        on laptop-sleep must not hang) — but suppresses ALL display and touches NO
        shared display state (no ``self._code_formatter``, spinner, ``print``, or
        reasoning sink), accumulating chunks with a LOCAL formatter-free loop so it
        is safe to run several concurrently.
        """
        if quiet:
            return self._stream_once_quiet(active_model, messages, config)
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
            for chunk in self._iter_stream_with_idle_timeout(
                active_model, messages, config
            ):
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
                        # Record that this turn has shown a visible answer, so the
                        # invoke() safety net doesn't re-emit it.
                        self._answer_displayed = True
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
            self._stop_spinner()
            # Context-window overflow: let the caller (_call_model) handle it — it
            # can compact and re-invoke on the shrunken history so the in-flight
            # task CONTINUES, instead of this turn dead-ending. Re-raise a typed
            # marker so the caller distinguishes it from a generic stream error.
            if self._is_context_overflow_error(e):
                if sink is not None:
                    sink.stop()
                raise _ContextOverflow(e) from e
            # Any other stream error (idle timeout, transient network drop, or a
            # mid-stream API error like a 500/overloaded/api_error): RE-RAISE so
            # the retry wrapper (_stream_response) re-runs the turn on a fresh
            # streaming connection with abortable backoff. We deliberately do NOT
            # fall back to a blocking `active_model.invoke()` here: that call can't
            # be cancelled (Esc/Ctrl+C can't preempt a C-level request), so it
            # wedged the turn on exactly the errors seen in the field; retrying the
            # STREAM keeps the turn interruptible and idle-timeout-protected. Any
            # partial output is discarded — a mid-stream failure can't be resumed
            # and partial reasoning would be invalid anyway.
            if sink is not None:
                sink.stop()
            raise
        finally:
            # Never leave a lingering transient block (cancel / error / no flush).
            if sink is not None:
                sink.stop()

        return response, had_reasoning

    @staticmethod
    def _answer_marker() -> str:
        """The cyan ● prefix for a streamed answer (prepended to the first chunk)."""
        return "\033[36m●\033[0m "

    def _emit_answer(self, text: str) -> None:
        """Display an answer that was PRODUCED WITHOUT STREAMING, rendered exactly
        like a streamed one (``●`` marker + markdown via ``CodeFormatter``).

        The turn's answer is normally shown live inside ``_stream_once``. A few
        paths instead compute the final text without a visible stream — the
        orchestrator single-subtask result, the aggregation-fallback concatenation,
        and the context-overflow / stream-error / recursion terminal messages —
        and would otherwise reach the user as a silent turn (only ``[Context: N]``
        prints). ``invoke()`` calls this as a safety net when ``_answer_displayed``
        is still False, so every path shows its answer. Idempotent-safe: it sets
        ``_answer_displayed`` so it can't double-print."""
        if not text or self._answer_displayed:
            return
        self._stop_spinner()
        fmt = CodeFormatter()
        fmt.process_chunk(self._answer_marker() + text)
        fmt.flush()
        # Commit the final line to scrollback (patch_stdout only commits on a
        # newline; the pinned UI would otherwise erase an uncommitted tail).
        if getattr(self, "styled_turn_view", False):
            print(flush=True)
        self._answer_displayed = True

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
        for cb in getattr(self, "callbacks", None) or []:
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
        for cb in getattr(self, "callbacks", None) or []:
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
            "web_crawler": "Crawling the page",
            "web_search": "Searching the web",
            "describe_image": "Analyzing image",
            "start_background_task": "Starting background task",
        }
        return labels.get(tool_name, f"Running {tool_name}")

    def _invoke_tool(self, tool, tool_name: str, tool_args: dict, quiet: bool = False):
        """Invoke a tool with a progress spinner, unless it's self-reporting
        (``_SELF_REPORTING_TOOLS`` emit their own progress → spinner left off).

        ``quiet=True`` (a spawned sub-agent's tool call) touches NO spinner: the
        run may be one of several concurrent sub-agents, and the parent/batch owns
        the shared spinner — per-tool start/stop from a pool thread would clobber
        it (and race the other sub-agents)."""
        if quiet:
            return tool.invoke(tool_args)
        if tool_name in self._SELF_REPORTING_TOOLS:
            self._stop_spinner()
            return tool.invoke(tool_args)

        self._start_spinner(self._tool_progress_label(tool_name, tool_args))
        try:
            return tool.invoke(tool_args)
        finally:
            self._stop_spinner()

    @staticmethod
    def _reasoning_content_text(block: dict) -> str:
        """Delegates to :func:`response_parsing.reasoning_content_text`."""
        return response_parsing.reasoning_content_text(block)

    @staticmethod
    def _reasoning_summary_text(block: dict) -> str:
        """Delegates to :func:`response_parsing.reasoning_summary_text`."""
        return response_parsing.reasoning_summary_text(block)

    def _extract_thinking(self, response) -> Optional[str]:
        """Delegates to :func:`response_parsing.extract_thinking`."""
        return response_parsing.extract_thinking(response)

    @staticmethod
    def _was_truncated_by_tokens(response) -> bool:
        """Delegates to :func:`response_parsing.was_truncated_by_tokens`."""
        return response_parsing.was_truncated_by_tokens(response)

    def _extract_visible(self, content) -> str:
        """Delegates to :func:`response_parsing.extract_visible`."""
        return response_parsing.extract_visible(content)

    def _disable_reasoning(self) -> dict:
        """Temporarily disable reasoning; returns saved state for _restore."""
        return disable_reasoning(self.model)

    def _restore_reasoning(self, saved: dict) -> None:
        """Restore reasoning settings saved by _disable_reasoning()."""
        restore_reasoning(self.model, saved)

    def _extract_content(self, chunk) -> tuple[str, str]:
        """Delegates to :func:`response_parsing.extract_content`."""
        return response_parsing.extract_content(chunk)

    @staticmethod
    def _elide_middle(text: str, limit: int = 72) -> str:
        """Delegates to :func:`tool_formatting.elide_middle`."""
        return tool_formatting.elide_middle(text, limit)

    @staticmethod
    def _format_tool_call(tool_call: dict) -> str:
        """Delegates to :func:`tool_formatting.format_tool_call`."""
        return tool_formatting.format_tool_call(tool_call)

    def _print_tool_marker(self, tool_call: dict) -> None:
        """Print one tool-call marker: the styled name + ↳arg block (pinned UI)
        or the plain ``[⚙ …]`` marker. Shared by both tool-exec chokepoints so
        the main loop and the worker loop render identically."""
        # exit_plan_mode is a client-side meta tool — the approval UI renders the
        # plan as a formatted block, so skip the flattened ↳ plan=… marker.
        if tool_call.get("name") == "exit_plan_mode":
            return
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

    @staticmethod
    def _normalize_tool_args(args: Any) -> Any:
        """Delegates to :func:`tool_formatting.normalize_tool_args`."""
        return tool_formatting.normalize_tool_args(args)

    def _truncate_tool_result(self, text: str) -> str:
        """Cap a tool result to the configured char budget (0 disables)."""
        return tool_formatting.truncate_tool_result(
            text, getattr(self, "_max_tool_result_chars", 100_000)
        )

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

    def _handle_exit_plan_mode(self, plan: str, allowed_bash: list = None) -> str:
        """Drive the plan-approval flow for an ``exit_plan_mode`` call; returns
        the ToolMessage content the model sees next.

        Shows the plan + an inline approval prompt via ``_plan_approval_ui``
        (approve|edit|keep_planning). On approve, ``_exit_plan_mode_provider``
        flips plan mode off + persists the plan, and the model is told to execute
        it now. Without a UI hook (non-TTY/tests) it auto-approves so scripted
        runs never block. ``edit`` is resolved inside the UI (it re-prompts after
        editing), so only approve/keep_planning reach here.

        ``allowed_bash`` (from the tool call) is the list of commands the plan
        pre-declared; on approval they are registered so they auto-confirm during
        execution instead of re-prompting per command."""
        plan = (plan or "").strip()
        allowed_bash = [str(c).strip() for c in (allowed_bash or []) if str(c).strip()]
        ui = getattr(self, "_plan_approval_ui", None)
        if ui is not None:
            verdict, plan = ui(plan)  # UI may return an edited plan
        else:
            verdict = "approve"  # non-TTY/tests: auto-approve

        if verdict == "keep_planning":
            return (
                "The user wants to keep planning — plan mode is still ON "
                "(read-only). Refine the plan based on their feedback and call "
                "exit_plan_mode again when it's ready."
            )
        # approve (the default, incl. the no-UI auto-approve path).
        provider = getattr(self, "_exit_plan_mode_provider", None)
        if provider is not None:
            try:
                provider(plan)
            except Exception as e:
                logger.error(f"exit_plan_mode approval provider failed: {e}")
        # Pin the full toolset for the execution turns: routing must not narrow
        # tools now that the model may edit/run anything to carry out the plan.
        self._execute_plan_route = True
        # Pre-approved commands auto-confirm during execution (plan mode is now
        # off, so they'd otherwise hit the per-command Proceed? gate).
        if allowed_bash:
            self._preapproved_bash = list(allowed_bash)
        note = ""
        if allowed_bash:
            note = (
                "\n\nThese commands were pre-approved and will run without a "
                "confirmation prompt: " + ", ".join(allowed_bash)
            )
        return (
            "The user APPROVED the plan. Plan mode is now OFF — you may make "
            "changes. Execute the approved plan now, following it step by step. "
            "The approved plan:\n\n" + plan + note
        )

    def _subagent_tools(self, agent) -> List[BaseTool]:
        """Delegates to :func:`subagent_runner.subagent_tools` (``agent`` here is
        the sub-agent type definition)."""
        return subagent_runner.subagent_tools(self, agent)

    def _run_spawn_batch(self, tool_calls: list) -> Dict[str, str]:
        """Delegates to :func:`subagent_runner.run_spawn_batch`."""
        return subagent_runner.run_spawn_batch(self, tool_calls)

    def _handle_spawn_agent(
        self, agent_type: str, prompt: str, description: str = "",
        in_batch: bool = False, run_in_background: bool = False,
    ) -> str:
        """Delegates to :func:`subagent_runner.handle_spawn_agent`."""
        return subagent_runner.handle_spawn_agent(
            self, agent_type, prompt, description,
            in_batch=in_batch, run_in_background=run_in_background,
        )

    @staticmethod
    def _wrap_subagent_result(
        agent_name: str, result: str, resumed: bool = False
    ) -> str:
        """Delegates to :func:`subagent_runner.wrap_subagent_result`."""
        return subagent_runner.wrap_subagent_result(agent_name, result, resumed=resumed)

    def _launch_background_subagent(self, agent, prompt: str, label: str) -> str:
        """Delegates to :func:`subagent_runner.launch_background_subagent`
        (``agent`` here is the sub-agent type definition)."""
        return subagent_runner.launch_background_subagent(self, agent, prompt, label)

    def _handle_resume_agent(
        self, agent_id: str, prompt: str, run_in_background: bool = True
    ) -> str:
        """Delegates to :func:`subagent_runner.handle_resume_agent`."""
        return subagent_runner.handle_resume_agent(
            self, agent_id, prompt, run_in_background=run_in_background
        )

    def drain_background_completions(self) -> List[BaseMessage]:
        """Delegates to :func:`subagent_runner.drain_background_completions`."""
        return subagent_runner.drain_background_completions(self)

    def has_undelivered_background(self) -> bool:
        """Delegates to :func:`subagent_runner.has_undelivered_background`."""
        return subagent_runner.has_undelivered_background(self)

    def _callback_free_model(self):
        """Delegates to :func:`subagent_runner.callback_free_model`."""
        return subagent_runner.callback_free_model(self)

    def _subagent_base_model(self, agent):
        """Delegates to :func:`subagent_runner.subagent_base_model` (``agent``
        here is the sub-agent type definition)."""
        return subagent_runner.subagent_base_model(self, agent)

    def _run_one_subagent(
        self, agent, prompt: str, label: str, drive_spinner: bool = True,
        kind: str = "spawn",
    ) -> str:
        """Delegates to :func:`subagent_runner.run_one_subagent` (``agent`` here
        is the sub-agent type definition)."""
        return subagent_runner.run_one_subagent(
            self, agent, prompt, label, drive_spinner=drive_spinner, kind=kind
        )

    @staticmethod
    def _tool_error_message(tool_name: str, exc: Exception) -> str:
        """Delegates to :func:`tool_formatting.tool_error_message`."""
        return tool_formatting.tool_error_message(tool_name, exc)

    def _is_preapproved_bash(self, command: str) -> bool:
        """Delegates to :func:`confirmation_gate.is_preapproved_bash`."""
        return confirmation_gate.is_preapproved_bash(self, command)

    def _confirm_tool(self, tool_name: str, tool_args: dict) -> bool:
        """Delegates to :func:`confirmation_gate.confirm`."""
        return confirmation_gate.confirm(self, tool_name, tool_args)

    def _prompt_confirm(self, header: str, detail: str, category: str) -> bool:
        """Delegates to :func:`confirmation_gate.prompt_confirm`."""
        return confirmation_gate.prompt_confirm(self, header, detail, category)

    def _spinner_snapshot(self) -> tuple:
        """Return ``(active, label)`` for the shared spinner, or ``(False, …)``.

        Reads the pinned-UI sink when present (the spinner's own ``spinning`` flag
        stays False in sink mode), else the stdout spinner's own state.
        """
        for cb in getattr(self, "callbacks", None) or []:
            sp = getattr(cb, "spinner", None)
            if sp is None:
                continue
            sink = getattr(sp, "_sink", None)
            if sink is not None:
                return sink.snapshot()
            return getattr(sp, "spinning", False), getattr(sp, "label", "Thinking")
        return False, "Thinking"

    def _client_side_tool_message(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_id: str,
        spawn_results: Optional[Dict[str, str]] = None,
    ) -> Optional[ToolMessage]:
        """Dispatch the three client-side stub tools (exit_plan_mode / spawn_agent
        / resume_agent) that are intercepted here rather than routed via MCP.

        Returns the ToolMessage to append for one of those, or None if ``tool_name``
        isn't client-side (the caller then runs it as a normal tool). When
        ``spawn_results`` holds ``tool_id`` (a batched, possibly parallel spawn),
        that precomputed result is used instead of spawning inline.
        """
        if tool_name == "exit_plan_mode":
            content = self._handle_exit_plan_mode(
                str(tool_args.get("plan", "")),
                tool_args.get("allowed_bash"),
            )
        elif tool_name == "spawn_agent":
            content = (
                spawn_results[tool_id]
                if spawn_results and tool_id in spawn_results
                else self._handle_spawn_agent(
                    str(tool_args.get("agent_type", "")),
                    str(tool_args.get("prompt", "")),
                    str(tool_args.get("description", "")),
                    run_in_background=tool_args.get("run_in_background", True),
                )
            )
        elif tool_name == "resume_agent":
            content = self._handle_resume_agent(
                str(tool_args.get("agent_id", "")),
                str(tool_args.get("prompt", "")),
                run_in_background=tool_args.get("run_in_background", True),
            )
        else:
            return None
        return ToolMessage(content=content, tool_call_id=tool_id, name=tool_name)

    def _execute_tools(self, state: AgentState) -> Dict[str, Any]:
        """Execute the tool calls on the last AI message."""
        last_message = state["messages"][-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": []}

        self._stop_spinner()

        route_tools = self._get_route_tools(state)

        # Visible tool marker so reasoning before/after a call is separated.
        if self.verbose:
            for tool_call in last_message.tool_calls:
                self._print_tool_marker(tool_call)

        # If the model requested several sub-agents in ONE turn, run them
        # concurrently (bounded pool) and stash each result by tool_id; the loop
        # below then just picks up its precomputed result. A single spawn is run
        # inline by the loop (no pool overhead).
        spawn_results = self._run_spawn_batch(last_message.tool_calls)

        tool_results = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = self._normalize_tool_args(tool_call["args"])
            tool_id = tool_call["id"]

            # exit_plan_mode / spawn_agent / resume_agent are handled client-side
            # (not via MCP); spawn_agent may reuse a batched parallel result.
            client_msg = self._client_side_tool_message(
                tool_name, tool_args, tool_id, spawn_results
            )
            if client_msg is not None:
                tool_results.append(client_msg)
                continue

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
                            content=self._truncate_tool_result(str(result)),
                            tool_call_id=tool_id,
                            name=tool_name,
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

        # Mid-turn steering: fold any messages the user typed while this tool
        # batch ran in AFTER the tool results, so the next model call sees them
        # and addresses them without the turn ending.
        steering = self._drain_steering()
        if steering and self.verbose:
            self._stop_spinner()
            for _ in steering:
                print(
                    "\n\033[90m[↳ steering: folding your message into this turn]\033[0m",
                    flush=True,
                )
            self._start_spinner()

        return {"messages": tool_results + steering}

    def _should_continue(self, state: AgentState) -> str:
        """"continue" if the last AI message has tool calls, else "end".

        A cancelled turn ends immediately even with pending tool calls — else the
        graph loops back into another model call after the tools (or a stopped
        spawn batch) return, so "stop all" wouldn't actually stop the turn. The
        worker/stream loops already honor the cancel; this closes the graph loop."""
        if self._cancelled():
            return "end"
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        return "end"

    def __call__(self, prompt: str) -> str:
        """Invoke the agent with a prompt."""
        return self.invoke(prompt)

    @classmethod
    def _strip_ephemeral(cls, text: str) -> str:
        """Remove ephemeral per-turn reminder blocks from a prompt for storage."""
        return cls._EPHEMERAL_BLOCK_RE.sub("", text)

    def invoke(self, prompt: str) -> str:
        """Invoke the agent with a prompt; returns the response string.

        A **delivery-only** turn (empty ``prompt`` with pending background
        completions) is supported: the finished sub-agent reports become the
        turn's input and the model addresses them with no user message. This is
        how a background completion auto-triggers a turn while the user is idle.
        """
        # Fresh cancel token for this turn (see request_cancel): a stale set flag
        # from a prior cancelled turn must not abort this one.
        cancel_ev = getattr(self, "_cancel_event", None)
        if cancel_ev is not None:
            cancel_ev.clear()

        # Reset the "answer shown" flag: streaming sets it True as it prints; the
        # safety net at the end of this method emits the answer if it's still False.
        self._answer_displayed = False

        # Deliver any background sub-agent that finished since the last turn: its
        # report is folded into history as a user message so THIS turn's model
        # call addresses it alongside the prompt (or on its own, if empty).
        bg = getattr(self, "drain_background_completions", None)
        delivered = 0
        if bg is not None:
            for m in bg():
                self._messages.append(m)
                delivered += 1

        # Buffer this turn's messages for the session log and flush ONCE at the
        # end. Buffering (rather than slicing _messages by a start index) is what
        # makes the log survive compaction: compaction REPLACES _messages
        # mid-turn, which would invalidate any index taken here.
        turn_log: List[BaseMessage] = []

        stored_prompt = self._strip_ephemeral(prompt)
        if not stored_prompt.strip():
            if delivered == 0:
                # Nothing to do — no prompt and no completion to deliver.
                return ""
            # Delivery-only turn: the drained completion messages ARE the input;
            # don't append an empty user turn.
        else:
            # Model sees the full prompt this turn; only the clean prompt is stored.
            user_msg = HumanMessage(content=stored_prompt)
            self._messages.append(user_msg)
            turn_log.append(user_msg)

        # Pre-flight compaction: shrink the accumulated history before it seeds this 
        # turn's graph run if it's over the high-water mark. The client provider owns 
        # the token-budget check and summarizes older turns into the system prompt. 
        # In-turn growth is bounded by the per-result cap (_truncate_tool_result) and 
        # the overflow backstop.
        compact = getattr(self, "_compact_provider", None)
        if compact is not None:
            compact()

        # The turn the model runs on: clean history + the full current prompt.
        # Delivery-only turn (no user prompt): the history already ends with the
        # drained completion message(s) — run on it as-is, no prompt substitution.
        if not stored_prompt.strip():
            model_messages = list(self._messages)
        else:
            model_messages = self._messages[:-1] + [HumanMessage(content=prompt)]
        initial_state: AgentState = {
            "messages": model_messages,
            "thinking": None,
            "route": None,
            "subtasks": None,
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
        except KeyboardInterrupt:
            # User cancelled mid-turn (the UI injects KeyboardInterrupt into this
            # worker thread). CLOSE the turn out rather than deleting it: keep the
            # user message and append an explicit interrupted marker.
            marker = AIMessage(content=INTERRUPTED_MARKER)
            self._messages.append(marker)
            # Log the cancelled turn too — resuming should show the question was
            # asked and interrupted, matching what the live history says.
            turn_log.append(marker)
            self._log_turn(turn_log)
            self._last_input_tokens = None  # stale after the appended marker
            raise
        except GraphRecursionError:
            logger.warning(
                "Agent stopped after the safety step limit (%d); the task may be "
                "looping. Returning the work so far — raise LLM.RECURSION_LIMIT "
                "if a legitimate task needs more steps.",
                self.recursion_limit,
            )
            self._stop_spinner()
            partial = self._last_visible_from(self._messages)
            msg = partial or (
                "I reached my safety step limit while working on that and "
                "couldn't finish. Try narrowing the request, or raise "
                "LLM.RECURSION_LIMIT in config if the task legitimately needs "
                "more steps."
            )
            self._emit_answer(msg)  # never streamed on this path — show it
            return msg

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
        turn_log.extend(new_messages)
        self._log_turn(turn_log)

        # Prefer the most recent AI turn with visible text. Emit it if the turn
        # produced it WITHOUT streaming (orchestrator single-subtask, aggregation
        # fallback, context-overflow / stream-error terminal messages) — the safety
        # net that guarantees no silent turn. `_emit_answer` is a no-op when the
        # answer already streamed (`_answer_displayed`), so a normal turn isn't
        # double-printed.
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                visible = self._extract_visible(msg.content)
                if visible:
                    self._emit_answer(visible)
                    return visible

        # Empty final turn: salvage the last tool result rather than return "".
        last_tool = self._last_tool_result(final_messages)
        if last_tool:
            salvaged = f"The last tool reported:\n{last_tool}"
            self._emit_answer(salvaged)
            return salvaged
        fallback = (
            "I wasn't able to produce a response for that. Could you rephrase "
            "or give me a bit more detail?"
        )
        self._emit_answer(fallback)
        return fallback

    def _log_turn(self, messages: List[BaseMessage]) -> None:
        """Append this turn to the session log, if one is attached (best-effort).

        The client sets ``session_log`` when session persistence is on; a plain
        agent (unit tests, embedded use) has none and this is a no-op.
        """
        log = getattr(self, "session_log", None)
        if log is None or not messages:
            return
        try:
            log.log_turn(messages)
        except Exception as e:  # noqa: BLE001 — a log must never break a turn
            logger.debug(f"Session log write failed: {e}")

    def _last_visible_from(self, messages: List[BaseMessage]) -> str:
        """Most recent visible AI text (to salvage a cut-short answer), or ""."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                visible = self._extract_visible(msg.content)
                # The interrupted marker is bookkeeping, not an answer — never
                # surface it to the user as "the work so far".
                if visible and visible.strip() != INTERRUPTED_MARKER:
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

    def clear_messages(self) -> None:
        """Clear the message history."""
        self._messages.clear()
        self._thinking = None
        self._last_input_tokens = None  # context is small again
        self._preapproved_bash = []  # plan-scoped approvals don't outlive a clear
        self._execute_plan_route = False  # plan-execution route pin is plan-scoped
        self._activity.clear()  # drop the agents-panel feed for a fresh context

