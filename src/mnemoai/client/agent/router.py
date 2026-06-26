"""Query router for classifying user queries into workflow categories."""

import re
from typing import Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from mnemoai.client.agent.reasoning_utils import disable_reasoning, restore_reasoning
from mnemoai.utils.config import config
from mnemoai.utils.logger import logger

# --- Deterministic fast-path signals -----------------------------------------
# Before spending an LLM round-trip on classification, we look for unambiguous
# signals in the query. A SINGLE clear signal routes deterministically (free,
# instant, and not subject to classifier misclassification). When TWO OR MORE
# signals appear — or none — we fall back to "full" / the LLM, because the safe
# failure mode is OVER-binding tools (harmless) not UNDER-binding (drops
# capability, which is the bug class we keep hitting).

# Image extensions -> a vision question. describe_image is an always-available
# meta tool, so any route can use it; we route these to "knowledge" (read-only,
# document-ish) which keeps the tool set tight while the vision tool is bound.
_IMAGE_EXT_RE = re.compile(
    r"\.(png|jpe?g|gif|bmp|webp|tiff?|ico|heic)\b", re.IGNORECASE
)
# A URL -> live web content (research).
_URL_RE = re.compile(r"https?://", re.IGNORECASE)
# A filesystem-ish path or code file reference -> code route.
_PATH_RE = re.compile(
    r"(?:^|\s)(?:~|\.{1,2})?/[\w./\-]+"            # /abs, ./rel, ~/home paths
    r"|[\w\-./]+\.(?:py|js|ts|tsx|jsx|go|rs|java|c|cpp|h|hpp|rb|php|sh|"
    r"yaml|yml|toml|json|md|txt|cfg|ini|sql)\b",   # bare code/config filenames
    re.IGNORECASE,
)
# Document formats handled by the knowledge route's readers.
_DOC_EXT_RE = re.compile(r"\.(pdf|docx?|csv|xlsx?|jsonl?)\b", re.IGNORECASE)
# Greetings / thanks / trivially short chit-chat -> simple_qa (no tools).
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|yo|thanks|thank you|thx|ok(ay)?|cool|great|nice|"
    r"good (morning|afternoon|evening)|bye|goodbye)\b[\s!.?]*$",
    re.IGNORECASE,
)
# Any of the deterministic content signals above (path, url, doc, image). A
# query carrying one of these is doing real work, not trivial chit-chat.
_SIGNAL_RES = (_IMAGE_EXT_RE, _URL_RE, _PATH_RE, _DOC_EXT_RE)


def is_trivial_query(query: str) -> bool:
    """True for a short, signal-free query that isn't worth orchestrating.

    The orchestrator (decompose → workers → aggregate) only earns its overhead
    on genuinely complex, multi-step tasks. A brief conversational prompt with no
    file/URL/doc signal ("can you do it?", "please do it", "what do you think?")
    should go straight to the normal call_model path — which binds the same tools
    and has the empty-turn safety net — instead of being decomposed into a single
    trivial subtask (which previously could surface a blank answer). Conservative:
    only very short, signal-free queries qualify; anything substantial or carrying
    a content signal is left to normal routing.
    """
    q = (query or "").strip()
    if not q:
        return True
    if any(rx.search(q) for rx in _SIGNAL_RES):
        return False
    # Word-count gate: short prompts are chit-chat / clarifications. Real
    # decomposable tasks are longer and more specific.
    return len(q.split()) <= 6

# Route definitions: maps route names to tool name lists.
# None means all tools (fallback).
ROUTE_TOOLS: Dict[str, Optional[List[str]]] = {
    "simple_qa": [],
    # code: everything for local development — filesystem writes/exec, git, and
    # the COMPLETE task-support suites (todos, plan-mode bookkeeping, background
    # tasks). fs_read / glob need not be listed: fs_read and describe_image are
    # always-available meta tools, and glob_search is bound here for code search.
    "code": [
        "fs_write",
        "file_edit",
        "glob_search",
        "grep_search",
        "execute_bash",
        "git_safe",
        "git_status_safe",
        "git_commit_safe",
        # Todos (complete suite)
        "todo_write",
        "todo_read",
        "todo_clear",
        # Plan-mode bookkeeping (complete suite)
        "enter_plan_mode",
        "add_plan_step",
        "add_plan_file",
        "add_plan_risk",
        "present_plan",
        "approve_plan",
        "exit_plan_mode",
        "get_plan_status",
        # Background tasks (complete suite — start/inspect/list/cancel/clear)
        "start_background_task",
        "get_task_status",
        "get_task_output",
        "list_background_tasks",
        "cancel_background_task",
        "wait_for_task",
        "clear_completed_tasks",
    ],
    "research": [
        "web_search",
        "web_crawler",
    ],
    # knowledge: querying user-provided documents via the RAG index. Reading a
    # specific file by path is fs_read (a meta tool, always available) — and
    # fs_read handles every format through its `mode` (PDF/CSV/JSON/DOCX/Line),
    # so there are no separate per-format reader tools to bind.
    "knowledge": [
        "list_documents",
        "search_in_documents",
        "clear_documents",
        "glob_search",
    ],
    "full": None,
}


def get_classifier_prompt() -> str:
    """Get the routing/classifier prompt from prompts.yaml.

    Returns:
        The ROUTING_PROMPT string.

    Raises:
        PromptError: if ROUTING_PROMPT is missing — routing is enabled, so its
        prompt is required (no in-code fallback).
    """
    return config.require_prompt("ROUTING_PROMPT")


class QueryRouter:
    """Routes queries to appropriate tool subsets based on classification."""

    def __init__(self, model: BaseChatModel) -> None:
        """Initialize the query router.

        Args:
            model: LangChain chat model for classification
        """
        self.model = model
        self._valid_routes = set(ROUTE_TOOLS.keys())

    def fast_route(self, query: str) -> Optional[str]:
        """Route deterministically from unambiguous signals, or None.

        Returns a route only when the query has exactly ONE clear signal (or is
        a plain greeting). With multiple signals it returns ``full`` (the task
        spans categories — bind everything); with none it returns None so the
        caller falls back to the LLM classifier. The bias is deliberate: a lone
        signal routes for free and reliably, but anything mixed/ambiguous never
        under-binds tools.

        Args:
            query: The user's query (the latest message text).

        Returns:
            A route name, or None to defer to the LLM classifier.
        """
        q = (query or "").strip()
        if not q:
            return None

        # Plain greeting / thanks / very short chit-chat -> no tools needed.
        if _GREETING_RE.match(q):
            return "simple_qa"

        # Collect signals. Each maps to the route whose tools it needs.
        signals = set()
        if _IMAGE_EXT_RE.search(q):
            signals.add("knowledge")  # vision tool is always-available; tight set
        if _URL_RE.search(q):
            signals.add("research")
        if _DOC_EXT_RE.search(q):
            signals.add("knowledge")
        if _PATH_RE.search(q):
            signals.add("code")

        if len(signals) == 1:
            return next(iter(signals))
        # 2+ signals -> spans categories, which is exactly what "full" is for
        # (bind everything; never under-bind). 0 signals -> defer to the LLM.
        if len(signals) >= 2:
            return "full"
        return None

    def classify(
        self,
        query: str,
        conversation_context: str = "",
    ) -> str:
        """Classify a query into a route category.

        First tries a deterministic heuristic fast-path (:meth:`fast_route`) to
        avoid an LLM round-trip on obvious cases; falls back to the LLM
        classifier only when the heuristics are inconclusive.

        Args:
            query: The user's query
            conversation_context: Recent conversation for context

        Returns:
            Route name (one of ROUTE_TOOLS keys)
        """
        # Deterministic fast-path — free, instant, not subject to classifier
        # misclassification. Skipped when there's conversation context, so short
        # follow-ups ("now commit it") still reach the LLM, which uses context
        # to disambiguate.
        if not conversation_context:
            fast = self.fast_route(query)
            if fast:
                logger.debug(f"Router fast-path -> {fast}")
                return fast

        messages: List[BaseMessage] = [
            SystemMessage(content=get_classifier_prompt()),
        ]

        if conversation_context:
            messages.append(
                HumanMessage(
                    content=(f"[Recent conversation context]\n{conversation_context}")
                )
            )

        messages.append(HumanMessage(content=query))

        try:
            # Temporarily suppress instance-level callbacks so the
            # spinner keeps running during classification, and disable
            # reasoning so the route lands in response.content (reasoning
            # models otherwise leave content empty).
            saved_callbacks = getattr(self.model, "callbacks", None)
            self.model.callbacks = None
            saved_reasoning = disable_reasoning(self.model)
            try:
                # Some endpoints (e.g. Mantle reasoning models) intermittently
                # return an empty/null response. A single retry recovers the
                # common case before we fall back to "full".
                route = ""
                for _ in range(2):
                    response = self.model.invoke(messages, config={"callbacks": []})
                    route = self._parse_route(response.content)
                    if route:
                        break
            finally:
                restore_reasoning(self.model, saved_reasoning)
                self.model.callbacks = saved_callbacks

            if not route:
                # Empty classification is recoverable (we route to "full", which
                # binds every tool), so this is debug-level, not a warning.
                logger.debug(
                    "Router produced no route after retry; falling back to 'full'"
                )
                return "full"

            logger.debug(f"Router classified query as: {route}")
            return route

        except Exception as e:
            logger.error(f"Router classification failed: {e}")
            return "full"

    def _parse_route(self, content: str) -> str:
        """Parse the model's classification response into a valid route.

        Handles thinking tags, extra whitespace, quotes, etc.

        Args:
            content: Raw model response (str or content-block list)

        Returns:
            A valid route name, or "" if the response was empty / unrecognized
            (the caller decides whether to retry or fall back to "full").
        """
        # Handle Bedrock-/Responses-style list content blocks (reasoning on)
        if isinstance(content, list):
            text = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            text = content or ""

        # Strip thinking tags if present
        text = re.sub(
            r"<think(?:ing)?>.*?</think(?:ing)?>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        route = text.strip().lower().strip("\"'.,!").strip()

        if route in self._valid_routes:
            return route

        if route:
            # Non-empty but not a known route: a genuine misclassification,
            # worth a debug note. Empty content is handled silently upstream.
            logger.debug(f"Router returned unrecognized route '{route}'")
        return ""
