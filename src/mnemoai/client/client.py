"""LangGraph-based client implementation."""

import asyncio
import json
import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from mcp import StdioServerParameters

from mnemoai.client import (
    context_injection,
    session_artifacts,
    transcript_export,
    usage_tracker,
)
from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.agent.message_codec import (
    convert_langchain_messages_to_strands,
    convert_strands_messages_to_langchain,
)
from mnemoai.client.agent.router import ROUTE_TOOLS, QueryRouter
from mnemoai.client.managers.agent_conversation_manager import (
    AgentConversationManager,
    messages_to_dict_list,
)
from mnemoai.client.managers.user_profile_manager import UserProfileManager
from mnemoai.client.mcp_config import load_external_servers
from mnemoai.client.mcp_tool_wrapper import MultiMCPClient
from mnemoai.client.memory.episodic_memory import EpisodicMemoryManager
from mnemoai.client.memory.playbook_store import PlaybookStore
from mnemoai.client.memory.reflector import Reflector, current_turn_messages
from mnemoai.client.session_log import (
    SessionLog,
    branch_session,
    first_user_prompt,
    read_session,
    turn_summaries,
)
from mnemoai.client.ui import turn_view
from mnemoai.client.ui.spinner import Spinner
from mnemoai.client.ui.streaming_callback import StreamingCallbackHandler
from mnemoai.models.controllers.llm_controller import LangChainLLMController
from mnemoai.utils.config import config
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import (
    SESSION_MAX_AGE_DAYS,
    conversations_dir,
    instance_id,
    model_dir,
    plans_dir,
    sanitize_model_name,
    sweep_old_plans,
    sweep_old_rag_artifacts,
    sweep_old_sessions,
)


class LangGraphClient:
    """LangGraph-based client for AI assistant."""

    def __init__(
        self,
        verbose: bool = False,
    ) -> None:
        self.verbose_mode = verbose

        # The MCP server runs as a subprocess (`python -m mnemoai.server.server`).
        # Prepend the package's parent dir to PYTHONPATH so the child can import
        # mnemoai from a checkout (<repo>/src) or an install (site-packages).
        import mnemoai

        pkg_parent = os.path.dirname(
            os.path.dirname(os.path.abspath(mnemoai.__file__))
        )
        # Pin this instance's id BEFORE copying the env so the MCP subprocess
        # inherits the same MNEMOAI_INSTANCE_ID — both halves then resolve the
        # same per-instance session-pointer file (multiple tabs don't clobber).
        instance_id()
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            pkg_parent + (os.pathsep + existing if existing else "")
        )

        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mnemoai.server.server"],
            env=env,
        )
        # Built-in server always launched; servers in ~/.mnemoai/mcp.json are
        # launched alongside it and their tools merged.
        external_servers = load_external_servers()
        self.mcp_client = MultiMCPClient(self.server_params, external_servers)

        self.profile_manager = UserProfileManager()
        self.system_prompt = self._build_system_prompt()

        self.session_id = self._new_session_id()

        self.agent: Optional[LangGraphAgent] = None
        self.tools = None
        self.model = None

        # User-toggled (/plan), enforced client-side: hard-blocks mutating/exec
        # tools and tells the model to research + present a plan.
        self.plan_mode_active: bool = False

        self.llm_controller = LangChainLLMController(verbose=self.verbose_mode)
        # Cache of same-provider models built for a custom sub-agent's 'model'
        # override (keyed by model name), so repeated spawns don't rebuild.
        self._subagent_model_cache = {}

        self.conversation_manager = AgentConversationManager(
            max_tokens=config.get("MAX_CONVERSATION_TOKENS", 1024 * 4)
        )

        self.spinner = Spinner()
        self.spinner_lock = threading.Lock()
        self.callback_handler = StreamingCallbackHandler(
            spinner=self.spinner,
            spinner_lock=self.spinner_lock,
        )

        self.episodic_memory = None
        if config.get("ENABLE_EPISODIC_MEMORY", False):
            self._initialize_episodic_memory()

        self.reflector = None
        self.playbook = None
        if config.get("ENABLE_PLAYBOOK", False):
            self._initialize_playbook()

        self.previous_query = None
        self.previous_response = None
        self.previous_messages = None

        # The currently-"open" conversation file: set on load or first save so a
        # bare /save overwrites it; reset by /clear.
        self.current_conversation_path = None

    def _build_system_prompt(self) -> str:
        """Delegates to :func:`context_injection.build_system_prompt`."""
        return context_injection.build_system_prompt(self)

    def _system_prompt_with_playbook(self) -> str:
        """``system_prompt`` plus the playbook block, as handed to the agent."""
        if not self.playbook:
            return self.system_prompt
        playbook_context = self._get_playbook_context()
        if not playbook_context:
            return self.system_prompt
        return f"{self.system_prompt}\n\n{playbook_context}"

    @staticmethod
    def _sanitize_for_path(name: str) -> str:
        """Delegate to ``utils.paths.sanitize_model_name`` (kept for callers)."""
        return sanitize_model_name(name)

    def _model_scoped_dir(self) -> str:
        """``<app_home>/{profile}/models/{model}/`` (created) — episodic memory
        and the playbook are model-scoped so switching models doesn't mix them."""
        model_name = config.get("MODEL_ID", {}).get("NAME", "default")
        return str(model_dir(model_name))

    def _initialize_episodic_memory(self) -> None:
        """Initialize episodic memory (model-scoped)."""
        logger.debug("Initializing episodic memory...")

        embed_model_config = config.get("RAG", {}).get("EMBED_MODEL_ID")
        if not embed_model_config:
            raise ValueError(
                "RAG.EMBED_MODEL_ID must be configured for episodic memory"
            )

        # Model-scoped so switching models doesn't contaminate the vector store.
        episodic_path = os.path.join(self._model_scoped_dir(), "episodic_memory")
        os.makedirs(episodic_path, exist_ok=True)

        store_type = (
            config.get("EPISODIC_MEMORY", {}).get("STORE_TYPE", "chromadb").lower()
        )

        from mnemoai.models.controllers.embeddings_controller import (
            EmbeddingsController,
        )

        embeddings_controller = EmbeddingsController(embed_model_config)

        self.episodic_memory = EpisodicMemoryManager(
            persist_path=episodic_path,
            store_type=store_type,
            embeddings_controller=embeddings_controller,
        )
        self.episodic_memory.cleanup(max_episodes=1000, max_age_days=90)
        logger.debug(f"✓ {store_type.upper()} episodic memory initialized")

    def _initialize_playbook(self) -> None:
        """Initialize ACE Reflector and Playbook store."""
        logger.debug("Initializing ACE Playbook...")

        # Model-scoped so one model's strategies don't leak into another's runs.
        playbook_path = os.path.join(self._model_scoped_dir(), "playbook")
        os.makedirs(playbook_path, exist_ok=True)

        # Embeddings for semantic deduplication, if available.
        embeddings = None
        if config.get("RAG", {}).get("EMBED_MODEL_ID"):
            try:
                from mnemoai.models.controllers.embeddings_controller import (
                    EmbeddingsController,
                )

                embeddings = EmbeddingsController()
            except Exception as e:
                logger.warning(f"Could not initialize embeddings for playbook: {e}")

        self.reflector = Reflector(persist_path=playbook_path)
        self.playbook = PlaybookStore(
            persist_path=playbook_path,
            embeddings_controller=embeddings,
            max_entries=config.get("PLAYBOOK", {}).get("MAX_ENTRIES", 500),
            similarity_threshold=config.get("PLAYBOOK", {}).get(
                "SIMILARITY_THRESHOLD", 0.85
            ),
        )

        stats = self.playbook.get_stats()
        logger.debug(f"✓ Playbook initialized ({stats['total_entries']} entries)")

    def start(self, verbose: bool = False) -> None:
        """Start the client and initialize the agent."""
        try:
            self.verbose_mode = verbose

            # Startup housekeeping: prune stale approved-plan files so the plans
            # dir doesn't grow without bound (best-effort, never blocks startup).
            try:
                sweep_old_plans()
            except Exception as e:
                logger.debug(f"Plan sweep skipped: {e}")

            # Prune ORPHANED per-session RAG/chunk artifacts left by a crashed
            # instance. Age-based so a concurrently-running instance's fresh
            # files are never deleted (fixes the multi-tab delete-all bug — exit
            # no longer wildcard-sweeps, so this bounds crash leftovers instead).
            try:
                sweep_old_rag_artifacts()
            except Exception as e:
                logger.debug(f"RAG artifact sweep skipped: {e}")

            # Expire old resumable session transcripts (age-based, every project
            # dir). `/save` files live elsewhere and are never swept.
            try:
                sweep_old_sessions(self._session_max_age_days())
            except Exception as e:
                logger.debug(f"Session sweep skipped: {e}")

            # Fail fast if prompts.yaml is missing a required prompt (a feature's
            # prompt is required when that feature is enabled).
            config.validate_prompts(
                routing=config.get("ENABLE_ROUTING", False),
                orchestration=config.get("ENABLE_ORCHESTRATION", False),
            )

            with self.mcp_client:
                self.tools = self.mcp_client.list_tools_sync()
                logger.info(f"Loaded {len(self.tools)} tools from MCP server")

                if config.get("ENABLE_RAG", False):
                    self._initialize_rag_session()
                self._initialize_chunk_cache()

                self.llm_controller.initialize_model(callbacks=[self.callback_handler])
                self.model = self.llm_controller.get_model()

                # Append playbook context to the system prompt.
                system_prompt_with_context = self._system_prompt_with_playbook()

                router = None
                tool_routes = None
                if config.get("ENABLE_ROUTING", False):
                    router = QueryRouter(self.model)
                    tool_routes = ROUTE_TOOLS
                    logger.info("Query routing enabled")

                # Orchestration requires routing.
                orchestrator_enabled = (
                    config.get("ENABLE_ORCHESTRATION", False) and router is not None
                )
                if orchestrator_enabled:
                    logger.info("Orchestrator enabled for complex tasks")

                self.agent = LangGraphAgent(
                    model=self.model,
                    tools=self.tools,
                    system_prompt=system_prompt_with_context,
                    verbose=self.verbose_mode,
                    callbacks=[self.callback_handler],
                    router=router,
                    tool_routes=tool_routes,
                    orchestrator_enabled=orchestrator_enabled,
                    plan_mode_provider=lambda: self.plan_mode_active,
                )
                # Mid-loop compaction hook: the agent calls this (sync) before a
                # model call when history exceeds its high-water mark, reusing the
                # same manager as the post-turn /compact path.
                self.agent._compact_provider = self._compact_now
                # Attribute usage to the configured model, and let the router
                # (a model call the user never sees) count toward the same totals.
                self.agent.usage_model_name = self.model_name_for_log() or "model"
                if router is not None:
                    router.usage = self.agent.usage
                    router.usage_model_name = self.agent.usage_model_name
                # Per-agent model override (custom sub-agent frontmatter 'model'):
                # build a same-provider model with the NAME swapped, on demand.
                self.agent._subagent_model_factory = self._subagent_model_factory
                # Append-only session transcript for `--resume`, scoped to the
                # launch directory. Independent of /save (user-curated, never
                # swept); this one expires on age.
                self._attach_session_log()
                # Let a blocking MCP tool call notice Esc. The worker parks in an
                # uninterruptible wait, so without this probe a cancel can't land
                # until the call's deadline — up to ten minutes on a long tool.
                probe = getattr(self.mcp_client, "set_cancel_probe", None)
                if probe is not None:
                    probe(lambda: self.agent is not None and self.agent._cancelled())

        except Exception as e:
            logger.error(traceback.format_exc())
            raise e

    def query(self, prompt: str) -> str:
        """Send a query to the agent and return its response."""
        if not self.agent:
            raise RuntimeError("Client not started. Call start() first.")

        self.callback_handler.reset()
        with self.spinner_lock:
            self.spinner.start()

        # Delivery-only turn (empty prompt): a background sub-agent finished and
        # we're auto-triggering a turn to surface it. Skip the per-turn prompt
        # injections (episodic/plan/steering) — there's no user prompt to frame —
        # and let the agent run on the drained completion messages alone.
        delivery_only = not (prompt or "").strip()

        try:
            if not delivery_only and self.episodic_memory:
                prompt = self._inject_episodic_context(prompt)

            # Plan mode: remind the model per-turn that it's read-only (the
            # system prompt is frozen at session start).
            if not delivery_only and self.plan_mode_active:
                prompt = self._plan_mode_reminder() + prompt

            # STEERING.md: user-authored always-on instructions, prepended last so
            # they LEAD the prompt (highest priority). Re-read each turn; stripped
            # before storage, so compaction never summarizes them away.
            if not delivery_only:
                steering = self._steering_reminder()
                if steering:
                    prompt = steering + prompt

            with self.mcp_client:
                response = self.agent(prompt)

                if hasattr(self.agent, "_code_formatter"):
                    self.agent._code_formatter.flush()

                asyncio.run(
                    self.conversation_manager.manage_messages(
                        self, self._summary_model(), self.agent
                    )
                )

                self._profile_turn()

                token_count = self._count_context_tokens()
                print(f"\n\033[90m[Context: {token_count} tokens]\033[0m")

                if self.episodic_memory:
                    self.previous_query = prompt
                    self.previous_response = response
                    self.previous_messages = self.agent.messages.copy()

                return response

        except KeyboardInterrupt:
            with self.spinner_lock:
                self.spinner.stop()
            return "Operation was cancelled."

        except Exception as e:
            # Clean user-facing message for any model/MCP/runtime failure.
            with self.spinner_lock:
                self.spinner.stop()
            logger.error(f"Query failed: {e}", exc_info=True)
            msg = (
                "Something went wrong while processing that request "
                f"({type(e).__name__}). Your conversation is intact — "
                "please try again or rephrase."
            )
            # The turn errored before/without streaming an answer, so PRINT this
            # (nothing else will) — otherwise the turn ends silently with only
            # the traceback in the log and no user-facing message.
            print(f"\n\033[91m{msg}\033[0m", flush=True)
            return msg

        finally:
            with self.spinner_lock:
                self.spinner.stop()

    def compact_conversation(self, focus_instructions: str = "") -> bool:
        """Manually compact the conversation (/compact); True if it ran.

        Summarizes older messages, keeping recent turns verbatim.
        ``focus_instructions`` optionally guides what the summary emphasizes.
        """
        if not self.agent:
            return False
        return asyncio.run(
            self.conversation_manager.compact(
                self, self._summary_model(), self.agent, focus_instructions
            )
        )

    def _summary_model(self):
        """The model used for compaction summaries.

        Defaults to a reasoning-DISABLED variant of the main model (built once,
        lazily): a summary doesn't benefit from a slow extended-thinking pass, and
        on a max-reasoning model that pass is the dominant cost of compaction.
        Provider-agnostic (the controller clears REASONING/REASONING_EFFORT, a
        no-op for providers without thinking). Set ``LLM.SUMMARIZATION_THINK: true``
        to keep thinking on (then the main model is reused). Falls back to the main
        model if building the variant fails."""
        if config.get("LLM", {}).get("SUMMARIZATION_THINK", False):
            return self.model
        cached = getattr(self, "_summary_model_cached", "unset")
        if cached == "unset":
            try:
                cached = self.llm_controller.build_non_reasoning_model()
            except Exception as e:
                logger.warning(f"Non-reasoning summary model unavailable, using main model: {e}")
                cached = self.model
            self._summary_model_cached = cached
        return cached

    def _compact_now(self, force: bool = False) -> bool:
        """Mid-loop compaction hook the agent calls before each model call.

        Compacts when history exceeds the high-water mark (default 80% of
        ``MAX_CONVERSATION_TOKENS``; ``LLM.COMPACT_HIGH_WATER_TOKENS`` overrides,
        0 disables the proactive check). ``force`` (the overflow backstop) skips
        the budget check and keeps a smaller recent window. Safe to call from the
        synchronous agent loop — no event loop runs on this thread during
        ``agent(prompt)`` (MCP has its own daemon-thread loop). Returns True if it
        summarized anything."""
        if not self.agent:
            return False
        mgr = self.conversation_manager
        if not force:
            high_water = config.get("LLM", {}).get(
                "COMPACT_HIGH_WATER_TOKENS", int(mgr.max_tokens * 0.8)
            )
            if high_water <= 0:
                return False
            # Prefer the provider's EXACT input_tokens from the last turn (ground
            # truth — the same number shown as [Context: N]). Our own estimate
            # tiktoken-counts the serialized message JSON and applies a per-provider
            # safety multiplier (e.g. 1.5x for mantle/anthropic), so it over-counts
            # the real prompt ~2x and would fire compaction far too early. Use the
            # estimate ONLY as a fallback before any turn has run (no actual yet).
            actual = getattr(self.agent, "_last_input_tokens", None) or 0
            if actual:
                current = actual
            else:
                current = mgr.count_tokens(messages_to_dict_list(self.agent.messages))
            if current <= high_water:
                return False
            # Cheapest layer first: evict OLD tool-result bodies (no LLM call).
            # Eviction shrinks history and nulls the provider's exact count, so
            # re-measure with the (conservative) estimate; if that alone brings us
            # back under the high-water mark, skip the expensive full summary.
            if mgr.evict_old_tool_results(self.agent):
                if mgr.count_tokens(messages_to_dict_list(self.agent.messages)) <= high_water:
                    return True
        keep = 2 if force else config.get("LLM", {}).get("KEEP_RECENT_MESSAGES", 6)
        try:
            return asyncio.run(
                mgr._compact(self, self._summary_model(), self.agent, keep_recent=keep)
            )
        except Exception as e:
            logger.error(f"Mid-loop compaction failed: {e}")
            return False

    def reflect_and_learn(self, task: str) -> None:
        """Reflect on the last interaction and update the playbook."""
        if not self.reflector or not self.playbook:
            return

        if not self.agent or not self.agent.messages:
            return

        try:
            entries = self.reflector.reflect_on_trajectory(
                messages=self.agent.messages,
                task=task,
            )

            if entries:
                self.playbook.append_batch(entries)
                logger.debug(f"Reflector: learned {len(entries)} strategies")
        except Exception as e:
            logger.error(f"Reflection failed: {e}")

    def auto_extract_memory(self, query: str, response: str) -> None:
        """Distill durable facts from the last exchange into MEMORY.md, in the
        background (the auto-learning counterpart to the model calling the
        ``memory`` tool itself).

        Opt-in via ``ENABLE_MEMORY_AUTO_EXTRACTION`` (default off): unlike the
        ``memory`` tool, this writes WITHOUT a confirmation prompt, so it's gated
        behind its own toggle and confined to ``MEMORY.md`` (it can only add /
        consolidate curated facts, never touch anything else). Runs on a daemon
        thread so the turn-end path doesn't block the UI on an extra model call.
        No-op when memory or the toggle is disabled, or the model isn't ready.
        """
        if not config.get("ENABLE_MEMORY", True):
            return
        if not config.get("ENABLE_MEMORY_AUTO_EXTRACTION", False):
            return
        if not self.model or not query or not response:
            return

        t = threading.Thread(
            target=self._auto_extract_memory_worker,
            args=(query, response),
            daemon=True,
        )
        t.start()

    def _auto_extract_memory_worker(self, query: str, response: str) -> None:
        """Background worker for :meth:`auto_extract_memory` (see its docstring)."""
        from mnemoai.client.memory.memory_store import MemoryError, MemoryStore

        try:
            prompt_template = config.prompt("MEMORY_EXTRACTION_PROMPT")
            if not prompt_template:
                return  # prompt unavailable → silently skip

            store = MemoryStore()
            existing = store.read().strip() or "(empty)"
            exchange = f"User: {query}\n\nAssistant: {response}"
            prompt = prompt_template.format(
                existing_memory=existing, exchange=exchange
            )

            raw = self._invoke_model_once(prompt)
            ops = self._parse_memory_ops(raw)
            if not ops:
                return

            applied = 0
            for op in ops:
                action = str(op.get("action", "")).strip().lower()
                try:
                    if action == "add" and op.get("text"):
                        store.add(str(op["text"]).strip())
                        applied += 1
                    elif action == "replace" and op.get("old_text") and op.get("text"):
                        store.replace(str(op["old_text"]), str(op["text"]).strip())
                        applied += 1
                except MemoryError as e:
                    # Cap reached or ambiguous match — skip this op, keep the rest.
                    logger.debug(f"Auto-memory op skipped: {e}")
            if applied:
                logger.debug(f"Auto-memory: applied {applied} operation(s)")
        except Exception as e:
            logger.error(f"Auto memory extraction failed: {e}")

    def _subagent_model_factory(self, model_name: str):
        """Build a callback-free chat model for a custom sub-agent's ``model``
        override. Reuses the configured provider/TYPE/params, swapping only
        ``MODEL_ID.NAME`` (provider-agnostic — a cheap sub-agent runs a cheaper
        model of the SAME provider). Cached by name; returns None on failure so
        the caller falls back to the parent model."""
        if model_name in self._subagent_model_cache:
            return self._subagent_model_cache[model_name]
        model = None
        try:
            ctrl = LangChainLLMController(verbose=self.verbose_mode)
            # Copy model_id (never mutate the shared config dict) and set both
            # forms of the name — mantle_factory reads model_id['NAME'], the
            # other providers read self.model_name.
            ctrl.model_id = {**ctrl.model_id, "NAME": model_name}
            ctrl.model_name = model_name
            ctrl.initialize_model(callbacks=None)
            model = ctrl.get_model()
        except Exception as e:
            logger.error(
                f"Sub-agent model override '{model_name}' failed to build; "
                f"using the default model: {e}"
            )
        self._subagent_model_cache[model_name] = model
        return model

    def _invoke_model_once(self, prompt: str) -> str:
        """Run a single, isolated model call for ``prompt`` and return its text.

        Used for background side-tasks (memory extraction) — deliberately does NOT
        touch the agent's conversation state. Supports both the LangChain
        (``invoke``) and Strands (``stream``) model shapes.
        """
        try:
            from langchain_core.messages import HumanMessage

            if hasattr(self.model, "invoke"):
                result = self.model.invoke([HumanMessage(content=prompt)])
                return str(getattr(result, "content", result))
        except Exception as e:
            logger.debug(f"LangChain single-invoke failed ({e}); trying stream")

        # Strands fallback: drain the stream into text.
        try:
            text = ""

            async def _run() -> str:
                nonlocal text
                async for event in self.model.stream(
                    [{"role": "user", "content": [{"text": prompt}]}]
                ):
                    delta = (
                        event.get("contentBlockDelta", {})
                        .get("delta", {})
                        .get("text")
                    )
                    if delta:
                        text += delta
                return text

            return asyncio.run(_run())
        except Exception as e:
            logger.debug(f"Single model invoke failed: {e}")
            return ""

    @staticmethod
    def _parse_memory_ops(raw: str) -> list:
        """Parse the extractor's reply into a list of memory-op dicts.

        Tolerant: strips code fences, extracts the outermost JSON array, and
        returns [] on anything malformed (a background task must never raise).
        """
        if not raw:
            return []
        text = raw.strip()
        # Strip ```json … ``` fences if the model added them despite instructions.
        if text.startswith("```"):
            text = text.split("```", 2)[1] if text.count("```") >= 2 else text
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:]
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            ops = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return []
        return [op for op in ops if isinstance(op, dict)] if isinstance(ops, list) else []

    def _inject_memory_context(self) -> str:
        """Delegates to :func:`context_injection.inject_memory_context`."""
        return context_injection.inject_memory_context(self)

    def _inject_skills_context(self) -> str:
        """Delegates to :func:`context_injection.inject_skills_context`."""
        return context_injection.inject_skills_context(self)

    def _inject_subagents_context(self) -> str:
        """Delegates to :func:`context_injection.inject_subagents_context`."""
        return context_injection.inject_subagents_context(self)

    def _get_playbook_context(self) -> str:
        """Delegates to :func:`context_injection.get_playbook_context`."""
        return context_injection.get_playbook_context(self)

    def _get_conversation_context(self) -> str:
        """Delegates to :func:`context_injection.get_conversation_context`."""
        return context_injection.get_conversation_context(self)

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Delegates to :func:`context_injection.compute_similarity`."""
        return context_injection.compute_similarity(self, text1, text2)

    def _plan_mode_reminder(self) -> str:
        """Delegates to :func:`context_injection.plan_mode_reminder`."""
        return context_injection.plan_mode_reminder(self)

    def _steering_reminder(self) -> str:
        """Delegates to :func:`context_injection.steering_reminder`."""
        return context_injection.steering_reminder(self)

    def _approve_plan(self, plan: str) -> None:
        """Approve the current plan (the agent's _exit_plan_mode_provider): turn
        plan mode OFF and persist the approved plan to the plans dir so it
        survives compaction and is re-readable later. Execution follows the
        in-context plan handed back by the exit_plan_mode tool result."""
        self.plan_mode_active = False
        try:
            ts = self.session_id.split("_", 1)[1] if "_" in self.session_id else ""
            fname = f"plan_{ts}.md" if ts else "plan.md"
            path = plans_dir() / fname
            path.write_text((plan or "").strip() + "\n")
            logger.info("Approved plan saved to %s", path)
            print(f"\n\033[92m🔓 Plan approved\033[0m — saved to {path}\n")
        except Exception as e:
            logger.error(f"Failed to persist approved plan: {e}")
            print("\n\033[92m🔓 Plan approved\033[0m — executing now.\n")

    def _inject_episodic_context(self, prompt: str) -> str:
        """Delegates to :func:`context_injection.inject_episodic_context`."""
        return context_injection.inject_episodic_context(self, prompt)

    def _count_context_tokens(self) -> int:
        """Delegates to :func:`context_injection.count_context_tokens`."""
        return context_injection.count_context_tokens(self)

    def clear_context(self) -> None:
        """Clear conversation history but keep system prompt."""
        if self.agent:
            self.agent.clear_messages()

        # Flush THIS session's RAG/chunk artifacts (capture the id before we mint
        # a new one) — never a wildcard sweep, which would delete other live
        # instances' data.
        old_session_id = self.session_id
        if config.get("ENABLE_RAG", False):
            self._flush_rag_store(old_session_id)
        self._flush_chunk_cache_store(old_session_id)

        self.session_id = self._new_session_id()
        self.system_prompt = self._build_system_prompt()
        if self.agent:
            # Mirror start(): the AGENT's prompt carries the playbook, while
            # self.system_prompt stays playbook-free (session-log replay uses it).
            # Without this, /clear silently drops the playbook block.
            self.agent.system_prompt = self._system_prompt_with_playbook()

        # Fresh conversation: the next /save makes a new file.
        self.current_conversation_path = None

        # /usage counts THIS conversation, so a cleared context starts from zero.
        tracker = getattr(self.agent, "usage", None) if self.agent else None
        if tracker is not None:
            tracker.reset()

    def _new_session_id(self) -> str:
        """Delegates to :func:`session_artifacts.new_session_id`."""
        return session_artifacts.new_session_id(self)

    def _prev_session_from_pointer(self, pointer_path) -> Optional[str]:
        """Delegates to :func:`session_artifacts.prev_session_from_pointer`."""
        return session_artifacts.prev_session_from_pointer(self, pointer_path)

    def _repoint_session(self, pointer_path, flush_fn) -> None:
        """Delegates to :func:`session_artifacts.repoint_session`."""
        session_artifacts.repoint_session(self, pointer_path, flush_fn)

    def _initialize_rag_session(self) -> None:
        """Delegates to :func:`session_artifacts.initialize_rag_session`."""
        session_artifacts.initialize_rag_session(self)

    def _initialize_chunk_cache(self) -> None:
        """Delegates to :func:`session_artifacts.initialize_chunk_cache`."""
        session_artifacts.initialize_chunk_cache(self)

    def _flush_chunk_cache_store(self, session_id: str = None) -> None:
        """Delegates to :func:`session_artifacts.flush_chunk_cache_store`."""
        session_artifacts.flush_chunk_cache_store(self, session_id)

    def _flush_rag_store(self, session_id: str = None) -> None:
        """Delegates to :func:`session_artifacts.flush_rag_store`."""
        session_artifacts.flush_rag_store(self, session_id)

    def save_conversation(self, timestamp: str = None, path: str = None) -> None:
        """Save the conversation to a JSON file.

        ``path`` may be a directory (saves ``conversation_<ts>.json`` there) or a
        file path (``.json`` added if missing); omitted → the profile's
        ``conversations/`` dir (or the open conversation, if any).
        """
        if not self.agent:
            logger.error("Agent not initialized")
            return

        try:
            timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"conversation_{timestamp}.json"

            if path:
                expanded = os.path.expanduser(path)
                # Directory if it exists as one or ends with a separator.
                if os.path.isdir(expanded) or path.endswith(("/", os.sep)):
                    os.makedirs(expanded, exist_ok=True)
                    filepath = os.path.join(expanded, default_name)
                else:
                    parent = os.path.dirname(expanded)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    filepath = expanded if expanded.endswith(".json") else expanded + ".json"
            elif self.current_conversation_path:
                # Overwrite the open conversation rather than spawn a copy.
                filepath = self.current_conversation_path
            else:
                filepath = os.path.join(str(conversations_dir()), default_name)

            strands_messages = convert_langchain_messages_to_strands(
                self.agent.messages
            )
            conversation_data = {
                "messages": [
                    {"role": "system", "content": [{"text": self.system_prompt}]}
                ]
                + strands_messages,
                "tools": (
                    [{"name": t.name, "description": t.description} for t in self.tools]
                    if self.tools
                    else []
                ),
            }

            with open(filepath, "w") as f:
                json.dump(conversation_data, f, indent=2, default=str)

            # Remember it so a bare /save updates it in place.
            self.current_conversation_path = filepath

            print(f"Conversation saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")

    def list_saved_conversations(self) -> list:
        """``Path`` objects for saved ``*.json`` conversations, newest first."""
        try:
            d = conversations_dir()
            files = [p for p in d.glob("*.json") if p.is_file()]
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return files
        except Exception as e:
            logger.error(f"Failed to list saved conversations: {e}")
            return []

    def delete_conversation(self, file_path) -> bool:
        """Delete a saved conversation file; True on success.

        Guarded to only remove a regular ``*.json`` file inside the profile's
        ``conversations/`` dir (so a bad/hand-typed path can't delete something
        elsewhere). If the deleted file is the currently-open conversation, forget
        it so a later bare ``/save`` starts fresh."""
        try:
            p = Path(os.path.expanduser(str(file_path))).resolve()
            conv_dir = conversations_dir().resolve()
            if p.parent != conv_dir or p.suffix != ".json" or not p.is_file():
                logger.warning(f"Refusing to delete non-conversation path: {p}")
                return False
            p.unlink()
            if (
                self.current_conversation_path
                and Path(self.current_conversation_path).resolve() == p
            ):
                self.current_conversation_path = None
            logger.info(f"Deleted conversation: {p.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete conversation: {e}")
            return False

    @staticmethod
    def conversation_title(file_path, max_len: int = 60) -> str:
        """A short title from a saved conversation's first real user message;
        "" if unreadable/empty.

        Skips injected context (episodic-memory block, plan-mode reminder) so the
        title is the user's actual words, not a prepended block.
        """
        try:
            with open(os.path.expanduser(str(file_path)), "r") as f:
                data = json.load(f)
            messages = data if isinstance(data, list) else data.get("messages", [])
            # Same extraction the --resume picker uses, so a conversation and a
            # session transcript of the same chat get the same label.
            return first_user_prompt(messages, max_len=max_len)
        except Exception:
            pass
        return ""

    def load_conversation(self, file_path: str) -> bool:
        """Load a conversation from ``file_path``; True on success."""
        try:
            normalized_path = os.path.expanduser(file_path)
            if not os.path.exists(normalized_path):
                logger.error(f"File not found: {normalized_path}")
                return False

            with open(normalized_path, "r") as f:
                conversation_data = json.load(f)

            messages = (
                conversation_data
                if isinstance(conversation_data, list)
                else conversation_data.get("messages", [])
            )
            conversation_messages = [m for m in messages if m.get("role") != "system"]
            langchain_messages = convert_strands_messages_to_langchain(
                conversation_messages
            )

            # Strip any stored <plan-mode-active> block (older saves) so a
            # reloaded chat doesn't believe it's still in plan mode.
            for m in langchain_messages:
                content = getattr(m, "content", None)
                if isinstance(content, str) and "<plan-mode-active>" in content:
                    m.content = LangGraphAgent._strip_ephemeral(content)

            if self.agent:
                self.agent.messages.clear()
                self.agent.messages.extend(langchain_messages)
                # Mark as open so a bare /save writes back to the same file.
                self.current_conversation_path = normalized_path
                self._seed_session_log(langchain_messages, normalized_path)
                logger.info(
                    f"Loaded {len(langchain_messages)} messages from {normalized_path}"
                )
                # Replay the transcript to scrollback.
                transcript = turn_view.render_conversation(langchain_messages)
                if transcript:
                    print("\n" + transcript)
                print(
                    f"\n\033[90m[Context: {self._count_context_tokens()} tokens]\033[0m"
                )
                return True

            logger.error("Agent not initialized")
            return False

        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return False

    def _session_max_age_days(self) -> int:
        """Days a resumable session log is kept (0 disables persistence).

        Falls back to a code default so the knob reaches existing installs
        without rewriting their ``config.yaml``.
        """
        try:
            return int(config.get("SESSION_MAX_AGE_DAYS", SESSION_MAX_AGE_DAYS))
        except (TypeError, ValueError):
            return SESSION_MAX_AGE_DAYS

    def _attach_session_log(self) -> None:
        """Start this session's transcript and attach it to the agent.

        Best-effort and opt-out: ``SESSION_MAX_AGE_DAYS: 0`` disables session
        persistence entirely (nothing is written), matching the way the sweep
        knob turns the feature off.
        """
        if not self.agent or self._session_max_age_days() <= 0:
            return
        try:
            self.agent.session_log = SessionLog(model=self.model_name_for_log())
        except Exception as e:  # noqa: BLE001 — never block startup
            logger.debug(f"Session log unavailable: {e}")

    def model_name_for_log(self) -> str:
        """Best-effort model id, recorded in the session log's meta record."""
        try:
            return str(config.get("MODEL_ID", {}).get("NAME", "") or "")
        except Exception:  # noqa: BLE001
            return ""

    def _seed_session_log(self, messages: list, source: str = "") -> None:
        """Copy a just-restored conversation into THIS run's transcript.

        Both ``--resume`` and ``/load`` replace the live history wholesale, so
        without this the new session file records only what happens afterwards —
        resuming it later would restore a fragment of the conversation the user
        can plainly see on screen. Best-effort: a transcript must never break a
        restore that already succeeded.
        """
        log = getattr(self.agent, "session_log", None)
        if log is None:
            return
        try:
            log.seed_history(messages, source=source)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Session log seed failed: {e}")

    def _profile_turn(self) -> None:
        """Feed THIS turn to the user profile (no-op when profiling is off).

        Extracted from ``query`` so it can be tested directly: the scoping is the
        whole point of the method, and a test that scopes its own input before
        calling ``analyze_conversation`` proves nothing about the real call path.

        Profiling runs after every turn while ``agent.messages`` holds the whole
        session, so passing all of it re-analyzed every earlier prompt:
        ``interaction_count`` grew as N²/2 (an observed profile reached 62,977) and
        the trait EMAs washed out from folding the same messages repeatedly. Same
        bug, same fix, as the reflector's ``scope_to_last_turn``.
        """
        if not config.get("PROFILE", {}).get("USE_PROFILING", False):
            return
        if not self.agent or not self.agent.messages:
            return
        try:
            messages_for_profile = convert_langchain_messages_to_strands(
                current_turn_messages(self.agent.messages)
            )
            self.profile_manager.analyze_conversation(messages_for_profile)
        except Exception as e:  # noqa: BLE001
            # Profiling is a side effect of a turn the user has ALREADY seen
            # answered; it must never surface as "something went wrong".
            logger.debug(f"Profiling this turn failed: {e}")

    def usage_report(self) -> str:
        """The ``/usage`` report: reported token totals for this session."""
        tracker = getattr(self.agent, "usage", None) if self.agent else None
        if tracker is None:
            return "Usage tracking is unavailable (no agent running)."
        return usage_tracker.render(tracker, self._count_context_tokens())

    def export_transcript(
        self, path: str = None, fmt: str = None, include_reasoning: bool = False
    ) -> Optional[str]:
        """Write a shareable Markdown/text transcript; returns the path written.

        Unlike ``save_conversation`` (re-importable JSON for ``/load``), this is a
        one-way human-readable artifact for pasting into a bug report or PR — so it
        defaults to the CURRENT DIRECTORY, not the profile's conversations dir: an
        export you can't find is useless. Returns None when there's nothing
        visible to export, so the caller can say so rather than write an empty file.
        """
        if not self.agent:
            logger.error("Agent not initialized")
            return None
        messages = list(self.agent.messages or [])
        # An explicit extension picks the format; otherwise Markdown.
        if not fmt and path:
            suffix = os.path.splitext(os.path.expanduser(path))[1].lstrip(".").lower()
            if suffix in ("md", "txt"):
                fmt = suffix
        fmt = (fmt or "md").lower().lstrip(".")

        # Title from the LIVE messages (conversation_title reads a saved FILE).
        try:
            title = first_user_prompt(
                convert_langchain_messages_to_strands(messages), max_len=60
            )
        except Exception:  # noqa: BLE001 — a title is cosmetic
            title = ""

        text = transcript_export.render(
            messages,
            fmt,
            title=title,
            model=self.model_name_for_log(),
            include_reasoning=include_reasoning,
        )
        if not text.strip():
            return None

        default_name = transcript_export.suggest_filename(messages, fmt)
        try:
            if path:
                expanded = os.path.expanduser(path)
                if os.path.isdir(expanded) or path.endswith(("/", os.sep)):
                    os.makedirs(expanded, exist_ok=True)
                    filepath = os.path.join(expanded, default_name)
                else:
                    parent = os.path.dirname(expanded)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    filepath = (
                        expanded
                        if os.path.splitext(expanded)[1]
                        else f"{expanded}.{fmt}"
                    )
            else:
                filepath = os.path.join(os.getcwd(), default_name)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            return filepath
        except OSError as e:
            logger.error(f"Failed to export transcript: {e}")
            return None

    def branch_conversation(self, through_turn: int = None) -> Optional[str]:
        """Fork THIS session at ``through_turn`` and switch the live chat to it.

        Returns the new session's path. The current transcript is copied (the
        source file is never modified, so the original stays resumable), the live
        history is truncated to the branch point, and this run keeps writing into
        the new file — so continuing here diverges without disturbing what came
        before.

        Requires a session log: with ``SESSION_MAX_AGE_DAYS: 0`` there is no
        transcript to fork, and a branch of nothing would silently look like it
        worked.
        """
        log = getattr(self.agent, "session_log", None) if self.agent else None
        if log is None or log.path is None:
            return None
        try:
            new_path = branch_session(log.path, through_turn)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to branch session: {e}")
            return None
        if new_path is None:
            return None

        # Re-point this run at the branch and rehydrate the truncated history, so
        # the live context matches the file we're now appending to.
        try:
            data = read_session(new_path)
            raw = [m for m in data["messages"] if m.get("role") != "system"]
            messages = convert_strands_messages_to_langchain(raw)
            self.agent.messages.clear()
            self.agent.messages.extend(messages)
            self.agent.session_log = SessionLog.reopen(new_path)
            # A branch is not the open /save file — a bare /save must not
            # overwrite the conversation this was forked from.
            self.current_conversation_path = None
        except Exception as e:  # noqa: BLE001
            logger.error(f"Branch created but could not be opened: {e}")
            return None
        return str(new_path)

    def session_turns(self) -> list:
        """Per-turn labels for the ``/branch`` picker (empty when not recording)."""
        log = getattr(self.agent, "session_log", None) if self.agent else None
        if log is None or log.path is None:
            return []
        try:
            return turn_summaries(log.path)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not read turn summaries: {e}")
            return []

    def resume_session(self, path: str) -> bool:
        """Rehydrate a session transcript into the live agent; True on success.

        Reuses the same decode + replay path as ``/load`` so a resumed chat looks
        identical to a loaded one. Deliberately does NOT set
        ``current_conversation_path`` — a resumed session is not an open ``/save``
        file, so a bare ``/save`` must not overwrite one.
        """
        try:
            data = read_session(path)
            raw = [m for m in data["messages"] if m.get("role") != "system"]
            if not raw:
                logger.error(f"Session has no messages: {path}")
                return False
            if not self.agent:
                logger.error("Agent not initialized")
                return False

            messages = convert_strands_messages_to_langchain(raw)
            for m in messages:
                content = getattr(m, "content", None)
                if isinstance(content, str) and "<plan-mode-active>" in content:
                    m.content = LangGraphAgent._strip_ephemeral(content)

            self.agent.messages.clear()
            self.agent.messages.extend(messages)
            self._seed_session_log(messages, str(path))
            logger.info(f"Resumed {len(messages)} messages from {path}")

            transcript = turn_view.render_conversation(messages)
            if transcript:
                print("\n" + transcript)
            print(f"\n\033[90m[Context: {self._count_context_tokens()} tokens]\033[0m")
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to resume session: {e}")
            return False

    def __enter__(self):
        """Context manager entry."""
        self.start(verbose=self.verbose_mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
