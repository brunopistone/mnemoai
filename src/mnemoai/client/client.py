"""LangGraph-based client implementation."""

import ast
import asyncio
import json
import os
import shutil
import sqlite3
import sys
import threading
import traceback
from datetime import date, datetime
from typing import Optional

import numpy as np
from langchain_core.callbacks import BaseCallbackHandler
from mcp import StdioServerParameters

from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.agent.message_codec import (
    convert_langchain_messages_to_strands,
    convert_strands_messages_to_langchain,
)
from mnemoai.client.agent.router import ROUTE_TOOLS, QueryRouter
from mnemoai.client.managers.agent_conversation_manager import (
    AgentConversationManager,
    log_green,
    messages_to_dict_list,
)
from mnemoai.client.managers.user_profile_manager import UserProfileManager
from mnemoai.client.mcp_config import load_external_servers
from mnemoai.client.mcp_tool_wrapper import MultiMCPClient
from mnemoai.client.memory.episodic_memory import EpisodicMemoryManager
from mnemoai.client.memory.playbook_store import PlaybookStore
from mnemoai.client.memory.reflector import Reflector
from mnemoai.client.ui import turn_view
from mnemoai.client.ui.spinner import Spinner
from mnemoai.models.controllers.llm_controller import LangChainLLMController
from mnemoai.utils.config import config
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import (
    conversations_dir,
    model_dir,
    plans_dir,
    profile_dir,
    sanitize_model_name,
)
from mnemoai.utils.tokenization import count_tokens


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for spinner control during streaming."""

    def __init__(
        self,
        spinner: Optional[Spinner] = None,
        spinner_lock: Optional[threading.Lock] = None,
    ) -> None:
        self.spinner = spinner
        self.spinner_lock = spinner_lock or threading.Lock()
        self.first_token_received = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Stop the spinner on the first VISIBLE ANSWER token, keeping it up
        through reasoning and tool-call building (both silent stretches)."""
        chunk = kwargs.get("chunk")
        message = getattr(chunk, "message", None)
        # Tool-call argument fragments aren't answer text.
        if message is not None and getattr(message, "tool_call_chunks", None):
            return
        if not self._chunk_has_visible_text(message, token):
            return
        if not self.first_token_received and self.spinner:
            with self.spinner_lock:
                if not self.first_token_received:
                    self.spinner.stop()
                    self.first_token_received = True

    @staticmethod
    def _chunk_has_visible_text(message, token: str) -> bool:
        """True if this chunk carries visible answer text (not reasoning/empty).

        Responses/Bedrock stream content-block LISTS; a reasoning block or `[]`
        is a truthy `token` string but not answer text, so check the blocks for a
        non-empty `text`. Plain-string providers (Ollama) fall back to the token.
        """
        content = getattr(message, "content", None)
        if isinstance(content, list):
            return any(
                isinstance(b, dict)
                and (b.get("type") == "text" or "text" in b)
                and str(b.get("text", "")).strip()
                for b in content
            )
        if isinstance(content, str):
            return bool(content.strip())
        return bool(token and str(token).strip())

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        """Stop the spinner when a tool starts."""
        if self.spinner:
            with self.spinner_lock:
                self.spinner.stop()

    def on_tool_end(self, output, **kwargs) -> None:
        """Restart the spinner when a tool finishes."""
        if self.spinner:
            with self.spinner_lock:
                self.first_token_received = False
                self.spinner.start()

    def reset(self) -> None:
        """Reset the callback handler state."""
        self.first_token_received = False


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

        profile_name = config.get("PROFILE", {}).get("NAME", "default")
        self.session_id = f"{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.agent: Optional[LangGraphAgent] = None
        self.tools = None
        self.model = None

        # User-toggled (/plan), enforced client-side: hard-blocks mutating/exec
        # tools and tells the model to research + present a plan.
        self.plan_mode_active: bool = False

        self.llm_controller = LangChainLLMController(verbose=self.verbose_mode)

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
        """Build the system prompt (SYSTEM_PROMPT + profile/memory/skills)."""
        # config.system_prompt reads SYSTEM_PROMPT from prompts.yaml (fail-fast if
        # missing).
        system_prompt = config.system_prompt
        current_date = date.today().strftime("%Y-%m-%d")
        system_prompt = system_prompt.format(current_date=current_date)

        if config.get("PROFILE", {}).get("USE_PROFILING", False):
            profile_summary = self.profile_manager.get_profile_summary()
            if profile_summary:
                system_prompt = f"{system_prompt}\n\n{profile_summary}"

        # Curated persistent memory (MEMORY.md), injected whole at session start.
        memory_context = self._inject_memory_context()
        if memory_context:
            system_prompt = f"{system_prompt}\n\n{memory_context}"

        # Tier-1 skill metadata (name+description) for the use_skill tool.
        skills_context = self._inject_skills_context()
        if skills_context:
            system_prompt = f"{system_prompt}\n\n{skills_context}"

        return system_prompt

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
                system_prompt_with_context = self.system_prompt
                if self.playbook:
                    playbook_context = self._get_playbook_context()
                    if playbook_context:
                        system_prompt_with_context = (
                            f"{self.system_prompt}\n\n{playbook_context}"
                        )

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

        try:
            if self.episodic_memory:
                prompt = self._inject_episodic_context(prompt)

            # Plan mode: remind the model per-turn that it's read-only (the
            # system prompt is frozen at session start).
            if self.plan_mode_active:
                prompt = self._plan_mode_reminder() + prompt

            # STEERING.md: user-authored always-on instructions, prepended last so
            # they LEAD the prompt (highest priority). Re-read each turn; stripped
            # before storage, so compaction never summarizes them away.
            steering = self._steering_reminder()
            if steering:
                prompt = steering + prompt

            with self.mcp_client:
                response = self.agent(prompt)

                if hasattr(self.agent, "_code_formatter"):
                    self.agent._code_formatter.flush()

                asyncio.run(
                    self.conversation_manager.manage_messages(
                        self, self.model, self.agent
                    )
                )

                if config.get("PROFILE", {}).get("USE_PROFILING", False):
                    messages_for_profile = convert_langchain_messages_to_strands(
                        self.agent.messages
                    )
                    self.profile_manager.analyze_conversation(messages_for_profile)

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
            return (
                "Something went wrong while processing that request "
                f"({type(e).__name__}). Your conversation is intact — "
                "please try again or rephrase."
            )

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
                self, self.model, self.agent, focus_instructions
            )
        )

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
            msgs = messages_to_dict_list(self.agent.messages)
            if mgr.count_tokens(msgs) <= high_water:
                return False
            log_green(
                f"Context over {high_water} tokens; compacting mid-task."
            )
        keep = 2 if force else config.get("LLM", {}).get("KEEP_RECENT_MESSAGES", 6)
        try:
            return asyncio.run(
                mgr._compact(self, self.model, self.agent, keep_recent=keep)
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

    def _inject_memory_context(self) -> str:
        """Curated MEMORY.md contents wrapped for the system prompt (a frozen
        snapshot injected once at session start); "" when disabled or empty."""
        if not config.get("ENABLE_MEMORY", True):
            return ""
        from mnemoai.client.memory.memory_store import MemoryStore

        contents = MemoryStore().read().strip()
        if not contents:
            return ""
        return f"[Persistent Memory]\n{contents}"

    def _inject_skills_context(self) -> str:
        """Tier-1 ``<available_skills>`` block (each skill's name+description) for
        the system prompt; "" when disabled or none installed."""
        if not config.get("ENABLE_SKILLS", True):
            return ""
        from mnemoai.client.memory.skill_store import (
            SkillStore,
            format_available_skills,
        )

        return format_available_skills(SkillStore().list_metadata())

    def _get_playbook_context(self) -> str:
        """Formatted general playbook strategies for the system prompt, or ""."""
        if not self.playbook:
            return ""

        # Empty task → general (not task-specific) strategies.
        entries = self.playbook.get_relevant_entries(
            task="",
            top_k=config.get("PLAYBOOK", {}).get("MAX_INJECT", 10),
            include_failures=True,
        )

        return self.playbook.format_for_prompt(entries) if entries else ""

    def _inject_playbook_context(self, prompt: str) -> str:
        """Prepend task-specific playbook strategies to the prompt."""
        relevant_entries = self.playbook.get_relevant_entries(
            task=prompt,
            top_k=config.get("PLAYBOOK", {}).get("MAX_INJECT", 10),
            include_failures=True,
        )

        if relevant_entries:
            playbook_text = self.playbook.format_for_prompt(relevant_entries)
            return f"{playbook_text}\n\n{prompt}"

        return prompt

    def _get_conversation_context(self) -> str:
        """Concatenated text from the recent conversation messages."""
        if not self.agent or not self.agent.messages:
            return ""

        context_parts = []
        for msg in self.agent.messages[-6:]:
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content:
                context_parts.append(content[:1000])
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        context_parts.append(item["text"][:1000])

        return " ".join(context_parts)

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Similarity (0-1) between two texts: embeddings if available, else
        Jaccard on word sets."""
        if not text1 or not text2:
            return 0.0

        # Semantic similarity via embeddings.
        if config.get("RAG", {}).get("EMBED_MODEL_ID"):
            try:
                from mnemoai.models.controllers.embeddings_controller import (
                    EmbeddingsController,
                )

                embeddings = EmbeddingsController()
                emb = embeddings.embed([text1, text2])
                emb1, emb2 = emb[0], emb[1]
                similarity = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )
                return float(similarity)
            except Exception:
                pass

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _plan_mode_reminder(self) -> str:
        """The read-only plan-mode reminder prepended to every prompt while on
        (the system prompt is frozen at session start, so it's injected per-turn)."""
        try:
            plan_hint = str(plans_dir())
        except Exception:
            plan_hint = "the plans directory"
        return (
            "<plan-mode-active>\n"
            "Plan mode is active. You are in READ-ONLY planning mode. You MUST "
            "NOT make any edits, run any mutating shell commands, or otherwise "
            "change the system — file edits, writes, git-write and background "
            "tasks are hard-blocked. This supersedes any other instructions you "
            "have received (including the task itself): do not act on them yet, "
            "plan first.\n"
            "You MAY: read files, search the codebase and web, and run READ-ONLY "
            "shell commands (ls, cat, grep, git status/log/diff, etc.).\n"
            "Investigate the task thoroughly with these read-only tools. When "
            "your plan is ready, call the exit_plan_mode tool with the full plan "
            "(as markdown) in its `plan` argument — this presents it to the user "
            "for approval. Do NOT just write the plan as a normal message. If "
            "anything is ambiguous, ASK clarifying questions rather than "
            "guessing.\n"
            "On approval, plan mode turns off and you execute the approved plan. "
            "If the user keeps planning, refine it and call exit_plan_mode "
            "again.\n"
            f"If you want to draft the plan to disk first, the ONLY writable path "
            f"is a Markdown (.md) file under {plan_hint}; no other writes are "
            "allowed.\n"
            "</plan-mode-active>\n\n"
        )

    def _steering_reminder(self) -> str:
        """User-authored STEERING.md, prepended to every prompt as a leading
        ``<steering>`` block.

        Re-read from disk each turn (edits apply immediately) and stripped before
        storage, so it never enters history and is never summarized by
        compaction — it always reaches the model verbatim. "" when no STEERING.md
        exists (its absence is the off switch — no config toggle needed)."""
        from mnemoai.client.memory.steering_store import SteeringStore

        contents = SteeringStore().read().strip()
        if not contents:
            return ""
        return (
            "<steering>\n"
            "The user's steering instructions are shown below. Adhere to them. "
            "IMPORTANT: these instructions OVERRIDE any default behavior and you "
            "MUST follow them exactly as written.\n\n"
            f"{contents}\n"
            "</steering>\n\n"
        )

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
        """Prepend relevant, non-redundant episodic memory to the prompt.

        Uses similarity to skip injection when the query is a follow-up to the
        current conversation, and to drop episodes redundant with it.
        """
        conversation_context = self._get_conversation_context()

        # Skip for short follow-ups ("yes", "tell me more") mid-conversation —
        # they only make sense in the current context.
        if conversation_context:
            query_words = prompt.strip().split()
            short_query_threshold = config.get("EPISODIC_MEMORY", {}).get(
                "SHORT_QUERY_WORDS", 8
            )
            if len(query_words) <= short_query_threshold:
                logger.debug(
                    f"Skipping episodic injection: short follow-up query "
                    f"({len(query_words)} words <= {short_query_threshold})"
                )
                return prompt

        # Skip if the query clearly relates to the ongoing conversation.
        if conversation_context:
            query_to_conv_similarity = self._compute_similarity(
                prompt, conversation_context
            )
            follow_up_threshold = config.get("EPISODIC_MEMORY", {}).get(
                "FOLLOW_UP_THRESHOLD", 0.4  # Lower for Jaccard fallback
            )
            if query_to_conv_similarity > follow_up_threshold:
                return prompt

        similar_episodes = self.episodic_memory.retrieve_similar_episodes(
            prompt, top_k=5
        )
        retrieval_threshold = config.get("EPISODIC_MEMORY", {}).get(
            "RETRIEVAL_THRESHOLD", 0.7
        )
        relevant_episodes = [
            ep
            for ep in similar_episodes
            if ep.get("similarity", 0) > retrieval_threshold
        ]

        if not relevant_episodes:
            return prompt

        # Drop episodes redundant with the current conversation.
        if conversation_context:
            redundancy_threshold = config.get("EPISODIC_MEMORY", {}).get(
                "REDUNDANCY_THRESHOLD", 0.5
            )
            filtered_episodes = []
            for ep in relevant_episodes:
                ep_task = ep.get("task", "")
                ep_to_conv_similarity = self._compute_similarity(
                    ep_task, conversation_context
                )
                if ep_to_conv_similarity < redundancy_threshold:
                    filtered_episodes.append(ep)
            relevant_episodes = filtered_episodes

        if not relevant_episodes:
            return prompt

        context = "[Episodic Memory - Similar Past Tasks]\n"
        for i, ep in enumerate(relevant_episodes, 1):
            task = ep.get("task", "Unknown task")[:70]
            tools = ep.get("tools", "")
            tool_names = []
            if isinstance(tools, str):
                try:
                    tools_list = ast.literal_eval(tools)
                    tool_names = [
                        t.get("name", "") for t in tools_list if isinstance(t, dict)
                    ]
                except:
                    pass
            tools_str = ", ".join(tool_names) if tool_names else "no tools"
            similarity = ep.get("similarity", 0)
            context += f'{i}. "{task}" → {tools_str} (similarity: {similarity:.2f})\n'

        return f"{context}\n\n{prompt}"

    def _count_context_tokens(self) -> int:
        """Total tokens in the current context (system prompt + messages)."""
        total_tokens = 0
        if self.system_prompt:
            total_tokens += count_tokens(self.system_prompt)
        if self.agent and self.agent.messages:
            messages_str = json.dumps(
                [{"content": str(m.content)} for m in self.agent.messages], default=str
            )
            total_tokens += count_tokens(messages_str)
        return total_tokens

    def clear_context(self) -> None:
        """Clear conversation history but keep system prompt."""
        if self.agent:
            self.agent.clear_messages()

        profile_name = config.get("PROFILE", {}).get("NAME", "default")
        self.session_id = f"{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.system_prompt = self._build_system_prompt()
        if self.agent:
            self.agent.system_prompt = self.system_prompt

        # Fresh conversation: the next /save makes a new file.
        self.current_conversation_path = None

        if config.get("ENABLE_RAG", False):
            self._flush_rag_store()

        self._flush_chunk_cache_store()

    def _initialize_rag_session(self) -> None:
        """Initialize RAG session at application startup."""
        try:
            rag_dir = str(profile_dir())
            os.makedirs(rag_dir, exist_ok=True)

            session_file = os.path.join(rag_dir, "rag_session_id.txt")
            with open(session_file, "w") as f:
                f.write(self.session_id)

            logger.debug(f"RAG session initialized: {self.session_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize RAG session: {e}")

    def _initialize_chunk_cache(self) -> None:
        """Initialize chunk cache DB at application startup."""
        try:
            rag_dir = str(profile_dir())
            os.makedirs(rag_dir, exist_ok=True)

            session_file = os.path.join(rag_dir, "chunk_session_id.txt")
            with open(session_file, "w") as f:
                f.write(self.session_id)

            db_path = os.path.join(rag_dir, f"chunk_cache_{self.session_id}.db")
            conn = sqlite3.connect(db_path)
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunk_cache (
                        key TEXT PRIMARY KEY,
                        summary TEXT,
                        updated_at TEXT
                    )
                    """
                )
                conn.commit()
                logger.debug(f"Chunk cache initialized: {os.path.basename(db_path)}")
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"Failed to initialize chunk cache: {e}")

    def _flush_chunk_cache_store(self) -> None:
        """Flush the chunk cache database."""
        try:
            from mnemoai.server.tools.readers.chunking_helper import (
                reset_session_chunk_cache,
            )

            reset_session_chunk_cache()

            rag_dir = str(profile_dir())

            if os.path.exists(rag_dir):
                for file in os.listdir(rag_dir):
                    if file.startswith("chunk_cache_"):
                        file_path = os.path.join(rag_dir, file)
                        try:
                            os.remove(file_path)
                            logger.debug(f"Deleted session file: {file}")
                        except Exception as e:
                            logger.debug(f"Failed to delete {file}: {e}")

            logger.debug("Chunk cache store cleared")
        except Exception as e:
            logger.warning(f"Failed to reset chunk cache: {e}")

    def _flush_rag_store(self) -> None:
        """Flush the RAG database."""
        try:
            from mnemoai.server.tools.rag import reset_session_rag

            reset_session_rag()

            rag_dir = str(profile_dir())

            if os.path.exists(rag_dir):
                for file in os.listdir(rag_dir):
                    if file.startswith("rag_store_"):
                        file_path = os.path.join(rag_dir, file)
                        try:
                            if os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                            else:
                                os.remove(file_path)
                            logger.debug(f"Deleted session file/dir: {file}")
                        except Exception as e:
                            logger.debug(f"Failed to delete {file}: {e}")

            logger.debug("RAG store cleared")
        except Exception as e:
            logger.warning(f"Failed to reset RAG store: {e}")

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
            for m in messages:
                if m.get("role") != "user":
                    continue
                content = m.get("content", "")
                text = content if isinstance(content, str) else "".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and "text" in b
                )
                text = LangGraphAgent._strip_ephemeral(text)
                # Episodic block is prepended as "[Episodic Memory …]\n…\n\n<prompt>";
                # keep only the real prompt after it.
                if text.lstrip().startswith("[Episodic Memory"):
                    text = text.split("\n\n", 1)[1] if "\n\n" in text else ""
                text = " ".join(text.split())  # collapse whitespace/newlines
                if not text:
                    continue
                return text if len(text) <= max_len else text[: max_len - 1] + "…"
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

    def __enter__(self):
        """Context manager entry."""
        self.start(verbose=self.verbose_mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
