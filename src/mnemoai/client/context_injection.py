"""System-prompt building + per-turn context injection (client collaborators).

Assembles the frozen session-start system prompt (base prompt + profile summary +
curated MEMORY.md + skill/sub-agent metadata) and the per-turn context the client
prepends to a query: episodic-memory recall, the plan-mode banner, and the
STEERING.md block. Also the similarity helper and the context-token count.

Functions take the ``LangGraphClient`` as the first arg and reach its
collaborators (``profile_manager``/``playbook``/``episodic_memory``/``agent``)
and read-only ``system_prompt``/``agent.messages`` through it, reading the same
``config`` singleton the client uses. The client keeps thin delegating methods —
the ``plan_policy``/``cancellation`` collaborator pattern. No import of the client
class (functions receive the instance), so there is no import cycle.
"""

import ast
import json
from datetime import date

from mnemoai.client.agent.subagents import available_subagents_block
from mnemoai.utils.config import config
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import plans_dir
from mnemoai.utils.tokenization import count_tokens

# Warn once per process when embedding similarity is unavailable (see
# compute_similarity) instead of on every turn.
_EMBED_SIMILARITY_WARNED = False


def build_session_blocks(client, include_playbook: bool = False) -> list[str]:
    """The ordered session-start context blocks, non-empty ones only.

    SINGLE SOURCE OF TRUTH for "what accompanies the base system prompt".
    Both assembly paths must use it:

    * session start -- :func:`build_system_prompt` (playbook excluded here; the
      client appends it separately so ``client.system_prompt`` stays
      playbook-free for the session-log/replay paths),
    * after compaction -- ``AgentConversationManager._build_system_with_summary``
      rebuilds the prompt from scratch and MUST re-add every block, playbook
      included, or the learned context silently vanishes mid-session.

    Args:
        client: The ``LangGraphClient`` (read for profile/playbook collaborators).
        include_playbook: Append the ACE playbook block. The compaction rebuild
            sets this because it replaces the client's own playbook append.

    Returns:
        Blocks in injection order: profile, MEMORY.md, skills, sub-agents
        [, playbook].
    """
    blocks = [
        inject_profile_context(client),
        # MEMORY.md (curated persistent memory, injected whole),
        inject_memory_context(client),
        # skill metadata (name+description for use_skill),
        inject_skills_context(client),
        # sub-agent types (built-in + custom, for spawn_agent discovery).
        inject_subagents_context(client),
    ]
    if include_playbook:
        blocks.append(get_playbook_context(client))
    return [block for block in blocks if block]


def build_system_prompt(client) -> str:
    """Build the system prompt (SYSTEM_PROMPT + profile/memory/skills)."""
    # config.system_prompt reads SYSTEM_PROMPT from prompts.yaml (fail-fast if
    # missing).
    system_prompt = config.system_prompt
    current_date = date.today().strftime("%Y-%m-%d")
    system_prompt = system_prompt.format(current_date=current_date)

    for block in build_session_blocks(client):
        system_prompt = f"{system_prompt}\n\n{block}"

    return system_prompt


def inject_profile_context(client) -> str:
    """The learned user-profile summary block; "" when profiling is disabled."""
    if not config.get("PROFILE", {}).get("USE_PROFILING", False):
        return ""
    return client.profile_manager.get_profile_summary() or ""


def inject_memory_context(client) -> str:
    """Curated MEMORY.md contents wrapped for the system prompt (read at session
    start and again on the compaction rebuild); "" when disabled or empty."""
    if not config.get("ENABLE_MEMORY", True):
        return ""
    from mnemoai.client.memory.memory_store import MemoryStore

    contents = MemoryStore().read().strip()
    if not contents:
        return ""
    return f"[Persistent Memory]\n{contents}"


def inject_skills_context(client) -> str:
    """Tier-1 ``<available_skills>`` block (each skill's name+description) for
    the system prompt; "" when disabled or none installed."""
    if not config.get("ENABLE_SKILLS", True):
        return ""
    from mnemoai.client.memory.skill_store import (
        SkillStore,
        format_available_skills,
    )

    return format_available_skills(SkillStore().list_skills())


def inject_subagents_context(client) -> str:
    """``<available_subagents>`` block for the spawn_agent tool (built-in +
    custom types), so the model can discover custom ~/.mnemoai/agents/ types."""
    return available_subagents_block()


def get_playbook_context(client) -> str:
    """Formatted general playbook strategies for the system prompt, or ""."""
    if not client.playbook:
        return ""

    # Empty task → general (not task-specific) strategies.
    entries = client.playbook.get_relevant_entries(
        task="",
        top_k=config.get("PLAYBOOK", {}).get("MAX_INJECT", 10),
        include_failures=True,
    )

    return client.playbook.format_for_prompt(entries) if entries else ""


def get_conversation_context(client) -> str:
    """Concatenated text from the recent conversation messages."""
    if not client.agent or not client.agent.messages:
        return ""

    context_parts = []
    for msg in client.agent.messages[-6:]:
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content:
            context_parts.append(content[:1000])
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    context_parts.append(item["text"][:1000])

    return " ".join(context_parts)


def compute_similarity(client, text1: str, text2: str) -> float:
    """Similarity (0-1) between two texts: embeddings if available, else
    Jaccard on word sets."""
    if not text1 or not text2:
        return 0.0

    # Semantic similarity via embeddings.
    if config.get("RAG", {}).get("EMBED_MODEL_ID"):
        try:
            import numpy as np

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
        except Exception as e:
            # Falling through to Jaccard means the episodic thresholds are now
            # being applied to word-overlap scores, which behave nothing like
            # cosine. Say so once instead of degrading retrieval in silence.
            global _EMBED_SIMILARITY_WARNED
            if not _EMBED_SIMILARITY_WARNED:
                _EMBED_SIMILARITY_WARNED = True
                logger.warning(
                    f"Embedding similarity unavailable ({e}); falling back to "
                    "word-overlap (Jaccard) for episodic relevance. Recall will "
                    "be noticeably worse until the embedding model works."
                )

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def plan_mode_reminder(client) -> str:
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
        "for approval. Do NOT just write the plan as a normal message. If the "
        "plan will run specific shell commands during execution (tests, "
        "builds, installs), list them in the `allowed_bash` argument so the "
        "user pre-approves them and they don't prompt one-by-one. If "
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


def steering_reminder(client) -> str:
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


def inject_episodic_context(client, prompt: str) -> str:
    """Prepend relevant, non-redundant episodic memory to the prompt.

    Uses similarity to skip injection when the query is a follow-up to the
    current conversation, and to drop episodes redundant with it.
    """
    conversation_context = get_conversation_context(client)

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
        query_to_conv_similarity = compute_similarity(
            client, prompt, conversation_context
        )
        follow_up_threshold = config.get("EPISODIC_MEMORY", {}).get(
            "FOLLOW_UP_THRESHOLD", 0.4  # Lower for Jaccard fallback
        )
        if query_to_conv_similarity > follow_up_threshold:
            return prompt

    similar_episodes = client.episodic_memory.retrieve_similar_episodes(
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
            ep_to_conv_similarity = compute_similarity(
                client, ep_task, conversation_context
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


def count_context_tokens(client) -> int:
    """Total tokens in the current context. Prefers the provider's exact
    ``input_tokens`` from the last turn (ground truth — includes system
    prompt, tool calls, everything the API saw); falls back to the
    conservative estimate when no turn has run yet."""
    actual = getattr(client.agent, "_last_input_tokens", None) if client.agent else None
    if actual:
        return int(actual)
    total_tokens = 0
    if client.system_prompt:
        total_tokens += count_tokens(client.system_prompt)
    if client.agent and client.agent.messages:
        messages_str = json.dumps(
            [{"content": str(m.content)} for m in client.agent.messages], default=str
        )
        total_tokens += count_tokens(messages_str)
    return total_tokens
