"""Model-initiated sub-agent lifecycle — spawn / resume / background (helpers).

The envelope AROUND the shared model↔tool loop for ``spawn_agent`` /
``resume_agent``: tool-subset resolution from a sub-agent type's allowlist,
callback-free model copies, single + parallel-batch + background(daemon)
execution, and background-completion draining. The loop itself
(``agent._run_worker_loop``) stays ON the agent and is invoked THROUGH the agent
handle — it is NOT duplicated here, so the shared-loop coupling never moves and a
test that monkeypatches ``a._run_worker_loop`` still intercepts.

Functions take the ``LangGraphAgent`` as the first arg (``agent``); the sub-agent
type DEFINITION (a ``subagents.SubAgent``) is ``subagent``. They dispatch back
through the agent's own methods (``_run_one_subagent``, ``_subagent_tools``,
``_launch_background_subagent``, ``_wrap_subagent_result``, ``_run_worker_loop``,
spinner, ``_activity``, ``_bg_agents``) so overrides on a bare ``__new__`` test
stub still intercept. The agent keeps thin delegating methods — the
``plan_policy``/``confirmation_gate`` collaborator pattern.

``_bg_agents`` (BackgroundAgentRegistry) and ``_activity`` (AgentActivityStore)
remain agent-owned fields, so background threads and the TUI activity panel are
unchanged.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from mnemoai.client.agent import subagents
from mnemoai.utils.logger import logger


def subagent_tools(agent, subagent) -> List[BaseTool]:
    """Resolve a spawned sub-agent's tool objects from its type's allowlist.

    ``subagent.tools`` is a name allowlist (or None = all). Meta tools (fs_read,
    describe_image) are always included; ``spawn_agent`` is always removed so a
    sub-agent can't spawn its own sub-agents, and ``ask_user_question`` because a
    sub-agent has no direct user — a background one has no terminal at all, so a
    picker would block its daemon thread on a prompt that never paints. Mirrors
    the route/worker tool-subset selection in ``_orchestrate``."""
    meta = {"fs_read", "describe_image"}
    if subagent.tools is None:
        allowed = None  # all tools
    else:
        allowed = set(subagent.tools) | meta
    denied = set(getattr(subagent, "disallowed_tools", None) or [])
    deny_all = "*" in denied  # the deny-everything sentinel
    subset = []
    for t in agent.tools:
        if t.name in ("spawn_agent", "ask_user_question"):
            continue  # no nested spawning; no user to ask
        if deny_all or t.name in denied:
            continue  # per-agent denylist, applied AFTER the allowlist
        if allowed is None or t.name in allowed:
            subset.append(t)
    return subset


def run_spawn_batch(agent, tool_calls: list) -> Dict[str, str]:
    """Run multiple ``spawn_agent`` calls from one turn concurrently.

    Returns ``{tool_id: result_text}`` for the spawns run here. Returns ``{}``
    when there are 0 or 1 spawn calls — the caller then handles a lone spawn
    inline (no pool overhead). Bounded by ``_max_subagent_concurrency`` (a
    failing sub-agent yields an error string, never aborting its siblings).
    The sub-agent loops are ``quiet`` (they stream but suppress display and
    touch no shared display state), so running them on pool threads is safe."""
    # Background spawns return immediately (they don't block), so they're not
    # part of the concurrent-wait batch — the inline path launches them. Only
    # explicit run_in_background=false spawns wait here (background is now the
    # default, so an omitted arg means background → excluded from the batch).
    spawns = [
        tc for tc in tool_calls
        if tc.get("name") == "spawn_agent"
        and (tc.get("args") or {}).get("run_in_background", True) is False
    ]
    max_workers = getattr(agent, "_max_subagent_concurrency", 1)
    if len(spawns) <= 1 or max_workers <= 1:
        return {}  # inline path handles a single (or forced-sequential) spawn

    def _one(tc) -> tuple:
        args = agent._normalize_tool_args(tc["args"])
        content = agent._handle_spawn_agent(
            str(args.get("agent_type", "")),
            str(args.get("prompt", "")),
            str(args.get("description", "")),
            in_batch=True,
        )
        return tc["id"], content

    if agent.verbose:
        print(
            f"\n\033[90m[↳ running {len(spawns)} sub-agents in parallel]\033[0m",
            flush=True,
        )
    agent._start_spinner(f"{len(spawns)} sub-agents running…")
    results: Dict[str, str] = {}
    try:
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(spawns))
        ) as pool:
            for tool_id, content in pool.map(_one, spawns):
                results[tool_id] = content
    finally:
        agent._stop_spinner()
    return results


def handle_spawn_agent(
    agent, agent_type: str, prompt: str, description: str = "",
    in_batch: bool = False, run_in_background: bool = False,
) -> str:
    """Run a spawned sub-agent to completion and return only its final report.

    The sub-agent runs on an isolated context (its own message list, via
    ``_run_worker_loop``) with its type's system prompt + tool allowlist; the
    parent sees only the returned text, never the sub-agent's tool calls.
    Nested spawning is blocked (a sub-agent's tool set drops ``spawn_agent``,
    and ``_spawn_depth`` guards against it regardless).

    ``run_in_background=True`` launches it on a daemon thread and returns
    IMMEDIATELY with an agent id; the parent turn continues, and the result is
    delivered later (see ``_launch_background_subagent`` + ``drain_background_*``)."""
    if getattr(agent, "_spawn_depth", 0) > 0:
        return (
            "A sub-agent cannot spawn its own sub-agents. Do the work directly "
            "with your tools."
        )
    subagent = subagents.get_subagent(agent_type)
    if subagent is None:
        available = ", ".join(a.name for a in subagents.list_subagents())
        return (
            f"Unknown agent_type '{agent_type}'. Available types: {available}."
        )
    prompt = (prompt or "").strip()
    if not prompt:
        return "spawn_agent needs a non-empty prompt describing the task."

    label = description.strip() or subagent.name

    if run_in_background:
        return agent._launch_background_subagent(subagent, prompt, label)

    if agent.verbose:
        print(
            f"\n\033[90m[↳ spawn_agent: {subagent.name} — {label}]\033[0m\n",
            flush=True,
        )

    # In a parallel batch the aggregate "N sub-agents running…" spinner is
    # owned by _run_spawn_batch and shared across pool threads, so a single
    # sub-agent must NOT drive its own per-tool spinner (it would race the
    # others and clobber the aggregate label). Solo spawns own the spinner.
    result = agent._run_one_subagent(subagent, prompt, label, drive_spinner=not in_batch)
    return agent._wrap_subagent_result(subagent.name, result)


def wrap_subagent_result(agent_name: str, result: str, resumed: bool = False) -> str:
    """Wrap a sub-agent's report with the header + not-shown-to-user footer."""
    header = f"[{agent_name} sub-agent result{' — resumed' if resumed else ''}]"
    return (
        f"{header}\n{result}\n\n"
        "(This result is not shown to the user — summarize what matters for "
        "them yourself.)"
    )


def launch_background_subagent(agent, subagent, prompt: str, label: str) -> str:
    """Start a sub-agent on a daemon thread and return immediately.

    The thread runs the quiet loop in HEADLESS mode (untrusted destructive
    tools auto-deny — no TTY to prompt on), records the result in the registry
    on completion, and never raises into the parent. Returns an ack string
    with the agent id the parent can reference."""
    rec = agent._bg_agents.register(subagent.name, label, prompt)

    def _run() -> None:
        agent._set_headless(True)
        try:
            result = agent._run_one_subagent(
                subagent, prompt, label, drive_spinner=False, kind="background"
            )
            agent._bg_agents.complete(rec.agent_id, result)
        except Exception as e:  # never crash the daemon
            logger.error(f"Background sub-agent {rec.agent_id} failed: {e}")
            agent._bg_agents.complete(
                rec.agent_id, f"The {subagent.name} sub-agent failed: {e}",
                failed=True,
            )
        # Wake the UI so it can auto-deliver this completion when idle (or
        # let a running turn pick it up at its next boundary). Best-effort:
        # absent hook (plain loop/tests) → delivered on the user's next turn.
        hook = getattr(agent, "_on_background_complete", None)
        if hook is not None:
            try:
                hook(rec.agent_id)
            except Exception as e:
                logger.debug(f"background-complete hook failed: {e}")

    threading.Thread(target=_run, daemon=True).start()
    return (
        f"Started background sub-agent '{rec.agent_id}' ({subagent.name}: {label}). "
        "It runs while you continue; you'll be notified when it finishes, and "
        "its result will be delivered then. Do not wait for it — carry on."
    )


def handle_resume_agent(
    agent, agent_id: str, prompt: str, run_in_background: bool = True
) -> str:
    """Resume a prior sub-agent with a follow-up, using its saved record.

    Reconstructs a brief from the recorded run's original task + prior report
    and runs the same type's quiet loop with the new prompt. Defaults to
    **background** (the original background sub-agent's mode): returns
    immediately and delivers the report on completion; ``run_in_background=
    False`` waits for the report inline."""
    agent_id = (agent_id or "").strip()
    prompt = (prompt or "").strip()
    if not prompt:
        return "resume_agent needs a non-empty follow-up prompt."
    rec = agent._bg_agents.get(agent_id)
    if rec is None:
        # Not in this session's registry — fall back to the persisted record
        # on disk, so a finished sub-agent stays resumable after a restart /
        # on a loaded conversation (the in-memory registry doesn't survive).
        rec = agent._bg_agents.load_from_disk(agent_id)
    if rec is None:
        known = ", ".join(r.agent_id for r in agent._bg_agents.list_all()) or "none"
        return (
            f"Unknown agent_id '{agent_id}' (no live or persisted record). "
            f"Known sub-agents this session: {known}."
        )
    if rec.status == "running":
        return (
            f"Sub-agent '{agent_id}' is still running — wait for it to finish "
            "before resuming it."
        )
    subagent = subagents.get_subagent(rec.agent_type)
    if subagent is None:
        return f"The '{rec.agent_type}' agent type no longer exists."

    # Re-brief: original task + prior report as context, then the follow-up.
    # Omit empty sections (a disk-loaded record from before prompts were
    # persisted may lack the original task).
    parts = []
    if rec.prompt:
        parts.append(f"You previously worked on this task:\n{rec.prompt}")
    if rec.result:
        parts.append(f"Your prior report was:\n{rec.result}")
    parts.append(f"Follow-up instruction:\n{prompt}")
    resume_prompt = "\n\n".join(parts)
    label = f"{rec.description} (resumed)" if rec.description else "resumed"

    if run_in_background:
        # Launch detached — same as a background spawn (returns immediately,
        # report delivered on completion). This is the default so resuming a
        # background sub-agent stays background.
        return agent._launch_background_subagent(subagent, resume_prompt, label)

    if agent.verbose:
        print(
            f"\n\033[90m[↳ resume_agent: {subagent.name} — {label}]\033[0m\n",
            flush=True,
        )
    result = agent._run_one_subagent(subagent, resume_prompt, label)
    return agent._wrap_subagent_result(subagent.name, result, resumed=True)


def drain_background_completions(agent) -> List[BaseMessage]:
    """Pop newly-finished background sub-agents as wrapped user messages.

    Called by the chat loop at the start of a turn: each just-completed
    background agent becomes a HumanMessage carrying its report, so the model
    addresses it as new input (reusing the steering framing). Returns [] when
    nothing finished since the last drain."""
    registry = getattr(agent, "_bg_agents", None)
    if registry is None:
        return []
    ready = registry.drain_completed_unnotified()
    msgs: List[BaseMessage] = []
    for rec in ready:
        verb = "failed" if rec.status == "failed" else "finished"
        msgs.append(
            HumanMessage(
                content=(
                    f"Your background sub-agent '{rec.agent_id}' "
                    f"({rec.agent_type}: {rec.description}) {verb} while you "
                    f"were working. Its report:\n\n{rec.result}\n\n"
                    "Address this now: summarize what matters for the user."
                )
            )
        )
    return msgs


def has_undelivered_background(agent) -> bool:
    """True if a finished background sub-agent's report hasn't been surfaced
    yet (drives the UI's auto-delivery / delivery-only turn)."""
    registry = getattr(agent, "_bg_agents", None)
    return registry.any_undelivered() if registry is not None else False


def callback_free_model(agent):
    """A copy of the base chat model with instance-level callbacks stripped.

    The streaming callback handler is bound to ``agent.model`` at init and
    drives the shared spinner; a quiet sub-agent must not fire it. Returns an
    independent copy (never mutating ``agent.model`` — that would race parallel
    sub-agents) via pydantic ``model_copy``; falls back to the shared model if
    copying isn't supported (then the quiet stream's empty config is the only
    guard, acceptable for a single sub-agent)."""
    model = agent.model
    try:
        return model.model_copy(update={"callbacks": None})
    except Exception:
        return model


def subagent_base_model(agent, subagent):
    """Callback-free base model for a spawned sub-agent, honoring a custom
    type's per-agent ``model`` override (a same-provider model with the NAME
    swapped, built by the client-set factory). Falls back to the parent model
    when there's no override, no factory, or the build fails — so built-in
    types and the default path are unchanged. The factory builds with
    callbacks=None, so the override is already callback-free (safe for the
    quiet/parallel/background sub-agent loops)."""
    override = getattr(subagent, "model", None)
    factory = getattr(agent, "_subagent_model_factory", None)
    if override and factory:
        model = factory(override)
        if model is not None:
            return model
    return agent._callback_free_model()


def run_one_subagent(
    agent, subagent, prompt: str, label: str, drive_spinner: bool = True,
    kind: str = "spawn",
) -> str:
    """Run a single spawned sub-agent to completion (quiet) and return its
    final report text. Used both for a lone spawn and inside the parallel
    pool. Increments the (thread-local) spawn depth so a nested spawn from
    THIS thread is refused; restores it on the way out. ``drive_spinner`` is
    False inside a parallel batch (the batch owns one shared spinner).
    ``kind`` ("spawn"|"background") tags the activity-panel row."""
    # Bind tools onto a CALLBACK-FREE copy of the base model: the chat model
    # carries the streaming callback handler at the instance level (bound at
    # init), which LangChain MERGES with per-call config — so an empty config
    # can't silence it. A quiet sub-agent's stream would otherwise fire
    # on_llm_new_token/on_tool_start → spinner.stop(), tearing down the batch's
    # shared "N running…" spinner (and racing siblings). A per-instance copy
    # (not mutating the shared agent.model) is concurrency-safe.
    sub_tools = agent._subagent_tools(subagent)
    base = agent._subagent_base_model(subagent)
    sub_model = agent._bind_tools(base, sub_tools)
    sys_prompt = subagents.subagent_system_prompt(subagent)
    # A spawned sub-agent is handed a whole self-contained task (esp. the
    # search-heavy explore/plan types), so it needs the same generous turn
    # budget as the main agent loop — NOT the orchestrator-worker default of
    # 10, which starves exploration. Reuse RECURSION_LIMIT (default 200):
    # the main loop's own bound, and the same value as a fresh full agent.
    sub_max_iterations = getattr(agent, "recursion_limit", None) or 200

    if drive_spinner:
        agent._start_spinner(f"{subagent.name}: starting…")

        def _progress(note: str) -> None:
            agent._start_spinner(f"{subagent.name} ({label}): {note}")
    else:
        _progress = None  # batch owns the shared "N running…" spinner

    # Live activity run for the agents panel/detail view (covers foreground
    # solo + batch spawns, background spawns, and inline resume).
    sink = agent._activity.open_run(subagent.name, label, kind)

    agent._spawn_depth += 1
    try:
        result, _ = agent._run_worker_loop(
            sub_model,
            sub_tools,
            prompt,
            max_iterations=sub_max_iterations,
            system_prompt=sys_prompt,
            quiet=True,
            progress=_progress,
            activity=sink,
        )
        # A stopped agent returns cleanly via the loop's cancel path; finish_ok
        # marks it "stopped" vs. "done" and records the final answer.
        sink.finish_ok(result)
        return result
    except Exception as e:
        logger.error(f"spawn_agent ({subagent.name}) failed: {e}")
        sink.finish("failed")
        return f"The {subagent.name} sub-agent failed: {e}"
    finally:
        agent._spawn_depth -= 1
        if drive_spinner:
            agent._stop_spinner()
        # Backstop: if the run exited WITHOUT a normal finish (e.g. a
        # KeyboardInterrupt cancel, which is BaseException — not caught by
        # `except Exception` above), mark it stopped so the panel doesn't
        # show it "running" forever with the timer ticking.
        sink.finish_if_running("failed")
