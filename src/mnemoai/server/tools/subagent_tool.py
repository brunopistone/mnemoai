"""The ``spawn_agent`` tool — hand a self-contained task to a fresh sub-agent.

The parent model calls ``spawn_agent(agent_type, prompt)`` to delegate a task to
a sub-agent that runs in its OWN isolated context (a fresh model↔tool loop): it
investigates/acts with the tools its type allows, and only its final report comes
back to the parent — the sub-agent's intermediate tool calls never enter the
parent's context. This keeps the parent's window clean during search-heavy or
multi-step work.

Thin server surface, client-side logic — the same split as ``exit_plan_mode`` /
``use_skill``. The body here is a stub; the real behavior is the client-side
interception in ``client/agent/agent.py`` (which owns the live agent + models).
If driven directly (no client), the stub reply is harmless.
"""

from mcp.server.fastmcp import FastMCP

from ..error_handler import tool_error_handler


def register_subagent_tools(mcp: FastMCP) -> None:
    """Register the ``spawn_agent`` sub-agent tool.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    @tool_error_handler
    async def spawn_agent(
        agent_type: str,
        prompt: str,
        description: str = "",
        run_in_background: bool = False,
    ) -> str:
        """Delegate a self-contained task to a fresh sub-agent and get its report.

        The sub-agent runs in its OWN isolated context — it does NOT see this
        conversation, only the ``prompt`` you give it — and returns a single final
        report. Its intermediate steps stay out of your context, so this is ideal
        for search-heavy or multi-step work whose details you don't need to keep.

        Available agent types:
          - general-purpose: full toolset; research + make changes across files.
          - explore: READ-ONLY; locate code, trace how things work, gather context.
          - plan: READ-ONLY; investigate then return a step-by-step plan.

        Guidance:
          - **Brief it completely.** The sub-agent starts fresh — include the goal,
            the why, relevant file paths, and what you've already ruled out. It
            can't ask you follow-ups.
          - **Pick the right type.** Use ``explore``/``plan`` (read-only) for
            investigation; ``general-purpose`` when it must edit or run things.
          - **Run several in parallel.** For independent investigations, emit
            multiple ``spawn_agent`` calls in the SAME turn — they run concurrently
            and you get all reports back together. Only do this when the tasks
            don't depend on each other's results.
          - **Run in the background** (``run_in_background=true``) for a long task
            you don't need to wait on: the call returns immediately with an agent
            id, you keep working, and its report is delivered when it finishes. A
            background sub-agent CANNOT ask for confirmation, so it will auto-skip
            any destructive tool that isn't already approved — only use it for
            work that's read-only or whose actions were pre-approved.
          - **The result is NOT shown to the user.** When the sub-agent returns,
            summarize what matters for the user yourself.
          - Don't use it for a single quick file read — call fs_read directly.

        Args:
            agent_type: One of the available types above.
            prompt: The complete, self-contained task/brief for the sub-agent.
            description: Optional 3-5 word summary of the task (for display).
            run_in_background: If true, run detached and return an id immediately.

        Returns:
            The sub-agent's final report (or, for background, an ack + id). Handled
            client-side; this stub reply only appears if driven without the client.
        """
        return (
            "(spawn_agent is handled by the client; no sub-agent was run in this "
            "direct call.)"
        )

    @mcp.tool()
    @tool_error_handler
    async def resume_agent(
        agent_id: str, prompt: str, run_in_background: bool = True
    ) -> str:
        """Resume a previously-run sub-agent with a follow-up instruction.

        Continues the sub-agent identified by ``agent_id`` (from an earlier
        ``spawn_agent`` / a completion notification) — it keeps its prior work as
        context and runs the new ``prompt``, returning a fresh report. Use this to
        iterate on a sub-agent's work (e.g. "now also check the tests") without
        re-briefing it from scratch.

        By default the resumed run is **in the background** (like the original
        background sub-agent): it returns immediately and its report is delivered
        when it finishes. Pass ``run_in_background=false`` to wait for the report
        inline instead. A background resumed run, like any background sub-agent,
        auto-skips destructive tools that aren't pre-approved.

        Args:
            agent_id: The id of the sub-agent to resume.
            prompt: The follow-up instruction for the sub-agent.
            run_in_background: Run detached (default true) vs. wait inline (false).

        Returns:
            The resumed sub-agent's report (or a background ack + id). Handled
            client-side; this stub reply only appears if driven without the client.
        """
        return (
            "(resume_agent is handled by the client; nothing was resumed in this "
            "direct call.)"
        )
