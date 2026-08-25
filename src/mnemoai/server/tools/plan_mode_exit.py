"""The ``exit_plan_mode`` tool — the plan-mode readiness signal.

When the enforced read-only plan mode (`/plan`, see the client-side
``plan_policy``) is active and the model has finished researching, it calls
``exit_plan_mode(plan=…)`` with its plan as markdown INSTEAD of just printing the
plan as text. That call is intercepted client-side (the MCP server is a piped
subprocess and can't prompt the terminal): the client shows the plan with an
inline approval prompt, and on approval flips plan mode off and hands the
approved plan back so the model executes it in the same turn.

Thin server surface, client-side logic — the same split as ``use_skill``
(``skill_tool.py``). The body here is a stub; the client interception in
``client/agent/agent.py`` supplies the real behavior. If the tool is ever driven
directly (no client gate), the stub reply is harmless.
"""

from mcp.server.fastmcp import FastMCP

from ..error_handler import tool_error_handler


def register_plan_mode_exit_tools(mcp: FastMCP) -> None:
    """Register the plan-mode exit/approval tool.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    @tool_error_handler
    def exit_plan_mode(plan: str, allowed_bash: list[str] = None) -> str:
        """Present your finished plan for approval and exit read-only plan mode.

        Call this the MOMENT your plan is ready while plan mode is active — pass
        the full plan as markdown in ``plan``, instead of just writing the plan
        as a normal message. The user is shown the plan and asked to approve it;
        on approval, plan mode turns off and you should execute the approved plan
        immediately. If the user chooses to keep planning, stay read-only, refine
        the plan, and call ``exit_plan_mode`` again when ready.

        Only call this once you have thoroughly investigated with read-only tools
        and have a concrete, actionable plan. Do NOT call it to ask a clarifying
        question — ask that directly instead.

        Args:
            plan: The complete implementation plan, as markdown.
            allowed_bash: Optional list of shell commands the plan will run during
                execution (e.g. ``["pytest", "npm run build"]``). When the user
                approves the plan they also pre-approve these, so they run without
                a per-command confirmation prompt. A command auto-confirms when it
                starts with one of these entries; anything else still prompts. Omit
                for plans that only edit files or run nothing notable.

        Returns:
            The user's decision (approved → proceed, or keep planning). Handled
            client-side; this stub reply only appears if plan mode isn't active.
        """
        return (
            "Plan recorded. (Plan mode is not active, so there is nothing to "
            "approve — proceed normally.)"
        )
