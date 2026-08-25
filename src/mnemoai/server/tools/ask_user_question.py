"""The ``ask_user_question`` tool — a model-initiated multiple-choice question.

When a decision is genuinely the user's to make (and can't be resolved from the
request, the code, or a sensible default), the model calls
``ask_user_question(question=…, options=[…])`` instead of guessing or writing a
wall of alternatives. The call is intercepted client-side (the MCP server is a
piped subprocess and can't prompt the terminal): the client shows a picker and
returns the chosen option as the tool result.

Thin server surface, client-side logic — the same split as ``exit_plan_mode``
(``plan_mode_exit.py``) and ``use_skill`` (``skill_tool.py``). The body here is a
stub; ``client/agent/ask_user.py`` supplies the real behavior. If the tool is ever
driven directly (no client gate), the stub reply tells the model to decide itself
rather than leaving it waiting for an answer that will never come.
"""

from mcp.server.fastmcp import FastMCP

from ..error_handler import tool_error_handler


def register_ask_user_tools(mcp: FastMCP) -> None:
    """Register the user-question tool.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    @tool_error_handler
    def ask_user_question(question: str, options: list[str]) -> str:
        """Ask the user to choose between concrete options, and wait for the answer.

        Use this ONLY when you are blocked on a decision that is genuinely the
        user's to make — one you cannot resolve from their request, the code, or a
        sensible default. Prefer acting on a reasonable default and saying what you
        assumed; a question the user has to answer costs them more than a choice
        they can correct.

        Good: a real fork with different tradeoffs ("which provider should the new
        backend target?"). Bad: asking permission to proceed, asking which file to
        read, re-asking something already answered, or offering options you could
        rank yourself.

        Ask at most one question at a time, and only when already-running work
        doesn't answer it. A sub-agent cannot use this tool (it has no direct user)
        and must decide for itself.

        Args:
            question: The specific question, phrased so the options are the answer.
            options: 2-8 short, distinct, mutually exclusive choices. Put the one
                you'd recommend first. Don't add a "something else" entry — the
                user can always dismiss the question and reply in their own words.

        Returns:
            The option the user chose. Handled client-side; this stub reply only
            appears when the tool is driven without the client interception.
        """
        return (
            "No interactive user is attached, so this question cannot be answered. "
            "Decide yourself using your best judgment and state the assumption "
            "you made."
        )
