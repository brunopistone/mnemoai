"""MCP tool for the agent's self-curated persistent memory (MEMORY.md).

Exposes a single ``memory`` tool the model calls during a turn to maintain a
small, durable set of facts about the user, environment, and project. The file
is injected whole into the system prompt at the start of each session, so it
must stay compact — a hard character cap forces consolidation rather than
unbounded growth.

The actual file logic lives in ``client/memory/memory_store.py`` (shared with
the client-side ``/memory`` command); this module is just the MCP surface.
"""

from mcp.server.fastmcp import FastMCP

from mnemoai.client.memory.memory_store import MemoryError, MemoryStore

from ..error_handler import tool_error_handler


def register_memory_tools(mcp: FastMCP) -> None:
    """Register the persistent-memory tool.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    @tool_error_handler
    async def memory(action: str, text: str = "", old_text: str = "") -> str:
        """Maintain durable, cross-session memory about the user and environment.

        This is your long-term notebook. Its full contents are injected into your
        system prompt at the start of every session, so keep it small, dense, and
        high-signal. Save PROACTIVELY — the moment you learn something durable,
        call this tool; the user does not have to ask you to remember.

        WHEN TO SAVE (call this as soon as one of these appears in a turn):
        - User preferences ("prefers pytest over unittest", "wants concise replies")
        - Environment / setup facts (OS, key tool versions, paths, services)
        - Project conventions (naming, structure, workflows, branch rules)
        - Corrections the user makes to you (so you don't repeat the mistake)
        - Hard-won lessons or tool quirks discovered while working
        - Notable completed work worth recalling later
        WHEN TO SKIP:
        - Trivia or one-off details that won't matter next session
        - Anything easily re-discovered (file contents, command output)
        - Raw data dumps; long narratives — store the conclusion, not the log
        - Facts already visible in the current context

        WRITE DENSE ENTRIES. Pack related facts into ONE entry rather than many
        thin ones; prefer a compact statement over a dated story.
          GOOD: "Env: macOS, Python 3.13 via conda 'mnemoai', zsh, VS Code."
          GOOD: "User prefers pytest over unittest and concise answers."
          BAD:  "User has a project."            (vague, low-signal)
          BAD:  "On 2026-06-19 I discovered after several tries that the repo
                 uses Go 1.21 ..."               (verbose; just store "Repo uses Go 1.21")

        Actions:
        - action="add", text="..."            — add a new entry.
        - action="replace", old_text="...", text="..."  — replace the entry that
          uniquely contains old_text with text (use to consolidate or update).
        - action="remove", old_text="..."     — remove the entry uniquely
          containing old_text.

        Memory is capped. If an add would overflow, you'll get an error telling
        you to consolidate first: merge or remove stale entries with
        replace/remove, then add again. ``old_text`` need only be a unique
        substring identifying exactly one entry (an ambiguous match errors).

        Args:
            action: One of "add", "replace", "remove".
            text: The entry text (for add) or the replacement text (for replace).
            old_text: The unique substring identifying the entry (replace/remove).

        Returns:
            A short status string, or an error explaining what to fix.
        """
        store = MemoryStore()
        act = (action or "").strip().lower()
        try:
            if act == "add":
                return store.add(text)
            if act == "replace":
                return store.replace(old_text, text)
            if act == "remove":
                return store.remove(old_text)
            return (
                f"Unknown action {action!r}. Use 'add', 'replace', or 'remove'."
            )
        except MemoryError as e:
            # Surface the guidance to the model so it can self-correct.
            return f"Memory not updated: {e}"
