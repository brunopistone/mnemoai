"""MCP tool for loading an agent skill's full instructions on demand.

Skills use three-tier progressive disclosure (see
``client/memory/skill_store.py``): only each skill's name+description is in the
system prompt (the ``<available_skills>`` block). When the model decides a skill
matches the task, it calls ``use_skill`` — this tool returns the full ``SKILL.md``
body, which lands in context as a tool result and guides the rest of the turn.

The file logic lives in ``client/memory/skill_store.py`` (shared with the
client's metadata injection and ``/skills`` command); this module is just the MCP
surface, mirroring ``memory_tool.py``.
"""

from mcp.server.fastmcp import FastMCP

from mnemoai.client.memory.skill_store import SkillStore

from ..error_handler import tool_error_handler


def register_skill_tools(mcp: FastMCP) -> None:
    """Register the skill-loading tool.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    @tool_error_handler
    async def use_skill(name: str) -> str:
        """Load a skill's full step-by-step instructions into context.

        Call this the MOMENT the user's request matches one of the skills listed
        in the <available_skills> block of your system prompt — BEFORE you start
        working. A skill is an authored procedure: load it and follow it rather
        than guessing the steps. It is normal and expected to call this for any
        non-trivial task that matches a listed skill.

        After loading, the body may point you at bundled resources (a reference
        file or a script) living in the skill's directory — read them with
        fs_read or run them with execute_bash as instructed.

        Args:
            name: The skill's name, exactly as shown in <available_skills>.

        Returns:
            The skill's full instructions, or an error listing available skills
            if the name is unknown.
        """
        store = SkillStore()
        skill = store.load_body((name or "").strip())
        if skill is None:
            available = ", ".join(n for n, _ in store.list_metadata()) or "(none)"
            return (
                f"No skill named {name!r}. Available skills: {available}. "
                "Use the exact name shown in <available_skills>."
            )
        return (
            f"# Skill: {skill.name}\n\n"
            f"{skill.body}\n\n"
            "---\n"
            f"(Bundled resources for this skill, if any, live in: {skill.path} — "
            "read reference files with fs_read and run scripts with execute_bash there.)"
        )
