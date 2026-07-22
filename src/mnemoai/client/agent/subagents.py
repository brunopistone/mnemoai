"""Sub-agent type registry — the built-in agent types ``spawn_agent`` can run.

A *sub-agent* is a fresh, isolated model↔tool loop the parent model spawns via
the ``spawn_agent`` tool to hand off a self-contained task (research, planning,
a focused edit). Each type carries its own system prompt and a tool allowlist, so
e.g. an ``explore`` agent is read-only. The parent only ever sees the sub-agent's
final answer — its intermediate tool calls stay out of the parent's context, which
is the whole point (context isolation).

This module is pure data + resolution logic (no LLM, no MCP), mirroring
``plan_policy``/``skill_store``: the agent runner (`client/agent/agent.py`) and the
`/agents` command consume it. Built-in prompts live in ``prompts.yaml`` (via
``config.prompt`` with the bundled-fallback loader, so a new type reaches existing
installs); a tiny in-code default is the crash-guard when a prompt is absent.
"""

from pathlib import Path
from typing import List, NamedTuple, Optional

import yaml

from mnemoai.client.memory.skill_store import _parse_frontmatter
from mnemoai.utils.config import config
from mnemoai.utils.console import print_error
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import agents_dir


class SubAgentType(NamedTuple):
    """A spawnable sub-agent type (built-in or custom).

    Attributes:
        name: The id the model passes to ``spawn_agent(agent_type=…)``.
        description: One-line "when to use", shown to the model in the tool prompt.
        tools: Allowlist of tool names the sub-agent may use; ``None`` = all the
            parent's tools. Meta tools (fs_read, describe_image) are always added,
            and spawn_agent is always removed (no nested spawning).
        prompt_key: ``prompts.yaml`` key for a built-in type's system prompt
            ("" for a custom type, which carries its prompt inline).
        fallback_prompt: In-code system prompt used only if ``prompt_key`` is absent
            (crash-guard; the bundled prompts.yaml normally supplies it).
        inline_prompt: A custom type's system prompt (the ``.md`` body); "" for a
            built-in (which resolves via ``prompt_key``).
        source: "built-in" or "custom" (from ``~/.mnemoai/agents/``).
        disallowed_tools: Optional denylist of tool names removed AFTER the
            allowlist (and after the always-added meta tools) when resolving the
            sub-agent's toolset; ``None`` = nothing denied. The counterpart to
            the ``tools`` allowlist.
        model: Optional per-agent model NAME override (custom types only). When
            set, the sub-agent runs on a same-provider model with only
            ``MODEL_ID.NAME`` swapped (a cheap agent can use a cheap model);
            ``None`` reuses the parent model.
    """

    name: str
    description: str
    tools: Optional[List[str]]
    prompt_key: str
    fallback_prompt: str
    inline_prompt: str = ""
    source: str = "built-in"
    disallowed_tools: Optional[List[str]] = None
    model: Optional[str] = None


# Read-only tools an explore/plan agent may use (no writes, no exec, no git-write).
_READONLY_TOOLS = [
    "fs_read",
    "glob_search",
    "grep_search",
    "search_in_documents",
    "list_documents",
    "git_status_safe",
    "web_search",
    "web_crawler",
    "describe_image",
]


_BUILTIN_SUBAGENTS = [
    SubAgentType(
        name="general-purpose",
        description=(
            "General-purpose agent for multi-step tasks: researching, searching, "
            "and making changes across files. Has the full toolset."
        ),
        tools=None,  # all parent tools
        prompt_key="SUBAGENT_GENERAL_PROMPT",
        fallback_prompt=(
            "You are a sub-agent. Given the task message, use the tools available to complete "
            "it. Complete the task fully—don't gold-plate, but don't leave it half-done. When "
            "you complete the task, respond with a concise report covering what was done and "
            "any key findings — the caller will relay this to the user, so it only needs the "
            "essentials. "
            "Your strengths: "
            "- Searching for code, configurations, and patterns across large codebases "
            "- Analyzing multiple files to understand system architecture "
            "- Investigating complex questions that require exploring many files "
            "- Performing multi-step research tasks "
            "Guidelines: "
            "- For file searches: search broadly when you don't know where something lives. Use fs_read when you know the specific file path. "
            "- For analysis: Start broad and narrow down. Use multiple search strategies if the first doesn't yield results. "
            "- Be thorough: Check multiple locations, consider different naming conventions, look for related files. "
            "- NEVER create files unless they're absolutely necessary for achieving your goal. ALWAYS prefer editing an existing file to creating a new one. "
            "- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested. "
        ),
    ),
    SubAgentType(
        name="explore",
        description=(
            "Read-only agent for exploring a codebase or documents — locating "
            "code, tracing how things work, gathering context. Cannot edit or run "
            "commands. Use it to answer 'where/how is X' without cluttering your "
            "own context."
        ),
        tools=_READONLY_TOOLS,
        prompt_key="SUBAGENT_EXPLORE_PROMPT",
        fallback_prompt=(
            "You are a file search specialist. You excel at thoroughly navigating and "
            "exploring codebases. "
            "=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS === "
            "This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from: "
            "- Creating new files (no writes, touch, or file creation of any kind) "
            "- Modifying existing files (no edit operations) "
            "- Deleting files (no rm or deletion) "
            "- Moving or copying files (no mv or cp) "
            "- Creating temporary files anywhere, including /tmp "
            "- Using redirect operators (>, >>, |) or heredocs to write to files "
            "- Running ANY commands that change system state "
            "Your role is EXCLUSIVELY to search and analyze existing code. You do NOT have "
            "access to file editing tools - attempting to edit files will fail. "
            "Your strengths: "
            "- Rapidly finding files using glob patterns "
            "- Searching code and text with powerful regex patterns "
            "- Reading and analyzing file contents "
            "Guidelines: "
            "- Use glob_search for broad file pattern matching "
            "- Use grep_search for searching file contents with regex "
            "- Use fs_read when you know the specific file path you need to read "
            "- Adapt your search approach based on the thoroughness level specified by the caller "
            "- Communicate your final report directly as a regular message - do NOT attempt to create files "
            "Complete the search request efficiently and report your findings clearly: exact "
            "file paths, line numbers, and how the pieces fit together. "
        ),
    ),
    SubAgentType(
        name="plan",
        description=(
            "Read-only architect agent: investigates, then returns a concrete "
            "step-by-step implementation plan. Cannot edit or run commands."
        ),
        tools=_READONLY_TOOLS,
        prompt_key="SUBAGENT_PLAN_PROMPT",
        fallback_prompt=(
            "You are a software architect and planning specialist. Your role is to explore "
            "the codebase and design implementation plans. "
            "=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS === "
            "This is a READ-ONLY planning task. You are STRICTLY PROHIBITED from: "
            "- Creating new files (no writes, touch, or file creation of any kind) "
            "- Modifying existing files (no edit operations) "
            "- Deleting files (no rm or deletion) "
            "- Moving or copying files (no mv or cp) "
            "- Creating temporary files anywhere, including /tmp "
            "- Using redirect operators (>, >>, |) or heredocs to write to files "
            "- Running ANY commands that change system state "
            "Your role is EXCLUSIVELY to explore the codebase and design implementation "
            "plans. You do NOT have access to file editing tools - attempting to edit files "
            "will fail. "
            "1. **Understand Requirements**: Focus on the requirements provided. "
            "2. **Explore Thoroughly**: Read any files provided in the prompt; find existing "
            "    patterns and conventions using glob_search, grep_search, and fs_read; "
            "    understand the current architecture; identify similar features as reference; "
            "    trace through relevant code paths. "
            "3. **Design Solution**: Create an implementation approach; consider trade-offs "
            "    and architectural decisions; follow existing patterns where appropriate. "
            "4. **Detail the Plan**: Provide a step-by-step implementation strategy; identify "
            "    dependencies and sequencing; anticipate potential challenges. "
            "End your response with: "
            "List 3-5 files most critical for implementing this plan. "
            "REMEMBER: You can ONLY explore and plan. You CANNOT and MUST NOT write, edit, or "
            "modify any files. "
        ),
    ),
]

def _parse_tools(raw) -> Optional[List[str]]:
    """Normalize a frontmatter ``tools`` value to an allowlist, or None (= all).

    Accepts a YAML list, a comma-separated string, or ``"*"``/absent (→ all tools).
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if raw in ("", "*", "all"):
            return None
        return [t.strip() for t in raw.split(",") if t.strip()]
    if isinstance(raw, list):
        names = [str(t).strip() for t in raw if str(t).strip()]
        return None if not names or "*" in names else names
    return None


def _parse_denylist(raw) -> Optional[List[str]]:
    """Normalize a frontmatter ``disallowed-tools`` value to a denylist, or None.

    Denylist semantics are the INVERSE of ``_parse_tools``: ``"*"``/``"all"``
    means deny EVERYTHING (returned as the sentinel ``["*"]``), not "deny
    nothing" — so a lockdown intent isn't silently inverted. Absent/empty → None.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if raw == "":
            return None
        if raw in ("*", "all"):
            return ["*"]
        return [t.strip() for t in raw.split(",") if t.strip()]
    if isinstance(raw, list):
        names = [str(t).strip() for t in raw if str(t).strip()]
        if not names:
            return None
        return ["*"] if "*" in names or "all" in names else names
    return None


def _load_custom_subagents(root: Optional[Path] = None) -> List[SubAgentType]:
    """Load custom sub-agent types from ``~/.mnemoai/agents/*.md``.

    Each file: YAML frontmatter (``name`` optional — defaults to the filename;
    ``description`` required; optional ``tools``, ``model``) + a markdown body used
    as the system prompt. Tolerant scan (mirrors ``SkillStore``): a bad/incomplete
    file is reported and skipped, never fatal. An absent dir yields [].
    """
    if root is None:
        root = agents_dir()
    root = Path(root)
    if not root.is_dir():
        return []

    agents: List[SubAgentType] = []
    for entry in sorted(root.iterdir()):
        if entry.suffix.lower() != ".md" or not entry.is_file():
            continue
        try:
            text = entry.read_text()
        except (OSError, UnicodeDecodeError) as e:
            print_error(f"Agent '{entry.name}': could not read ({e}); skipping.")
            continue
        try:
            front, body = _parse_frontmatter(text)
        except yaml.YAMLError as e:
            print_error(f"Agent '{entry.name}': invalid YAML frontmatter ({e}); skipping.")
            continue
        if not front:
            print_error(f"Agent '{entry.name}': no YAML frontmatter (--- block); skipping.")
            continue
        description = str(front.get("description", "")).strip()
        if not description:
            print_error(f"Agent '{entry.name}': missing 'description'; skipping.")
            continue
        body = body.strip()
        if not body:
            print_error(f"Agent '{entry.name}': empty body (no system prompt); skipping.")
            continue
        name = str(front.get("name", "") or entry.stem).strip().lower()
        # Optional per-agent denylist (accept hyphen or underscore) and model
        # override; both tolerant (absent -> None). Denylist reuses _parse_tools.
        disallowed = _parse_denylist(
            front.get("disallowed_tools") or front.get("disallowed-tools")
        )
        model_override = str(front.get("model", "") or "").strip() or None
        agents.append(
            SubAgentType(
                name=name,
                description=description,
                tools=_parse_tools(front.get("tools")),
                prompt_key="",
                fallback_prompt="",
                inline_prompt=body,
                source="custom",
                disallowed_tools=disallowed,
                model=model_override,
            )
        )
    if agents:
        logger.debug("Loaded %d custom sub-agent(s): %s",
                     len(agents), ", ".join(a.name for a in agents))
    return agents


def list_subagents() -> List[SubAgentType]:
    """All available sub-agent types: built-ins + custom (``~/.mnemoai/agents``),
    with a custom type overriding a built-in of the same name."""
    by_name = {a.name: a for a in _BUILTIN_SUBAGENTS}
    for custom in _load_custom_subagents():
        by_name[custom.name] = custom  # custom wins on name collision
    return list(by_name.values())


def get_subagent(name: str) -> Optional[SubAgentType]:
    """Resolve a sub-agent type by name (case-insensitive), or None if unknown."""
    key = (name or "").strip().lower()
    for a in list_subagents():
        if a.name == key:
            return a
    return None


def subagent_system_prompt(agent: SubAgentType) -> str:
    """The system prompt for a type: a custom type's inline body, else the
    ``prompts.yaml`` value, else the in-code fallback (crash-guard)."""
    if agent.inline_prompt:
        return agent.inline_prompt
    return config.prompt(agent.prompt_key) or agent.fallback_prompt


def _tools_description(agent: SubAgentType) -> str:
    """Short tool-scope annotation for a type, so the model can pick by capability.
    ``all`` when the type has the full toolset, else the comma-joined allowlist."""
    if agent.tools is None:
        return "all"
    return ", ".join(agent.tools)


def format_available_subagents(agents: List[SubAgentType]) -> str:
    """One ``- name: description (Tools: …)`` line per type, for the spawn_agent
    listing — the tool-scope suffix lets the model pick by capability."""
    return "\n".join(
        f"  - {a.name}: {a.description} (Tools: {_tools_description(a)})"
        for a in agents
    )


def available_subagents_block() -> str:
    """The ``<available_subagents>`` system-prompt block listing all spawnable
    types (built-in + custom), so the model discovers custom agents. Single source
    used by the client's session-start injection and the compaction re-injection.
    "" only if there are somehow no types (built-ins always exist)."""
    listing = format_available_subagents(list_subagents())
    if not listing:
        return ""
    return (
        "<available_subagents>\n"
        "These agent types can be spawned with the spawn_agent tool to handle a "
        "self-contained task in an isolated context (only their final report "
        "returns to you). Pick the type whose description fits:\n"
        f"{listing}\n"
        "</available_subagents>"
    )
