"""Agent Skills: authored, on-demand instruction packs.

A *skill* is a directory under the skills root containing a ``SKILL.md`` file:
YAML frontmatter (at least ``name`` + ``description``) followed by a markdown
body of instructions, optionally bundling ``reference.md`` docs and ``scripts/``
executables alongside it.

Skills implement **three-tier progressive disclosure**:

1. *Metadata* — only each skill's name+description is injected into the system
   prompt at session start (cheap, always-on). See :func:`format_available_skills`.
2. *Body* — the full ``SKILL.md`` body is loaded into context only when the model
   calls the ``use_skill`` tool (which returns :meth:`SkillStore.load_body`).
3. *Resources* — bundled ``reference.md`` / ``scripts/`` are read/run on demand via
   the existing ``fs_read`` / ``execute_bash`` tools.

This module is pure file logic — no MCP, no LLM — so it is shared by the
server-side ``use_skill`` tool and the client-side metadata injection + ``/skills``
command, and is unit-testable on its own. Scanning is tolerant: a malformed or
incomplete skill is skipped (logged), never fatal — mirroring
``client/mcp_config.py``'s external-server loading.

The **directory name is the canonical skill id** used by ``use_skill(name)``; the
frontmatter ``name`` is display only.
"""

from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import yaml

from mnemoai.utils.console import print_error
from mnemoai.utils.logger import logger

# Cap each description in the always-on metadata block so a verbose, user-authored
# description can't bloat the system prompt.
_MAX_DESC_CHARS = 200

# Aggregate cap on the whole <available_skills> listing (it is injected every
# turn). With many skills installed, the per-description cap alone is unbounded —
# N skills produce N lines — so we also bound the total: once the accumulated
# lines exceed this many chars, the rest are summarized as a "+N more" line.
_MAX_LISTING_CHARS = 4000

# `name`/`description` are the only required frontmatter keys. Other keys —
# are tolerated (so skills authored elsewhere parse cleanly) and not acted on.
# We never reject a skill for extra keys; that would be more friction than help 
# on a local tool. Sanity cap on description length, so a runaway description is 
# reported rather than silently injected.
_MAX_DESC_LEN = 1024


class Skill(NamedTuple):
    """A parsed skill.

    Attributes:
        name: The directory name (canonical id used by ``use_skill``).
        description: One-line trigger description from the frontmatter.
        body: The markdown body after the frontmatter.
        path: The skill's directory (where bundled resources live).
        argument_hint: Optional short hint (from frontmatter ``argument_hint``)
            describing what the skill expects, shown in the tier-1 listing so the
            model knows what to gather before invoking. "" when absent.
    """

    name: str
    description: str
    body: str
    path: Path
    argument_hint: str = ""


class SkillIssue(NamedTuple):
    """A skill directory that was found but rejected, and why.

    Surfaced by ``/skills`` so a malformed skill isn't silently invisible — the
    authoring-feedback loop, mnemoai-style.

    Attributes:
        name: The directory name.
        reason: A short, user-facing explanation of why it was skipped.
    """

    name: str
    reason: str


def _parse_frontmatter(text: str) -> Tuple[dict, str]:
    """Split a ``SKILL.md`` into (frontmatter dict, body).

    Expects a leading ``---`` fenced YAML block. Returns ``({}, text)`` when no
    frontmatter is present so the caller can reject it for missing fields.
    """
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return {}, text
    # Drop the opening fence, then split on the closing one.
    after_open = stripped[len("---"):].lstrip("\n")
    parts = after_open.split("\n---", 1)
    if len(parts) != 2:
        return {}, text
    front_raw, body = parts
    data = yaml.safe_load(front_raw)
    if not isinstance(data, dict):
        return {}, text
    return data, body.lstrip("\n")


class SkillStore:
    """Scan and read skills from a skills root directory."""

    def __init__(self, root: Optional[Path] = None) -> None:
        """Initialize the store.

        Args:
            root: Skills root directory; defaults to ``paths.skills_dir()``.
        """
        if root is None:
            from mnemoai.utils.paths import skills_dir

            root = skills_dir()
        self.root = Path(root)

    def _scan(self) -> Tuple[List[Skill], List[SkillIssue]]:
        """Scan the root once, returning (valid skills, rejected issues).

        Each skill is one ``<name>/SKILL.md``. A directory whose ``SKILL.md`` is
        missing/unreadable, has malformed YAML, or lacks the required
        ``name``/``description`` is collected as a :class:`SkillIssue` (with a
        user-facing reason) rather than silently dropped — so ``/skills`` can
        surface it. A non-directory entry, or a directory with no ``SKILL.md`` at
        all, is ignored (not a skill attempt). An absent root yields empties.
        """
        if not self.root.is_dir():
            return [], []
        skills: List[Skill] = []
        issues: List[SkillIssue] = []
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir():
                continue
            skill_md = entry / "SKILL.md"
            if not skill_md.is_file():
                continue  # a dir without SKILL.md isn't a skill attempt
            try:
                text = skill_md.read_text()
            except (OSError, UnicodeDecodeError) as e:
                issues.append(SkillIssue(entry.name, f"could not read SKILL.md ({e})"))
                continue
            try:
                front, body = _parse_frontmatter(text)
            except yaml.YAMLError as e:
                issues.append(SkillIssue(entry.name, f"invalid YAML frontmatter ({e})"))
                continue
            if not front:
                issues.append(SkillIssue(entry.name, "no YAML frontmatter (--- block)"))
                continue
            description = str(front.get("description", "")).strip()
            if not description:
                issues.append(SkillIssue(entry.name, "missing 'description' in frontmatter"))
                continue
            if len(description) > _MAX_DESC_LEN:
                issues.append(
                    SkillIssue(entry.name, f"description too long ({len(description)} > {_MAX_DESC_LEN} chars)")
                )
                continue
            argument_hint = str(front.get("argument_hint", "")).strip().replace("\n", " ")
            skills.append(
                Skill(
                    name=entry.name,
                    description=description,
                    body=body.strip(),
                    path=entry,
                    argument_hint=argument_hint,
                )
            )
        if skills:
            logger.debug("Loaded %d skill(s): %s", len(skills), ", ".join(s.name for s in skills))
        for issue in issues:
            print_error(f"Skill '{issue.name}': {issue.reason}; skipping.")
        return skills, issues

    def list_skills(self) -> List[Skill]:
        """Return all valid skills under the root (sorted by name)."""
        return self._scan()[0]

    def list_issues(self) -> List[SkillIssue]:
        """Return skill directories that were found but rejected, with reasons."""
        return self._scan()[1]

    def list_metadata(self) -> List[Tuple[str, str]]:
        """Return ``[(name, description)]`` for the tier-1 system-prompt block."""
        return [(s.name, s.description) for s in self.list_skills()]

    def load_body(self, name: str) -> Optional[Skill]:
        """Return the full skill for ``name`` (directory id), or None if unknown."""
        name = (name or "").strip()
        if not name:
            return None
        for skill in self.list_skills():
            if skill.name == name:
                return skill
        return None


def _skill_line(entry) -> str:
    """Render one listing line from either a ``Skill`` or a legacy ``(name, desc)``.

    Accepts both so callers that only have name+description (the historical
    ``list_metadata`` shape) keep working, while injection callers can pass full
    ``Skill`` objects to surface ``argument_hint``.
    """
    if isinstance(entry, Skill):
        name, desc, hint = entry.name, entry.description, entry.argument_hint
    else:
        name, desc = entry[0], entry[1]
        hint = ""
    d = desc.strip().replace("\n", " ")
    if len(d) > _MAX_DESC_CHARS:
        d = d[: _MAX_DESC_CHARS - 1].rstrip() + "…"
    line = f"  - {name}: {d}"
    if hint:
        line += f" (expects: {hint})"
    return line


def format_available_skills(meta) -> str:
    """Build the always-on ``<available_skills>`` system-prompt block.

    ``meta`` is a list of ``Skill`` objects or legacy ``(name, description)``
    tuples. Returns "" when there are no skills. Each description is truncated
    and the whole listing is bounded by ``_MAX_LISTING_CHARS`` (it is injected on
    every turn); skills beyond the budget are collapsed into a "+N more" line so
    a large skills library can't dominate the prompt. Used by both the client's
    session-start injection and the compaction re-injection so the format is
    defined once.
    """
    if not meta:
        return ""
    lines = []
    used = 0
    remaining = 0
    for i, entry in enumerate(meta):
        line = _skill_line(entry)
        # Always keep the first line; otherwise stop once over the aggregate cap.
        if lines and used + len(line) > _MAX_LISTING_CHARS:
            remaining = len(meta) - i
            break
        lines.append(line)
        used += len(line) + 1  # +1 for the newline
    if remaining:
        lines.append(f"  … (+{remaining} more — see /skills)")
    body = "\n".join(lines)
    return (
        "<available_skills>\n"
        "You have these skills — authored, step-by-step procedures. When the "
        "user's request matches one, call the `use_skill` tool with its name to "
        "load the full instructions BEFORE acting. Do not guess the procedure.\n"
        f"{body}\n"
        "</available_skills>"
    )
