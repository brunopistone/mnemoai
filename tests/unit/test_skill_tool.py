"""Unit tests for the use_skill MCP tool (server/tools/skill_tool.py).

The tool loads a skill's full SKILL.md body (tier-2) by name, or returns an
actionable error listing available skills for an unknown name. It reads through
SkillStore against the default skills_dir(), so the tests point that at a tmp
dir via monkeypatch. No LLM involved — unit tier.
"""

import asyncio

import pytest

from mnemoai.server.tools.skill_tool import register_skill_tools


class _CapturingMCP:
    """Minimal stand-in for FastMCP that captures registered tool functions."""

    def __init__(self):
        self.registered = {}

    def tool(self):
        def decorator(func):
            self.registered[func.__name__] = func
            return func

        return decorator


def _write_skill(root, name, desc, body="# Body\nDo the thing."):
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n{body}\n")


@pytest.fixture
def use_skill(tmp_path, monkeypatch):
    # Point SkillStore's default root at the tmp skills dir.
    monkeypatch.setattr(
        "mnemoai.utils.paths.skills_dir", lambda: tmp_path, raising=True
    )
    _write_skill(tmp_path, "alpha", "Use when the user asks for alpha.")
    mcp = _CapturingMCP()
    register_skill_tools(mcp)
    return mcp.registered["use_skill"]


def run(coro):
    return asyncio.run(coro)


class TestUseSkill:
    def test_known_skill_returns_body(self, use_skill):
        out = run(use_skill("alpha"))
        assert "# Skill: alpha" in out
        assert "Do the thing." in out
        # Footer points the model at the skill dir for bundled resources.
        assert "alpha" in out and "fs_read" in out

    def test_unknown_skill_lists_available(self, use_skill):
        out = run(use_skill("nope"))
        assert "No skill named" in out
        assert "alpha" in out  # lists what IS available
        # Must not raise — it returns a model-facing string.

    def test_blank_name_lists_available(self, use_skill):
        out = run(use_skill(""))
        assert "No skill named" in out
