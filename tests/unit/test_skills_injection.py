"""Unit tests for skills system-prompt injection (tier-1) and its survival
through conversation compaction.

The <available_skills> block is injected at session start and MUST be
re-injected after compaction (which re-fetches the base prompt fresh, dropping
session-start injections). These test the building blocks without an LLM:
- AgentConversationManager._skills_block() / _build_system_with_summary()
- the format used is shared via skill_store.format_available_skills
"""

import pytest

from mnemoai.client.managers.agent_conversation_manager import (
    AgentConversationManager,
)


def _write_skill(root, name, desc):
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\nBody.\n")


@pytest.fixture
def with_skill(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mnemoai.utils.paths.skills_dir", lambda: tmp_path, raising=True
    )
    _write_skill(tmp_path, "alpha", "Use when the user asks for alpha.")
    return tmp_path


class TestSkillsBlock:
    def test_block_built_when_skills_present(self, with_skill):
        mgr = AgentConversationManager(max_tokens=1000)
        block = mgr._skills_block()
        assert "<available_skills>" in block
        assert "alpha" in block

    def test_block_empty_without_skills(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mnemoai.utils.paths.skills_dir", lambda: tmp_path, raising=True
        )
        mgr = AgentConversationManager(max_tokens=1000)
        assert mgr._skills_block() == ""

    def test_block_empty_when_disabled(self, with_skill, monkeypatch):
        # ENABLE_SKILLS=False -> no block even if skills exist on disk.
        from mnemoai.utils.config import config

        monkeypatch.setattr(
            config, "get",
            lambda k, d=None: False if k == "ENABLE_SKILLS" else config.get(k, d),
        )
        mgr = AgentConversationManager(max_tokens=1000)
        assert mgr._skills_block() == ""


class TestCompactionReinjection:
    def test_summary_prompt_includes_skills_block(self, with_skill):
        # The rebuilt system prompt after compaction must carry the skills block,
        # else skills silently vanish after the first auto-compact.
        mgr = AgentConversationManager(max_tokens=1000)
        rebuilt = mgr._build_system_with_summary("Earlier we did X and Y.")
        assert "<available_skills>" in rebuilt
        assert "alpha" in rebuilt
        # The summary itself is still present.
        assert "Earlier we did X and Y." in rebuilt

    def test_summary_prompt_without_skills_has_no_block(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mnemoai.utils.paths.skills_dir", lambda: tmp_path, raising=True
        )
        mgr = AgentConversationManager(max_tokens=1000)
        rebuilt = mgr._build_system_with_summary("Earlier we did X.")
        assert "<available_skills>" not in rebuilt
