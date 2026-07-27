"""Unit tests: the session-start context blocks must survive compaction.

Compaction rebuilds the system prompt from the bare ``SYSTEM_PROMPT``, so every
session-start injection has to be re-added or it silently disappears for the
rest of the session. Regression guard for the profile / MEMORY.md / playbook
blocks (skills + sub-agents are covered by test_skills_injection.py).

``context_injection.build_session_blocks`` is the single source of truth used by
both assembly paths; these tests pin that contract without an LLM.
"""

from types import SimpleNamespace

import pytest

from mnemoai.client import context_injection
from mnemoai.client.managers.agent_conversation_manager import (
    AgentConversationManager,
)

MEMORY_TEXT = "[user] Prefers pytest over unittest."
PROFILE_TEXT = "<profile>\nStyle: concise\n</profile>"
PLAYBOOK_TEXT = "[Playbook - Learned Strategies]\nAvoid: guessing paths"


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A fake client with all four collaborators + on-disk MEMORY.md and a skill."""
    memory_file = tmp_path / "MEMORY.md"
    memory_file.write_text(MEMORY_TEXT)
    monkeypatch.setattr(
        "mnemoai.utils.paths.memory_file_path", lambda: memory_file, raising=True
    )

    skills_root = tmp_path / "skills"
    skill = skills_root / "alpha"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Use when the user asks for alpha.\n---\nBody.\n"
    )
    monkeypatch.setattr(
        "mnemoai.utils.paths.skills_dir", lambda: skills_root, raising=True
    )

    from mnemoai.utils.config import config

    real_get = config.get
    monkeypatch.setattr(
        config,
        "get",
        lambda k, d=None: (
            {"USE_PROFILING": True} if k == "PROFILE" else real_get(k, d)
        ),
    )

    playbook = SimpleNamespace(
        get_relevant_entries=lambda task, top_k, include_failures: ["entry"],
        format_for_prompt=lambda entries: PLAYBOOK_TEXT,
    )
    return SimpleNamespace(
        profile_manager=SimpleNamespace(get_profile_summary=lambda: PROFILE_TEXT),
        playbook=playbook,
    )


class TestBuildSessionBlocks:
    def test_all_blocks_present_with_playbook(self, wired):
        blocks = context_injection.build_session_blocks(wired, include_playbook=True)
        joined = "\n\n".join(blocks)
        assert PROFILE_TEXT in joined
        assert MEMORY_TEXT in joined
        assert "<available_skills>" in joined
        assert "<available_subagents>" in joined
        assert PLAYBOOK_TEXT in joined

    def test_playbook_excluded_by_default(self, wired):
        # The session-start path appends the playbook itself, so the default must
        # leave it out to avoid a duplicate block.
        joined = "\n\n".join(context_injection.build_session_blocks(wired))
        assert MEMORY_TEXT in joined
        assert PLAYBOOK_TEXT not in joined

    def test_injection_order_is_stable(self, wired):
        joined = "\n\n".join(
            context_injection.build_session_blocks(wired, include_playbook=True)
        )
        positions = [
            joined.index(PROFILE_TEXT),
            joined.index(MEMORY_TEXT),
            joined.index("<available_skills>"),
            joined.index("<available_subagents>"),
            joined.index(PLAYBOOK_TEXT),
        ]
        assert positions == sorted(positions)

    def test_empty_blocks_dropped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mnemoai.utils.paths.memory_file_path",
            lambda: tmp_path / "absent.md",
            raising=True,
        )
        monkeypatch.setattr(
            "mnemoai.utils.paths.skills_dir", lambda: tmp_path, raising=True
        )
        client = SimpleNamespace(
            profile_manager=SimpleNamespace(get_profile_summary=lambda: ""),
            playbook=None,
        )
        blocks = context_injection.build_session_blocks(client, include_playbook=True)
        assert all(block for block in blocks)
        assert not any("[Persistent Memory]" in block for block in blocks)


class TestCompactionRebuild:
    def test_rebuilt_prompt_keeps_every_block(self, wired):
        mgr = AgentConversationManager(max_tokens=1000)
        rebuilt = mgr._build_system_with_summary("Earlier we did X.", client=wired)
        assert MEMORY_TEXT in rebuilt, "MEMORY.md lost after compaction"
        assert PROFILE_TEXT in rebuilt, "user profile lost after compaction"
        assert PLAYBOOK_TEXT in rebuilt, "playbook lost after compaction"
        assert "<available_skills>" in rebuilt
        assert "<available_subagents>" in rebuilt
        assert "Earlier we did X." in rebuilt

    def test_summary_block_stays_last(self, wired):
        mgr = AgentConversationManager(max_tokens=1000)
        rebuilt = mgr._build_system_with_summary("Earlier we did X.", client=wired)
        assert rebuilt.index("<conversation_summary>") > rebuilt.index(MEMORY_TEXT)

    def test_client_read_failure_does_not_abort_compaction(self, wired):
        # A broken memory/profile read must degrade the prompt, never raise: the
        # compaction it would abort is what keeps the turn inside the window.
        boom = SimpleNamespace(
            profile_manager=SimpleNamespace(
                get_profile_summary=lambda: (_ for _ in ()).throw(OSError("disk"))
            ),
            playbook=None,
        )
        mgr = AgentConversationManager(max_tokens=1000)
        rebuilt = mgr._build_system_with_summary("Earlier we did X.", client=boom)
        assert "Earlier we did X." in rebuilt
        # Falls back to the client-independent blocks.
        assert "<available_subagents>" in rebuilt

    def test_without_client_falls_back_to_shared_blocks(self, wired):
        mgr = AgentConversationManager(max_tokens=1000)
        rebuilt = mgr._build_system_with_summary("Earlier we did X.")
        assert "<available_skills>" in rebuilt
        assert "<available_subagents>" in rebuilt
        assert "Earlier we did X." in rebuilt
