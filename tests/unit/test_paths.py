"""Unit tests for the centralized path helper (utils/paths.py).

All persistent state lives under a single app-home dir
(~/.mnemoai by default, overridable via
$MNEMOAI_HOME).
"""

import pytest

from mnemoai.utils import paths


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    """Point the app home at a temp dir for the duration of a test."""
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))
    return tmp_path


class TestAppHome:
    def test_env_override_used_and_created(self, tmp_home):
        home = paths.app_home()
        assert home == tmp_home
        assert home.is_dir()

    def test_default_is_dot_mnemoai(self, monkeypatch):
        monkeypatch.delenv("MNEMOAI_HOME", raising=False)
        # Don't actually create it in the real home; just check the path shape.
        from pathlib import Path

        expected = Path.home() / ".mnemoai"
        # app_home() creates it; tolerate that but assert location.
        assert paths.app_home() == expected


class TestSubdirs:
    def test_config_path_under_home(self, tmp_home):
        # config.yaml now lives in the config/ subfolder (created on access).
        assert paths.config_path() == tmp_home / "config" / "config.yaml"
        assert (tmp_home / "config").is_dir()

    def test_legacy_config_path_is_flat(self, tmp_home):
        # The pre-subfolder fallback still points at the flat location.
        assert paths.legacy_config_path() == tmp_home / "config.yaml"

    def test_mcp_paths_under_home(self, tmp_home):
        assert paths.mcp_config_path() == tmp_home / "mcp" / "mcp.json"
        assert paths.legacy_mcp_config_path() == tmp_home / "mcp.json"
        assert (tmp_home / "mcp").is_dir()

    def test_seed_example_files_copies_examples(self, tmp_home):
        paths.seed_example_files()
        # Examples land in the subfolders; live files are NOT created.
        assert (tmp_home / "config" / "config.yaml.example").is_file()
        assert (tmp_home / "mcp" / "mcp.json.example").is_file()
        assert not (tmp_home / "config" / "config.yaml").exists()
        assert not (tmp_home / "mcp" / "mcp.json").exists()
        # Idempotent + non-destructive: a user edit survives a re-seed.
        edited = tmp_home / "config" / "config.yaml.example"
        edited.write_text("EDITED")
        paths.seed_example_files()
        assert edited.read_text() == "EDITED"

    def test_seed_skills_are_bundled_examples(self, tmp_home):
        # Bundled example skills land in skills/ out of the box.
        paths.seed_example_files()
        skills = tmp_home / "skills"
        seeded = {p.name for p in skills.iterdir() if p.is_dir()}
        assert "steering-creator" in seeded  # the new bundled skill
        assert "skill-creator" in seeded

    def test_seed_skills_per_skill_reaches_populated_dir(self, tmp_home):
        # Regression: a NEW bundled skill must reach an EXISTING (non-empty)
        # skills dir on upgrade — seeding is per-skill, not all-or-nothing.
        skills = tmp_home / "skills"
        skills.mkdir(parents=True, exist_ok=True)
        (skills / "my-own-skill").mkdir()  # user already has a skill
        paths.seed_example_files()
        names = {p.name for p in skills.iterdir() if p.is_dir()}
        assert "my-own-skill" in names          # user's skill untouched
        assert "steering-creator" in names      # bundled skill still seeded

    def test_seed_skills_never_clobbers_user_edit(self, tmp_home):
        # A user's edit to a bundled skill survives re-seeding (dir exists → skip).
        paths.seed_example_files()
        sc = tmp_home / "skills" / "steering-creator" / "SKILL.md"
        sc.write_text("USER EDITED")
        paths.seed_example_files()
        assert sc.read_text() == "USER EDITED"

    def test_plans_and_tasks_created(self, tmp_home):
        assert paths.plans_dir() == tmp_home / "plans"
        assert paths.tasks_dir() == tmp_home / "tasks"
        assert (tmp_home / "plans").is_dir()
        assert (tmp_home / "tasks").is_dir()

    def test_mcp_log_path_under_logs_dir(self, tmp_home):
        assert paths.logs_dir() == tmp_home / "logs"
        assert (tmp_home / "logs").is_dir()
        assert paths.mcp_log_path() == tmp_home / "logs" / "mcp.log"

    def test_open_mcp_log_appends_when_small(self, tmp_home):
        # Below the cap: keep appending to the same file, no rotation.
        with paths.open_mcp_log() as f:
            f.write("first\n")
        with paths.open_mcp_log() as f:
            f.write("second\n")
        assert paths.mcp_log_path().read_text() == "first\nsecond\n"
        assert not (tmp_home / "logs" / "mcp.log.1").exists()

    def test_open_mcp_log_rotates_when_oversized(self, tmp_home, monkeypatch):
        # At/over the cap: the current log becomes mcp.log.1 and a fresh log opens.
        monkeypatch.setattr(paths, "MCP_LOG_MAX_BYTES", 50)
        log = paths.mcp_log_path()
        log.write_text("X" * 60)  # over the cap
        with paths.open_mcp_log() as f:
            f.write("new content\n")
        assert (tmp_home / "logs" / "mcp.log.1").read_text() == "X" * 60
        assert log.read_text() == "new content\n"

    def test_open_mcp_log_single_backup_generation(self, tmp_home, monkeypatch):
        # Rotating twice keeps only ONE backup (mcp.log.1 is replaced, not stacked).
        monkeypatch.setattr(paths, "MCP_LOG_MAX_BYTES", 50)
        paths.mcp_log_path().write_text("A" * 60)
        with paths.open_mcp_log() as f:
            f.write("B" * 60)
        with paths.open_mcp_log() as f:
            f.write("newest\n")
        assert (tmp_home / "logs" / "mcp.log.1").read_text() == "B" * 60  # A gone
        assert not (tmp_home / "logs" / "mcp.log.2").exists()

    def test_sweep_old_plans_removes_stale_only(self, tmp_home):
        import os
        import time

        d = paths.plans_dir()
        old = d / "plan_20260101_000000.md"
        old.write_text("old plan")
        recent = d / "plan_20260714_000000.md"
        recent.write_text("recent plan")
        stale = time.time() - 10 * 86400
        os.utime(old, (stale, stale))

        removed = paths.sweep_old_plans(max_age_days=7)

        assert removed == 1
        assert not old.exists()
        assert recent.exists()  # within window, kept

    def test_sweep_old_plans_ignores_non_plan_files(self, tmp_home):
        import os
        import time

        d = paths.plans_dir()
        (d / "notes.md").write_text("keep")          # not plan_*.md
        keep_json = d / "custom.json"
        keep_json.write_text("keep")
        stale = time.time() - 30 * 86400
        for f in (d / "notes.md", keep_json):
            os.utime(f, (stale, stale))

        paths.sweep_old_plans(max_age_days=7)

        assert (d / "notes.md").exists()
        assert keep_json.exists()

    def test_sweep_old_plans_removes_legacy_current_plan_json(self, tmp_home):
        d = paths.plans_dir()
        legacy = d / "current_plan.json"
        legacy.write_text("{}")  # retired plan_mode.py artifact
        paths.sweep_old_plans(max_age_days=7)
        assert not legacy.exists()

    def test_profile_dir_explicit(self, tmp_home):
        d = paths.profile_dir("alice")
        assert d == tmp_home / "alice"
        assert d.is_dir()

    def test_conversations_dir_under_profile(self, tmp_home):
        # Regression: /save must write to <profile>/conversations/, not the
        # profile root.
        d = paths.conversations_dir("alice")
        assert d == tmp_home / "alice" / "conversations"
        assert d.is_dir()

    def test_model_dir_nested_and_sanitized(self, tmp_home):
        d = paths.model_dir("brnpistone/Qwen3.5-4B:latest", profile="bob")
        assert d == tmp_home / "bob" / "models" / "brnpistone_Qwen3.5-4B_latest"
        assert d.is_dir()


class TestSanitizeModelName:
    def test_slashes_colons_spaces(self):
        assert paths.sanitize_model_name("a/b:c d") == "a_b_c_d"

    def test_dotted_id_preserved(self):
        assert (
            paths.sanitize_model_name("global.anthropic.claude-fable-5")
            == "global.anthropic.claude-fable-5"
        )

    def test_empty_and_none_default(self):
        assert paths.sanitize_model_name("") == "default"
        assert paths.sanitize_model_name(None) == "default"

    def test_result_has_no_separators(self):
        out = paths.sanitize_model_name("x/y:z w")
        assert "/" not in out and ":" not in out and " " not in out
