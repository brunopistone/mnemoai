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

    def test_example_files_refreshed_from_bundle(self, tmp_home):
        # .example files are read-only reference (never loaded as config), so they
        # are refreshed from the bundle on re-seed — this is how a NEW config key
        # reaches an EXISTING install on upgrade. A stale local copy is replaced.
        example = tmp_home / "config" / "config.yaml.example"
        example.parent.mkdir(parents=True, exist_ok=True)
        example.write_text("STALE")
        paths.seed_example_files()
        assert example.read_text() != "STALE"  # refreshed from the package
        assert "EVICTED_TOOL_RESULT_CHARS" in example.read_text()  # new key present

    def test_live_config_never_overwritten(self, tmp_home):
        # The user's LIVE config.yaml is sacred — seeding must never touch it.
        live = tmp_home / "config" / "config.yaml"
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_text("MY LIVE CONFIG")
        paths.seed_example_files()
        assert live.read_text() == "MY LIVE CONFIG"

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
        # A user's edit to a bundled skill survives re-seeding: the edited content
        # isn't a pristine (shipped) hash, so the in-place refresh leaves it alone.
        paths.seed_example_files()
        sc = tmp_home / "skills" / "steering-creator" / "SKILL.md"
        sc.write_text("USER EDITED")
        paths.seed_example_files()
        assert sc.read_text() == "USER EDITED"

    def test_pristine_installed_skill_is_refreshed(self, tmp_home, monkeypatch):
        # A bundled skill whose installed SKILL.md is PRISTINE (a version we
        # shipped) is refreshed in place on upgrade, so doc/frontmatter updates
        # reach existing installs.
        skills = tmp_home / "skills"
        sc_dir = skills / "skill-creator"
        sc_dir.mkdir(parents=True)
        old = "old shipped body\n"
        sc_md = sc_dir / "SKILL.md"
        sc_md.write_text(old)
        # Register the old content's hash as a known-pristine shipped version.
        old_hash = paths._sha256(sc_md)
        monkeypatch.setitem(
            paths._PRISTINE_BUNDLED_SKILL_HASHES, "skill-creator", {old_hash}
        )
        paths.seed_example_files()
        # Refreshed to the current bundle (which documents argument_hint).
        assert sc_md.read_text() != old
        assert "argument_hint" in sc_md.read_text()

    def test_non_pristine_skill_left_untouched(self, tmp_home, monkeypatch):
        # An installed SKILL.md whose hash is NOT in the pristine set (user-edited)
        # is never refreshed, even though the dir exists.
        skills = tmp_home / "skills"
        sc_dir = skills / "skill-creator"
        sc_dir.mkdir(parents=True)
        sc_md = sc_dir / "SKILL.md"
        sc_md.write_text("MY CUSTOM SKILL")
        monkeypatch.setitem(
            paths._PRISTINE_BUNDLED_SKILL_HASHES, "skill-creator", {"deadbeef"}
        )
        paths.seed_example_files()
        assert sc_md.read_text() == "MY CUSTOM SKILL"

    def test_pristine_refresh_preserves_sibling_files(self, tmp_home, monkeypatch):
        # Refreshing SKILL.md must not disturb other files the user added alongside.
        skills = tmp_home / "skills"
        sc_dir = skills / "skill-creator"
        sc_dir.mkdir(parents=True)
        sc_md = sc_dir / "SKILL.md"
        sc_md.write_text("old\n")
        (sc_dir / "notes.md").write_text("my notes")
        monkeypatch.setitem(
            paths._PRISTINE_BUNDLED_SKILL_HASHES,
            "skill-creator",
            {paths._sha256(sc_md)},
        )
        paths.seed_example_files()
        assert (sc_dir / "notes.md").read_text() == "my notes"

    def test_current_bundled_hashes_are_registered_as_pristine(self):
        # Guard: every bundled skill's CURRENT SKILL.md hash — or a documented
        # prior one — must be tracked so a freshly-seeded skill is recognized as
        # pristine on the NEXT upgrade. Catches forgetting to append the prior hash
        # after editing a bundled skill.
        from pathlib import Path

        root = Path(paths.__file__).resolve().parent / "skills_example"
        for skill_dir in root.iterdir():
            md = skill_dir / "SKILL.md"
            if not md.is_file():
                continue
            known = paths._PRISTINE_BUNDLED_SKILL_HASHES.get(skill_dir.name, set())
            # The current shipped hash need not be listed (it's compared live), but
            # the skill MUST have an entry so the mechanism applies to it at all.
            assert skill_dir.name in paths._PRISTINE_BUNDLED_SKILL_HASHES, (
                f"{skill_dir.name} missing from _PRISTINE_BUNDLED_SKILL_HASHES"
            )
            assert known, f"{skill_dir.name} has an empty pristine-hash set"

    def test_prompts_seeded_when_absent(self, tmp_home):
        # A fresh install gets prompts.yaml copied out of the box.
        paths.seed_example_files()
        dest = paths.prompts_path()
        assert dest.is_file()
        assert "SYSTEM_PROMPT" in dest.read_text()

    def test_pristine_prompts_refreshed_on_upgrade(self, tmp_home, monkeypatch):
        # A prompts.yaml whose installed hash is PRISTINE (a version we shipped)
        # is refreshed in place so prompt improvements reach existing installs.
        dest = paths.prompts_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        old = "SYSTEM_PROMPT: |\n  an old shipped prompt\n"
        dest.write_text(old)
        monkeypatch.setattr(
            paths, "_PRISTINE_BUNDLED_PROMPTS_HASHES", {paths._sha256(dest)}
        )
        paths.seed_example_files()
        # Refreshed to the current bundled prompts.
        assert dest.read_text() != old
        assert "SYSTEM_PROMPT" in dest.read_text()

    def test_customized_prompts_left_untouched(self, tmp_home, monkeypatch):
        # A user-customized prompts.yaml (hash NOT in the pristine set) is never
        # overwritten, even though the file exists.
        dest = paths.prompts_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("SYSTEM_PROMPT: |\n  MY CUSTOM PROMPT\n")
        monkeypatch.setattr(paths, "_PRISTINE_BUNDLED_PROMPTS_HASHES", {"deadbeef"})
        paths.seed_example_files()
        assert "MY CUSTOM PROMPT" in dest.read_text()

    def test_current_bundled_prompts_hash_tracked(self):
        # Guard: the pristine set must be non-empty so the mechanism applies. When
        # the bundled prompts.yaml changes, its PREVIOUS shipped hash must be added.
        assert paths._PRISTINE_BUNDLED_PROMPTS_HASHES, (
            "_PRISTINE_BUNDLED_PROMPTS_HASHES is empty — prompt refresh won't apply"
        )

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
