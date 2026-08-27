"""Unit tests for the centralized path helper (utils/paths.py).

All persistent state lives under a single app-home dir
(~/.mnemoai by default, overridable via
$MNEMOAI_HOME).
"""

import os
import subprocess
from pathlib import Path

import pytest

from mnemoai.utils import paths

# The repo, located from THIS file — never from the process cwd. The
# shipped-hash guards below query git history, and a test that trusts cwd stops
# checking anything the moment something moves it (the integration tier chdirs
# for its own isolation): `git tag` then returns nothing, both guards skip, and
# the suite still reports green while protecting nothing.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args, text=True):
    """Run git against this repo regardless of the process cwd."""
    return subprocess.run(
        ["git", "-C", str(_REPO_ROOT), *args], capture_output=True, text=text
    )


def _shipped_tags():
    """Every release tag, or [] in a shallow/exported checkout (no git history)."""
    return [t for t in _git("tag", "--list", "v*").stdout.split() if t]


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
        assert (tmp_home / "hooks" / "hooks.json.example").is_file()
        assert not (tmp_home / "config" / "config.yaml").exists()
        assert not (tmp_home / "mcp" / "mcp.json").exists()
        assert not (tmp_home / "hooks" / "hooks.json").exists()

    def test_hooks_example_reaches_populated_home(self, tmp_home):
        # A new bundled example must reach an install that ALREADY exists — the
        # per-item "seed if absent" rule, not a first-run-only condition.
        (tmp_home / "config").mkdir(parents=True)
        (tmp_home / "config" / "config.yaml").write_text("MY LIVE CONFIG")
        paths.seed_example_files()
        assert (tmp_home / "hooks" / "hooks.json.example").is_file()

    def test_hooks_config_is_app_home_only(self, tmp_home):
        # Hooks are arbitrary code: the path must be under the app home, so a
        # hooks.json arriving with a git clone can never be picked up.
        assert paths.hooks_config_path() == tmp_home / "hooks" / "hooks.json"

    def test_seeding_never_creates_a_live_hooks_file(self, tmp_home):
        # Presence is the switch — seeding a live hooks.json would silently arm it.
        paths.seed_example_files()
        paths.seed_example_files()
        assert not paths.hooks_config_path().exists()

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

    def test_previously_shipped_skill_hashes_are_tracked(self):
        # The check above only proves each skill HAS an entry — it cannot notice a
        # missing version, which is the failure that actually costs users: an
        # installed SKILL.md whose hash we shipped but never registered reads as
        # "user-edited", so the refresh skips it and an improved skill never
        # reaches that install. Tags are enumerated from git rather than
        # hardcoded, for the same reason as the prompts.yaml guard below.
        import hashlib
        from pathlib import Path

        tags = _shipped_tags()
        if not tags:
            pytest.skip("no tags available (shallow or exported checkout)")

        root = Path(paths.__file__).resolve().parent / "skills_example"
        names = [d.name for d in root.iterdir() if (d / "SKILL.md").is_file()]
        current = {n: paths._sha256(root / n / "SKILL.md") for n in names}

        missing = []
        for tag in tags:
            for name in names:
                out = _git(
                    "show",
                    f"{tag}:src/mnemoai/utils/skills_example/{name}/SKILL.md",
                    text=False,
                )
                if out.returncode != 0 or not out.stdout:
                    continue  # skill didn't exist at that tag
                digest = hashlib.sha256(out.stdout).hexdigest()
                # The hash still bundled today needn't be listed (it's compared
                # live against the bundle), only superseded ones.
                if digest == current[name]:
                    continue
                if digest not in paths._PRISTINE_BUNDLED_SKILL_HASHES.get(name, set()):
                    missing.append(f"{name}@{tag} ({digest[:12]}…)")
        assert not missing, (
            "these shipped SKILL.md versions are not in "
            f"_PRISTINE_BUNDLED_SKILL_HASHES, so the refresh will treat them as "
            f"user-edited and skip them: {sorted(set(missing))}"
        )

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

    def test_previously_shipped_prompts_hashes_are_tracked(self):
        # The non-empty check above is too weak: it passed all through 1.6.3–1.7.6
        # while the hash of the THEN-shipped prompts.yaml was missing from the set,
        # so an edit to an existing prompt key would silently never reach those
        # installs (the bundled-fallback loader only fills MISSING keys). Every
        # version we have shipped must be recognized as pristine.
        #
        # Tags are ENUMERATED from git, never hardcoded: a fixed list stops
        # covering the releases made after it was written, which is exactly how
        # the 1.8.4–1.8.7 hash went four releases unregistered while this test
        # stayed green.
        import hashlib
        from pathlib import Path

        tags = _shipped_tags()
        if not tags:
            pytest.skip("no tags available (shallow or exported checkout)")

        # The CURRENTLY bundled content is deliberately absent from the set — it
        # is compared against the bundle at runtime instead. Skipping it matters
        # once the release tag exists: the tag we just cut ships this very file,
        # so without this the test fails on every release (and the same rule is
        # already applied by the SKILL.md guard above).
        current = paths._sha256(Path(paths.__file__).resolve().parent / "prompts.yaml")

        missing = []
        for tag in tags:
            out = _git("show", f"{tag}:src/mnemoai/utils/prompts.yaml", text=False)
            if out.returncode != 0 or not out.stdout:
                continue  # tag not present in a shallow/exported checkout
            digest = hashlib.sha256(out.stdout).hexdigest()
            if digest == current:
                continue
            if digest not in paths._PRISTINE_BUNDLED_PROMPTS_HASHES:
                missing.append(f"{tag} ({digest[:12]}…)")
        assert not missing, (
            "prompts.yaml shipped in these versions is not in "
            f"_PRISTINE_BUNDLED_PROMPTS_HASHES, so a prompt edit will NOT reach "
            f"those installs: {missing}"
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

    def test_app_log_path_does_not_create_the_dir(self, tmp_home):
        # /doctor reports this path; a read-only report must not write anything.
        assert paths.app_log_path() == tmp_home / "logs" / "mnemoai.log"
        assert not (tmp_home / "logs").exists()

    def test_sweep_old_logs_removes_stale_only(self, tmp_home):
        import os
        import time

        d = paths.logs_dir()
        old, rotated, recent = d / "mnemoai.log.2", d / "mcp.log.1", d / "mnemoai.log"
        for f in (old, rotated, recent):
            f.write_text("x")
        stale = time.time() - 30 * 86400
        os.utime(old, (stale, stale))
        os.utime(rotated, (stale, stale))

        removed = paths.sweep_old_logs(max_age_days=7)

        assert removed == 2
        assert not old.exists() and not rotated.exists()
        assert recent.exists()  # the live instance's log is never the one swept

    def test_sweep_old_logs_disabled_by_zero(self, tmp_home):
        import os
        import time

        f = paths.logs_dir() / "mnemoai.log"
        f.write_text("x")
        stale = time.time() - 365 * 86400
        os.utime(f, (stale, stale))
        assert paths.sweep_old_logs(max_age_days=0) == 0
        assert f.exists()

    def test_sweep_old_logs_no_dir_is_a_noop(self, tmp_home):
        # Must not CREATE logs/ just to sweep it (a launch that never logs).
        assert paths.sweep_old_logs(max_age_days=7) == 0
        assert not (tmp_home / "logs").exists()

    def test_instance_id_stable_within_process(self, tmp_home, monkeypatch):
        # Cached in the env so both halves of one instance (client + its MCP
        # subprocess, which copies os.environ) resolve the same id.
        monkeypatch.delenv("MNEMOAI_INSTANCE_ID", raising=False)
        first = paths.instance_id()
        assert paths.instance_id() == first
        assert os.environ["MNEMOAI_INSTANCE_ID"] == first

    def test_instance_id_inherited_from_env(self, tmp_home, monkeypatch):
        # A subprocess that inherited MNEMOAI_INSTANCE_ID reuses it verbatim.
        monkeypatch.setenv("MNEMOAI_INSTANCE_ID", "parent_123")
        assert paths.instance_id() == "parent_123"

    def test_session_pointers_are_per_instance(self, tmp_home, monkeypatch):
        # The multi-tab clobber bug: two instances must NOT share a pointer file.
        monkeypatch.setenv("MNEMOAI_INSTANCE_ID", "tab_A")
        rag_a = paths.rag_session_pointer_path()
        chunk_a = paths.chunk_session_pointer_path()
        monkeypatch.setenv("MNEMOAI_INSTANCE_ID", "tab_B")
        rag_b = paths.rag_session_pointer_path()
        chunk_b = paths.chunk_session_pointer_path()
        assert rag_a != rag_b
        assert chunk_a != chunk_b
        assert "tab_A" in rag_a.name and "tab_B" in rag_b.name

    def test_sweep_old_rag_artifacts_removes_stale_only(self, tmp_home, monkeypatch):
        import time

        monkeypatch.setenv("MNEMOAI_INSTANCE_ID", "iid")
        d = paths.profile_dir()
        stale_files = [
            d / "rag_store_default_20260101.faiss",
            d / "chunk_cache_default_20260101.db",
            d / "rag_session_id_dead.txt",
            d / "chunk_session_id_dead.txt",
        ]
        for f in stale_files:
            f.write_text("x")
        stale_store_dir = d / "rag_store_default_20260101"  # chromadb dir form
        stale_store_dir.mkdir()
        (stale_store_dir / "data").write_text("x")
        # A file belonging to a concurrently-running instance (fresh mtime).
        fresh = d / "rag_store_default_20260720.faiss"
        fresh.write_text("x")

        old = time.time() - 30 * 86400
        for f in stale_files + [stale_store_dir]:
            os.utime(f, (old, old))

        removed = paths.sweep_old_rag_artifacts(max_age_days=7)

        assert removed == 5  # 4 files + 1 dir
        for f in stale_files:
            assert not f.exists()
        assert not stale_store_dir.exists()
        assert fresh.exists()  # a live instance's fresh file is untouched

    def test_sweep_old_rag_artifacts_ignores_unrelated(self, tmp_home):
        import time

        d = paths.profile_dir()
        keep = d / "MEMORY.md"
        keep.write_text("facts")
        os.utime(keep, (time.time() - 30 * 86400, time.time() - 30 * 86400))
        paths.sweep_old_rag_artifacts(max_age_days=7)
        assert keep.exists()  # only rag_store_/chunk_cache_/*_session_id_ touched

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
