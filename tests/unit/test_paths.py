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

    def test_plans_and_tasks_created(self, tmp_home):
        assert paths.plans_dir() == tmp_home / "plans"
        assert paths.tasks_dir() == tmp_home / "tasks"
        assert (tmp_home / "plans").is_dir()
        assert (tmp_home / "tasks").is_dir()

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
