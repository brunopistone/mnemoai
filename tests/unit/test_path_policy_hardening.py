"""Path-policy hardening: symlink resolution on the write policy.

``_normalize`` follows symlinks, so a link into a protected directory can't
launder a system write into a "home" write. This was the actual gap: the policy
used ``abspath``, which collapses ``..`` but does NOT follow links. The write
policy is also applied to ``start_background_task`` working directories.

There is deliberately NO read-side denylist: a hard block the user cannot
override is the wrong shape for a local single-user tool (see CHANGELOG 1.8.0).
"""

import os

import pytest

from mnemoai.server.tools.safety import classify_write_path


class TestSymlinkResolution:
    def test_symlink_into_etc_is_blocked(self, tmp_path):
        """The bypass that motivated realpath: a home symlink pointing at /etc."""
        link = tmp_path / "innocent"
        try:
            os.symlink("/etc", link)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported here")

        verdict = classify_write_path(str(link / "passwd"))
        assert verdict.blocked
        assert "/etc" in verdict.reason

    def test_symlinked_parent_dir_is_resolved(self, tmp_path):
        """A link one level up still resolves through to the real target."""
        link = tmp_path / "sys"
        try:
            os.symlink("/usr", link)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported here")

        assert classify_write_path(str(link / "local" / "bin" / "x")).blocked

    def test_dotdot_traversal_still_blocked(self):
        """The pre-existing .. collapsing must not regress."""
        assert classify_write_path("/usr/../etc/passwd").blocked

    def test_ordinary_paths_still_allowed(self, tmp_path):
        for target in (str(tmp_path / "notes.md"), "~/projects/app/main.py"):
            assert not classify_write_path(target).blocked

    def test_nonexistent_file_in_real_dir_allowed(self, tmp_path):
        """A not-yet-created file resolves through its real parent."""
        assert not classify_write_path(str(tmp_path / "brand_new.txt")).blocked


class TestBackgroundTaskCwd:
    def test_system_cwd_is_blocked(self):
        assert classify_write_path("/etc").blocked

    def test_project_cwd_is_allowed(self, tmp_path):
        assert not classify_write_path(str(tmp_path)).blocked
