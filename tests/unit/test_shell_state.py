"""Unit tests for the shell session state (server/tools/shell_state.py).

Which shell the command tools run (bash, not whatever ``/bin/sh`` is on the host)
and where it runs (the directory the previous command ended in). Pure state +
string logic — no MCP, no subprocess needed for most of it.
"""

import os

import pytest

from mnemoai.server.tools import shell_state


@pytest.fixture(autouse=True)
def _clean_tracked_cwd():
    """The tracked directory is process-wide; don't leak it between tests."""
    shell_state.reset_cwd()
    yield
    shell_state.reset_cwd()


class TestBashPath:
    def test_resolves_to_a_bash_executable_or_none(self):
        path = shell_state.bash_path()
        # None is legal (no bash on the host → default shell); anything else must
        # actually be bash and actually exist.
        assert path is None or (os.path.exists(path) and "bash" in os.path.basename(path))

    def test_resolution_is_cached(self):
        assert shell_state.bash_path() == shell_state.bash_path()


class TestTrackedCwd:
    def test_defaults_to_the_process_cwd(self):
        assert shell_state.current_cwd() == os.getcwd()

    def test_set_then_read_round_trips(self, tmp_path):
        assert shell_state.set_cwd(str(tmp_path)) is True
        assert shell_state.current_cwd() == str(tmp_path)

    def test_a_nonexistent_directory_is_rejected(self, tmp_path):
        assert shell_state.set_cwd(str(tmp_path / "nope")) is False
        assert shell_state.current_cwd() == os.getcwd()

    def test_empty_and_none_are_rejected(self):
        assert shell_state.set_cwd("") is False
        assert shell_state.set_cwd(None) is False

    def test_a_file_is_not_a_directory(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("hi")
        assert shell_state.set_cwd(str(f)) is False

    def test_a_deleted_tracked_directory_falls_back_to_the_process_cwd(self, tmp_path):
        gone = tmp_path / "temp"
        gone.mkdir()
        assert shell_state.set_cwd(str(gone)) is True
        gone.rmdir()
        # Otherwise every later command would fail to even start.
        assert shell_state.current_cwd() == os.getcwd()


class TestCwdProbe:
    def test_probe_preserves_the_exit_status(self):
        wrapped = shell_state.wrap_with_cwd_probe("false", "/tmp/p.txt")
        assert "exit $" in wrapped

    def test_probe_path_is_quoted(self):
        wrapped = shell_state.wrap_with_cwd_probe("pwd", "/tmp/a b/p.txt")
        assert "'/tmp/a b/p.txt'" in wrapped

    def test_the_command_is_kept_verbatim_on_its_own_line(self):
        wrapped = shell_state.wrap_with_cwd_probe("echo hi  # trailing comment", "/tmp/p")
        # A trailing comment must not swallow the probe lines.
        assert wrapped.startswith("echo hi  # trailing comment\n")

    def test_a_trailing_line_continuation_disables_the_probe(self):
        # Appending would splice the probe into the command's own last line.
        cmd = "echo a \\"
        assert shell_state.wrap_with_cwd_probe(cmd, "/tmp/p") == cmd

    def test_an_empty_command_is_left_alone(self):
        assert shell_state.wrap_with_cwd_probe("", "/tmp/p") == ""

    def test_reading_a_missing_probe_is_none_not_an_error(self, tmp_path):
        assert shell_state.read_cwd_probe(str(tmp_path / "absent")) is None

    def test_an_empty_probe_is_none(self, tmp_path):
        p = tmp_path / "probe"
        p.write_text("")
        # The shape when a command ends the shell itself (`exit`, `exec`).
        assert shell_state.read_cwd_probe(str(p)) is None

    def test_a_probe_naming_a_gone_directory_is_none(self, tmp_path):
        p = tmp_path / "probe"
        p.write_text(str(tmp_path / "vanished") + "\n")
        assert shell_state.read_cwd_probe(str(p)) is None

    def test_a_valid_probe_is_returned_stripped(self, tmp_path):
        p = tmp_path / "probe"
        p.write_text(f"{tmp_path}\n")
        assert shell_state.read_cwd_probe(str(p)) == str(tmp_path)

    def test_only_the_first_line_is_read(self, tmp_path):
        p = tmp_path / "probe"
        p.write_text(f"{tmp_path}\nnoise the command wrote itself\n")
        assert shell_state.read_cwd_probe(str(p)) == str(tmp_path)
