"""Unit tests for the server-side safety policies (server/tools/safety/).

These test the pure classifier logic directly — bypassing the agent/client
layer — because that's exactly the layer the policies are meant to protect
(an MCP server can be driven without the client's confirmation gate).

No config, LLM, or filesystem needed.
"""

from mnemoai.server.tools.safety import (
    classify_shell_command,
    classify_write_path,
)


class TestBashPolicyBlocks:
    """Catastrophic, irreversible commands must be blocked."""

    def test_rm_rf_root(self):
        assert classify_shell_command("rm -rf /").blocked is True

    def test_rm_rf_root_glob(self):
        assert classify_shell_command("rm -rf /*").blocked is True

    def test_rm_rf_root_with_sudo(self):
        assert classify_shell_command("sudo rm -rf /").blocked is True

    def test_rm_fr_home_tilde(self):
        assert classify_shell_command("rm -fr ~").blocked is True

    def test_rm_rf_home_tilde_trailing_slash(self):
        assert classify_shell_command("rm -rf ~/").blocked is True

    def test_rm_rf_home_var(self):
        assert classify_shell_command("rm -rf $HOME").blocked is True

    def test_rm_long_flags(self):
        assert classify_shell_command("rm --recursive --force /").blocked is True

    def test_rm_separate_flags(self):
        assert classify_shell_command("rm -r -f /").blocked is True

    def test_mkfs(self):
        r = classify_shell_command("mkfs.ext4 /dev/sda1")
        assert r.blocked is True and r.rule == "mkfs"

    def test_dd_to_device(self):
        r = classify_shell_command("dd if=/dev/zero of=/dev/sda bs=1M")
        assert r.blocked is True and r.rule == "dd_device"

    def test_shutdown(self):
        assert classify_shell_command("shutdown -h now").blocked is True

    def test_reboot(self):
        assert classify_shell_command("reboot").blocked is True

    def test_poweroff(self):
        assert classify_shell_command("poweroff").blocked is True

    def test_init_0(self):
        assert classify_shell_command("init 0").blocked is True

    def test_fork_bomb(self):
        assert classify_shell_command(":(){ :|:& };:").blocked is True

    def test_wipefs_device(self):
        assert classify_shell_command("wipefs -a /dev/sda").blocked is True

    def test_blocked_result_carries_reason_and_rule(self):
        r = classify_shell_command("rm -rf /")
        assert r.reason  # non-empty human message
        assert r.rule == "rm_rf_root"


class TestBashPolicyAllows:
    """Ordinary (even destructive-but-scoped) commands are allowed here.

    They remain gated by the client's confirmation prompt; this layer is only a
    floor against system-destroying actions, not a second confirmation gate.
    """

    def test_scoped_recursive_delete(self):
        assert classify_shell_command("rm -rf build/").blocked is False

    def test_delete_single_file(self):
        assert classify_shell_command("rm file.txt").blocked is False

    def test_delete_node_modules(self):
        assert classify_shell_command("rm -rf ./node_modules").blocked is False

    def test_rm_rf_under_tmp(self):
        assert classify_shell_command("rm -rf /tmp/mybuild").blocked is False

    def test_git_reset_hard(self):
        # Destructive to the worktree but not the system → not our concern here.
        assert classify_shell_command("git reset --hard").blocked is False

    def test_dd_to_regular_file(self):
        assert classify_shell_command("dd if=a of=backup.img").blocked is False

    def test_plain_read_commands(self):
        for cmd in ("ls -la", "cat /etc/hosts", "echo hi", "grep -rf x ."):
            assert classify_shell_command(cmd).blocked is False

    def test_empty_command(self):
        assert classify_shell_command("").blocked is False
        assert classify_shell_command("   ").blocked is False


class TestPathPolicyBlocks:
    """Writes into critical system directories must be refused."""

    def test_filesystem_root(self):
        assert classify_write_path("/").blocked is True

    def test_etc(self):
        assert classify_write_path("/etc/passwd").blocked is True

    def test_usr_bin(self):
        assert classify_write_path("/usr/bin/foo").blocked is True

    def test_bin(self):
        assert classify_write_path("/bin/ls").blocked is True

    def test_boot(self):
        assert classify_write_path("/boot/vmlinuz").blocked is True

    def test_macos_system(self):
        assert classify_write_path("/System/x").blocked is True

    def test_dev(self):
        assert classify_write_path("/dev/sda").blocked is True

    def test_macos_private_etc(self):
        assert classify_write_path("/private/etc/hosts").blocked is True

    def test_dotdot_escape_into_etc(self):
        # ../ segments are normalized before classification.
        assert classify_write_path("/usr/../etc/shadow").blocked is True

    def test_blocked_result_carries_reason(self):
        assert classify_write_path("/etc/passwd").reason


class TestPathPolicyAllows:
    """Home, project, temp and app paths are allowed here."""

    def test_home_file(self):
        assert classify_write_path("~/notes.md").blocked is False

    def test_tmp(self):
        assert classify_write_path("/tmp/out.txt").blocked is False

    def test_project_file(self):
        assert classify_write_path("/Users/me/proj/a.py").blocked is False

    def test_relative_path(self):
        assert classify_write_path("./rel.txt").blocked is False

    def test_user_library_is_allowed(self):
        # User's ~/Library is fine; only the system-wide /Library is blocked.
        assert classify_write_path("~/Library/App/state.json").blocked is False

    def test_empty_path(self):
        assert classify_write_path("").blocked is False

    def test_resolved_is_absolute(self):
        r = classify_write_path("~/notes.md")
        assert r.resolved.startswith("/")
