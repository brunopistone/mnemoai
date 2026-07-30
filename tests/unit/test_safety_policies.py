"""Unit tests for the server-side safety policies (server/tools/safety/).

These test the pure classifier logic directly — bypassing the agent/client
layer — because that's exactly the layer the policies are meant to protect
(an MCP server can be driven without the client's confirmation gate).

No config, LLM, or filesystem needed.
"""

from mnemoai.server.tools.safety import (
    classify_shell_command,
    classify_shell_write_targets,
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


class TestRmRfRootAnchor:
    """The root-wipe rule must not be defeated by what FOLLOWS the target.

    `rm -rf /` alone is refused by GNU rm, so the form that actually wipes the
    filesystem is `rm -rf / --no-preserve-root` — which the original rule missed
    because it anchored on end-of-command immediately after the target.
    """

    def test_no_preserve_root_after_target(self):
        r = classify_shell_command("rm -rf / --no-preserve-root")
        assert r.blocked is True and r.rule == "rm_rf_root"

    def test_no_preserve_root_before_flags(self):
        assert classify_shell_command("rm --no-preserve-root -rf /").blocked is True

    def test_redirection_after_target(self):
        assert classify_shell_command("rm -rf / 2>/dev/null").blocked is True

    def test_quoted_target(self):
        assert classify_shell_command('rm -rf "/"').blocked is True
        assert classify_shell_command("rm -rf '/'").blocked is True

    def test_scoped_delete_still_allowed_with_trailing_flags(self):
        # The anchor still does its job: a real subdirectory is not a root wipe.
        assert classify_shell_command("rm -rf build/ --verbose").blocked is False
        assert classify_shell_command("rm -rf /tmp/x 2>/dev/null").blocked is False


class TestShellWriteTargets:
    """A shell write must obey the same path policy as fs_write/file_edit.

    `execute_bash` never consulted classify_write_path, so a redirection into a
    system directory was simply not a "write tool" call.
    """

    def test_redirect_into_etc(self):
        r = classify_shell_command("echo x > /etc/hosts")
        assert r.blocked is True and r.rule == "system_write"

    def test_append_into_etc(self):
        assert classify_shell_command("echo x >> /etc/sudoers").blocked is True

    def test_nested_shell_c_payload(self):
        assert classify_shell_command("sh -c 'echo x > /etc/hosts'").blocked is True

    def test_tee_through_a_pipe(self):
        assert classify_shell_command("echo x | sudo tee /etc/hosts").blocked is True

    def test_copy_destination(self):
        assert classify_shell_command("cp evil /usr/local/bin/ls").blocked is True

    def test_sed_in_place(self):
        assert classify_shell_command("sed -i 's/a/b/' /etc/passwd").blocked is True

    def test_second_segment_of_a_chain(self):
        assert classify_shell_command("ls; echo y > /private/etc/hosts").blocked is True

    def test_reason_names_the_target(self):
        assert "/etc/hosts" in classify_shell_command("echo x > /etc/hosts").reason

    def test_dev_null_is_exempt(self):
        # /dev is a protected prefix, but the standard sinks must stay writable
        # or nearly every real command would be blocked.
        assert classify_shell_command("ls -la 2>/dev/null").blocked is False
        assert classify_shell_command("make 1>/dev/null 2>&1").blocked is False

    def test_fd_duplication_is_not_a_path(self):
        assert classify_shell_command("echo oops 1>&2").blocked is False

    def test_quoting_is_respected(self):
        # A redirection inside a quoted string is text, not a redirection.
        assert classify_shell_command("echo 'x > /etc/hosts'").blocked is False
        assert classify_shell_command("git commit -m 'fix > /etc/x typo'").blocked is False

    def test_ordinary_writes_allowed(self):
        for cmd in (
            "echo hi > /tmp/out.txt",
            "pytest -q > out.log 2>&1",
            "cp a.py b.py",
            "mkdir -p ~/scratch/x",
            "npm run build 2>&1 | tee build.log",
            "make install",
        ):
            assert classify_shell_command(cmd).blocked is False, cmd

    def test_reads_are_not_writes(self):
        for cmd in ("cat /etc/hosts", "diff /etc/hosts /tmp/h", "wc -l < /etc/hosts"):
            assert classify_shell_write_targets(cmd).blocked is False, cmd

    def test_unbalanced_quoting_still_scanned(self):
        # Can't tokenize -> fall back to the raw-string redirection scan rather
        # than skipping the check entirely.
        assert classify_shell_write_targets('echo "oops > /etc/hosts').blocked is True


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
