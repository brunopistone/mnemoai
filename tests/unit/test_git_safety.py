"""Unit tests for git safety guardrails (server/tools/git_safety.py).

These test the pure command-classification + argv-building logic (no real git
calls).
"""

from mnemoai.server.tools.git_safety import (
    BLOCKED_COMMANDS,
    DANGEROUS_PATTERNS,
    PROTECTED_BRANCHES,
    build_git_argv,
    build_scan_string,
    check_dangerous_command,
)


class TestScanStringMatchesWhatGitSees:
    """The danger patterns must run on the argv git will actually receive.

    Scanning the raw string let quoting hide a flag from the check while `shlex`
    still handed the real flag to git, and made a commit message that merely
    NAMED a dangerous operation trip that operation's pattern.
    """

    def test_quoting_cannot_hide_a_flag(self):
        assert build_scan_string('push origin main --for"ce"') == (
            "push origin main --force"
        )
        assert check_dangerous_command('push origin main --for"ce"')["blocked"] is True

    def test_fully_quoted_flag_is_seen(self):
        assert check_dangerous_command('reset "--hard" HEAD~1')["dangerous"] is True
        assert check_dangerous_command('branch "-D" old')["dangerous"] is True

    def test_commit_message_is_not_scanned(self):
        for command in (
            "commit -m 'reset --hard fix'",
            'commit -m "revert the push --force change"',
            "commit -m'--no-verify in the message'",
            "commit --message='clean -fd cleanup'",
        ):
            result = check_dangerous_command(command)
            assert result["dangerous"] is False, command
            assert result["blocked"] is False, command

    def test_real_flags_alongside_a_message_still_caught(self):
        assert check_dangerous_command("commit --amend -m 'x'")["dangerous"] is True
        assert check_dangerous_command("commit --no-verify -m 'x'")["dangerous"] is True

    def test_unparseable_quoting_falls_back_to_raw_string(self):
        # build_git_argv refuses this separately; the scan must not crash.
        assert build_scan_string("commit -m 'unterminated") == "commit -m 'unterminated"


class TestBlockedCommands:
    def test_force_push_to_main_is_blocked(self):
        result = check_dangerous_command("push -f origin main")
        assert result["blocked"] is True

    def test_force_push_to_master_is_blocked(self):
        result = check_dangerous_command("push --force origin master")
        assert result["blocked"] is True

    def test_force_push_main_flag_after_branch_is_blocked(self):
        result = check_dangerous_command("push origin main --force")
        assert result["blocked"] is True


class TestDangerousCommands:
    def test_hard_reset_is_dangerous_not_blocked(self):
        result = check_dangerous_command("reset --hard HEAD~1")
        assert result["blocked"] is False
        assert result["dangerous"] is True
        assert any(w["type"] == "hard_reset" for w in result["warnings"])

    def test_force_push_to_feature_branch_is_dangerous_not_blocked(self):
        result = check_dangerous_command("push --force origin my-feature")
        assert result["blocked"] is False
        assert result["dangerous"] is True
        assert any(w["type"] == "force_push" for w in result["warnings"])

    def test_force_delete_branch_is_dangerous(self):
        result = check_dangerous_command("branch -D old-feature")
        assert result["dangerous"] is True
        assert any(w["type"] == "force_delete_branch" for w in result["warnings"])

    def test_safe_lowercase_delete_branch_is_not_flagged(self):
        # -d only deletes already-merged branches and must stay unflagged,
        # even though -D (force delete) is dangerous.
        result = check_dangerous_command("branch -d merged-feature")
        force_del = [
            w
            for w in result.get("warnings", [])
            if w["type"] == "force_delete_branch"
        ]
        assert force_del == []

    def test_clean_force_is_dangerous(self):
        result = check_dangerous_command("clean -fd")
        assert result["dangerous"] is True

    def test_no_verify_commit_is_dangerous(self):
        result = check_dangerous_command("commit --no-verify -m 'x'")
        assert result["dangerous"] is True
        assert any(w["type"] == "skip_hooks" for w in result["warnings"])

    def test_amend_is_dangerous(self):
        result = check_dangerous_command("commit --amend -m 'x'")
        assert result["dangerous"] is True
        assert any(w["type"] == "amend" for w in result["warnings"])

    def test_force_with_lease_is_not_flagged_as_force_push(self):
        # --force-with-lease is the safe variant and must NOT trip force_push.
        result = check_dangerous_command("push --force-with-lease origin feature")
        force_warnings = [
            w for w in result.get("warnings", []) if w["type"] == "force_push"
        ]
        assert force_warnings == []


class TestSafeCommands:
    def test_status_is_safe(self):
        result = check_dangerous_command("status")
        assert result["blocked"] is False
        assert result["dangerous"] is False

    def test_plain_commit_is_safe(self):
        result = check_dangerous_command("commit -m 'normal commit'")
        assert result["dangerous"] is False

    def test_push_to_feature_branch_is_safe(self):
        result = check_dangerous_command("push origin my-feature")
        assert result["dangerous"] is False

    def test_case_insensitive_detection(self):
        result = check_dangerous_command("RESET --HARD HEAD~1")
        assert result["dangerous"] is True


class TestConstants:
    def test_protected_branches_include_main_and_master(self):
        assert "main" in PROTECTED_BRANCHES
        assert "master" in PROTECTED_BRANCHES

    def test_pattern_tables_nonempty(self):
        assert len(DANGEROUS_PATTERNS) > 0
        assert len(BLOCKED_COMMANDS) > 0


class TestBuildGitArgv:
    """argv construction: quoting must survive, option injection must not."""

    def test_quoted_commit_message_stays_one_argument(self):
        # The tool's own documented example. With str.split() this became
        # ["commit", "-m", "'Add", "feature'"] and committed the message "'Add".
        argv, refusal = build_git_argv("commit -m 'Add feature'")
        assert refusal == ""
        assert argv == ["git", "commit", "-m", "Add feature"]

    def test_double_quoted_message_stays_one_argument(self):
        argv, refusal = build_git_argv('commit -m "fix: two words"')
        assert refusal == ""
        assert argv == ["git", "commit", "-m", "fix: two words"]

    def test_multiline_message_preserved(self):
        argv, _ = build_git_argv("commit -m 'subject\n\nbody line'")
        assert argv[-1] == "subject\n\nbody line"

    def test_plain_command_unchanged(self):
        argv, refusal = build_git_argv("status --short")
        assert refusal == ""
        assert argv == ["git", "status", "--short"]

    def test_path_with_spaces_in_quotes(self):
        argv, _ = build_git_argv("add 'docs/my notes.md'")
        assert argv == ["git", "add", "docs/my notes.md"]

    def test_unbalanced_quote_is_refused(self):
        argv, refusal = build_git_argv("commit -m 'unterminated")
        assert argv == []
        assert "quoting" in refusal.lower()

    def test_empty_command_is_refused(self):
        argv, refusal = build_git_argv("   ")
        assert argv == []
        assert refusal

    def test_config_option_is_refused(self):
        # -c core.pager / -c alias.x='!sh -c ...' is arbitrary code execution
        # that no DANGEROUS_PATTERNS entry can see.
        argv, refusal = build_git_argv("-c core.pager=sh status")
        assert argv == []
        assert "-c" in refusal

    def test_alias_config_injection_is_refused(self):
        argv, refusal = build_git_argv("-c alias.x=!touch /tmp/pwned x")
        assert argv == []
        assert refusal

    def test_exec_path_is_refused(self):
        argv, refusal = build_git_argv("--exec-path=/tmp/evil status")
        assert argv == []
        assert "--exec-path" in refusal

    def test_repo_retargeting_options_are_refused(self):
        for command in (
            "-C /etc status",
            "--git-dir=/tmp/other/.git log",
            "--work-tree=/ checkout .",
        ):
            argv, refusal = build_git_argv(command)
            assert argv == [], command
            assert refusal, command

    def test_upload_pack_is_refused_anywhere(self):
        argv, refusal = build_git_argv(
            "fetch --upload-pack='touch /tmp/pwned' origin main"
        )
        assert argv == []
        assert "--upload-pack" in refusal

    def test_unknown_leading_option_is_refused(self):
        argv, refusal = build_git_argv("--bogus status")
        assert argv == []
        assert "--bogus" in refusal

    def test_subcommand_options_still_allowed(self):
        # Options AFTER the subcommand are the normal case and must pass.
        argv, refusal = build_git_argv("log --oneline -n 5 --no-merges")
        assert refusal == ""
        assert argv == ["git", "log", "--oneline", "-n", "5", "--no-merges"]
