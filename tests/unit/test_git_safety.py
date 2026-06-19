"""Unit tests for git safety guardrails (server/tools/git_safety.py).

These test the pure command-classification logic (no real git calls).
"""

from personal_ai_assistant.server.tools.git_safety import (
    check_dangerous_command,
    PROTECTED_BRANCHES,
    DANGEROUS_PATTERNS,
    BLOCKED_COMMANDS,
)


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
