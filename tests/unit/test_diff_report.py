"""Unit tests for the `/diff` report (uncommitted changes, session edits marked).

Most of these are pure — numstat parsing, rename unwrapping, rendering, unified-diff
colorization — so they need no repository. The handful that do drive a real `git`
build a throwaway repo in ``tmp_path`` and skip when git isn't installed.
"""

import os
import re
import shutil
import subprocess

import pytest

from mnemoai.client import diff_report
from mnemoai.client.diff_report import Change
from mnemoai.client.file_ledger import WRITTEN, FileLedger

_ANSI = re.compile(r"\033\[[0-9;]*m")


def plain(text: str) -> str:
    return _ANSI.sub("", text)


def client_with(*written) -> object:
    """A stand-in client whose agent ledger already wrote ``written``."""
    ledger = FileLedger()
    for path in written:
        ledger.record(path, WRITTEN)

    class _Agent:
        files = ledger

    class _Client:
        agent = _Agent()

    return _Client()


@pytest.fixture
def repo(tmp_path):
    """A tiny git repo with one committed file, one edit, one untracked file."""
    if not shutil.which("git"):
        pytest.skip("git not installed")
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "T")
    git("config", "commit.gpgsign", "false")
    (root / "tracked.txt").write_text("one\ntwo\n")
    git("add", "-A")
    git("commit", "-qm", "first")
    (root / "tracked.txt").write_text("one\ntwo changed\nthree\n")
    (root / "fresh.txt").write_text("a\nb\nc\n")
    return root


class TestParseNumstat:
    def test_added_and_deleted_counts(self):
        changes = diff_report.parse_numstat("14\t2\tsrc/a.py\n0\t9\tb.md\n")
        assert changes == [
            Change("src/a.py", 14, 2),
            Change("b.md", 0, 9),
        ]

    def test_binary_dashes_do_not_crash_and_are_flagged(self):
        (change,) = diff_report.parse_numstat("-\t-\tlogo.png\n")
        assert change.binary and change.added == 0 and change.deleted == 0

    def test_blank_and_short_lines_are_skipped(self):
        assert diff_report.parse_numstat("\n\nnot-numstat\n1\t2\n") == []

    def test_a_tab_in_the_path_is_kept(self):
        (change,) = diff_report.parse_numstat("1\t0\tweird\tname.py")
        assert change.path == "weird\tname.py"

    def test_empty_input(self):
        assert diff_report.parse_numstat("") == []
        assert diff_report.parse_numstat(None) == []


class TestRenamePaths:
    def test_plain_rename_takes_the_new_name(self):
        assert diff_report._numstat_path("old.py => new.py") == "new.py"

    def test_braced_rename_is_rebuilt(self):
        assert diff_report._numstat_path("src/{old => new}/x.py") == "src/new/x.py"

    def test_braced_rename_at_the_tail(self):
        assert diff_report._numstat_path("src/a/{x.py => y.py}") == "src/a/y.py"

    def test_a_path_without_an_arrow_is_untouched(self):
        assert diff_report._numstat_path("src/a.py") == "src/a.py"


class TestRender:
    def test_clean_tree_says_so(self):
        out = plain(diff_report.render("/repo", "main", []))
        assert "Nothing uncommitted" in out
        assert "(main)" in out

    def test_counts_and_totals(self):
        out = plain(
            diff_report.render(
                "/repo", "dev", [Change("a.py", 10, 3), Change("b.py", 1, 0)]
            )
        )
        assert "+10 -3" in out
        assert "2 files · +11 -3" in out

    def test_singular_file_count(self):
        out = plain(diff_report.render("/repo", "dev", [Change("a.py", 1, 0)]))
        assert "1 file · " in out

    def test_this_sessions_files_are_marked_and_come_first(self):
        out = plain(
            diff_report.render(
                "/repo",
                "dev",
                [Change("mine.py", 1, 0, mine=True), Change("theirs.py", 99, 0)],
            )
        )
        rows = [line for line in out.splitlines() if ".py" in line]
        assert rows[0].strip().startswith("✎ mine.py")
        assert not rows[1].strip().startswith("✎")
        assert "1 written this session" in out

    def test_no_marker_legend_when_nothing_is_ours(self):
        out = plain(diff_report.render("/repo", "dev", [Change("a.py", 1, 0)]))
        assert "written this session" not in out

    def test_untracked_files_are_tagged_new(self):
        out = plain(
            diff_report.render("/repo", "dev", [Change("n.py", 4, 0, untracked=True)])
        )
        assert "new" in out

    def test_binary_file_reads_as_binary(self):
        out = plain(diff_report.render("/repo", "dev", [Change("x.png", 0, 0, binary=True)]))
        assert "binary" in out

    def test_a_zero_line_change_is_not_blank(self):
        out = plain(diff_report.render("/repo", "dev", [Change("mode.sh", 0, 0)]))
        assert "no change" in out

    def test_counts_column_is_aligned(self):
        out = plain(
            diff_report.render(
                "/repo", "dev", [Change("a.py", 1, 0, untracked=True), Change("b.py", 100, 20)]
            )
        )
        rows = [line for line in out.splitlines() if ".py" in line]
        assert rows[0].index("new") > rows[1].index("+100")

    def test_long_list_elides_with_the_command_that_shows_the_rest(self, monkeypatch):
        monkeypatch.setattr(diff_report, "_MAX_ROWS", 2)
        changes = [Change(f"f{i}.py", i + 1, 0) for i in range(5)]
        out = plain(diff_report.render("/repo", "dev", changes))
        assert "… +3 more (git status)" in out
        # The totals still count every file, not just the listed rows.
        assert "5 files" in out

    def test_home_relative_header(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        out = plain(diff_report.render(str(tmp_path / "code" / "app"), "", []))
        assert os.path.join("~", "code", "app") in out


class TestColorize:
    def test_additions_deletions_and_hunks_are_colored_differently(self):
        colored, dropped = diff_report.colorize(
            "diff --git a/x b/x\n@@ -1 +1 @@\n-old\n+new\n context\n"
        )
        assert dropped == 0
        assert "\033[32m+new" in colored
        assert "\033[31m-old" in colored
        assert "\033[38;5;111m@@" in colored

    def test_file_headers_are_not_read_as_add_delete(self):
        colored, _ = diff_report.colorize("--- a/x\n+++ b/x\n")
        assert "\033[32m+++" not in colored
        assert "\033[31m---" not in colored

    def test_the_text_survives_unchanged(self):
        colored, _ = diff_report.colorize("@@ -1 +1 @@\n-a\n+b\n")
        assert plain(colored).splitlines() == ["@@ -1 +1 @@", "-a", "+b"]

    def test_long_diff_is_capped_and_reports_what_was_dropped(self):
        colored, dropped = diff_report.colorize("\n".join(f"+{i}" for i in range(50)), limit=10)
        assert dropped == 40
        assert len(colored.splitlines()) == 10

    def test_empty_diff(self):
        assert diff_report.colorize("") == ("", 0)


class TestCollect:
    def test_tracked_edit_and_untracked_file_are_both_reported(self, repo, monkeypatch):
        monkeypatch.chdir(repo)
        root, branch, changes = diff_report.collect(client_with())
        assert os.path.realpath(root) == os.path.realpath(str(repo))
        assert branch  # some name; git's default varies by version
        by_path = {c.path: c for c in changes}
        assert by_path["tracked.txt"].added == 2 and by_path["tracked.txt"].deleted == 1
        assert by_path["fresh.txt"].untracked and by_path["fresh.txt"].added == 3

    def test_the_ledgers_writes_are_marked(self, repo, monkeypatch):
        monkeypatch.chdir(repo)
        _, _, changes = diff_report.collect(client_with(str(repo / "tracked.txt")))
        by_path = {c.path: c for c in changes}
        assert by_path["tracked.txt"].mine
        assert not by_path["fresh.txt"].mine

    def test_an_ignored_file_is_not_reported(self, repo, monkeypatch):
        (repo / ".gitignore").write_text("ignored.txt\n")
        (repo / "ignored.txt").write_text("noise\n")
        monkeypatch.chdir(repo)
        _, _, changes = diff_report.collect(client_with())
        assert "ignored.txt" not in {c.path: c for c in changes}

    def test_a_binary_untracked_file_is_flagged_not_counted(self, repo, monkeypatch):
        (repo / "blob.bin").write_bytes(b"\x00\x01\x02")
        monkeypatch.chdir(repo)
        _, _, changes = diff_report.collect(client_with())
        assert {c.path: c for c in changes}["blob.bin"].binary

    def test_an_oversized_untracked_file_is_not_read(self, repo, monkeypatch):
        monkeypatch.setattr(diff_report, "_MAX_UNTRACKED_BYTES", 4)
        monkeypatch.chdir(repo)
        _, _, changes = diff_report.collect(client_with())
        assert {c.path: c for c in changes}["fresh.txt"].binary  # "too big to open"

    def test_outside_a_repo_there_is_nothing_to_diff(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert diff_report.collect(client_with(), cwd=str(tmp_path)) is None
        out = plain(diff_report.report(client_with()))
        assert "not inside a git repository" in out
        assert "/files" in out  # points at the report that still works

    def test_a_missing_ledger_is_tolerated(self, repo, monkeypatch):
        monkeypatch.chdir(repo)

        class _Client:
            agent = None

        _, _, changes = diff_report.collect(_Client())
        assert changes and not any(c.mine for c in changes)


class TestFileReport:
    def test_one_files_diff_is_shown(self, repo, monkeypatch):
        monkeypatch.chdir(repo)
        out = plain(diff_report.report(client_with(), "tracked.txt"))
        assert "+two changed" in out
        assert "-two" in out

    def test_a_repo_relative_path_works_from_a_subdirectory(self, repo, monkeypatch):
        (repo / "sub").mkdir()
        monkeypatch.chdir(repo / "sub")
        out = plain(diff_report.report(client_with(), "tracked.txt"))
        assert "two changed" in out

    def test_an_untracked_file_reads_as_all_additions(self, repo, monkeypatch):
        monkeypatch.chdir(repo)
        out = plain(diff_report.report(client_with(), "fresh.txt"))
        assert "new file · 3 lines" in out
        assert "+a" in out and "+c" in out

    def test_an_unchanged_file_says_so(self, repo, monkeypatch):
        (repo / "same.txt").write_text("x\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-qm", "second"], cwd=repo, check=True, capture_output=True
        )
        monkeypatch.chdir(repo)
        assert "No uncommitted changes" in plain(diff_report.report(client_with(), "same.txt"))

    def test_a_missing_path_is_reported_not_raised(self, repo, monkeypatch):
        monkeypatch.chdir(repo)
        assert "No such file" in diff_report.report(client_with(), "nope.txt")

    def test_a_directory_is_refused_with_the_reason(self, repo, monkeypatch):
        (repo / "sub").mkdir()
        monkeypatch.chdir(repo)
        assert "is a directory" in diff_report.report(client_with(), "sub")

    def test_a_file_outside_the_repo_is_refused(self, repo, tmp_path, monkeypatch):
        outside = tmp_path / "elsewhere.txt"
        outside.write_text("x\n")
        monkeypatch.chdir(repo)
        assert "is outside" in plain(diff_report.report(client_with(), str(outside)))

    def test_our_own_edit_is_marked_in_the_header(self, repo, monkeypatch):
        monkeypatch.chdir(repo)
        out = plain(diff_report.report(client_with(str(repo / "tracked.txt")), "tracked.txt"))
        assert "written this session" in out.splitlines()[0]

    def test_a_long_diff_names_the_command_that_shows_the_rest(self, repo, monkeypatch):
        monkeypatch.setattr(diff_report, "_MAX_DIFF_LINES", 5)
        (repo / "tracked.txt").write_text("\n".join(str(i) for i in range(80)))
        monkeypatch.chdir(repo)
        out = plain(diff_report.report(client_with(), "tracked.txt"))
        assert "git diff HEAD -- tracked.txt" in out


class TestReportIsSafe:
    def test_a_failure_becomes_a_message_not_a_traceback(self, monkeypatch):
        monkeypatch.setattr(diff_report, "repo_root", lambda cwd: 1 / 0)
        out = diff_report.report(client_with())
        assert "Could not read the working tree" in out

    def test_git_that_fails_is_treated_as_absent(self, tmp_path):
        assert diff_report._git(["not-a-git-command"], str(tmp_path)) is None

    def test_no_git_invocation_can_write(self):
        # A guard, not a formality: /diff must stay a report. Every argv this module
        # passes to git is inspected here, so a mutating verb can't slip in later.
        source = open(diff_report.__file__).read()
        verbs = re.findall(r'_git\(\[([^\]]*)\]', source)
        for verb in verbs:
            first = verb.split(",")[0].strip().strip("\"'")
            assert first in ("rev-parse", "diff", "ls-files", "not-a-git-command"), first


class TestWiring:
    def test_diff_is_a_builtin_command(self):
        from mnemoai.client.ui.chat_interface import ChatInterface
        from mnemoai.client.user_commands import BUILTIN_COMMANDS

        assert "diff" in BUILTIN_COMMANDS
        assert any(cmd == "/diff" for cmd, _ in ChatInterface._COMMANDS)
        assert any(
            cmd.startswith("/diff")
            for _, items in ChatInterface._COMMAND_GROUPS
            for cmd, _ in items
        )
