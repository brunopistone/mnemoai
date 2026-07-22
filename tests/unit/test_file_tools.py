"""Unit tests for the file tools (server/tools/): file_edit, fs_write path
resolution, and glob_search / grep_search.

Exercise the real filesystem logic via temp files/dirs — no LLM involved.
"""

import asyncio
import json
import os
import shutil

import pytest

from mnemoai.server.tools import read_state
from mnemoai.server.tools.file_edit import register_edit_tools
from mnemoai.server.tools.file_search import register_search_tools
from mnemoai.server.tools.fs_write import _resolve_path

# grep_search shells out to ripgrep; skip its tests where rg isn't installed so
# the unit tier stays environment-robust. CI installs rg (see tests.yml) so the
# behavior is still genuinely covered there, not silently skipped.
_needs_rg = pytest.mark.skipif(
    shutil.which("rg") is None, reason="ripgrep (rg) not installed"
)


@pytest.fixture(autouse=True)
def _reset_read_state():
    # The read-before-write gate keeps process-global state; isolate each test.
    read_state.reset()
    yield
    read_state.reset()


class _CapturingMCP:
    def __init__(self):
        self.registered = {}

    def tool(self):
        def decorator(func):
            self.registered[func.__name__] = func
            return func

        return decorator


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def file_edit():
    mcp = _CapturingMCP()
    register_edit_tools(mcp)
    return mcp.registered["file_edit"]


@pytest.fixture
def glob_search():
    mcp = _CapturingMCP()
    register_search_tools(mcp)
    return mcp.registered["glob_search"]


@pytest.fixture
def grep_search():
    mcp = _CapturingMCP()
    register_search_tools(mcp)
    return mcp.registered["grep_search"]


@pytest.fixture
def grep_tree(tmp_path):
    (tmp_path / "a.txt").write_text("foo\nfoo\nbar\n")
    (tmp_path / "b.txt").write_text("foo\nbaz\n")
    (tmp_path / "c.txt").write_text("nothing here\n")
    return tmp_path


class TestFileEdit:
    def test_simple_replacement(self, file_edit, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello world")
        read_state.record_read(str(f))  # simulate the required prior fs_read
        result = json.loads(run(file_edit(str(f), "world", "there")))
        assert result["success"] is True
        assert f.read_text() == "hello there"

    def test_missing_file_errors(self, file_edit, tmp_path):
        result = json.loads(run(file_edit(str(tmp_path / "nope.txt"), "a", "b")))
        assert result["error"] is True
        assert "not found" in result["message"].lower()

    def test_string_not_found_errors(self, file_edit, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("content")
        read_state.record_read(str(f))
        result = json.loads(run(file_edit(str(f), "absent", "x")))
        assert result["error"] is True
        assert "not found" in result["message"].lower()
        # File must be untouched.
        assert f.read_text() == "content"

    def test_non_unique_without_replace_all_errors(self, file_edit, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("x\nx\nx")
        read_state.record_read(str(f))
        result = json.loads(run(file_edit(str(f), "x", "y", replace_all=False)))
        assert result["error"] is True
        assert "occurrences" in result["message"].lower()
        # Nothing changed.
        assert f.read_text() == "x\nx\nx"

    def test_replace_all_replaces_every_occurrence(self, file_edit, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("x\nx\nx")
        read_state.record_read(str(f))
        result = json.loads(run(file_edit(str(f), "x", "y", replace_all=True)))
        assert result["success"] is True
        assert result["replacements"] == 3
        assert f.read_text() == "y\ny\ny"

    def test_directory_path_errors(self, file_edit, tmp_path):
        result = json.loads(run(file_edit(str(tmp_path), "a", "b")))
        assert result["error"] is True

    def test_lines_delta_reported(self, file_edit, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("line1")
        read_state.record_read(str(f))
        result = json.loads(run(file_edit(str(f), "line1", "line1\nline2\nline3")))
        assert result["success"] is True
        assert result["lines_delta"] == 2


class TestGlobSearch:
    def test_finds_files_by_extension(self, glob_search, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = json.loads(run(glob_search("*.py", path=str(tmp_path))))
        assert result["success"] is True
        assert result["count"] == 2
        assert all(m.endswith(".py") for m in result["matches"])

    def test_recursive_pattern(self, glob_search, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("")
        (tmp_path / "top.py").write_text("")
        result = json.loads(run(glob_search("**/*.py", path=str(tmp_path))))
        assert result["count"] == 2

    def test_nonexistent_path_errors(self, glob_search, tmp_path):
        result = json.loads(run(glob_search("*.py", path=str(tmp_path / "nope"))))
        assert result["error"] is True

    def test_max_results_truncation(self, glob_search, tmp_path):
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text("")
        result = json.loads(run(glob_search("*.py", path=str(tmp_path), max_results=2)))
        assert result["count"] == 2
        assert result.get("truncated") is True

    def test_no_matches_returns_empty(self, glob_search, tmp_path):
        result = json.loads(run(glob_search("*.rs", path=str(tmp_path))))
        assert result["success"] is True
        assert result["count"] == 0


class TestBinaryFileSteering:
    """A text reader on a binary/image file must fail fast with a message that
    steers the model to describe_image — not a raw UnicodeDecodeError.
    """

    def test_looks_like_binary_detects_image_and_text(self, tmp_path):
        from mnemoai.server.tools import looks_like_binary

        png = tmp_path / "x.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00stuff")
        txt = tmp_path / "x.txt"
        txt.write_text("hello\nworld\n")
        assert looks_like_binary(str(png)) is True
        assert looks_like_binary(str(txt)) is False

    def test_binary_file_error_steers_images_to_describe_image(self):
        from mnemoai.server.tools import binary_file_error

        err = binary_file_error("/tmp/x.png")
        assert err["error"] is True and err["is_image"] is True
        assert "describe_image" in err["message"]

    def test_read_lines_on_png_returns_steering_message(self, tmp_path):
        from mnemoai.server.tools.readers.line_reader import read_lines

        png = tmp_path / "img.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00binary\x00data")
        out = json.loads(run(read_lines(str(png), 1, 5)))
        assert out["error"] is True
        assert "describe_image" in out["message"]

    def test_read_lines_on_text_still_works(self, tmp_path, monkeypatch):
        import mnemoai.server.tools.readers.line_reader as lr
        from mnemoai.server.tools.readers.line_reader import read_lines

        # read_lines needs DOC_MAX_TOKENS; provide it (no config.yaml in tests).
        monkeypatch.setattr(
            lr.config,
            "get",
            lambda key, default=None: 16384 if key == "DOC_MAX_TOKENS" else default,
        )
        f = tmp_path / "a.txt"
        f.write_text("line1\nline2\nline3\n")
        out = json.loads(run(read_lines(str(f), 1, 2)))
        assert "content" in out and "line1" in out["content"]


class TestLineNumberGutters:
    """fs_read Line mode prefixes cat -n gutters; file_edit strips a pasted
    gutter so a numbered block still matches raw content."""

    def _patch_tokens(self, monkeypatch):
        import mnemoai.server.tools.readers.line_reader as lr

        monkeypatch.setattr(
            lr.config,
            "get",
            lambda key, default=None: 16384 if key == "DOC_MAX_TOKENS" else default,
        )

    def test_read_lines_prefixes_gutter(self, tmp_path, monkeypatch):
        import re

        from mnemoai.server.tools.readers.line_reader import read_lines

        self._patch_tokens(monkeypatch)
        f = tmp_path / "a.txt"
        f.write_text("line1\nline2\nline3\n")
        out = json.loads(run(read_lines(str(f), 1, -1)))
        lines = out["content"].split("\n")
        assert re.match(r"^\s*1\tline1$", lines[0])
        assert re.match(r"^\s*2\tline2$", lines[1])
        assert "line1" in out["content"]  # substring assertions still hold

    def test_gutter_reflects_start_line(self, tmp_path, monkeypatch):
        import re

        from mnemoai.server.tools.readers.line_reader import read_lines

        self._patch_tokens(monkeypatch)
        f = tmp_path / "a.txt"
        f.write_text("l1\nl2\nl3\nl4\n")
        out = json.loads(run(read_lines(str(f), 2, 3)))
        first = out["content"].split("\n")[0]
        assert re.match(r"^\s*2\t", first)  # numbering starts at start_line

    def test_file_edit_matches_pasted_numbered_block(self, file_edit, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello world")
        read_state.record_read(str(f))
        # old_string carries a gutter as the model would copy it from fs_read.
        result = json.loads(run(file_edit(str(f), "     1\thello world", "hi there")))
        assert result["success"] is True
        assert f.read_text() == "hi there"  # gutter stripped, raw content written

    def test_file_edit_new_string_written_verbatim(self, file_edit, tmp_path):
        # old_string carries display gutters (relocated by stripping), but a RAW
        # new_string is written verbatim — never de-guttered — so a legitimate
        # leading "digits<TAB>" (TSV key) in the replacement is NOT deleted.
        f = tmp_path / "data.tsv"
        f.write_text("id\tname\n100\tApple\n200\tBanana\n")
        read_state.record_read(str(f))
        result = json.loads(
            run(file_edit(str(f), "     2\t100\tApple", "100\tCherry"))
        )
        assert result["success"] is True
        # The '100\t' key column in the replacement must survive.
        assert f.read_text() == "id\tname\n100\tCherry\n200\tBanana\n"

    def test_raw_old_string_with_digit_tab_not_falsely_stripped(self, file_edit, tmp_path):
        # A file whose RAW content legitimately begins with "digits<TAB>": the
        # exact match succeeds first, so new_string is written verbatim (never
        # gutter-stripped).
        f = tmp_path / "data.tsv"
        f.write_text("42\tvalue\n99\tother\n")
        read_state.record_read(str(f))
        result = json.loads(run(file_edit(str(f), "42\tvalue", "42\tCHANGED")))
        assert result["success"] is True
        assert "42\tCHANGED" in f.read_text()  # digit+TAB new content preserved


@_needs_rg
class TestGrepModes:
    """grep_search: true TOTAL result cap across all modes, and that
    files_with_matches / count return results at all (they used to be silently
    empty because --files-with-matches/--count override --json)."""

    def test_files_with_matches_not_empty(self, grep_search, grep_tree):
        # Regression: this default mode returned [] because --files-with-matches
        # overrides --json and the parser expected JSON match events.
        r = json.loads(
            run(grep_search("foo", path=str(grep_tree), output_mode="files_with_matches"))
        )
        assert r["success"] is True
        assert r["count"] == 2
        assert sorted(os.path.basename(f) for f in r["files"]) == ["a.txt", "b.txt"]

    def test_count_mode_not_empty(self, grep_search, grep_tree):
        r = json.loads(
            run(grep_search("foo", path=str(grep_tree), output_mode="count"))
        )
        assert r["success"] is True
        assert r["total_matches"] == 3
        assert r["files_with_matches"] == 2
        assert set(os.path.basename(f) for f in r["counts"]) == {"a.txt", "b.txt"}

    def test_content_mode_returns_matches(self, grep_search, grep_tree):
        r = json.loads(
            run(grep_search("foo", path=str(grep_tree), output_mode="content", max_results=0))
        )
        assert r["count"] == 3

    def test_content_total_cap_across_files(self, grep_search, grep_tree):
        # max_results is a TOTAL cap, not per-file: 2 caps the 3 foo matches.
        r = json.loads(
            run(grep_search("foo", path=str(grep_tree), output_mode="content", max_results=2))
        )
        assert r["count"] == 2
        assert r.get("truncated") is True

    def test_files_total_cap(self, grep_search, grep_tree):
        r = json.loads(
            run(
                grep_search(
                    "foo", path=str(grep_tree), output_mode="files_with_matches", max_results=1
                )
            )
        )
        assert r["count"] == 1
        assert r.get("truncated") is True

    def test_unlimited_when_zero(self, grep_search, grep_tree):
        r = json.loads(
            run(grep_search("foo", path=str(grep_tree), output_mode="content", max_results=0))
        )
        assert r["count"] == 3
        assert "truncated" not in r

    def test_no_matches(self, grep_search, grep_tree):
        r = json.loads(
            run(grep_search("zzz", path=str(grep_tree), output_mode="files_with_matches"))
        )
        assert r["success"] is True
        assert r["count"] == 0

    def test_leading_dash_pattern_does_not_crash(self, grep_search, grep_tree):
        # A pattern beginning with '-' must be treated as a pattern (via -e),
        # not parsed as a ripgrep flag.
        r = json.loads(
            run(grep_search("-bar", path=str(grep_tree), output_mode="content"))
        )
        assert "error" not in r  # no flag-parse failure

    def test_bad_output_mode_errors(self, grep_search, grep_tree):
        r = json.loads(
            run(grep_search("foo", path=str(grep_tree), output_mode="bogus"))
        )
        assert r["error"] is True

    def test_non_utf8_matching_line_does_not_abort_search(self, grep_search, tmp_path):
        # ripgrep emits {lines:{bytes:...}} (no "text") for a match on a line it
        # can't decode; that must NOT KeyError and kill the whole search.
        (tmp_path / "plain.txt").write_text("match here\n")
        (tmp_path / "bin.txt").write_bytes(b"match \xff\xfe end\n")
        for mode in ("files_with_matches", "content", "count"):
            r = json.loads(
                run(grep_search("match", path=str(tmp_path), output_mode=mode))
            )
            assert "error" not in r, f"{mode} aborted on a non-UTF-8 match: {r}"
        # Both files are found in files mode (bin.txt's line is decoded lossily).
        r = json.loads(
            run(grep_search("match", path=str(tmp_path), output_mode="files_with_matches"))
        )
        assert r["count"] == 2

    def test_invalid_regex_reports_error_not_zero_matches(self, grep_search, grep_tree):
        # A malformed regex (rg exit 2) must read as an error, not "0 matches".
        r = json.loads(
            run(grep_search("[", path=str(grep_tree), output_mode="content"))
        )
        assert r.get("error") is True
        assert "failed" in r["message"].lower()

    def test_unreadable_file_does_not_discard_valid_matches(self, grep_search, tmp_path):
        # rg exits 2 when a searched file is unreadable, but still streams the
        # matches from readable files. Those must NOT be discarded; the error is
        # surfaced non-fatally in a `warnings` field.
        (tmp_path / "readable.txt").write_text("hello world\n")
        noperm = tmp_path / "noperm.txt"
        noperm.write_text("hello secret\n")
        os.chmod(str(noperm), 0o000)
        try:
            r = json.loads(
                run(grep_search("hello", path=str(tmp_path),
                                output_mode="files_with_matches"))
            )
            assert "error" not in r
            assert r["count"] >= 1  # readable.txt kept
            assert any(os.path.basename(f) == "readable.txt" for f in r["files"])
        finally:
            os.chmod(str(noperm), 0o644)


@_needs_rg
class TestGrepContextAndPaging:
    """grep_search: asymmetric -A/-B/-C context (actually returned), offset
    paging (counts matches only), and the long-line width cap."""

    def test_no_context_shape_unchanged(self, grep_search, grep_tree):
        # Default (no context): every row is a match, count == len(matches).
        r = json.loads(
            run(grep_search("foo", path=str(grep_tree), output_mode="content", max_results=0))
        )
        assert r["count"] == 3
        assert all(not m["is_context"] for m in r["matches"])

    def test_content_includes_context_lines(self, grep_search, grep_tree):
        # a.txt is "foo\nfoo\nbar"; context around 'bar' pulls the preceding foo.
        r = json.loads(
            run(
                grep_search(
                    "bar", path=str(grep_tree), output_mode="content",
                    context_before=1, context_after=1, max_results=0,
                )
            )
        )
        ctx = [m for m in r["matches"] if m["is_context"]]
        matches = [m for m in r["matches"] if not m["is_context"]]
        assert len(matches) == 1 and matches[0]["text"] == "bar"
        assert any(m["text"] == "foo" for m in ctx)  # neighbouring line shown
        assert r["count"] == 1  # count is matches only

    def test_symmetric_context_shorthand(self, grep_search, grep_tree):
        sym = json.loads(
            run(grep_search("bar", path=str(grep_tree), output_mode="content",
                            context_lines=1, max_results=0))
        )
        asym = json.loads(
            run(grep_search("bar", path=str(grep_tree), output_mode="content",
                            context_before=1, context_after=1, max_results=0))
        )
        assert [m["text"] for m in sym["matches"]] == [m["text"] for m in asym["matches"]]

    def test_cap_counts_matches_only(self, grep_search, grep_tree):
        r = json.loads(
            run(grep_search("foo", path=str(grep_tree), output_mode="content",
                            context_before=1, context_after=1, max_results=1))
        )
        assert r["count"] == 1  # one match
        assert r.get("truncated") is True

    def test_offset_content_mode(self, grep_search, grep_tree):
        # foo appears 3x (a:1, a:2, b:1). offset=1 skips the first match.
        r = json.loads(
            run(grep_search("foo", path=str(grep_tree), output_mode="content",
                            max_results=1, offset=1))
        )
        assert r["count"] == 1
        assert r["matches"][0]["line"] == 2  # second foo in a.txt

    def test_offset_files_mode(self, grep_search, grep_tree):
        # foo matches 2 files (a.txt, b.txt); offset=1 returns the 2nd (last),
        # the tail of the set — not truncated.
        r = json.loads(
            run(grep_search("foo", path=str(grep_tree), output_mode="files_with_matches",
                            max_results=1, offset=1))
        )
        assert r["count"] == 1
        assert "truncated" not in r
        # offset=0 with a cap below the total IS truncated.
        r0 = json.loads(
            run(grep_search("foo", path=str(grep_tree), output_mode="files_with_matches",
                            max_results=1, offset=0))
        )
        assert r0["count"] == 1 and r0.get("truncated") is True

    def test_offset_beyond_total_is_empty(self, grep_search, grep_tree):
        r = json.loads(
            run(grep_search("foo", path=str(grep_tree), output_mode="files_with_matches",
                            max_results=5, offset=99))
        )
        assert r["count"] == 0

    def test_context_does_not_leak_across_pages(self, grep_search, tmp_path):
        # Regression: before-context of an OFF-window match must not leak into a
        # page, and a context line must not duplicate across consecutive pages.
        # 3 matches (lines 9,18,26); -B3, max_results=1.
        lines = ["L%d" % i if i not in (9, 18, 26) else "HIT_%d" % i for i in range(1, 30)]
        (tmp_path / "f.txt").write_text("\n".join(lines) + "\n")
        p = str(tmp_path / "f.txt")

        page0 = json.loads(
            run(grep_search("HIT_", path=p, output_mode="content",
                            context_before=3, max_results=1, offset=0))
        )
        t0 = [m["text"] for m in page0["matches"]]
        assert t0 == ["L6", "L7", "L8", "HIT_9"]  # only HIT_9's own before-context

        page1 = json.loads(
            run(grep_search("HIT_", path=p, output_mode="content",
                            context_before=3, max_results=1, offset=1))
        )
        t1 = [m["text"] for m in page1["matches"]]
        assert t1 == ["L15", "L16", "L17", "HIT_18"]  # HIT_18's context, no leak
        assert not (set(t0) & set(t1))  # no context line on both pages

    def test_after_context_attribution(self, grep_search, tmp_path):
        lines = ["L%d" % i if i not in (9, 18) else "HIT_%d" % i for i in range(1, 25)]
        (tmp_path / "f.txt").write_text("\n".join(lines) + "\n")
        r = json.loads(
            run(grep_search("HIT_", path=str(tmp_path / "f.txt"), output_mode="content",
                            context_after=2, max_results=1, offset=0))
        )
        # Only HIT_9 and its two after-context lines — not HIT_18's.
        assert [m["text"] for m in r["matches"]] == ["HIT_9", "L10", "L11"]

    def test_long_line_capped(self, grep_search, tmp_path):
        (tmp_path / "big.txt").write_text("x" * 600 + " needle\n")
        r = json.loads(
            run(grep_search("needle", path=str(tmp_path), output_mode="content", max_results=0))
        )
        text = r["matches"][0]["text"]
        assert len(text) < 600 and "chars]" in text

    def test_gitignore_and_git_dir_skipped_by_default(self, grep_search, tmp_path):
        # rg honors .gitignore / skips .git by default — no manual excludes needed.
        (tmp_path / "tracked.txt").write_text("needle here\n")
        (tmp_path / ".gitignore").write_text("ignored.txt\n")
        (tmp_path / "ignored.txt").write_text("needle here\n")
        gitdir = tmp_path / ".git"
        gitdir.mkdir()
        (gitdir / "config").write_text("needle here\n")
        r = json.loads(
            run(grep_search("needle", path=str(tmp_path), output_mode="files_with_matches"))
        )
        names = {os.path.basename(f) for f in r["files"]}
        assert names == {"tracked.txt"}


class TestGlobExclusionAndSorting:
    """glob_search: default noise-dir exclusion (opt-out) + sort-before-truncate
    so newest-first is honest when capped."""

    def test_excludes_noise_dirs_by_default(self, glob_search, tmp_path):
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "x.py").write_text("")
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "y.py").write_text("")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "z.py").write_text("")
        r = json.loads(run(glob_search("**/*.py", path=str(tmp_path))))
        assert r["count"] == 1
        assert os.path.basename(r["matches"][0]) == "z.py"

    def test_include_ignored_returns_noise_dirs(self, glob_search, tmp_path):
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "x.py").write_text("")
        (tmp_path / "z.py").write_text("")
        r = json.loads(run(glob_search("**/*.py", path=str(tmp_path), include_ignored=True)))
        assert r["count"] == 2

    def test_sort_before_truncate_returns_globally_newest(self, glob_search, tmp_path):
        # 5 files with staggered mtimes: f0 oldest ... f4 newest.
        for i in range(5):
            f = tmp_path / f"f{i}.py"
            f.write_text("")
            os.utime(str(f), (1000 + i * 100, 1000 + i * 100))
        r = json.loads(run(glob_search("*.py", path=str(tmp_path), max_results=2)))
        assert r["count"] == 2
        assert [os.path.basename(m) for m in r["matches"]] == ["f4.py", "f3.py"]
        assert r.get("truncated") is True

    def test_unsorted_still_truncates(self, glob_search, tmp_path):
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text("")
        r = json.loads(
            run(glob_search("*.py", path=str(tmp_path), max_results=2, sort_by_mtime=False))
        )
        assert r["count"] == 2
        assert r.get("truncated") is True

    def test_max_results_zero_unlimited(self, glob_search, tmp_path):
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text("")
        r = json.loads(run(glob_search("*.py", path=str(tmp_path), max_results=0)))
        assert r["count"] == 5
        assert "truncated" not in r


class TestResolvePath:
    """fs_write._resolve_path: relative paths resolve against the CWD, not ~
    (the old extension-based relocation was a surprise-overwrite footgun)."""

    def test_relative_resolves_against_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _resolve_path("notes.txt") == str(tmp_path / "notes.txt")

    def test_relative_python_file_not_forced_into_home(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resolved = _resolve_path("script.py")
        assert resolved == str(tmp_path / "script.py")
        assert not resolved.startswith(os.path.expanduser("~/script.py"))

    def test_relative_subdir_preserved_under_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _resolve_path("a/b/c.py") == str(tmp_path / "a" / "b" / "c.py")

    def test_absolute_unchanged(self):
        assert _resolve_path("/etc/hosts") == "/etc/hosts"

    def test_home_tilde_expands(self):
        assert _resolve_path("~/x.txt") == os.path.expanduser("~/x.txt")


def _patch_doc_tokens(monkeypatch):
    import mnemoai.server.tools.readers.line_reader as lr

    monkeypatch.setattr(
        lr.config,
        "get",
        lambda key, default=None: 16384 if key == "DOC_MAX_TOKENS" else default,
    )


class TestStreamedRead:
    """read_lines streams (no whole-file readlines) but keeps identical
    semantics for positive and negative (from-EOF) ranges."""

    def test_positive_slice_only(self, tmp_path, monkeypatch):
        from mnemoai.server.tools.readers.line_reader import read_lines

        _patch_doc_tokens(monkeypatch)
        f = tmp_path / "f.txt"
        f.write_text("\n".join(f"l{i}" for i in range(1, 11)) + "\n")  # 10 lines
        out = json.loads(run(read_lines(str(f), 3, 5)))
        assert out["start_line"] == 3 and out["end_line"] == 5
        assert out["total_lines"] == 10 and out["lines_requested"] == 3
        assert "l3" in out["content"] and "l4" in out["content"]
        assert "l2" not in out["content"] and "l6" not in out["content"]

    def test_negative_end_from_eof(self, tmp_path, monkeypatch):
        from mnemoai.server.tools.readers.line_reader import read_lines

        _patch_doc_tokens(monkeypatch)
        f = tmp_path / "f.txt"
        f.write_text("\n".join(f"l{i}" for i in range(1, 11)) + "\n")
        out = json.loads(run(read_lines(str(f), 1, -1)))  # entire file
        assert out["end_line"] == 10 and out["total_lines"] == 10
        assert "l1" in out["content"] and "l10" in out["content"]

    def test_negative_start_from_eof(self, tmp_path, monkeypatch):
        from mnemoai.server.tools.readers.line_reader import read_lines

        _patch_doc_tokens(monkeypatch)
        f = tmp_path / "f.txt"
        f.write_text("\n".join(f"l{i}" for i in range(1, 11)) + "\n")
        out = json.loads(run(read_lines(str(f), -3, -1)))  # last 3 lines
        assert out["start_line"] == 8 and out["end_line"] == 10
        assert "l8" in out["content"] and "l10" in out["content"]
        assert "l7" not in out["content"]

    def test_empty_file(self, tmp_path, monkeypatch):
        from mnemoai.server.tools.readers.line_reader import read_lines

        _patch_doc_tokens(monkeypatch)
        f = tmp_path / "f.txt"
        f.write_text("")
        out = json.loads(run(read_lines(str(f), 1, -1)))
        assert out["error"] is True
        assert "0 lines" in out["message"]

    def test_token_budget_truncates_streamed_slice(self, tmp_path, monkeypatch):
        import mnemoai.server.tools.readers.line_reader as lr
        from mnemoai.server.tools.readers.line_reader import read_lines

        # Tiny token budget so the per-line budget loop fires on the slice.
        monkeypatch.setattr(
            lr.config,
            "get",
            lambda key, default=None: 30 if key == "DOC_MAX_TOKENS" else default,
        )
        f = tmp_path / "f.txt"
        f.write_text("\n".join("word " * 40 for _ in range(20)) + "\n")
        out = json.loads(run(read_lines(str(f), 1, -1)))
        assert out["truncated"] is True
        assert out["lines_processed"] < out["lines_requested"]

    def test_partial_line_truncation_keeps_gutter(self, tmp_path, monkeypatch):
        import mnemoai.server.tools.readers.line_reader as lr
        from mnemoai.server.tools.readers.line_reader import read_lines

        # Budget > 50 so the partial-line-fit branch fires on a single huge line.
        monkeypatch.setattr(
            lr.config,
            "get",
            lambda key, default=None: 100 if key == "DOC_MAX_TOKENS" else default,
        )
        f = tmp_path / "long.txt"
        f.write_text("word " * 500 + "\n")
        out = json.loads(run(read_lines(str(f), 1, 1)))
        first = out["content"].split("\n")[0]
        # The fragment must start with the "{n}\t" gutter, NOT a bare line number
        # mashed into the text ("1 word ...").
        assert first.startswith("     1\t")
        assert not first.startswith("1 ")


class TestEncodingPreservation:
    """file_edit / fs_write preserve the file's encoding, BOM, and line ending
    on an in-place edit (create stays plain UTF-8 LF)."""

    def test_file_edit_preserves_crlf(self, file_edit, tmp_path):
        f = tmp_path / "crlf.txt"
        f.write_bytes(b"foo\r\nbar\r\nbaz\r\n")
        read_state.record_read(str(f))
        result = json.loads(run(file_edit(str(f), "bar", "BAR")))
        assert result["success"] is True
        assert f.read_bytes() == b"foo\r\nBAR\r\nbaz\r\n"  # CRLF intact

    def test_file_edit_utf16_with_bom(self, file_edit, tmp_path):
        f = tmp_path / "u16.txt"
        f.write_bytes(b"\xff\xfe" + "hi\nthere\n".encode("utf-16-le"))
        read_state.record_read(str(f))
        result = json.loads(run(file_edit(str(f), "there", "THERE")))
        assert result["success"] is True  # previously bounced to binary steering
        raw = f.read_bytes()
        assert raw.startswith(b"\xff\xfe")  # BOM kept
        assert raw[2:].decode("utf-16-le") == "hi\nTHERE\n"

    def test_fs_write_str_replace_preserves_crlf(self, tmp_path):
        from mnemoai.server.tools.fs_write import register_fs_write_tools

        mcp = _CapturingMCP()
        register_fs_write_tools(mcp)
        fs_write = mcp.registered["fs_write"]
        f = tmp_path / "c.txt"
        f.write_bytes(b"a\r\nb\r\n")
        read_state.record_read(str(f))
        run(fs_write(str(f), "str_replace", old_str="a", new_str="A"))
        assert f.read_bytes() == b"A\r\nb\r\n"

    def test_fs_write_append_preserves_crlf(self, tmp_path):
        from mnemoai.server.tools.fs_write import register_fs_write_tools

        mcp = _CapturingMCP()
        register_fs_write_tools(mcp)
        fs_write = mcp.registered["fs_write"]
        f = tmp_path / "c.txt"
        f.write_bytes(b"a\r\nb\r\n")
        read_state.record_read(str(f))
        run(fs_write(str(f), "append", new_str="c"))
        # Appended content joins with the file's CRLF ending.
        assert b"\r\nc" in f.read_bytes()

    def test_fs_write_create_stays_utf8_lf(self, tmp_path):
        from mnemoai.server.tools.fs_write import register_fs_write_tools

        mcp = _CapturingMCP()
        register_fs_write_tools(mcp)
        fs_write = mcp.registered["fs_write"]
        f = tmp_path / "new.txt"
        run(fs_write(str(f), "create", file_text="x\ny\n"))
        assert f.read_bytes() == b"x\ny\n"  # no BOM, LF preserved

    def test_crlf_content_in_new_string_no_double_cr(self, file_edit, tmp_path):
        # A model-supplied replacement that itself contains a literal CRLF, on a
        # CRLF file, must not produce a stray \r\r\n.
        f = tmp_path / "c.txt"
        f.write_bytes(b"one\r\ntwo\r\n")
        read_state.record_read(str(f))
        run(file_edit(str(f), "two", "two\r\nthree"))
        assert b"\r\r\n" not in f.read_bytes()

    def test_utf16_file_is_readable_then_editable(self, tmp_path, monkeypatch):
        # A UTF-16/BOM file must NOT be treated as binary — it's readable (so the
        # read-before-write gate can be satisfied) and then editable end-to-end.
        from mnemoai.server.tools import looks_like_binary
        from mnemoai.server.tools.fs_write import register_fs_write_tools
        from mnemoai.server.tools.readers.line_reader import read_lines

        _patch_doc_tokens(monkeypatch)
        f = tmp_path / "u16.txt"
        f.write_bytes(b"\xff\xfe" + "hello\nworld\n".encode("utf-16-le"))
        assert looks_like_binary(str(f)) is False
        out = json.loads(run(read_lines(str(f), 1, -1)))
        assert "hello" in out["content"] and out["total_lines"] == 2

        mcp = _CapturingMCP()
        register_fs_write_tools(mcp)
        fs_write = mcp.registered["fs_write"]
        read_state.record_read(str(f))
        run(fs_write(str(f), "str_replace", old_str="world", new_str="WORLD"))
        raw = f.read_bytes()
        assert raw.startswith(b"\xff\xfe")  # BOM preserved
        assert "WORLD" in raw.decode("utf-16")

    def test_insert_matches_readlines_not_splitlines(self, tmp_path):
        # A file containing a form-feed (\x0c): str.splitlines() would break on it
        # (wrong line count), readlines() does not. Insert must use readlines
        # semantics so the index is right.
        from mnemoai.server.tools.fs_write import register_fs_write_tools

        mcp = _CapturingMCP()
        register_fs_write_tools(mcp)
        fs_write = mcp.registered["fs_write"]
        f = tmp_path / "ff.txt"
        f.write_bytes(b"line1\x0cstill1\nline2\n")  # readlines() sees 2 lines
        read_state.record_read(str(f))
        run(fs_write(str(f), "insert", insert_line=1, new_str="INSERTED"))
        # Inserted after the FIRST readlines-line (which includes the \x0c).
        assert "line1\x0cstill1\nINSERTED" in f.read_bytes().decode()
