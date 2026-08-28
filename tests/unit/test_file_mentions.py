"""Unit tests for ``@path`` file mentions (client/file_mentions.py).

Pure file logic: a tmp tree stands in for the project, so both halves are
testable with no terminal and no model. The cases worth pinning are the ones
where being wrong is worse than doing nothing — a mention that silently attaches
NOTHING (the model then answers about a file it never saw), a mention that
attaches a 200 MB log, and a completion menu that raises on a keystroke.
"""

from pathlib import Path

import pytest

from mnemoai.client import file_mentions as fm


@pytest.fixture(autouse=True)
def _clear_index_cache():
    """The completion index is process-global and TTL-cached; isolate each test."""
    fm._INDEX_CACHE.clear()
    yield
    fm._INDEX_CACHE.clear()


@pytest.fixture
def tree(tmp_path):
    """A small project: nested source, a dotfile, and a cache dir to ignore."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "chat_interface.py").write_text("one\ntwo\nthree\n")
    (tmp_path / "src" / "pkg" / "__pycache__").mkdir()
    (tmp_path / "src" / "pkg" / "__pycache__" / "cached.pyc").write_text("x")
    (tmp_path / "notes.md").write_text("# Notes\n")
    (tmp_path / ".hidden").write_text("secret\n")
    return tmp_path


class TestFragmentAtCursor:
    def test_a_bare_at_offers_the_directory(self):
        # "" and None must stay distinguishable: one opens the menu, one means
        # there is no mention under the cursor at all.
        assert fm.fragment_at_cursor("summarize @") == ""

    def test_mid_sentence(self):
        assert fm.fragment_at_cursor("explain @src/pkg/ch") == "src/pkg/ch"

    def test_no_mention_is_none(self):
        assert fm.fragment_at_cursor("nothing here") is None

    def test_an_email_is_not_a_mention(self):
        assert fm.fragment_at_cursor("write to bruno@exa") is None

    def test_after_an_opening_bracket(self):
        assert fm.fragment_at_cursor("see (@src/") == "src/"

    def test_a_windows_path_still_completes(self):
        assert fm.fragment_at_cursor("@src\\pkg") == "src\\pkg"

    def test_only_the_fragment_at_the_cursor_counts(self):
        assert fm.fragment_at_cursor("@one.py and @two") == "two"

    def test_empty_input(self):
        assert fm.fragment_at_cursor("") is None


class TestResolve:
    def test_relative_to_the_working_directory(self, tree):
        assert fm.resolve("notes.md", tree) == tree / "notes.md"

    def test_absolute_is_taken_as_is(self, tree):
        target = tree / "notes.md"
        assert fm.resolve(str(target), tree) == target

    def test_home_expands(self, tree, monkeypatch):
        monkeypatch.setenv("HOME", str(tree))
        monkeypatch.setenv("USERPROFILE", str(tree))
        assert fm.resolve("~/notes.md", Path("/nowhere")) == tree / "notes.md"

    def test_a_directory_resolves_too(self, tree):
        # Unlike a steering reference: a mentioned dir contributes its listing.
        assert fm.resolve("src", tree) == tree / "src"

    def test_missing_is_none(self, tree):
        assert fm.resolve("nope.md", tree) is None

    def test_nonsense_is_none_not_an_exception(self, tree):
        assert fm.resolve("\0bad", tree) is None


class TestExpand:
    def test_a_file_is_appended_under_its_own_header(self, tree):
        text, mentions = fm.expand("explain @notes.md", cwd=tree)
        assert "# Notes" in text
        assert str(tree / "notes.md") in text
        assert "mentioned as @notes.md" in text
        assert [m.kind for m in mentions] == ["file"]

    def test_the_mention_stays_in_the_prose(self, tree):
        text, _ = fm.expand("explain @notes.md please", cwd=tree)
        # The sentence still has to read as the user wrote it.
        assert text.startswith("explain @notes.md please")

    def test_nothing_to_do_returns_the_line_untouched(self, tree):
        assert fm.expand("plain question", cwd=tree) == ("plain question", [])
        assert fm.expand("", cwd=tree) == ("", [])

    def test_a_missing_path_is_reported(self, tree):
        _text, mentions = fm.expand("look at @src/gone.py", cwd=tree)
        assert [(m.kind, m.summary) for m in mentions] == [
            ("missing", "no such file")
        ]

    def test_a_decorator_is_not_a_failed_mention(self, tree):
        # The whole reason a non-resolving ref is left alone: @staticmethod,
        # @someone and a bare word must produce no output and no notice.
        _text, mentions = fm.expand("use @staticmethod on @helper", cwd=tree)
        assert mentions == []

    def test_a_version_number_still_warns(self, tree):
        # It has a dot, so it reads as a path — a gray "no such file" line is
        # honest noise; silently ignoring a typo'd path is not.
        _text, mentions = fm.expand("what changed in @1.15.1", cwd=tree)
        assert [m.kind for m in mentions] == ["missing"]

    def test_the_same_file_is_attached_once(self, tree):
        text, mentions = fm.expand("@notes.md vs @notes.md", cwd=tree)
        assert text.count("# Notes") == 1
        assert len(mentions) == 1

    def test_two_spellings_of_one_file_are_attached_once(self, tree):
        text, mentions = fm.expand("@notes.md and @./notes.md", cwd=tree)
        assert text.count("# Notes") == 1
        assert len(mentions) == 1

    def test_a_binary_file_is_named_not_inlined(self, tree):
        (tree / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00binary")
        text, mentions = fm.expand("what is @logo.png", cwd=tree)
        assert text == "what is @logo.png"
        assert [(m.kind, m.summary) for m in mentions] == [
            ("skipped", "not a text file")
        ]

    def test_a_large_file_is_truncated_and_says_so(self, tree, monkeypatch):
        monkeypatch.setattr(fm, "_max_file_chars", lambda: 100)
        (tree / "big.log").write_text("Q" * 5000)
        text, mentions = fm.expand("@big.log", cwd=tree)
        assert text.count("Q") == 100
        assert "truncated" in text and "do not assume" in text
        assert mentions[0].chars == 100
        assert mentions[0].summary == "first 100 chars of 5000 bytes"

    def test_a_cap_of_zero_means_no_cap(self, tree, monkeypatch):
        monkeypatch.setattr(fm, "_max_file_chars", lambda: 0)
        (tree / "big.log").write_text("Q" * 5000)
        text, mentions = fm.expand("@big.log", cwd=tree)
        assert text.count("Q") == 5000 and mentions[0].summary == "1 line"

    def test_a_directory_contributes_a_listing_not_its_files(self, tree):
        text, mentions = fm.expand("what is in @src/pkg", cwd=tree)
        assert "directory listing" in text
        assert "chat_interface.py" in text
        assert "one\ntwo\nthree" not in text  # the file's CONTENT stays out
        assert "__pycache__/" in text  # dirs keep the slash so they can be re-mentioned
        assert mentions[0].kind == "dir" and mentions[0].summary == "2 entries"

    def test_a_long_listing_is_bounded(self, tree, monkeypatch):
        monkeypatch.setattr(fm, "_MAX_DIR_ENTRIES", 3)
        for i in range(10):
            (tree / "src" / f"f{i}.txt").write_text("x")
        text, mentions = fm.expand("@src", cwd=tree)
        assert "more entries]" in text
        assert mentions[0].summary == "11 entries"  # counted in full, listed bounded

    def test_an_unreadable_directory_is_skipped(self, tree, monkeypatch):
        monkeypatch.setattr(fm, "_dir_listing", lambda _p: ("", 0))
        _text, mentions = fm.expand("@src", cwd=tree)
        assert [(m.kind, m.summary) for m in mentions] == [
            ("skipped", "empty or unreadable")
        ]

    def test_too_many_files_stops_inlining_and_says_so(self, tree, monkeypatch):
        monkeypatch.setattr(fm, "_MAX_FILES", 2)
        for i in range(4):
            (tree / f"f{i}.md").write_text(f"file {i}\n")
        line = " ".join(f"@f{i}.md" for i in range(4))
        text, mentions = fm.expand(line, cwd=tree)
        assert [m.kind for m in mentions] == ["file", "file", "skipped", "skipped"]
        assert "file 2" not in text
        assert all(m.summary == "over 2 files" for m in mentions if m.kind == "skipped")

    def test_the_per_line_budget_stops_at_the_offending_file(self, tree, monkeypatch):
        monkeypatch.setattr(fm, "_MAX_TOTAL_CHARS", 30)
        (tree / "small.md").write_text("y" * 10)
        (tree / "huge.md").write_text("z" * 100)
        text, mentions = fm.expand("@small.md @huge.md", cwd=tree)
        assert "y" * 10 in text and "z" * 100 not in text
        assert mentions[1].summary == "over the size budget"

    def test_expand_never_raises_on_a_vanished_file(self, tree, monkeypatch):
        def _boom(_self, *_a, **_kw):
            raise OSError("gone mid-turn")

        # It resolved a moment ago and is unreadable now — the race a mention is
        # most exposed to, since resolution and reading are separate syscalls.
        monkeypatch.setattr(Path, "open", _boom)
        _text, mentions = fm.expand("@notes.md", cwd=tree)
        assert [(m.kind, m.summary) for m in mentions] == [
            ("skipped", "could not be read")
        ]


class TestLabel:
    def test_a_file_reads_as_lines(self, tree):
        _text, mentions = fm.expand("@src/pkg/chat_interface.py", cwd=tree)
        assert mentions[0].label == "@src/pkg/chat_interface.py · 3 lines"
        assert mentions[0].attached

    def test_one_line_is_singular(self, tree):
        (tree / "one.txt").write_text("no trailing newline")
        _text, mentions = fm.expand("@one.txt", cwd=tree)
        assert mentions[0].label.endswith("· 1 line")

    def test_a_missing_file_is_not_attached(self, tree):
        _text, mentions = fm.expand("@gone/x.py", cwd=tree)
        assert mentions[0].label == "@gone/x.py · no such file"
        assert not mentions[0].attached

    def test_no_summary_falls_back_to_the_bare_ref(self):
        assert fm.Mention("x.py", None, "missing").label == "@x.py"


class TestCompletions:
    def test_a_bare_fragment_matches_a_nested_basename(self, tree):
        # The point of the index: you know the file's name, not its directory.
        assert ("src/pkg/chat_interface.py", "src/pkg") in fm.completions(
            "chat_int", cwd=tree
        )

    def test_an_empty_fragment_lists_the_working_directory(self, tree):
        names = [p for p, _ in fm.completions("", cwd=tree)]
        assert "notes.md" in names and "src/" in names
        assert ".hidden" not in names  # an empty fragment shouldn't open with dotfiles

    def test_a_dotfile_appears_once_it_is_asked_for(self, tree):
        assert ".hidden" in [p for p, _ in fm.completions(".hid", cwd=tree)]

    def test_a_path_fragment_completes_its_last_segment(self, tree):
        assert fm.completions("src/pk", cwd=tree) == [("src/pkg/", "dir")]

    def test_a_trailing_slash_lists_the_directory(self, tree):
        names = [p for p, _ in fm.completions("src/pkg/", cwd=tree)]
        assert "src/pkg/chat_interface.py" in names

    def test_an_absolute_fragment_reaches_outside_the_project(self, tree):
        deep = tree / "src" / "pkg"
        got = fm.completions(f"{deep}/chat", cwd=Path("/"))
        assert got == [(f"{deep}/chat_interface.py", "file")]

    def test_a_missing_parent_yields_nothing(self, tree):
        assert fm.completions("nope/deeper/x", cwd=tree) == []

    def test_the_limit_is_honored(self, tree):
        for i in range(30):
            (tree / f"item{i}.md").write_text("x")
        assert len(fm.completions("item", cwd=tree, limit=5)) == 5

    def test_a_directory_hit_keeps_its_trailing_slash(self, tree):
        # So the next keystroke completes INSIDE it instead of ending the mention.
        assert ("src/", "dir") in fm.completions("sr", cwd=tree)

    def test_ranking_prefers_a_basename_prefix(self, tree):
        (tree / "src" / "zzz_query.py").write_text("x")
        (tree / "src" / "pkg" / "query_router.py").write_text("x")
        first = fm.completions("query", cwd=tree)[0][0]
        assert first == "src/pkg/query_router.py"

    def test_a_broken_scan_returns_no_completions(self, tree, monkeypatch):
        def _boom(*_a, **_kw):
            raise OSError("no")

        monkeypatch.setattr(fm, "_dir_entries", _boom)
        # A completer that raises takes the whole input down with it.
        assert fm.completions("anything", cwd=tree) == []


class TestIndex:
    def test_the_walk_fallback_skips_caches_and_dotfiles(self, tree):
        files = fm._walk_index(tree)
        assert "src/pkg/chat_interface.py" in files
        assert not any("__pycache__" in f for f in files)
        assert ".hidden" not in files

    def test_the_walk_is_bounded(self, tree, monkeypatch):
        monkeypatch.setattr(fm, "_MAX_INDEX_FILES", 2)
        for i in range(10):
            (tree / f"f{i}.txt").write_text("x")
        assert len(fm._walk_index(tree)) == 2

    def test_git_declines_outside_a_repository(self, tmp_path):
        # tmp_path is not a work tree, so ls-files fails and the walk takes over.
        assert fm._git_index(tmp_path) is None

    def test_a_missing_git_binary_is_not_fatal(self, tree, monkeypatch):
        def _boom(*_a, **_kw):
            raise FileNotFoundError("git")

        monkeypatch.setattr(fm.subprocess, "run", _boom)
        assert fm._git_index(tree) is None
        assert fm.completions("chat_int", cwd=tree)  # still answers, via the walk

    def test_the_index_is_cached_between_keystrokes(self, tree, monkeypatch):
        calls = []
        monkeypatch.setattr(fm, "_git_index", lambda root: calls.append(root) or [])
        fm._index(tree)
        fm._index(tree)
        assert len(calls) == 1

    def test_the_cache_is_per_directory(self, tree, tmp_path, monkeypatch):
        other = tmp_path / "other"
        other.mkdir()
        calls = []
        monkeypatch.setattr(fm, "_git_index", lambda root: calls.append(root) or [])
        fm._index(tree)
        fm._index(other)
        assert len(calls) == 2


class TestDocumentedCap:
    def test_the_shipped_config_quotes_the_code_default(self):
        # The examples are what users read; the code default is what applies when
        # nobody edits config. A drift between them is a docs bug nothing else sees.
        import re

        examples = list(
            (Path(fm.__file__).parents[1] / "utils").glob("config.yaml*.example")
        )
        assert examples, "no config examples found"
        for ex in examples:
            quoted = re.search(
                r"^MENTIONS:\s*\n\s*MAX_FILE_CHARS:\s*(\d+)",
                ex.read_text(),
                re.MULTILINE,
            )
            assert quoted, f"{ex.name} does not document MENTIONS.MAX_FILE_CHARS"
            assert int(quoted.group(1)) == fm._DEFAULT_MAX_FILE_CHARS, ex.name
