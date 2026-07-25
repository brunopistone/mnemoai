"""Unit tests for path normalization (utils/path_utils.py).

Models are often handed a path the user copied from a shell — spaces
backslash-escaped or the whole path quoted. Tools aren't shells, so the literal
string fails to resolve. normalize_path resolves both forms without breaking
legitimate paths; clean_path_syntax does the syntactic-only cleanup for write
targets that may not exist yet.
"""


from mnemoai.utils.path_utils import clean_path_syntax, normalize_path


class TestNormalizePath:
    def test_plain_existing_path_unchanged(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        assert normalize_path(str(f)) == str(f)

    def test_shell_escaped_spaces_resolve(self, tmp_path):
        f = tmp_path / "My File.png"
        f.write_text("x")
        escaped = str(f).replace(" ", r"\ ")
        assert normalize_path(escaped) == str(f)

    def test_quoted_path_resolves(self, tmp_path):
        f = tmp_path / "My File.png"
        f.write_text("x")
        assert normalize_path(f'"{f}"') == str(f)
        assert normalize_path(f"'{f}'") == str(f)

    def test_escaped_and_quoted_resolves(self, tmp_path):
        f = tmp_path / "My File.png"
        f.write_text("x")
        # Both quoted AND backslash-escaped (belt and suspenders from some shells).
        weird = '"' + str(f).replace(" ", r"\ ") + '"'
        assert normalize_path(weird) == str(f)

    def test_missing_path_returns_expanded_literal(self):
        # Nothing matches on disk -> caller gets a clean literal to report.
        out = normalize_path("/no/such file.png")
        assert out == "/no/such file.png"

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        f = tmp_path / "f.txt"
        f.write_text("x")
        assert normalize_path("~/f.txt") == str(f)

    def test_empty_path(self):
        assert normalize_path("") == ""

    def test_literal_backslash_path_preferred_when_it_exists(self, tmp_path):
        # A real file whose name actually contains a backslash must NOT be
        # de-escaped away — the literal that exists wins.
        name = "weird\\name.txt"
        try:
            f = tmp_path / name
            f.write_text("x")
        except OSError:
            return  # filesystem disallows backslash in names; skip
        assert normalize_path(str(f)) == str(f)


class TestCleanPathSyntax:
    def test_strips_escaped_spaces(self):
        assert clean_path_syntax(r"/a/My\ File.txt") == "/a/My File.txt"

    def test_strips_surrounding_quotes(self):
        assert clean_path_syntax('"/a/My File.txt"') == "/a/My File.txt"
        assert clean_path_syntax("'/a/My File.txt'") == "/a/My File.txt"

    def test_plain_path_unchanged(self):
        assert clean_path_syntax("/a/plain.txt") == "/a/plain.txt"

    def test_no_filesystem_probe_needed(self):
        # Works for a not-yet-existing write target.
        assert clean_path_syntax(r"/brand/new\ file.md") == "/brand/new file.md"

    def test_empty(self):
        assert clean_path_syntax("") == ""
