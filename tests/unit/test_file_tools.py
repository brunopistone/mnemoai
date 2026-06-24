"""Unit tests for file_edit and glob_search (server/tools/).

Exercise the real filesystem logic via temp files/dirs — no LLM involved.
"""

import asyncio
import json
import os

import pytest

from mnemoai.server.tools.file_edit import register_edit_tools
from mnemoai.server.tools.file_search import register_search_tools


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


class TestFileEdit:
    def test_simple_replacement(self, file_edit, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello world")
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
        result = json.loads(run(file_edit(str(f), "absent", "x")))
        assert result["error"] is True
        assert "not found" in result["message"].lower()
        # File must be untouched.
        assert f.read_text() == "content"

    def test_non_unique_without_replace_all_errors(self, file_edit, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("x\nx\nx")
        result = json.loads(run(file_edit(str(f), "x", "y", replace_all=False)))
        assert result["error"] is True
        assert "occurrences" in result["message"].lower()
        # Nothing changed.
        assert f.read_text() == "x\nx\nx"

    def test_replace_all_replaces_every_occurrence(self, file_edit, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("x\nx\nx")
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
