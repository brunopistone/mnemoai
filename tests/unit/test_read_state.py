"""Unit tests for the read-before-write / staleness gate (read_state.py).

Two layers:
- the pure registry (record_read / check_write_allowed / forget / reset);
- end-to-end through fs_write + file_edit + fs_read, proving the gate blocks a
  modification of an existing file the model never read (or that changed on
  disk), while brand-new-file creation and post-read edits still work — and
  that every mnemoai extra (multi-command fs_write, file_edit's rich errors)
  keeps working behind the gate.
"""

import asyncio
import json
import os
import time

import pytest

from mnemoai.server.tools import read_state
from mnemoai.server.tools.file_edit import register_edit_tools
from mnemoai.server.tools.fs_read import register_fs_read_tools
from mnemoai.server.tools.fs_write import register_fs_write_tools


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


@pytest.fixture(autouse=True)
def _reset():
    read_state.reset()
    yield
    read_state.reset()


# --------------------------------------------------------------------------- #
# Pure registry
# --------------------------------------------------------------------------- #
class TestRegistry:
    def test_absent_file_allowed(self, tmp_path):
        # Brand-new file: create needs no prior read.
        assert read_state.check_write_allowed(str(tmp_path / "new.txt")) is None

    def test_existing_unread_blocked(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("data")
        verdict = read_state.check_write_allowed(str(f))
        assert verdict is not None and verdict.get("must_read_first") is True

    def test_allowed_after_record_read(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("data")
        read_state.record_read(str(f))
        assert read_state.check_write_allowed(str(f)) is None

    def test_stale_after_mtime_advances(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("data")
        read_state.record_read(str(f))
        # Advance the mtime to simulate an external change since the read.
        future = time.time() + 10
        os.utime(str(f), (future, future))
        verdict = read_state.check_write_allowed(str(f))
        assert verdict is not None and verdict.get("stale_read") is True

    def test_record_missing_path_is_noop(self, tmp_path):
        read_state.record_read(str(tmp_path / "ghost.txt"))  # must not raise
        # Still treated as a fresh create (absent), so allowed.
        assert read_state.check_write_allowed(str(tmp_path / "ghost.txt")) is None

    def test_forget_drops_entry(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("data")
        read_state.record_read(str(f))
        read_state.forget(str(f))
        assert read_state.check_write_allowed(str(f)) is not None

    def test_reset_clears_all(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("data")
        read_state.record_read(str(f))
        read_state.reset()
        assert read_state.check_write_allowed(str(f)) is not None

    def test_realpath_keying_unifies_spellings(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("data")
        read_state.record_read(str(f))
        # A different spelling of the same file (./ segment) must be allowed too.
        alt = str(tmp_path / "." / "f.txt")
        assert read_state.check_write_allowed(alt) is None

    def test_case_normalized_key_on_case_insensitive_fs(self, tmp_path):
        # On a case-insensitive FS (macOS/Windows) a read then a case-varied
        # write of the SAME file must not be falsely blocked. normcase folds
        # case; on case-sensitive Linux the two spellings are genuinely
        # different files, so we only assert the same-case path is allowed.
        f = tmp_path / "MixedCase.txt"
        f.write_text("data")
        read_state.record_read(str(f))
        assert read_state.check_write_allowed(str(f)) is None
        if os.path.normcase("A") == os.path.normcase("a"):  # case-insensitive FS
            alt = str(tmp_path / "mixedcase.txt")
            assert read_state.check_write_allowed(alt) is None


# --------------------------------------------------------------------------- #
# End-to-end through the tools
# --------------------------------------------------------------------------- #
@pytest.fixture
def tools():
    mcp = _CapturingMCP()
    register_fs_read_tools(mcp)
    register_fs_write_tools(mcp)
    register_edit_tools(mcp)
    return mcp.registered


class TestGateEndToEnd:
    def test_create_new_file_no_read_needed(self, tools, tmp_path):
        target = tmp_path / "brand_new.txt"
        out = json.loads(
            run(tools["fs_write"](str(target), "create", file_text="hi"))
        )
        assert out.get("success") is True
        assert target.read_text() == "hi"

    def test_create_over_existing_unread_blocked(self, tools, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("original")
        out = json.loads(
            run(tools["fs_write"](str(f), "create", file_text="CLOBBERED"))
        )
        assert out.get("must_read_first") is True
        assert f.read_text() == "original"  # untouched

    def test_str_replace_unread_then_allowed_after_record(self, tools, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("alpha beta")
        blocked = json.loads(
            run(tools["fs_write"](str(f), "str_replace", old_str="beta", new_str="gamma"))
        )
        assert blocked.get("must_read_first") is True
        read_state.record_read(str(f))
        ok = json.loads(
            run(tools["fs_write"](str(f), "str_replace", old_str="beta", new_str="gamma"))
        )
        assert ok.get("success") is True
        assert f.read_text() == "alpha gamma"

    def test_file_edit_unread_blocked_then_allowed(self, tools, tmp_path):
        f = tmp_path / "f.py"
        f.write_text("x = 1")
        blocked = json.loads(run(tools["file_edit"](str(f), "1", "2")))
        assert blocked.get("must_read_first") is True
        read_state.record_read(str(f))
        ok = json.loads(run(tools["file_edit"](str(f), "1", "2")))
        assert ok.get("success") is True
        assert f.read_text() == "x = 2"

    def test_file_edit_stale_after_external_change(self, tools, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("keep me")
        read_state.record_read(str(f))
        future = time.time() + 10
        os.utime(str(f), (future, future))
        out = json.loads(run(tools["file_edit"](str(f), "keep", "drop")))
        assert out.get("stale_read") is True
        assert f.read_text() == "keep me"

    def test_fs_read_line_blesses_a_following_edit(self, tools, tmp_path, monkeypatch):
        # read_lines needs DOC_MAX_TOKENS; provide it (no config.yaml in tests).
        import mnemoai.server.tools.readers.line_reader as lr

        monkeypatch.setattr(
            lr.config,
            "get",
            lambda key, default=None: 16384 if key == "DOC_MAX_TOKENS" else default,
        )
        f = tmp_path / "f.txt"
        f.write_text("hello world")
        # Reading through the real fs_read must record read-state...
        read_out = json.loads(run(tools["fs_read"](str(f), mode="Line")))
        assert "hello" in read_out.get("content", "")
        # ...so a following edit needs no explicit record_read.
        ok = json.loads(run(tools["file_edit"](str(f), "world", "there")))
        assert ok.get("success") is True
        assert f.read_text() == "hello there"

    def test_chained_edit_after_successful_edit(self, tools, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("a b c")
        read_state.record_read(str(f))
        first = json.loads(run(tools["file_edit"](str(f), "a", "A")))
        assert first.get("success") is True
        # The first write rebaselines the mtime, so a follow-up edit isn't stale.
        second = json.loads(run(tools["file_edit"](str(f), "b", "B")))
        assert second.get("success") is True
        assert f.read_text() == "A B c"

    def test_rich_error_payload_preserved_behind_gate(self, tools, tmp_path):
        # file_edit's non-unique error (an mnemoai extra) must still fire once
        # the read gate is satisfied.
        f = tmp_path / "f.txt"
        f.write_text("x\nx\nx")
        read_state.record_read(str(f))
        out = json.loads(run(tools["file_edit"](str(f), "x", "y")))
        assert out.get("error") is True
        assert "occurrences_sample" in out
