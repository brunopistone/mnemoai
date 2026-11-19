"""Concurrent edits and failed publication must preserve user file contents."""

import importlib
import json
import stat
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from mnemoai.server.tools import file_transaction, read_state
from mnemoai.server.tools.file_edit import register_edit_tools
from mnemoai.server.tools.fs_write import register_fs_write_tools


class Registry:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn
        return register


@pytest.fixture
def tools():
    registry = Registry()
    register_edit_tools(registry)
    register_fs_write_tools(registry)
    read_state.reset()
    yield registry.tools
    read_state.reset()


@pytest.mark.parametrize("argument", ["new_str", "file_text"])
@pytest.mark.parametrize("raw,encoding", [
    (b"existing/\n", "utf-8"),
    (b"existing/\r\n", "utf-8"),
    ("existing/\r\n".encode("utf-16"), "utf-16"),
])
def test_append_contract_preserves_content_encoding_and_read_baseline(tools, tmp_path, argument, raw, encoding):
    target = tmp_path / ".gitignore"
    target.write_bytes(raw)
    read_state.record_read(str(target))
    result = json.loads(tools["fs_write"](str(target), "append", **{argument: ".claude/\n.codex/\n"}))
    assert result["success"] and result["changed"]
    assert target.read_bytes().decode(encoding).replace("\r\n", "\n") == "existing/\n.claude/\n.codex/\n"
    assert read_state.check_write_allowed(str(target)) is None
    if b"\r\n" in raw or encoding == "utf-16":
        assert "\r\n.claude/" in target.read_bytes().decode(encoding)


@pytest.mark.parametrize("command,arguments", [
    ("append", {}),
    ("append", {"new_str": ""}),
    ("append", {"new_str": "one", "file_text": "two"}),
    ("append", {"new_str": "new", "old_str": "ignored"}),
    ("insert", {}),
    ("str_replace", {"old_str": "existing/"}),
    ("str_replace", {"old_str": "existing/", "new_str": "existing/"}),
    ("str_replace", {"old_str": "existing/", "new_str": "new/", "file_text": "ignored"}),
    ("create", {"new_str": "wrong payload field"}),
])
def test_invalid_write_arguments_never_touch_the_file(tools, tmp_path, command, arguments):
    target = tmp_path / ".gitignore"
    target.write_text("existing/\n")
    read_state.record_read(str(target))
    before = read_state.current_version(str(target))
    result = json.loads(tools["fs_write"](str(target), command, **arguments))
    assert result["error"] and result["invalid_arguments"]
    assert target.read_text() == "existing/\n"
    assert read_state.current_version(str(target)) == before


def test_explicit_empty_replacement_and_blank_insert_are_preserved(tools, tmp_path):
    target = tmp_path / "sample"
    target.write_text("delete me\nkeep\n")
    read_state.record_read(str(target))
    assert json.loads(tools["fs_write"](str(target), "str_replace", old_str="delete me\n", new_str=""))["changed"]
    assert target.read_text() == "keep\n"
    assert json.loads(tools["fs_write"](str(target), "insert", insert_line=0, new_str=""))["changed"]
    assert target.read_text() == "\nkeep\n"


def test_identical_create_is_explicitly_unchanged_and_does_not_replace_file(tools, tmp_path):
    target = tmp_path / "sample"
    target.write_text("already correct")
    read_state.record_read(str(target))
    before = read_state.current_version(str(target))
    result = json.loads(tools["fs_write"](str(target), "create", file_text="already correct"))
    assert result["success"] and result["changed"] is False
    assert read_state.current_version(str(target)) == before


@pytest.mark.parametrize("new_str", [None, "", "one addition\n"])
def test_append_alias_does_not_duplicate_text(tools, tmp_path, new_str):
    target = tmp_path / "sample"
    target.write_text("existing\n")
    read_state.record_read(str(target))
    result = json.loads(tools["fs_write"](
        str(target), "append", file_text="one addition\n", new_str=new_str,
    ))
    assert result["changed"]
    assert target.read_text() == "existing\none addition\n"

@pytest.mark.parametrize("second_tool", ["file_edit", "fs_write"])
def test_mutations_share_a_canonical_path_lock(tools, tmp_path, monkeypatch, second_tool):
    source = tmp_path / "file"
    source.write_text("a b")
    alias = tmp_path / "alias"
    alias.symlink_to(source)
    read_state.record_read(str(source))
    edit = importlib.import_module("mnemoai.server.tools.file_edit")
    original = edit.decode_to_lf
    reading, release, attempting = threading.Event(), threading.Event(), threading.Event()

    def paused_read(data):
        reading.set()
        assert release.wait(5)
        return original(data)

    monkeypatch.setattr(edit, "decode_to_lf", paused_read)

    def second():
        attempting.set()
        if second_tool == "file_edit":
            return tools[second_tool](str(alias), "b", "B")
        return tools[second_tool](str(alias), "str_replace", old_str="b", new_str="B")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(tools["file_edit"], str(source), "a", "A")
        assert reading.wait(5)
        other = pool.submit(second)
        try:
            assert attempting.wait(5)
            assert not other.done()
        finally:
            release.set()
        assert json.loads(first.result())["success"]
        assert json.loads(other.result())["success"]
    assert source.read_text() == "A B"
    assert alias.is_symlink()


@pytest.mark.parametrize("operation", ["edit", "create", "str_replace", "append", "insert"])
def test_failed_encoding_or_publication_never_truncates_original(tools, tmp_path, monkeypatch, operation):
    source = tmp_path / "file"
    source.write_text("original")
    read_state.record_read(str(source))

    def fail(*args):
        raise OSError("injected disk-full/replace failure")

    monkeypatch.setattr(file_transaction.os, "replace", fail)
    if operation == "edit":
        result = tools["file_edit"](str(source), "original", "new")
    else:
        result = tools["fs_write"](
            str(source), operation, file_text="new", old_str="original", new_str="new",
        )
    assert json.loads(result)["error"]
    assert source.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [source]


def test_encoding_failure_happens_before_opening_destination(tools, tmp_path, monkeypatch):
    source = tmp_path / "file"
    source.write_text("original")
    read_state.record_read(str(source))
    edit = importlib.import_module("mnemoai.server.tools.file_edit")

    def fail(*args):
        raise UnicodeError("injected encoding failure")

    monkeypatch.setattr(edit, "encode_from_lf", fail)
    assert json.loads(tools["file_edit"](str(source), "original", "new"))["error"]
    assert source.read_bytes() == b"original"


def test_edit_preserves_executable_mode_and_symlink(tools, tmp_path):
    source = tmp_path / "script"
    source.write_text("before")
    source.chmod(0o750)
    link = tmp_path / "link"
    link.symlink_to(source)
    read_state.record_read(str(source))
    assert json.loads(tools["file_edit"](str(link), "before", "after"))["success"]
    assert stat.S_IMODE(source.stat().st_mode) == 0o750
    assert link.is_symlink() and link.read_text() == "after"


def test_external_change_during_edit_is_not_overwritten(tools, tmp_path, monkeypatch):
    source = tmp_path / "file"
    source.write_text("original")
    read_state.record_read(str(source))
    edit = importlib.import_module("mnemoai.server.tools.file_edit")
    original = edit.encode_from_lf

    def external_edit(*args):
        source.write_text("external change")
        return original(*args)

    monkeypatch.setattr(edit, "encode_from_lf", external_edit)
    assert json.loads(tools["file_edit"](str(source), "original", "new"))["error"]
    assert source.read_text() == "external change"
