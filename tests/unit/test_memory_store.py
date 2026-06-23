"""Unit tests for the curated MEMORY.md store (client/memory/memory_store.py).

Pure file logic — no MCP, no LLM. Covers add/replace/remove, duplicate
rejection, the char-cap overflow → consolidation contract, unique-substring
matching, and multi-line entries.
"""

import pytest

from mnemoai.client.memory.memory_store import MemoryError, MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(path=tmp_path / "MEMORY.md", max_chars=200)


def test_add_and_read(store):
    store.add("prefers pytest over unittest")
    store.add("project uses src/ layout")
    assert store._entries() == [
        "prefers pytest over unittest",
        "project uses src/ layout",
    ]
    # Round-trips on disk.
    assert "prefers pytest" in store.read()


def test_add_rejects_blank(store):
    with pytest.raises(MemoryError):
        store.add("   ")


def test_add_rejects_exact_duplicate(store):
    store.add("same entry")
    with pytest.raises(MemoryError, match="already exists"):
        store.add("same entry")


def test_add_overflow_raises_consolidation_error(tmp_path):
    s = MemoryStore(path=tmp_path / "MEMORY.md", max_chars=40)
    s.add("first short entry")
    with pytest.raises(MemoryError, match="exceed the memory limit"):
        s.add("x" * 60)
    # The rejected entry was NOT written.
    assert "x" * 60 not in s.read()


def test_replace_unique_match(store):
    store.add("likes tabs")
    store.replace("tabs", "likes spaces, not tabs")
    assert store._entries() == ["likes spaces, not tabs"]


def test_replace_errors_on_no_match(store):
    store.add("alpha")
    with pytest.raises(MemoryError, match="No memory entry contains"):
        store.replace("zeta", "new")


def test_replace_errors_on_multiple_matches(store):
    store.add("alpha one")
    store.add("alpha two")
    with pytest.raises(MemoryError, match="be more specific"):
        store.replace("alpha", "merged")


def test_replace_overflow_rejected(tmp_path):
    s = MemoryStore(path=tmp_path / "MEMORY.md", max_chars=40)
    s.add("tiny")
    with pytest.raises(MemoryError, match="exceed the memory limit"):
        s.replace("tiny", "y" * 60)
    assert "y" * 60 not in s.read()


def test_remove_unique_match(store):
    store.add("keep me")
    store.add("drop me")
    store.remove("drop me")
    assert store._entries() == ["keep me"]


def test_remove_errors_on_ambiguous(store):
    store.add("alpha one")
    store.add("alpha two")
    with pytest.raises(MemoryError, match="be more specific"):
        store.remove("alpha")


def test_multiline_entry_preserved(store):
    store.add("line1\nline2\nline3")
    assert store._entries() == ["line1\nline2\nline3"]


def test_entries_separated_by_markdown_rule(store):
    # New format: entries are separated by a `---` thematic break on its own
    # line, and the file contains no stray `§` symbol.
    store.add("first fact")
    store.add("second fact")
    raw = store.read()
    assert "\n---\n" in raw
    assert "§" not in raw


def test_reads_legacy_section_sign_file(tmp_path):
    # Existing MEMORY.md files used a `§` delimiter; they must still parse.
    p = tmp_path / "MEMORY.md"
    p.write_text("legacy one.\n§\nlegacy two.\n")
    s = MemoryStore(path=p, max_chars=10000)
    assert s._entries() == ["legacy one.", "legacy two."]


def test_legacy_file_migrates_to_rule_on_write(tmp_path):
    # The first write after reading a `§` file rewrites it with `---`.
    p = tmp_path / "MEMORY.md"
    p.write_text("legacy one.\n§\nlegacy two.\n")
    s = MemoryStore(path=p, max_chars=10000)
    s.add("new three.")
    raw = p.read_text()
    assert "§" not in raw
    assert "\n---\n" in raw
    assert s._entries() == ["legacy one.", "legacy two.", "new three."]


def test_clear_empties_file(store):
    store.add("a")
    store.add("b")
    store.clear()
    assert store._entries() == []
    assert store.read().strip() == ""


def test_read_absent_file_is_empty(tmp_path):
    s = MemoryStore(path=tmp_path / "nope.md", max_chars=200)
    assert s.read() == ""
    assert s._entries() == []


# --- Client-side injection (_inject_memory_context) ---


def test_inject_memory_context(tmp_path, monkeypatch):
    """The client wraps MEMORY.md contents for the system prompt, and returns
    "" when disabled or empty."""
    import mnemoai.client.client as client_mod
    from mnemoai.client.client import LangGraphClient

    mem = tmp_path / "MEMORY.md"
    monkeypatch.setattr(client_mod, "MemoryStore", None, raising=False)
    # Point the store at our temp file and control the ENABLE_MEMORY toggle.
    monkeypatch.setattr(
        "mnemoai.utils.paths.memory_file_path", lambda profile=None: mem
    )
    enabled = {"v": True}
    monkeypatch.setattr(
        client_mod.config, "get",
        lambda k, d=None: enabled["v"] if k == "ENABLE_MEMORY" else d,
    )

    inject = LangGraphClient._inject_memory_context.__get__(
        object.__new__(LangGraphClient)
    )

    # Empty file -> "".
    assert inject() == ""

    # Populated -> wrapped block.
    mem.write_text("prefers pytest\n")
    out = inject()
    assert out.startswith("[Persistent Memory]")
    assert "prefers pytest" in out

    # Disabled -> "" regardless of contents.
    enabled["v"] = False
    assert inject() == ""
