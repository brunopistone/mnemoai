"""Unit tests for an episode's ``tools`` payload: what is stored, what a reader
gets back, and the one-shot on-disk compaction of stores written before the fix.

Pure logic plus a synthetic ChromaDB SQLite file — no config.yaml, no LLM.
"""

import json
import sqlite3

import pytest

from mnemoai.client.memory import episode_tools, episodic_compaction
from mnemoai.client.memory.episode_tools import (
    NO_TOOLS,
    clip_task,
    compact_tools,
    describe_tools,
    format_tools,
    parse_tools,
    tool_names,
)
from mnemoai.client.memory.episodic_compaction import (
    _is_oversized,
    compact_store,
    compact_stores,
)
from mnemoai.client.memory.episodic_memory import EpisodicMemoryManager


def _records(*names, result_chars: int = 0) -> list[dict]:
    """Live tool records the way ``extract_tools_from_messages`` builds them."""
    return [
        {
            "name": name,
            "args": {"path": f"/tmp/{name}"},
            "id": f"tooluse_{i}",
            "result": "x" * result_chars,
        }
        for i, name in enumerate(names)
    ]


class TestToolNames:
    def test_names_in_first_use_order(self):
        assert tool_names(_records("fs_read", "execute_bash")) == [
            "fs_read",
            "execute_bash",
        ]

    def test_repeats_collapse(self):
        # A task that read forty files said "fs_read", not forty things.
        assert tool_names(_records(*["fs_read"] * 40)) == ["fs_read"]

    def test_non_dict_records_ignored(self):
        assert tool_names(["fs_read", None, 7]) == []

    def test_non_list_input(self):
        assert tool_names(None) == []
        assert tool_names("fs_read") == []


class TestFormatTools:
    def test_the_stored_value_is_the_names(self):
        assert format_tools(_records("fs_read", "execute_bash")) == (
            "fs_read, execute_bash"
        )

    def test_no_tools_is_empty_string(self):
        assert format_tools([]) == ""

    def test_arguments_and_results_are_not_stored(self):
        stored = format_tools(_records("fs_read", result_chars=500_000))
        assert stored == "fs_read"
        assert len(stored) < episode_tools.MAX_TOOLS_CHARS

    def test_name_count_is_bounded(self):
        stored = format_tools(_records(*[f"tool_{i}" for i in range(50)]))
        assert len(stored.split(", ")) == episode_tools._MAX_NAMES

    def test_char_cap_cuts_between_names(self):
        long_names = [f"{'n' * 90}_{i}" for i in range(8)]
        stored = format_tools(_records(*long_names))
        assert len(stored) <= episode_tools.MAX_TOOLS_CHARS
        # Every surviving term is a whole name — a half name matches nothing.
        assert all(part in long_names for part in stored.split(", "))


class TestParseTools:
    def test_round_trip_of_the_stored_shape(self):
        stored = format_tools(_records("fs_read", "grep_search"))
        assert parse_tools(stored) == ["fs_read", "grep_search"]

    def test_legacy_record_repr(self):
        legacy = str(_records("web_search", "web_crawler", result_chars=200))
        assert parse_tools(legacy) == ["web_search", "web_crawler"]

    def test_legacy_repr_with_nested_escaped_results(self):
        # The shape that grew to megabytes: a result holding a repr of a result.
        inner = str([{"text": "a" * 100}])
        legacy = str([{"name": "fs_read", "args": {}, "result": str([{"text": inner}])}])
        assert parse_tools(legacy) == ["fs_read"]

    def test_legacy_empty_list(self):
        assert parse_tools(str([])) == []

    def test_legacy_argument_named_name_is_not_a_tool(self):
        legacy = str([{"name": "fs_write", "args": {"name": "not_a_tool"}}])
        assert parse_tools(legacy) == ["fs_write"]

    def test_oversized_legacy_value_is_not_parsed(self):
        # Refusing to parse is the point: it costs real time, per episode, per
        # turn — and the compaction is what removes the value for good.
        legacy = "[" + "{'name': 'fs_read'}," * 200_000 + "]"
        assert len(legacy) > episode_tools._MAX_LEGACY_CHARS
        assert parse_tools(legacy) == []

    def test_unparseable_legacy_value(self):
        assert parse_tools("[{'name': <object at 0x1>}]") == []

    def test_empty_and_wrong_types(self):
        assert parse_tools("") == []
        assert parse_tools("   ") == []
        assert parse_tools(None) == []
        assert parse_tools(42) == []
        assert parse_tools({"name": "fs_read"}) == []

    @pytest.mark.parametrize(
        "value",
        [
            "[",
            "{",
            "[{",
            "[[[[[[[[[[",
            "{'name': 'fs_read'}",
            ", , ,",
            "fs_read,,grep_search,",
            "\x00\x01",
        ],
    )
    def test_never_raises(self, value):
        assert isinstance(parse_tools(value), list)


class TestDescribeTools:
    def test_names(self):
        assert describe_tools("fs_read, execute_bash") == "fs_read, execute_bash"

    def test_legacy_value(self):
        assert describe_tools(str(_records("fs_read"))) == "fs_read"

    def test_nothing_reads_as_no_tools(self):
        assert describe_tools("") == NO_TOOLS
        assert describe_tools(str([])) == NO_TOOLS
        assert describe_tools(None) == NO_TOOLS


class TestCompactTools:
    def test_rewrites_a_legacy_value(self):
        legacy = str(_records("fs_read", "fs_write", result_chars=50_000))
        assert compact_tools(legacy) == "fs_read, fs_write"

    def test_is_idempotent(self):
        legacy = str(_records("fs_read", "fs_write", result_chars=1000))
        once = compact_tools(legacy)
        assert compact_tools(once) == once
        assert compact_tools(compact_tools(once)) == once

    def test_a_value_it_cannot_read_is_left_alone_not_emptied(self):
        # The writer's rule, and the opposite of the reader's: no name came back,
        # so there is nothing to write — and "" would destroy the field this
        # whole module exists to preserve.
        unreadable = "[{'name': <object at 0x1>}]"
        assert parse_tools(unreadable) == []
        assert compact_tools(unreadable) == unreadable
        assert compact_tools("[[[[[[") == "[[[[[["

    def test_a_value_past_the_readers_cap_is_still_rewritten(self):
        # The reader refuses to parse this per turn; the compaction runs once and
        # exists for exactly these, so the cap must not exempt them.
        legacy = "[" + "{'name': 'fs_read'}," * 200_000 + "]"
        assert len(legacy) > episode_tools._MAX_LEGACY_CHARS
        assert parse_tools(legacy) == []
        assert compact_tools(legacy) == "fs_read"

    def test_is_idempotent_on_a_value_it_could_not_read(self):
        unreadable = "[{'name': <object at 0x1>}]"
        assert compact_tools(compact_tools(unreadable)) == unreadable

    def test_no_tools_stays_empty(self):
        # An episode that really used none must not come back as a literal "[]".
        assert compact_tools(format_tools([])) == ""


class TestClipTask:
    def test_short_task_is_untouched(self):
        assert clip_task("fix the bug") == "fix the bug"

    def test_long_task_is_capped(self):
        clipped = clip_task("x" * 50_000)
        assert len(clipped) == episode_tools.MAX_TASK_CHARS + 1  # + the ellipsis
        assert clipped.endswith("…")

    def test_non_string(self):
        assert clip_task(None) == ""


class TestStoreEpisodeWritesTheCompactShape:
    """The writer's half: what actually lands in the store's metadata."""

    class _FakeStore:
        def __init__(self):
            self.added = []

        def search(self, query, top_k=1):
            return []

        def add(self, text, metadata):
            self.added.append((text, metadata))

    class _FakeEncoder:
        """4 chars ≈ 1 token, so nothing here downloads a tiktoken encoding."""

        def encode(self, text, disallowed_special=()):
            return [text[i : i + 4] for i in range(0, len(text), 4)]

        def decode(self, tokens):
            return "".join(tokens)

    def _manager(self):
        manager = object.__new__(EpisodicMemoryManager)
        manager.store = self._FakeStore()
        manager.encoder = self._FakeEncoder()
        manager.duplicate_threshold = 0.95
        manager.max_tokens = 400
        manager.config = {}
        # Token counting is not what this test is about (and must not depend on
        # the configured provider).
        manager.count_tokens = lambda text: len(text) // 4
        return manager

    def test_metadata_holds_names_not_records(self):
        manager = self._manager()
        manager.store_episode("fix the bug", _records("fs_read", result_chars=100_000))
        _text, metadata = manager.store.added[0]
        assert metadata["tools"] == "fs_read"
        assert parse_tools(metadata["tools"]) == ["fs_read"]

    def test_the_embedded_text_is_bounded_too(self):
        manager = self._manager()
        manager.store_episode("q" * 40_000, _records("fs_read", result_chars=100_000))
        text, metadata = manager.store.added[0]
        assert len(metadata["task"]) <= episode_tools.MAX_TASK_CHARS + 1
        assert len(text) <= episode_tools.MAX_TASK_CHARS + 200

    def test_no_tools_is_spelled_the_same_way_everywhere(self):
        manager = self._manager()
        manager.store_episode("what is this", [])
        text, metadata = manager.store.added[0]
        assert f"Tools used: {NO_TOOLS}" in text
        assert metadata["tools"] == ""
        assert describe_tools(metadata["tools"]) == NO_TOOLS


# --- The on-disk compaction ------------------------------------------------


def _make_store(tmp_path, episodes: int, payload_chars: int, name="model"):
    """A synthetic episodic store: Chroma's two metadata-carrying tables, filled
    with the legacy oversized ``tools`` payload."""
    store = tmp_path / name / "episodic_memory"
    store.mkdir(parents=True)
    con = sqlite3.connect(store / "chroma.sqlite3")
    con.executescript(
        """
        CREATE TABLE embedding_metadata (
            id INTEGER, key TEXT NOT NULL, string_value TEXT,
            int_value INTEGER, float_value REAL, bool_value INTEGER,
            PRIMARY KEY (id, key));
        CREATE INDEX embedding_metadata_string_value
            ON embedding_metadata (key, string_value)
            WHERE string_value IS NOT NULL;
        CREATE TABLE embeddings_queue (
            seq_id INTEGER PRIMARY KEY,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            operation INTEGER NOT NULL, topic TEXT NOT NULL, id TEXT NOT NULL,
            vector BLOB, encoding TEXT, metadata TEXT);
        """
    )
    for i in range(episodes):
        legacy = str(
            [
                {
                    "name": "fs_read",
                    "args": {"path": "p"},
                    "result": "r" * payload_chars,
                },
                {"name": "execute_bash", "args": {"cmd": "ls"}, "result": "o"},
            ]
        )
        meta = {
            "task": f"task {i}",
            "tools": legacy,
            "outcome": "success",
            "timestamp": "2026-01-01T00:00:00",
        }
        for key, value in meta.items():
            con.execute(
                "INSERT INTO embedding_metadata (id, key, string_value) VALUES (?,?,?)",
                (i, key, value),
            )
        con.execute(
            "INSERT INTO embeddings_queue (seq_id, operation, topic, id, vector,"
            " encoding, metadata) VALUES (?,?,?,?,?,?,?)",
            (
                i + 1,
                0,
                "persistent://x",
                f"episode_{i}",
                b"\x00" * 32,
                "float32",
                json.dumps(meta),
            ),
        )
    con.commit()
    con.execute("VACUUM")
    con.close()
    return store


def _rows(store):
    """The ``tools`` metadata rows and the log rows, as they are on disk."""
    con = sqlite3.connect(store / "chroma.sqlite3")
    try:
        meta = con.execute(
            "SELECT id, string_value FROM embedding_metadata "
            "WHERE key='tools' ORDER BY id"
        ).fetchall()
        queue = con.execute(
            "SELECT seq_id, id, metadata FROM embeddings_queue ORDER BY seq_id"
        ).fetchall()
    finally:
        con.close()
    return meta, queue


@pytest.fixture
def small_gate(monkeypatch):
    """Run the compaction on a small store: the 16 MB gate is about not reading
    every store on every startup, not about what the rewrite does."""
    monkeypatch.setattr(episodic_compaction, "_MIN_SIZE_BYTES", 200 * 1024)


class TestSizeGate:
    def test_a_big_file_qualifies(self, tmp_path):
        # Sparse — the gate reads the size, it never opens the file.
        big = tmp_path / "chroma.sqlite3"
        with open(big, "wb") as fh:
            fh.truncate(episodic_compaction._MIN_SIZE_BYTES + 1)
        assert _is_oversized(big) is True

    def test_a_small_store_is_left_alone_even_with_legacy_values(self, tmp_path):
        store = _make_store(tmp_path, episodes=8, payload_chars=40_000)
        assert (store / "chroma.sqlite3").stat().st_size < (
            episodic_compaction._MIN_SIZE_BYTES
        )
        assert compact_store(store) is None
        # And the values are still there, untouched.
        meta, _queue = _rows(store)
        assert all(len(value) > 40_000 for _id, value in meta)

    def test_missing_file(self, tmp_path):
        assert _is_oversized(tmp_path / "nope") is False


class TestCompactStore:
    def test_reclaims_the_payload_and_keeps_every_episode(self, tmp_path, small_gate):
        store = _make_store(tmp_path, episodes=8, payload_chars=40_000)
        before_meta, before_queue = _rows(store)
        assert len(before_meta) == 8

        result = compact_store(store)

        assert result is not None
        assert result.episodes == 8
        assert result.queue_rows == 8
        # Most of the file WAS the payload.
        assert result.after < result.before // 2
        assert result.reclaimed == result.before - result.after

        meta, queue = _rows(store)
        # Same episodes, same ids, same log rows — only the bytes inside them go.
        assert [r[0] for r in meta] == [r[0] for r in before_meta]
        assert [(q[0], q[1]) for q in queue] == [(q[0], q[1]) for q in before_queue]
        assert {value for _id, value in meta} == {"fs_read, execute_bash"}
        for _seq, _id, blob in queue:
            payload = json.loads(blob)
            assert payload["tools"] == "fs_read, execute_bash"
            # Everything else about the record is preserved.
            assert payload["outcome"] == "success"
            assert payload["task"].startswith("task ")

    def test_a_second_pass_changes_nothing(self, tmp_path, monkeypatch):
        # Gate off, so this is idempotence of the rewrite and not the size check
        # declining to look at an already-shrunk file.
        monkeypatch.setattr(episodic_compaction, "_MIN_SIZE_BYTES", 0)
        store = _make_store(tmp_path, episodes=8, payload_chars=40_000)
        assert compact_store(store) is not None
        assert compact_store(store) is None

    def test_missing_store(self, tmp_path):
        assert compact_store(tmp_path / "nope") is None

    def test_unreadable_database_is_skipped(self, tmp_path, small_gate):
        store = tmp_path / "broken" / "episodic_memory"
        store.mkdir(parents=True)
        (store / "chroma.sqlite3").write_bytes(b"not a database" * 30_000)
        assert compact_store(store) is None

    def test_a_value_it_cannot_read_is_left_alone_in_both_tables(
        self, tmp_path, small_gate
    ):
        # The rewrite must never trade an oversized field for an empty one: the
        # names are the only thing an episode records about its tools, and there
        # is no second copy to recover them from.
        store = _make_store(tmp_path, episodes=8, payload_chars=40_000)
        unreadable = "[{'name': <object at 0x1>, 'result': '" + "r" * 40_000 + "'}]"
        con = sqlite3.connect(store / "chroma.sqlite3")
        con.execute(
            "UPDATE embedding_metadata SET string_value = ? WHERE id = 0 AND key = ?",
            (unreadable, "tools"),
        )
        row = con.execute(
            "SELECT metadata FROM embeddings_queue WHERE seq_id = 1"
        ).fetchone()
        payload = json.loads(row[0])
        payload["tools"] = unreadable
        con.execute(
            "UPDATE embeddings_queue SET metadata = ? WHERE seq_id = 1",
            (json.dumps(payload),),
        )
        con.commit()
        con.close()

        result = compact_store(store)

        # Counted as untouched in both tables, and still there byte for byte.
        assert result.episodes == 7
        assert result.queue_rows == 7
        meta, queue = _rows(store)
        assert dict(meta)[0] == unreadable
        assert json.loads(queue[0][2])["tools"] == unreadable
        # The readable ones were still rewritten.
        assert {value for _id, value in meta if _id != 0} == {"fs_read, execute_bash"}

    def test_foreign_queue_metadata_is_left_exactly_as_is(self, tmp_path, small_gate):
        store = _make_store(tmp_path, episodes=8, payload_chars=40_000)
        alien = "~not json~" * 500
        con = sqlite3.connect(store / "chroma.sqlite3")
        con.execute(
            "UPDATE embeddings_queue SET metadata = ? WHERE seq_id = 1", (alien,)
        )
        con.commit()
        con.close()

        result = compact_store(store)

        assert result.queue_rows == 7
        _meta, queue = _rows(store)
        assert queue[0][2] == alien


class TestCompactStores:
    def test_sweeps_every_model_dir(self, tmp_path, small_gate):
        big_a = _make_store(tmp_path, 8, 40_000, name="model-a")
        big_b = _make_store(tmp_path, 8, 40_000, name="model-b")
        small = _make_store(tmp_path, 1, 5_000, name="model-c")

        results = compact_stores(tmp_path)

        assert {r.path for r in results} == {str(big_a), str(big_b)}
        assert all(r.after < r.before for r in results)
        # Below the gate: skipped, not emptied.
        meta, _queue = _rows(small)
        assert len(meta[0][1]) > 5_000

    def test_nothing_to_do(self, tmp_path, small_gate):
        _make_store(tmp_path, 1, 5_000, name="model-c")
        assert compact_stores(tmp_path) == []

    def test_missing_root(self, tmp_path):
        assert compact_stores(tmp_path / "nope") == []

    def test_the_work_is_announced_on_SCREEN_before_and_after(
        self, tmp_path, small_gate, capsys
    ):
        # The announcement exists because these are seconds of startup nobody
        # asked for and a silent pause reads as a hang — so it has to be PRINTED:
        # the console log handler sits at LOG_LEVEL (WARNING by default), which
        # keeps a logger.info in the file and off the screen.
        _make_store(tmp_path, 8, 40_000, name="model-a")

        compact_stores(tmp_path)

        out = capsys.readouterr().out
        assert "Compacting episodic memory storage" in out
        assert "1 store," in out  # not "1 stores"
        assert "reclaimed" in out
        # Before the work, not only after it.
        assert out.index("Compacting") < out.index("reclaimed")

    def test_nothing_is_announced_when_there_is_nothing_to_do(
        self, tmp_path, small_gate, capsys
    ):
        _make_store(tmp_path, 1, 5_000, name="model-c")
        assert compact_stores(tmp_path) == []
        assert capsys.readouterr().out == ""

    def test_a_broken_notice_does_not_break_the_compaction(
        self, tmp_path, small_gate, monkeypatch
    ):
        def boom(message):
            raise RuntimeError("no terminal")

        monkeypatch.setattr(episodic_compaction, "print_notice", boom)
        store = _make_store(tmp_path, 8, 40_000, name="model-a")

        assert len(compact_stores(tmp_path)) == 1

        meta, _queue = _rows(store)
        assert {value for _id, value in meta} == {"fs_read, execute_bash"}
