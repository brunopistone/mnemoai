"""Output caps must also bound allocation before rendering the result."""

import io
import json
import tracemalloc

import pytest

from mnemoai.server.tools.readers import json_reader, line_reader
from mnemoai.server.tools.readers.bounded_lines import count_lines, iter_lines


def test_complete_json_does_not_tell_the_model_that_it_was_truncated(tmp_path):
    path = tmp_path / "small.json"
    path.write_text('{"value": 1}\n')
    result = json.loads(json_reader.read_json(str(path)))
    assert result["truncated"] is False
    assert "TRUNCATED" not in result["message"]
    assert "read completely" in result["message"]


@pytest.mark.parametrize("text,expected", [("", 0), ("a", 1), ("a\n", 1), ("a\nb", 2), ("\n\n", 2)])
def test_chunked_line_count_keeps_eof_semantics(text, expected):
    assert count_lines(io.StringIO(text)) == expected


def test_single_huge_line_is_consumed_in_bounded_pieces():
    class Observed(io.StringIO):
        def readline(self, size=-1):
            assert 0 < size <= 65536
            return super().readline(size)

    lines = list(iter_lines(Observed("x" * 200_000 + "\nnext\n"), 32))
    assert lines == [("x" * 32, True), ("next\n", False)]


@pytest.mark.parametrize("kind", ["line", "json", "jsonl"])
def test_whole_file_and_single_line_reads_have_bounded_peak_memory(tmp_path, monkeypatch, kind):
    module = line_reader if kind == "line" else json_reader
    monkeypatch.setattr(module.config, "get", lambda key, default=None: 64 if key == "DOC_MAX_TOKENS" else default)
    monkeypatch.setattr(module, "count_tokens", lambda text: len(text) // 4)
    path = tmp_path / ("large." + kind)
    with path.open("w") as stream:
        stream.write('"')
        for _ in range(160):
            stream.write("word " * 16384)
        stream.write('"\n')
    tracemalloc.start()
    try:
        result = json.loads(
            module.read_lines(str(path), 1, -1) if kind == "line"
            else module.read_json(str(path))
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert not result.get("error"), result
    assert result["truncated"]
    assert result["total_lines"] == 1
    assert len(result["content"]) < 4096
    assert peak < 4_000_000, f"read buffered the 13 MB input ({peak} bytes allocated)"
    if kind != "line":
        assert not result["syntax_validated"]
