"""CSV reader reports complete row counts and caps every supported encoding."""

import json

from mnemoai.server.tools.readers import csv_reader


def test_the_first_omitted_row_counts_toward_total(tmp_path, monkeypatch):
    source = tmp_path / "rows.csv"
    source.write_text("name,value\nalpha,1\nbeta,2\n")
    monkeypatch.setattr(csv_reader.config, "get", lambda key, default=None: 1)
    monkeypatch.setattr(csv_reader, "count_tokens", lambda text: 1)
    result = json.loads(csv_reader.read_csv(str(source)))
    assert result["total_rows"] == 2
    assert result["rows_returned"] == 0
    assert result["truncated"] is True


def test_non_utf8_csv_obeys_the_same_budget(tmp_path, monkeypatch):
    source = tmp_path / "rows.csv"
    source.write_bytes("name,value\ncafé,1\ncrème,2\n".encode("latin-1"))
    monkeypatch.setattr(csv_reader.config, "get", lambda key, default=None: 1)
    monkeypatch.setattr(csv_reader, "count_tokens", lambda text: 1)
    result = json.loads(csv_reader.read_csv(str(source)))
    assert result["total_rows"] == 2
    assert result["rows_returned"] == 0
    assert result["truncated"] is True
