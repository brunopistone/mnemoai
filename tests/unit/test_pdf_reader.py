"""Read actual PDFs with the patched parser, retaining the reader's contract."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from mnemoai.server.tools.readers import pdf_reader


@pytest.fixture(autouse=True)
def no_rag(monkeypatch):
    monkeypatch.setitem(pdf_reader.config._config_data, "ENABLE_RAG", False)


def test_extracts_text_and_page_metadata_from_real_pdf(tmp_path):
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    for text in ("First page content", "Second page content"):
        page = writer.add_blank_page(width=300, height=200)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        contents = DecodedStreamObject()
        contents.set_data(f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(contents)
    path = tmp_path / "two pages.pdf"
    writer.write(path)

    result = json.loads(pdf_reader.read_pdf(str(path)))
    assert not result.get("error"), result
    assert result["total_pages"] == 2
    assert result["file_path"] == str(path)
    assert "--- Page 1 ---\nFirst page content" in result["content"]
    assert "--- Page 2 ---\nSecond page content" in result["content"]
    assert result["processing_metadata"]["chunked"] is False


def test_blank_pdf_remains_a_successful_empty_document(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=200)
    path = tmp_path / "blank.pdf"
    writer.write(path)
    result = json.loads(pdf_reader.read_pdf(str(path)))
    assert not result.get("error"), result
    assert result["content"] == ""
    assert result["total_pages"] == 1


def test_malformed_pdf_returns_structured_error(tmp_path):
    path = tmp_path / "malformed.pdf"
    path.write_bytes(b"%PDF-1.7\nThis is not a valid PDF.\n%%EOF\n")
    result = json.loads(pdf_reader.read_pdf(str(path)))
    assert result["error"] is True
    assert result["file_path"] == str(path)
    assert "Error reading PDF" in result["message"]


def _pdf_with_objects(objects):
    """Build small parser fixtures with correct offsets and an ordinary trailer."""
    data = bytearray(b"%PDF-1.7\n")
    offsets = []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += b"%d 0 obj\n" % index + obj + b"\nendobj\n"
    xref = len(data)
    data += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        data += b"%010d 00000 n \n" % offset
    data += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1, xref,
    )
    return bytes(data)


def _bounded_pdf_probe(tmp_path, data, operation):
    """Keep a parser-loop regression from hanging pytest or exhausting memory."""
    path = tmp_path / "parser-fixture.pdf"
    path.write_bytes(data)
    config = tmp_path / "empty.yaml"
    config.write_text("{}\n")
    script = """
import json
import sys
from pypdf import PdfReader, PdfWriter
from pypdf._configuration import overwrite_configuration
from pypdf.errors import LimitReachedError

path, operation = sys.argv[1:]
if operation == "read":
    from mnemoai.server.tools.readers.pdf_reader import read_pdf
    print(read_pdf(path))
elif operation == "clone":
    class BoundedObjects(list):
        def append(self, value):
            if len(self) >= 100:
                raise AssertionError("Cyclic clone kept allocating objects")
            super().append(value)
    writer = PdfWriter()
    writer._objects = BoundedObjects(writer._objects)
    writer.append(path)
    print(json.dumps({"pages": len(writer.pages)}))
elif operation == "pages":
    # Small allocation budget exercises the guard without a huge input/allocation.
    overwrite_configuration(page_tree_maximum_page_entries=40)
    try:
        pages = len(PdfReader(path).pages)
    except LimitReachedError as error:
        print(json.dumps({"limited": True, "message": str(error)}))
    else:
        print(json.dumps({"limited": False, "pages": pages}))
"""
    env = dict(
        os.environ,
        MNEMOAI_HOME=str(tmp_path / "home"),
        MNEMOAI_CONFIG=str(config),
        PYTHONPATH=str(Path(pdf_reader.__file__).resolve().parents[4]),
        AWS_EC2_METADATA_DISABLED="true",
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), operation],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_truncated_xref_without_trailer_returns_error_instead_of_hanging(tmp_path):
    # GHSA-5849-834g-hvw5: the parser used to rewind to the same EOF forever.
    data = b"%PDF-1.4\nstartxref\n28\n%%EOF\nxref\n1 0\naaa"
    result = _bounded_pdf_probe(tmp_path, data, "read")
    assert result["error"] is True
    assert "Xref parsing stuck" in result["message"]


def test_cyclic_dictionary_cloning_terminates_without_unbounded_objects(tmp_path):
    # GHSA-mfr5-77xh-9j7h: source object IDs must detect a two-node /N cycle.
    data = _pdf_with_objects([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents [4 0 R] >>",
        b"<< /Length 0 /N 5 0 R >>\nstream\n\nendstream",
        b"<< /N 6 0 R /T (first) >>",
        b"<< /N 5 0 R /T (second) >>",
    ])
    assert _bounded_pdf_probe(tmp_path, data, "clone")["pages"] == 1


def test_shared_page_tree_enforces_total_dictionary_entry_budget(tmp_path):
    # GHSA-cggf-277g-3q54: repeated references used to multiply copied dictionaries.
    keys = b"".join(b" /K%d 0" % i for i in range(10))
    data = _pdf_with_objects([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 3 0 R] /Count 6 >>",
        b"<< /Type /Pages /Kids [4 0 R 4 0 R 4 0 R] /Count 3 >>",
        b"<< /Type /Page /MediaBox [0 0 9 9]" + keys + b" >>",
    ])
    result = _bounded_pdf_probe(tmp_path, data, "pages")
    assert result["limited"] is True
    assert "page tree page entry limit" in result["message"]
