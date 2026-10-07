"""Read actual PDFs with the patched parser, retaining the reader's contract."""

import json

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
