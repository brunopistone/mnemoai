"""The patched PDF parser remains usable through the real MCP subprocess."""

import json

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

pytestmark = pytest.mark.integration


def test_live_pdf_reader_recovers_from_bad_input_and_extracts_text(live_client, _neutral_root):
    malformed = _neutral_root / "malformed-security-fixture.pdf"
    malformed.write_bytes(b"%PDF-1.7\nNot a valid PDF.\n%%EOF\n")
    valid = _neutral_root / "security-fixture.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({
            NameObject("/F1"): DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }),
        }),
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 100 Td (Patched PDF parser works.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(valid)
    messages = []

    def read(path):
        live_client.agent._run_tool_calls(
            [{"id": f"pdf-{len(messages)}", "name": "fs_read",
              "args": {"path": str(path), "mode": "PDF"}}],
            live_client.agent.tools, messages,
        )
        return json.loads(messages[-1].content)

    assert read(malformed)["error"] is True
    result = read(valid)
    assert not result.get("error"), result
    assert result["total_pages"] == 1
    assert "Patched PDF parser works." in result["content"]
