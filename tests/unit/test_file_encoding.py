"""Unit tests for file_encoding: detect/decode/encode round-trips.

The helper preserves a text file's encoding, BOM, and dominant line ending
across an in-place edit. These are pure-function tests (no I/O, no LLM).
"""

import pytest

from mnemoai.server.tools.file_encoding import (
    FileShape,
    bom_encoding,
    decode_to_lf,
    detect_shape,
    encode_from_lf,
)

CASES = {
    "utf8_lf": "hello\nworld\n".encode("utf-8"),
    "utf8_crlf": "hello\r\nworld\r\n".encode("utf-8"),
    "utf8_sig": b"\xef\xbb\xbf" + "a\nb\n".encode("utf-8"),
    "utf16le_crlf": b"\xff\xfe" + "x\r\ny\r\n".encode("utf-16-le"),
    "utf16be_lf": b"\xfe\xff" + "x\ny\n".encode("utf-16-be"),
}


class TestRoundTrip:
    @pytest.mark.parametrize("name", list(CASES))
    def test_decode_then_encode_is_byte_identical(self, name):
        raw = CASES[name]
        lf, shape = decode_to_lf(raw)
        assert "\r\n" not in lf  # decoded content is LF-normalized
        assert encode_from_lf(lf, shape) == raw  # exact round-trip

    def test_edit_in_lf_space_preserves_crlf(self):
        raw = "foo\r\nbar\r\nbaz\r\n".encode("utf-8")
        lf, shape = decode_to_lf(raw)
        out = encode_from_lf(lf.replace("bar", "BAR"), shape)
        assert out == "foo\r\nBAR\r\nbaz\r\n".encode("utf-8")

    def test_edit_preserves_utf16_and_bom(self):
        raw = b"\xff\xfe" + "keep\nme\n".encode("utf-16-le")
        lf, shape = decode_to_lf(raw)
        out = encode_from_lf(lf.replace("keep", "KEEP"), shape)
        assert out.startswith(b"\xff\xfe")
        assert out[2:].decode("utf-16-le") == "KEEP\nme\n"


class TestDetection:
    def test_plain_utf8_lf_shape(self):
        shape = detect_shape("a\nb\n".encode("utf-8"))
        assert shape == FileShape(codec="utf-8", newline="\n", bom=b"")

    def test_crlf_detected(self):
        assert detect_shape("a\r\nb\r\n".encode("utf-8")).newline == "\r\n"

    def test_dominant_ending_wins_on_mixed(self):
        # 2 CRLF vs 1 LF-only → CRLF dominant.
        raw = "a\r\nb\r\nc\n".encode("utf-8")
        assert detect_shape(raw).newline == "\r\n"

    def test_lf_only_when_no_crlf(self):
        assert detect_shape("a\nb\nc\n".encode("utf-8")).newline == "\n"


class TestBinary:
    def test_decode_raises_on_invalid_utf8(self):
        # An invalid UTF-8 byte sequence (no BOM) must raise so callers keep
        # their binary-file steering.
        with pytest.raises(UnicodeDecodeError):
            decode_to_lf(b"\xc3\x28")


class TestEncodeShortCircuit:
    def test_lf_file_not_reshaped(self):
        shape = FileShape(codec="utf-8", newline="\n", bom=b"")
        assert encode_from_lf("a\nb\n", shape) == b"a\nb\n"


class TestCrlfIdempotence:
    def test_literal_crlf_in_content_does_not_double(self):
        # A CRLF file whose new content already contains a literal \r\n must not
        # become \r\r\n when the CRLF ending is re-applied.
        shape = FileShape(codec="utf-8", newline="\r\n", bom=b"")
        assert encode_from_lf("a\r\nb\n", shape) == b"a\r\nb\r\n"
        assert b"\r\r\n" not in encode_from_lf("x\r\ny\r\nz\n", shape)


class TestBomEncoding:
    def test_utf32_le_not_mistaken_for_utf16(self):
        # The 4-byte UTF-32-LE BOM must win over its 2-byte UTF-16-LE prefix.
        assert bom_encoding(b"\xff\xfe\x00\x00data") == "utf-32-le"

    def test_utf16_le_still_detected(self):
        assert bom_encoding(b"\xff\xfeX\x00") == "utf-16-le"

    def test_utf8_sig_transparent(self):
        assert bom_encoding(b"\xef\xbb\xbfhi") == "utf-8-sig"

    def test_no_bom_is_empty(self):
        assert bom_encoding(b"plain text") == ""
