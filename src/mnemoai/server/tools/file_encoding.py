"""Detect and preserve a text file's encoding, BOM, and line ending on edit.

When EDITING an existing file we must not silently reshape it: a CRLF file has
to stay CRLF, and a UTF-16 / BOM file has to keep its BOM and codec. This module
sniffs those from the raw bytes, decodes to LF-normalized text for the string op
(the model always supplies LF), then re-applies the original line ending + BOM +
codec on write. It used only for in-place edits (creating a brand-new file stays 
plain UTF-8 LF).

Config-independent (only ``dataclasses``) so it stays unit-testable.
"""

from dataclasses import dataclass, field
from typing import Tuple

# BOM signature -> base codec (endianness-explicit so encode round-trips the
# exact bytes; we prepend/strip the BOM ourselves). Longest signature FIRST so a
# longer BOM wins over a prefix of it — utf-8-sig's 3 bytes over utf-16's 2, and
# utf-32-le's 4 bytes (b"\xff\xfe\x00\x00") over utf-16-le's 2-byte prefix.
_BOMS = (
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xef\xbb\xbf", "utf-8"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)


@dataclass
class FileShape:
    """The on-disk shape of a text file, to re-apply after an in-place edit."""

    codec: str = "utf-8"  # codec for the content bytes (BOM handled separately)
    newline: str = "\n"  # dominant line ending: "\r\n" or "\n"
    bom: bytes = field(default=b"")  # leading BOM bytes, or b"" if none


def bom_encoding(head: bytes) -> str:
    """Return the text codec implied by a leading BOM, or "" if none.

    ``"utf-8-sig"`` for a UTF-8 BOM so the signature is transparently consumed
    on read; the endianness-explicit ``utf-16``/``utf-32`` codec otherwise. Lets
    a reader open a BOM'd UTF-16/UTF-32 file as text instead of rejecting it as
    binary (it's full of NUL bytes).
    """
    for sig, enc in _BOMS:
        if head.startswith(sig):
            return "utf-8-sig" if enc == "utf-8" else enc
    return ""


def detect_shape(raw: bytes) -> FileShape:
    """Sniff codec, BOM, and dominant line ending from a file's raw bytes."""
    codec, bom = "utf-8", b""
    for sig, enc in _BOMS:
        if raw.startswith(sig):
            codec, bom = enc, sig
            break
    text = raw[len(bom):].decode(codec, errors="replace")
    crlf = text.count("\r\n")
    lf_only = text.count("\n") - crlf
    newline = "\r\n" if crlf > lf_only else "\n"
    return FileShape(codec=codec, newline=newline, bom=bom)


def decode_to_lf(raw: bytes) -> Tuple[str, FileShape]:
    """Decode raw bytes to LF-normalized text plus the shape to restore on write.

    Raises ``UnicodeDecodeError`` for content the detected codec can't decode, so
    callers keep their binary-file steering.
    """
    shape = detect_shape(raw)
    text = raw[len(shape.bom):].decode(shape.codec)
    return text.replace("\r\n", "\n"), shape


def encode_from_lf(lf_text: str, shape: FileShape) -> bytes:
    """Re-apply ``shape``'s line ending, codec, and BOM to LF-normalized text.

    Idempotent w.r.t. line endings: any CRLF already present in ``lf_text`` (e.g.
    a literal ``\\r\\n`` the model spliced into a replacement string) is collapsed
    to LF first, so re-applying a CRLF ending can't produce ``\\r\\r\\n``.
    """
    if shape.newline != "\n":
        text = lf_text.replace("\r\n", "\n").replace("\n", shape.newline)
    else:
        text = lf_text
    return shape.bom + text.encode(shape.codec)
