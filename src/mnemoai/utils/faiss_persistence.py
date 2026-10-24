"""Recoverable publication of a FAISS index and its JSON metadata."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from mnemoai.utils.atomic_write import atomic_write_json

JOURNAL_SUFFIX = ".pending-write.json"


def _digest(path):
    try:
        with open(path, "rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            return digest.hexdigest()
    except FileNotFoundError:
        return None


def recover_write(index_path, metadata_path):
    """Finish a committed index swap, or discard an uncommitted preparation."""
    journal = Path(str(index_path) + JOURNAL_SUFFIX)
    if not journal.exists():
        return
    pending = json.loads(journal.read_text())
    def valid_digest(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    if (
        not isinstance(pending, dict) or pending.get("version") != 1
        or not valid_digest(pending.get("after"))
        or (pending.get("before") is not None and not valid_digest(pending["before"]))
        or not isinstance(pending.get("metadata"), list)
        or any(not isinstance(row, dict) for row in pending["metadata"])
    ):
        raise ValueError("Invalid pending FAISS write; original files preserved")
    current = _digest(index_path)
    if current == pending["after"]:
        atomic_write_json(str(metadata_path), pending["metadata"])
    elif current != pending.get("before"):
        raise ValueError("FAISS index changed outside its pending write; recovery refused")
    journal.unlink()


def save_pair(index, metadata, index_path, metadata_path):
    """Stage first, journal metadata, then publish; never truncate the live index."""
    import faiss

    if not isinstance(metadata, list) or any(not isinstance(row, dict) for row in metadata):
        raise ValueError("FAISS metadata must be a list of objects")
    if index.ntotal != len(metadata):
        raise ValueError("FAISS index and metadata must have equal row counts")
    # Fail serialization before writing either live file.
    json.dumps(metadata)
    recover_write(index_path, metadata_path)
    parent = Path(index_path).absolute().parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".mnemoai-faiss-", suffix=".tmp", dir=parent)
    os.close(fd)
    try:
        faiss.write_index(index, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        atomic_write_json(str(index_path) + JOURNAL_SUFFIX, {
            "version": 1, "before": _digest(index_path),
            "after": _digest(temporary), "metadata": metadata,
        })
        os.replace(temporary, index_path)
        # If this fails, the journal retains all metadata needed to recover,
        # including replacements whose old/new indexes have identical row counts.
        recover_write(index_path, metadata_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
