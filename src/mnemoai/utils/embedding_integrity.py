"""Remove vectors matching the legacy fabricated SHA256 format, keeping backups."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from mnemoai.utils.atomic_write import atomic_write_json
from mnemoai.utils.logger import logger

_BATCH_SIZE = 256
_REPAIR_VERSION = 1
CHROMA_REPAIR_STAMP = "mnemoai_synthetic_checked"


def require_clean_vectors(store) -> None:
    """Never use or mutate an index whose repair/alignment could not be verified."""
    error = getattr(store, "integrity_error", None)
    if error:
        raise RuntimeError(
            f"Vector store needs repair; semantic search and writes are disabled: {error}"
        )


def mark_chroma_clean(collection) -> None:
    """Persist completion, including the row count to notice older append writers."""
    try:
        collection.modify(
            metadata={
                **(collection.metadata or {}),
                CHROMA_REPAIR_STAMP: f"{_REPAIR_VERSION}:{collection.count()}",
            }
        )
    except Exception as exc:
        # Scanning succeeded; an unwritable marker only means a later rescan.
        logger.warning("Could not save Chroma integrity check marker: %s", exc)


def _faiss_signature(index_path, metadata_path):
    try:
        stats = [Path(path).stat() for path in (index_path, metadata_path)]
    except OSError:
        return None
    return [[s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns] for s in stats]


def _faiss_is_checked(index_path, metadata_path) -> bool:
    try:
        stamp = json.loads(
            Path(str(index_path) + ".synthetic-checked.json").read_text()
        )
        signature = _faiss_signature(index_path, metadata_path)
        return signature is not None and stamp == {
            "version": _REPAIR_VERSION,
            "files": signature,
        }
    except (OSError, ValueError):
        return False


def mark_faiss_clean(index_path, metadata_path) -> None:
    """Refresh the O(1) file signature after a checked scan or a normal real write."""
    try:
        signature = _faiss_signature(index_path, metadata_path)
        if signature is not None:
            atomic_write_json(
                str(index_path) + ".synthetic-checked.json",
                {
                    "version": _REPAIR_VERSION,
                    "files": signature,
                },
            )
    except Exception as exc:
        logger.warning("Could not save FAISS integrity check marker: %s", exc)


def discard_faiss_repair_state(index_path) -> None:
    """Drop a pending journal/marker only when the user explicitly clears a store."""
    for suffix in (".synthetic-repair.json", ".synthetic-checked.json"):
        Path(str(index_path) + suffix).unlink(missing_ok=True)


def prepare_chroma(collection, store_path):
    """Return an integrity error instead of disabling access to readable metadata."""
    try:
        repair_chroma(collection, store_path)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Chroma repair deferred; using BM25-only, writes disabled: %s", error
        )
        return error
    return None


def load_faiss_pair(index_path, metadata_path):
    """Load/recover/repair without discarding readable metadata on failure.

    The caller can still build BM25 from metadata, but must reject semantic
    search and writes when the returned error is nonempty.
    """
    import faiss

    index, metadata, errors = None, [], []
    if Path(index_path).exists():
        try:
            index = faiss.read_index(str(index_path))
        except Exception as exc:
            errors.append(f"cannot read index: {exc}")
    if Path(str(index_path) + ".synthetic-repair.json").exists():
        try:
            if index is None:
                raise ValueError("pending repair has no readable index")
            recover_faiss_repair(index, index_path, metadata_path)
        except Exception as exc:
            errors.append(f"cannot recover repair journal: {exc}")
    if Path(metadata_path).exists():
        try:
            values = json.loads(Path(metadata_path).read_text())
            if not isinstance(values, list) or any(
                not isinstance(m, dict) for m in values
            ):
                raise ValueError("metadata must be a list of objects")
            metadata = values
        except Exception as exc:
            errors.append(f"cannot read metadata: {exc}")
    count = index.ntotal if index is not None else 0
    if count != len(metadata):
        errors.append(f"index/metadata length mismatch: {count} vs {len(metadata)}")
    if not errors:
        try:
            index, metadata = repair_faiss(index, metadata, index_path, metadata_path)
        except Exception as exc:
            errors.append(f"cannot finish synthetic-vector repair: {exc}")
    error = "; ".join(errors) or None
    if error:
        logger.warning(
            "FAISS integrity check failed; using BM25-only, writes disabled: %s", error
        )
    return index, metadata, error


def is_legacy_synthetic(vector) -> bool:
    """Match tiled, nonnegative, normalized byte vectors; sign alone is insufficient."""
    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1 or values.size < 64 or not np.isfinite(values).all():
        return False
    if np.any(values < 0) or not np.isclose(np.linalg.norm(values), 1.0, atol=1e-5):
        return False
    if not np.array_equal(values[32:], values[:-32]):
        return False
    block = values[:32]
    if np.unique(block).size < 8:
        return False
    ratios = block / block.max()
    candidates = ratios[None, :] * np.arange(1, 256)[:, None]
    return bool(np.any(np.all(np.abs(candidates - np.rint(candidates)) < 1e-4, axis=1)))


def _backup(path: Path, entries: list) -> None:
    """Preserve removed records before changing an index."""
    previous = json.loads(path.read_text()) if path.exists() else []
    if not isinstance(previous, list):
        raise TypeError(f"Cannot append repair backup to invalid file: {path}")
    # A prior attempt may have backed up the same rows before being interrupted.
    seen = {json.dumps(entry, sort_keys=True) for entry in previous}
    additions = [
        entry for entry in entries if json.dumps(entry, sort_keys=True) not in seen
    ]
    atomic_write_json(str(path), previous + additions)


def repair_chroma(collection, store_path) -> int:
    """Purge matching IDs through Chroma's API, preserving their metadata first."""
    count = collection.count()
    if (collection.metadata or {}).get(
        CHROMA_REPAIR_STAMP
    ) == f"{_REPAIR_VERSION}:{count}":
        return 0
    bad = []
    for offset in range(0, count, _BATCH_SIZE):
        batch = collection.get(
            limit=_BATCH_SIZE,
            offset=offset,
            include=["embeddings", "metadatas", "documents"],
        )
        vectors = batch.get("embeddings")
        if vectors is None:
            continue
        for i, vector in enumerate(vectors):
            if is_legacy_synthetic(vector):
                bad.append(
                    {
                        "id": batch["ids"][i],
                        "metadata": batch["metadatas"][i],
                        "document": batch["documents"][i]
                        if batch.get("documents")
                        else None,
                    }
                )
    if not bad:
        mark_chroma_clean(collection)
        return 0
    backup = Path(store_path) / "legacy-synthetic-metadata.json"
    _backup(backup, bad)
    for offset in range(0, len(bad), _BATCH_SIZE):
        collection.delete(
            ids=[entry["id"] for entry in bad[offset : offset + _BATCH_SIZE]]
        )
    logger.warning(
        "Removed %d legacy synthetic vectors; metadata backup: %s", len(bad), backup
    )
    mark_chroma_clean(collection)
    return len(bad)


def recover_faiss_repair(index, index_path, metadata_path) -> None:
    """Finish an interrupted repair before a loader validates index/metadata.

    FAISS and JSON cannot be replaced atomically together. The small journal
    records the replacement metadata before replacing the index. Its row count
    distinguishes the old index from the new one without guessing alignment.
    """
    journal = Path(str(index_path) + ".synthetic-repair.json")
    if not journal.exists():
        return
    pending = json.loads(journal.read_text())
    if (
        not isinstance(pending, dict)
        or not isinstance(pending.get("metadatas"), list)
        or any(not isinstance(m, dict) for m in pending["metadatas"])
        or type(pending.get("original_count")) is not int
        or pending["original_count"] <= len(pending["metadatas"])
    ):
        raise ValueError(f"Invalid FAISS repair journal: {journal}")
    clean = pending["metadatas"]
    if index.ntotal == len(clean):
        atomic_write_json(str(metadata_path), clean)
    elif index.ntotal != pending["original_count"]:
        raise ValueError(f"Cannot recover misaligned FAISS repair: {journal}")
    journal.unlink()


def repair_faiss(index, metadatas: list, index_path, metadata_path):
    """Return a repaired flat index and aligned metadata, or the original pair."""
    if index is None:
        if metadatas:
            raise ValueError("Cannot repair FAISS metadata without an index")
        return index, metadatas
    if index.ntotal != len(metadatas):
        raise ValueError("Cannot repair a FAISS index whose metadata is misaligned")
    if _faiss_is_checked(index_path, metadata_path):
        return index, metadatas
    bad = []
    for offset in range(0, index.ntotal, _BATCH_SIZE):
        vectors = index.reconstruct_n(offset, min(_BATCH_SIZE, index.ntotal - offset))
        bad.extend(
            offset + i
            for i, vector in enumerate(vectors)
            if is_legacy_synthetic(vector)
        )
    if not bad:
        mark_faiss_clean(index_path, metadata_path)
        return index, metadatas

    import faiss

    backup = Path(str(index_path) + ".legacy-synthetic-metadata.json")
    _backup(backup, [{"index": i, "metadata": metadatas[i]} for i in bad])
    removed = set(bad)
    kept = [i for i in range(index.ntotal) if i not in removed]
    repaired = (
        faiss.IndexFlatIP(index.d)
        if index.metric_type == faiss.METRIC_INNER_PRODUCT
        else faiss.IndexFlatL2(index.d)
    )
    for offset in range(0, len(kept), _BATCH_SIZE):
        repaired.add(
            np.vstack(
                [index.reconstruct(i) for i in kept[offset : offset + _BATCH_SIZE]]
            )
        )
    clean = [metadatas[i] for i in kept]
    journal = Path(str(index_path) + ".synthetic-repair.json")
    fd, temporary = tempfile.mkstemp(dir=Path(index_path).parent, suffix=".index.tmp")
    os.close(fd)
    try:
        faiss.write_index(repaired, temporary)
        atomic_write_json(
            str(journal),
            {
                "original_count": index.ntotal,
                "metadatas": clean,
            },
        )
        os.replace(temporary, index_path)
        recover_faiss_repair(repaired, index_path, metadata_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    logger.warning(
        "Removed %d legacy synthetic vectors; metadata backup: %s", len(bad), backup
    )
    mark_faiss_clean(index_path, metadata_path)
    return repaired, clean
