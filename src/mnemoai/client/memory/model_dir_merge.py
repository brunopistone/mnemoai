"""Folding a forked model-scoped memory directory back into one.

Episodic memory and the ACE playbook are model-scoped
(``{profile}/models/{model}/``). A Bedrock cross-region inference profile
prefixes the base model id with its routing scope, so the same weights arrive
under two ids (``anthropic.claude-opus-5`` and
``global.anthropic.claude-opus-5``) — and while the directory was keyed by the id
verbatim, switching between them silently started both stores from scratch. The
key is now normalized (:func:`mnemoai.utils.paths.normalize_model_key`); this
module carries the directories already written under the un-normalized key into
it, so the fix reaches the installs that grew the fork.

Nothing is deleted. A component the normalized dir doesn't have yet is MOVED; one
that has to be combined is copied and the donor directory is renamed aside —
which is also what stops a merge from re-running on every startup (its ids would
all collide, so it would copy nothing, slowly). Episodes are copied with their
STORED vectors, so no embedding model is called and nothing is re-embedded; a
store written by a different embedding model is left alone, because its vectors
aren't comparable to the target's — the same rule ``ChromaEpisodicStore`` applies
when it resets a collection instead of adopting it.

Best-effort throughout: this runs at startup before either store is opened, and a
failure here must leave the session working with the memory it already had.
"""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import chromadb

from mnemoai.client.memory.episode_tools import compact_tools
from mnemoai.utils.atomic_write import atomic_write_json
from mnemoai.utils.console import print_notice
from mnemoai.utils.logger import log_file_hint, logger
from mnemoai.utils.paths import normalize_model_key, profile_dir

try:  # Implementation detail, used only to close handles — never required.
    from chromadb.api.client import SharedSystemClient
except Exception:  # pragma: no cover - a chroma layout we don't know
    SharedSystemClient = None

_EPISODIC = "episodic_memory"
_PLAYBOOK = "playbook"
_DB_NAME = "chroma.sqlite3"
_PLAYBOOK_FILE = "playbook.json"
_COLLECTION = "episodic_memory"
# Chroma rejects an over-large add outright; the copy is batched well under it.
_BATCH = 500


@dataclass
class Unfork:
    """What one forked directory contributed to the normalized one."""

    donor: str
    target: str
    moved: List[str] = field(default_factory=list)  # components moved wholesale
    episodes: int = 0  # episodes copied by a merge
    strategies: int = 0  # playbook entries copied by a merge
    blocked: List[str] = field(default_factory=list)  # components left behind
    kept_as: Optional[str] = None  # the donor's new name, when it was kept

    @property
    def carried(self) -> bool:
        """True if anything actually moved — a no-op is not worth reporting."""
        return bool(self.moved or self.episodes or self.strategies)


def unfork_model_dirs(model_name: str, profile: str = None) -> List[Unfork]:
    """Fold every directory that normalizes to this model's key into it.

    Args:
        model_name: The configured chat-model id (prefix included).
        profile: Profile name, or None for the configured one.

    Returns:
        One record per donor directory that carried something over.
    """
    done: List[Unfork] = []
    try:
        models_root = profile_dir(profile) / "models"
        key = normalize_model_key(model_name)
        donors = forked_dirs(models_root, key)
        if not donors:
            return done
        target = models_root / key
        for donor in donors:
            try:
                record = _unfork_one(donor, target)
            except Exception:
                # One report, never zero: the traceback exists only in the log.
                logger.warning(
                    "Could not fold the memory of %s into %s; both are left as "
                    "they are. %s",
                    donor.name,
                    key,
                    log_file_hint(),
                    exc_info=True,
                )
                continue
            if record.carried:
                done.append(record)
                # On screen too: this reorganizes the user's memory directories
                # and leaves the donor aside under a new name, which they would
                # otherwise only ever discover by accident (an INFO record goes
                # to the log file alone — see console.print_notice).
                notice = _describe(record)
                logger.info(notice)
                try:
                    print_notice(notice)
                except Exception:
                    logger.debug("Could not print the fold notice", exc_info=True)
    except Exception:
        logger.debug("Model memory unfork skipped", exc_info=True)
    finally:
        # Close the handles this opened (donor AND target) before either store is
        # opened for real: the donor's directory is renamed under it, and the
        # aside copy is rewritten by the storage compaction later in this same
        # startup. Harmless if it doesn't work — a POSIX rename follows the inode.
        if done and SharedSystemClient is not None:
            try:
                SharedSystemClient.clear_system_cache()
            except Exception:
                logger.debug("Could not clear the chroma system cache", exc_info=True)
    return done


def forked_dirs(models_root, key: str) -> List[Path]:
    """Sibling model dirs that normalize to ``key`` without being named it."""
    try:
        entries = sorted(p for p in Path(models_root).iterdir() if p.is_dir())
    except OSError:
        return []
    return [p for p in entries if p.name != key and normalize_model_key(p.name) == key]


def merge_playbook_entries(donor: Any, target: Any) -> List[dict]:
    """Target entries, then every donor entry whose strategy isn't already there.

    Dedupes on the exact ``strategy`` text — the key ``PlaybookStore.append``
    already uses, so a merged file looks like one the store wrote itself.
    """
    merged = (
        [e for e in target if isinstance(e, dict)] if isinstance(target, list) else []
    )
    seen = {e.get("strategy") for e in merged}
    if not isinstance(donor, list):
        return merged
    for entry in donor:
        if not isinstance(entry, dict):
            continue
        strategy = entry.get("strategy")
        if not strategy or strategy in seen:
            continue
        seen.add(strategy)
        merged.append(entry)
    return merged


def _unfork_one(donor: Path, target: Path) -> Unfork:
    """Carry one donor directory's components into ``target``."""
    record = Unfork(donor=donor.name, target=target.name)
    target.mkdir(parents=True, exist_ok=True)

    if (donor / _EPISODIC / _DB_NAME).is_file():
        if (target / _EPISODIC / _DB_NAME).is_file():
            copied = _merge_episodic(donor / _EPISODIC, target / _EPISODIC)
            if copied is None:
                record.blocked.append(_EPISODIC)
            else:
                record.episodes = copied
        else:
            _adopt(donor / _EPISODIC, target / _EPISODIC)
            record.moved.append(_EPISODIC)

    if (donor / _PLAYBOOK / _PLAYBOOK_FILE).is_file():
        if (target / _PLAYBOOK / _PLAYBOOK_FILE).is_file():
            added = _merge_playbook(donor / _PLAYBOOK, target / _PLAYBOOK)
            if added is None:
                record.blocked.append(_PLAYBOOK)
            else:
                record.strategies = added
        else:
            _adopt(donor / _PLAYBOOK, target / _PLAYBOOK)
            record.moved.append(_PLAYBOOK)

    if record.blocked:
        # Something here is NOT in the target, so the donor keeps its own name:
        # renaming it aside would leave real memory under a key nothing resolves
        # to. The re-run this allows costs a dedupe that copies nothing.
        return record
    if _holds_files(donor):
        record.kept_as = _rename_aside(donor)
    else:
        # Only emptied component dirs are left, so this deletes no content.
        shutil.rmtree(donor, ignore_errors=True)
    return record


def _adopt(src: Path, dst: Path) -> None:
    """Move a component's contents across, never over an existing entry (the
    caller has established that the file which makes it real is absent)."""
    dst.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src.iterdir()):
        destination = dst / entry.name
        if destination.exists():
            continue
        # shutil.move, not os.replace: a profile dir can sit on another
        # filesystem than the app home (a symlinked ~/.mnemoai).
        shutil.move(str(entry), str(destination))


def _merge_episodic(donor_path: Path, target_path: Path) -> Optional[int]:
    """Copy the donor's episodes into the target collection.

    Returns the number copied, or None if they could not be — an unreadable
    store, or one whose vectors aren't comparable — which is what tells the
    caller to leave the donor directory under its own name.
    """
    source = _collection(donor_path)
    destination = _collection(target_path)
    if source is None or destination is None:
        return None
    if not _comparable_vectors(source, destination):
        logger.info(
            "Episodes in %s were embedded by a different model; they stay where "
            "they are (their vectors aren't comparable to %s's).",
            donor_path.parent.name,
            target_path.parent.name,
        )
        return None

    stored = source.get(include=["embeddings", "metadatas"])
    ids = list(stored.get("ids") or [])
    vectors = stored.get("embeddings")
    metadatas = stored.get("metadatas") or []
    if vectors is None or len(vectors) < len(ids):
        return None
    if not ids:
        return 0

    have = set(destination.get(include=[]).get("ids") or [])
    batch_ids: List[str] = []
    batch_vectors: List[list] = []
    batch_metadatas: List[dict] = []
    copied = 0
    for i, episode_id in enumerate(ids):
        metadata = metadatas[i] if i < len(metadatas) else None
        # An episode IS its metadata (task, tools, outcome) — one without any is
        # nothing to recall, and chroma rejects an empty dict anyway.
        if episode_id in have or not isinstance(metadata, dict) or not metadata:
            continue
        batch_ids.append(episode_id)
        batch_vectors.append(_as_list(vectors[i]))
        batch_metadatas.append(_compact_metadata(metadata))
        if len(batch_ids) >= _BATCH:
            destination.add(
                ids=batch_ids, embeddings=batch_vectors, metadatas=batch_metadatas
            )
            copied += len(batch_ids)
            batch_ids, batch_vectors, batch_metadatas = [], [], []
    if batch_ids:
        destination.add(
            ids=batch_ids, embeddings=batch_vectors, metadatas=batch_metadatas
        )
        copied += len(batch_ids)
    return copied


def _merge_playbook(donor_path: Path, target_path: Path) -> Optional[int]:
    """Append the donor's unseen strategies to the target's playbook.

    Returns how many were added, or None if either file couldn't be read.
    """
    target_file = target_path / _PLAYBOOK_FILE
    existing = _read_entries(target_file)
    donor = _read_entries(donor_path / _PLAYBOOK_FILE)
    if donor is None or existing is None:
        return None
    merged = merge_playbook_entries(donor, existing)
    added = len(merged) - len(existing)
    if added <= 0:
        return 0
    atomic_write_json(str(target_file), merged)
    return added


def _read_entries(path: Path) -> Optional[List[dict]]:
    """A playbook file's entries, or None if it can't be read as a list of them.

    An empty list is not the same answer as an unreadable file: one has nothing
    to carry, the other is content we must not treat as absent.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            entries = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(entries, list):
        return None
    return [e for e in entries if isinstance(e, dict)]


def _collection(store_path: Path):
    """The episodic collection at ``store_path``, or None if it has none."""
    try:
        client = chromadb.PersistentClient(path=str(store_path))
        return client.get_collection(name=_COLLECTION)
    except Exception:
        logger.debug("No episodic collection at %s", store_path, exc_info=True)
        return None


def _comparable_vectors(source, destination) -> bool:
    """True when both collections' vectors come from the same embedding model.

    Prefers the app's own stamp (``embed_fingerprint``); for a legacy unstamped
    collection the dimension is all there is to go on, and an empty collection
    has nothing to disagree about.
    """
    source_fp = (source.metadata or {}).get("embed_fingerprint")
    target_fp = (destination.metadata or {}).get("embed_fingerprint")
    if source_fp and target_fp:
        return source_fp == target_fp
    source_dim, target_dim = _dimension(source), _dimension(destination)
    return source_dim is None or target_dim is None or source_dim == target_dim


def _dimension(collection) -> Optional[int]:
    """The stored vector width, or None for an empty/unreadable collection."""
    try:
        if collection.count() == 0:
            return None
        vectors = collection.peek(1).get("embeddings")
        if vectors is None or len(vectors) == 0:
            return None
        return len(vectors[0])
    except Exception:
        return None


def _compact_metadata(metadata: dict) -> dict:
    """The metadata as it should land: the donor's ``tools`` value in the shape
    the current writer uses, so a pre-1.22 payload isn't copied forward."""
    if not isinstance(metadata.get("tools"), str):
        return dict(metadata)
    carried = dict(metadata)
    carried["tools"] = compact_tools(carried["tools"])
    return carried


def _as_list(vector) -> list:
    """A stored vector as a plain list (chroma hands back numpy)."""
    return vector.tolist() if hasattr(vector, "tolist") else list(vector)


def _holds_files(path: Path) -> bool:
    """True if any file remains anywhere under ``path``."""
    try:
        return any(p.is_file() for p in path.rglob("*"))
    except OSError:
        return True


def _rename_aside(donor: Path) -> Optional[str]:
    """Rename the donor out of the way, so the merge doesn't re-run. Returns the
    new name — the copy is kept, not deleted, and the storage compaction shrinks
    it later in this startup like any other store."""
    stamp = datetime.now().strftime("%Y%m%d")
    for attempt in range(1, 10):
        suffix = "" if attempt == 1 else f"-{attempt}"
        candidate = donor.with_name(f"{donor.name}.merged-{stamp}{suffix}")
        if not candidate.exists():
            shutil.move(str(donor), str(candidate))
            return candidate.name
    return None


def _describe(record: Unfork) -> str:
    """The one startup line for a folded directory."""
    parts = []
    if record.moved:
        parts.append("moved " + " + ".join(record.moved))
    if record.episodes:
        parts.append(f"carried {record.episodes} episodes")
    if record.strategies:
        parts.append(f"carried {record.strategies} strategies")
    # Mutually exclusive: a donor holding something we couldn't carry is never
    # renamed, so at most one of these tails applies.
    kept = f"; kept the old copy as {record.kept_as}" if record.kept_as else ""
    if record.blocked:
        kept = f"; {' + '.join(record.blocked)} stays in {record.donor}"
    return (
        f"Memory of {record.donor} folded into {record.target} "
        f"({', '.join(parts)}) — a cross-region routing prefix is not a "
        f"different model{kept}"
    )
