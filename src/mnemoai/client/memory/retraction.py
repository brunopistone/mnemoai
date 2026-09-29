"""Explicit note withdrawal and scoped evidence quarantine, not inferred causality."""

import copy
import hashlib
import json
import re
from datetime import datetime

PRIOR_STATUSES = frozenset({"active", "disabled", "archived"})
_OBS_ID = re.compile(r"obs-[a-f0-9]{64}")
_HASH = re.compile(r"[a-f0-9]{64}")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def reason_text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 500:
        raise ValueError("A reason of 1–500 characters is required")
    if any(not c.isprintable() and not c.isspace() for c in value):
        raise ValueError("Control characters are not allowed in the reason")
    return " ".join(value.split())


def source_id(ref):
    """Identify a recorded observation, never guess identity from similar text."""
    if not isinstance(ref, dict):
        return None
    session, turn = ref.get("session_id"), ref.get("turn")
    if not isinstance(session, str) or not session.strip() or type(turn) is not int or turn < 1:
        return None
    call = ref.get("tool_call_id")
    if isinstance(call, str) and call.strip():
        return "obs-" + _digest([session, turn, "call", call])
    tool, args, result = (ref.get(key) for key in ("tool", "args_sha256", "result_sha256"))
    if (
        isinstance(tool, str) and tool
        and isinstance(args, str) and _HASH.fullmatch(args)
        and isinstance(result, str) and _HASH.fullmatch(result)
    ):
        return "obs-" + _digest([session, turn, "content", tool, args, result])
    return None


def sources(entry):
    return {identity for ref in entry.get("source_refs", []) if (identity := source_id(ref))}


def validate(entry):
    """A malformed withdrawal cannot silently become an active memory."""
    meta = entry.get("retraction")
    if meta is None and entry.get("status") != "retracted":
        return
    if not isinstance(meta, dict):
        raise ValueError("Retraction metadata is missing or invalid")
    ids = meta.get("source_ids")
    if (
        not isinstance(meta.get("id"), str) or not meta["id"].startswith("ret-")
        or meta.get("actor") != "user"
        or meta.get("previous_status") not in PRIOR_STATUSES
        or meta.get("scope") != entry.get("scope")
        or type(meta.get("withdrawn_revision")) is not int
        or not 0 < meta["withdrawn_revision"] < entry["revision"]
        or not isinstance(ids, list)
        or any(not isinstance(i, str) or not _OBS_ID.fullmatch(i) for i in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Invalid retraction metadata")
    reason_text(meta.get("reason"))
    try:
        datetime.fromisoformat(meta["at"])
        if "restored_at" in meta:
            datetime.fromisoformat(meta["restored_at"])
            reason_text(meta.get("restore_reason"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid retraction audit record") from exc
    if entry.get("status") == "retracted":
        expected = sources(entry) if entry.get("scope") else set()
        if set(ids) != expected or "restored_at" in meta:
            raise ValueError("Retraction evidence does not match the withdrawn entry")
    elif "restored_at" not in meta:
        raise ValueError("A retraction can only be reversed with a restore record")


def quarantined(entries, scope):
    if not scope:
        return set()  # an unknown scope is not permission for a global quarantine
    return {
        identity for entry in entries
        if entry.get("status") == "retracted" and entry.get("scope") == scope
        for identity in entry["retraction"]["source_ids"]
    }


def allows_sources(refs, blocked):
    if not blocked:
        return True
    ids = [source_id(ref) for ref in refs]
    # When a scope has withdrawals, unverifiable evidence is not a fresh source.
    return bool(ids) and all(identity is not None and identity not in blocked for identity in ids)


def preview(entries, entry_id, action, reason=""):
    target = next((entry for entry in entries if entry["id"] == entry_id), None)
    if target is None:
        raise ValueError("Entry no longer exists")
    if action not in {"retract", "restore"}:
        raise ValueError("Unknown lifecycle action")
    if action == "retract" and target["status"] == "retracted":
        raise ValueError("Entry is already retracted")
    if action == "restore" and target["status"] != "retracted":
        raise ValueError("Entry has no active retraction")
    reason = reason_text(reason) if reason else ""
    scope, identities = target["scope"], sources(target)
    before = quarantined(entries, scope)
    others = [entry for entry in entries if entry["id"] != entry_id]
    after = before | identities if action == "retract" and scope else quarantined(others, scope)
    related = [
        {"id": entry["id"], "revision": entry["revision"], "status": entry["status"]}
        for entry in others
        if scope and entry.get("scope") == scope and identities.intersection(sources(entry))
    ]
    fields = (
        "id", "revision", "status", "scope", "context", "strategy", "source_refs",
        "retraction", "confidence", "helpful_count", "unhelpful_count",
    )
    manifest = sorted(
        [{key: entry.get(key) for key in fields} for entry in entries], key=lambda e: e["id"]
    )
    return {
        "entry_id": entry_id, "revision": target["revision"], "action": action, "reason": reason,
        "target": copy.deepcopy(target), "source_ids": sorted(identities if scope else []),
        "newly_quarantined": len(after - before), "released_sources": len(before - after),
        "sources_still_quarantined": len(identities & after) if action == "restore" else 0,
        "related": related, "other_entries_unchanged": len(others),
        "unverifiable_sources": sum(source_id(ref) is None for ref in target["source_refs"]),
        "lineage_unknown": not scope or not identities,
        "result_status": "retracted" if action == "retract" else target["retraction"]["previous_status"],
        "token": _digest({"entries": manifest, "id": entry_id, "action": action, "reason": reason}),
    }
