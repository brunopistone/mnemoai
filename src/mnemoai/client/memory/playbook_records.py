"""Versioned, evidence-linked playbook records; legacy fields are preserved."""

import copy
import json
import math
import uuid
from datetime import datetime, timezone

STATUSES = frozenset({"active", "disabled", "archived"})
COUNTERS = (
    "injection_count", "observed_success_count", "observed_failure_count",
    "helpful_count", "unhelpful_count",
)
CONFIDENCE_FLOOR = 0.2


class PlaybookEntry:
    """An extracted lesson plus the observed sources that support it."""

    def __init__(self, context, strategy, source, outcome="success", tools=None,
                 confidence=0.5, *, source_refs=None, scope="", provenance="legacy"):
        self.context, self.strategy, self.source = context, strategy, source
        self.outcome, self.tools, self.confidence = outcome, tools or [], confidence
        self.source_refs, self.scope = source_refs or [], scope
        self.provenance = provenance
        self.timestamp = now()
        self.id = "mem-" + uuid.uuid4().hex

    def to_dict(self):
        return normalize(vars(self))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(entry: dict) -> dict:
    """Add missing metadata without claiming provenance for historical records."""
    if not isinstance(entry, dict) or not all(
        isinstance(entry.get(key), str) for key in ("context", "strategy")
    ):
        raise ValueError("Invalid playbook entry; original file left untouched")
    out = copy.deepcopy(entry)
    out.setdefault("id", "mem-" + uuid.uuid4().hex)
    out.setdefault("revision", 1)
    out.setdefault("status", "active")
    out.setdefault("provenance", "legacy")
    out.setdefault("scope", "")
    out.setdefault("source_refs", [])
    out.setdefault("history", [])
    out.setdefault("confidence", 0.5)
    out.setdefault("tools", [])
    out.setdefault("outcome", "success")
    for key in COUNTERS:
        out.setdefault(key, 0)
    if (
        not isinstance(out["id"], str) or not out["id"].startswith("mem-")
        or type(out["revision"]) is not int or out["revision"] < 1
        or out["status"] not in STATUSES
        or not isinstance(out["scope"], str)
        or not isinstance(out["source_refs"], list)
        or not all(isinstance(ref, dict) for ref in out["source_refs"])
        or not isinstance(out["history"], list)
        or not all(isinstance(event, dict) for event in out["history"])
        or not isinstance(out["tools"], list)
        or not all(isinstance(tool, str) for tool in out["tools"])
        or not isinstance(out["confidence"], (float, int))
        or not math.isfinite(out["confidence"])
        or any(type(out[key]) is not int or out[key] < 0 for key in COUNTERS)
    ):
        raise ValueError("Invalid playbook metadata; original file left untouched")
    return out


def key(entry: dict) -> tuple:
    """Text similarity alone must never merge different applicability scopes."""
    return tuple(entry.get(field, "") for field in ("scope", "context", "strategy"))


def ref_key(ref: dict) -> str:
    return json.dumps(ref, sort_keys=True, ensure_ascii=True)


def eligible(entry: dict) -> bool:
    feedback = entry.get("helpful_count", 0) + entry.get("unhelpful_count", 0)
    return (
        entry.get("status", "active") == "active"
        and (not feedback or entry.get("confidence", 0.5) >= CONFIDENCE_FLOOR)
    )


def checkpoint(entry: dict, action: str) -> None:
    """Keep edited text and state recoverable without recursively copying history."""
    entry["history"].append({
        "revision": entry["revision"], "at": now(), "action": action,
        **{field: copy.deepcopy(entry.get(field)) for field in (
            "context", "strategy", "scope", "status", "provenance",
        )},
    })
    entry["revision"] += 1
