"""User-only playbook inspection and revision-checked edits; never a model tool."""

import json
import shlex

from mnemoai.client.memory import playbook_records

USAGE = (
    "Usage: /learned [inspect|edit|disable|restore|helpful|unhelpful <id> | "
    "on|off|clear]. IDs may be an unambiguous prefix."
)


def _plain(value):
    return " ".join("".join(c for c in str(value) if c.isprintable() or c.isspace()).split())


def render(entries, path, enabled=True, error=None):
    lines = [
        "Learned tool-use notes",
        f"Injection: {'on' if enabled else 'off for this session'}",
        f"Store: {_plain(path)}",
    ]
    if error:
        lines.append("Last reflection: " + _plain(error))
    if not entries:
        lines.append("No entries yet.")
    for entry in entries[:30]:
        status = entry["status"]
        if status == "active" and not playbook_records.eligible(entry):
            status = "dormant (negative feedback)"
        lines.append(
            f"  {_plain(entry['id'])} · {status} · {_plain(entry['provenance'])}\n"
            f"    [{_plain(entry['context'])[:200]}] {_plain(entry['strategy'])[:240]}"
        )
    if len(entries) > 30:
        lines.append(f"  … {len(entries) - 30} more entries; inspect by ID.")
    lines.extend([
        "Model-inferred and legacy notes are not proven strategies.", USAGE,
        "Settings: /config playbook · Model: /model · Enable/disable learning: /features",
    ])
    return "\n".join(lines)


def run(client, arguments, *, confirm, edit):
    store = getattr(client, "playbook", None)
    if store is None:
        error = getattr(client, "_playbook_error", None)
        if error:
            return (
                f"Playbook unavailable: {_plain(error)}. Stored data was not cleared. "
                "Fix storage access and restart to retry initialization."
            )
        return "Playbook is disabled. Enable it with /features to use /learned."
    try:
        parts = shlex.split(arguments)
        if not parts:
            return render(store.snapshot(), store.playbook_file, store.enabled,
                          getattr(getattr(client, "reflector", None), "last_error", None))
        action = parts[0].lower()
        if action in {"on", "off"} and len(parts) == 1:
            store.enabled = action == "on"
            client.refresh_playbook_context()
            return f"Playbook injection {action} for this session; stored entries unchanged."
        entries = store.snapshot()
        if action == "clear" and len(parts) == 1:
            if confirm(f"Clear all {len(entries)} playbook entries? A backup will be kept."):
                store.clear(expected=[(e["id"], e["revision"]) for e in entries])
                client.refresh_playbook_context()
                return "Playbook cleared; a backup is beside playbook.json."
            return "Cancelled; playbook unchanged."
        if len(parts) != 2 or action not in {
            "inspect", "edit", "disable", "restore", "helpful", "unhelpful",
        }:
            return USAGE
        matches = [e for e in entries if e["id"].startswith(parts[1])]
        if len(matches) != 1:
            return "ID is missing or ambiguous; use the full ID from /learned."
        entry = matches[0]
        if action == "inspect":
            return json.dumps(entry, indent=2, ensure_ascii=True)
        changed = {}
        if action == "edit":
            original = json.dumps(
                {k: entry[k] for k in ("context", "strategy")}, indent=2, ensure_ascii=True,
            )
            edited = edit(original)
            if edited == original:
                return "No changes made."
            changed = json.loads(edited)
            if not isinstance(changed, dict) or set(changed) != {"context", "strategy"}:
                return "Edit must contain only context and strategy; nothing changed."
        preview = json.dumps(changed or {"action": action}, ensure_ascii=True)
        if not confirm(
            f"{_plain(entry['id'])} revision {entry['revision']}: {preview}\nApply this change?"
        ):
            return "Cancelled; playbook unchanged."
        store.update(entry["id"], entry["revision"], action=action, **changed)
        client.refresh_playbook_context()
        return f"Updated {_plain(entry['id'])}. Use /learned inspect {_plain(entry['id'])} for its history."
    except (ValueError, OSError) as e:
        return f"Playbook operation failed: {_plain(e)}. No success is assumed."
