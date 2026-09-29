"""User-only playbook inspection and revision-checked edits; never a model tool."""

import json
import shlex

from mnemoai.client.memory import playbook_records, retraction

USAGE = (
    "Usage: /learned [inspect|preview|edit|disable|restore|helpful|unhelpful <id> | "
    "retract <id> <reason> | retracted | on|off|clear]. "
    "Restoring a retraction also requires a reason. IDs may be an unambiguous prefix."
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


def render_impact(plan):
    target = plan["target"]
    lines = [
        f"{plan['action'].capitalize()} {_plain(target['id'])} · revision {target['revision']}",
        f"Scope: {_plain(target['scope']) or 'unknown (legacy)'}",
        f"Context: {_plain(target['context'])}",
        f"Note: {_plain(target['strategy'])}",
        f"Reason: {_plain(plan['reason']) or '(supply a reason when applying)'}",
        f"Resulting status: {plan['result_status']}",
    ]
    if plan["action"] == "retract":
        lines.append(
            f"Quarantine {plan['newly_quarantined']} recorded observations from future "
            "automatic learning in this scope."
        )
    else:
        lines.append(
            f"Release {plan['released_sources']} observations; "
            f"{plan['sources_still_quarantined']} remain quarantined by other retractions."
        )
    if plan["lineage_unknown"] or plan["unverifiable_sources"]:
        lines.append(
            "Incomplete provenance: suppression of unlinked/paraphrased future lessons "
            "cannot be guaranteed."
        )
    lines.append(
        f"{len(plan['related'])} other notes share evidence; shared evidence is NOT a "
        "proven dependency. Review them manually:"
    )
    for related in plan["related"][:10]:
        lines.append(f"  {_plain(related['id'])} · {related['status']} (unchanged)")
    if len(plan["related"]) > 10:
        lines.append(f"  … {len(plan['related']) - 10} more; inspect the store for all IDs.")
    lines.append(f"All {plan['other_entries_unchanged']} other entries remain unchanged.")
    lines.append(
        "This changes playbook injection/automatic learning only, not conversation history, "
        "episodic recall, MEMORY.md, permissions, or actions already taken/sent."
    )
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
        head = arguments.split(maxsplit=2)
        if len(head) == 3 and head[0].lower() in {"retract", "restore"}:
            # Reasons are prose, not shell syntax: don't reject or remove
            # apostrophes in "the provider's failure isn't permanent".
            reason = head[2]
            if len(reason) >= 2 and reason[0] in {"'", '"'} and reason[-1] == reason[0]:
                reason = reason[1:-1]
            parts = shlex.split(" ".join(head[:2])) + [reason]
        else:
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
        if action == "retracted" and len(parts) == 1:
            return render([e for e in entries if e["status"] == "retracted"],
                          store.playbook_file, store.enabled)
        if action == "clear" and len(parts) == 1:
            withdrawn = sum(e["status"] == "retracted" for e in entries)
            notice = (
                f" This also removes {withdrawn} retractions and their learning quarantines; "
                "old sources may be learned from again." if withdrawn else ""
            )
            if confirm(f"Clear all {len(entries)} playbook entries? A backup will be kept.{notice}"):
                store.clear(expected=[(e["id"], e["revision"]) for e in entries])
                client.refresh_playbook_context()
                return "Playbook cleared; a backup is beside playbook.json."
            return "Cancelled; playbook unchanged."
        if len(parts) < 2 or action not in {
            "inspect", "preview", "retract", "edit", "disable", "restore", "helpful", "unhelpful",
        }:
            return USAGE
        matches = [e for e in entries if e["id"].startswith(parts[1])]
        if len(matches) != 1:
            return "ID is missing or ambiguous; use the full ID from /learned."
        entry = matches[0]
        if action == "preview" and len(parts) == 2:
            operation = "restore" if entry["status"] == "retracted" else "retract"
            return render_impact(store.preview_retraction(entry["id"], operation))
        if action == "preview":
            return USAGE
        if action == "retract" or (action == "restore" and entry["status"] == "retracted"):
            reason = retraction.reason_text(" ".join(parts[2:]))
            plan = store.preview_retraction(entry["id"], action, reason)
            if not confirm(render_impact(plan) + "\nApply this exact change?"):
                return "Cancelled; playbook unchanged."
            changed = store.apply_retraction(
                entry["id"], action=action, reason=reason, token=plan["token"],
            )
            client.refresh_playbook_context()
            return (
                f"Updated {_plain(entry['id'])}: {changed['status']}. History and backups retained; "
                "requests already sent are not undone."
            )
        if len(parts) != 2:
            return USAGE
        if action == "inspect":
            return json.dumps(entry, indent=2, ensure_ascii=True)
        if entry["status"] == "retracted":
            return "Restore the retraction with a reason before editing or rating this note."
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
