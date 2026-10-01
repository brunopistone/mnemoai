"""Compact review chrome, immutable detail snapshots, and actor answer labels."""

from dataclasses import dataclass

from prompt_toolkit.utils import get_cwidth

from mnemoai.utils.formatting.code_formatter import CodeFormatter

GRAY = "\033[90m"
RESET = "\033[0m"


def plain(value):
    """Keep readable text without letting report data control the terminal."""
    return "".join(c for c in str(value) if c.isprintable() or c in "\n\t")


def clipped(value, width):
    text = " ".join(plain(value).split())
    if get_cwidth(text) <= width:
        return text
    result = ""
    for char in text:
        if get_cwidth(result + char) > max(0, width - 1):
            break
        result += char
    return result.rstrip() + ("…" if width > 0 else "")


@dataclass(frozen=True)
class ReviewDisplay:
    """Publish strings atomically; painting must never inspect files or call Git."""

    key: tuple
    label: str
    summary: str
    details: str


def current_display(reviewer):
    display = getattr(reviewer, "view", None)
    key = (id(reviewer), getattr(reviewer, "_generation", None))
    return display if isinstance(display, ReviewDisplay) and display.key == key else None


def snapshot(key, phase, report=None, revision=0, details=""):
    report = report or {}
    if phase == "waiting":
        status = "waiting for chat model"
    elif phase == "reviewing":
        status = "checking answer"
    elif phase.startswith("reviewing_"):
        status = "checking " + phase.removeprefix("reviewing_")
    elif phase == "correcting":
        status = f"chat model revising · round {revision}"
    elif phase == "strategy":
        status = "chat model preparing strategy"
    elif report.get("unavailable"):
        status = "unavailable"
    else:
        status = plain(report.get("verdict", "inconclusive"))
        findings = len(report.get("findings") or [])
        if findings:
            status += f" · {findings} finding{'s' if findings != 1 else ''}"
        if report.get("coverage_gaps"):
            status += " · limited coverage"
    checkpoint = report.get("checkpoint", "completion")
    label = f"Peer review · {checkpoint} · {status}" if checkpoint != "completion" else f"Peer review · {status}"
    summary = plain(report.get("summary", ""))
    body = (
        "Peer review details · snapshot (work continues while this pane is open)\n"
        "Use /review last to recheck whether inspected artifacts have changed.\n\n"
        + (plain(details) if details else label)
    )
    return ReviewDisplay(key, label, summary, body)


def compact(display, width=80):
    """At most two grey lines; full evidence and findings stay in the detail pane."""
    width = max(1, width)
    lines = [clipped(display.label + " · /review last", width)]
    if display.summary:
        lines.append(clipped("  ↳ " + display.summary, width))
    return "\n".join(f"{GRAY}{line}{RESET}" for line in lines)


def draft_prefix():
    return f"{GRAY}Chat model · draft (review pending){RESET}\n"


def final_answer(answer, report):
    """Make the actor's chosen answer distinct from review advice and prior drafts."""
    if not isinstance(answer, str) or not answer.strip():
        return ""
    verdict = (report or {}).get("verdict", "inconclusive")
    if (report or {}).get("unavailable"):
        # The once-per-outage notice is emitted separately; don't nag on every
        # answer. Normal actor output does not claim that review passed.
        header, note = "Final answer · chat model", ""
    elif verdict == "pass":
        header = "Final answer · chat model"
        note = "Peer review passed in its inspected scope; not proof of correctness."
    elif verdict == "revise":
        header = "Chat model answer · unresolved review findings"
        note = "The review requested changes; do not treat this as verified completion."
    else:
        header = "Chat model answer · review incomplete"
        note = "This is the latest answer, not a verified completion."
    return (
        f"\033[1;36m{header}{RESET}\n"
        + (f"{GRAY}{note}{RESET}\n" if note else "") + "\n"
        + CodeFormatter.render_to_string(answer)
    )
