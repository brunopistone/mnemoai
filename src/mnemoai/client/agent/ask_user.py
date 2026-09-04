"""Model-initiated multiple-choice question (``ask_user_question``) — pure logic
plus the agent-arg driver.

The model calls ``ask_user_question`` when a decision is genuinely the user's to
make; the call is intercepted client-side (the MCP server is a piped subprocess
and can't prompt the terminal) and answered through a pinned-UI picker. Same
thin-stub + client-intercept split as ``exit_plan_mode`` / ``use_skill``.

Split like ``confirmation_gate``: the validation/formatting is pure and unit
tested, and the one impure function takes the agent as its first arg and
dispatches back through its methods (``_is_headless``, ``_question_ui``, the
spinner helpers) so a test overriding one on a ``__new__`` stub still intercepts.

**Every path returns a string** — a question that cannot be asked (headless
sub-agent, no TTY, malformed args) resolves to model-facing guidance rather than
blocking, because a prompt nobody can see would hang the turn forever.

**The offered options are never the whole answer.** A picker that only accepts
one of N rows forces the user to pick the closest wrong one whenever the real
answer is "neither, because…" — so every question also carries a free-text
**note** (rides along with whatever is picked) and an always-present escape row
that hands the decision back as a conversation instead of a choice. Those are
three distinct outcomes, not two: a chosen option, a refusal to choose (talk
about it), and a dismissal (decide for me) — see :func:`normalize_reply`.
"""

from typing import Any, List, Optional, Tuple

# Keep the picker usable and the option labels renderable on one row.
MAX_OPTIONS = 8
MAX_LABEL_CHARS = 120
MAX_QUESTION_CHARS = 400
# The note is prose the model reads, not a label to render, so it is far more
# generous than an option — but still bounded: it lands in the tool result.
MAX_NOTE_CHARS = 2000

# The escape row's value. Never a model-supplied label (picker_rows drops a
# colliding option), so identifying it can't depend on the wording below.
DISCUSS = "__mnemoai_discuss__"
DISCUSS_LABEL = "None of these — let's talk about it"


def normalize_options(options: Any) -> List[str]:
    """Coerce the ``options`` arg into clean, deduped, capped labels.

    Tolerates the shapes small models produce: a single string instead of a list,
    a list of dicts carrying a ``label``/``text``/``value`` key, ``None`` entries.
    Order is preserved (the picker's first row is the model's first option).
    """
    if options is None:
        return []
    if isinstance(options, str):
        options = [options]
    if isinstance(options, dict):
        # A mapping is ambiguous; its values read better as labels than its keys.
        options = list(options.values())
    if not isinstance(options, (list, tuple)):
        return []

    seen, out = set(), []
    for raw in options:
        if raw is None:
            continue  # str(None) would render as a literal "None" row
        if isinstance(raw, dict):
            raw = raw.get("label") or raw.get("text") or raw.get("value") or ""
        label = " ".join(str(raw).split())  # collapse newlines: one row per option
        if not label:
            continue
        if len(label) > MAX_LABEL_CHARS:
            label = label[: MAX_LABEL_CHARS - 1] + "…"
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(label)
        if len(out) == MAX_OPTIONS:
            break
    return out


def validate(question: Any, options: Any) -> Tuple[str, List[str], Optional[str]]:
    """Return ``(question, options, error)``; ``error`` is model-facing guidance.

    A malformed call must never reach the UI — it would show the user an empty or
    single-row picker for a question the model could have just typed.
    """
    text = " ".join(str(question or "").split())
    if len(text) > MAX_QUESTION_CHARS:
        text = text[: MAX_QUESTION_CHARS - 1] + "…"
    opts = normalize_options(options)

    if not text:
        return "", opts, (
            "ask_user_question needs a non-empty question. Ask the user directly "
            "in your reply instead."
        )
    if len(opts) < 2:
        return text, opts, (
            "ask_user_question needs at least 2 distinct options. Either offer "
            "real alternatives, or just ask the question in your reply."
        )
    return text, opts, None


def normalize_note(note: Any) -> str:
    """Coerce the free-text note into one bounded line of model-facing prose."""
    if note is None:
        return ""
    text = " ".join(str(note).split())
    if len(text) > MAX_NOTE_CHARS:
        text = text[: MAX_NOTE_CHARS - 1] + "…"
    return text


def picker_rows(options: List[str]) -> List[Tuple[str, str]]:
    """``(value, label)`` rows for the picker: the options plus the escape row.

    The escape row is appended HERE rather than by the UI so "there is always a
    way out of the choice" is one testable fact instead of a dialog detail. A
    model option that collides with the sentinel is dropped — the row's identity
    must not depend on its wording.
    """
    rows = [(o, o) for o in options if o != DISCUSS]
    rows.append((DISCUSS, DISCUSS_LABEL))
    return rows


def picker_reply(value: Any, note: Any) -> Optional[Tuple[Optional[str], str]]:
    """Map a picked row + the typed note into the ``question_ui`` return shape.

    A cancel (``value is None``) stays ``None`` — a dismissal even when a note was
    typed: the note accompanies an answer, it is not one on its own.
    """
    if value is None:
        return None
    text = normalize_note(note)
    if value == DISCUSS:
        return None, text
    return str(value), text


def normalize_reply(reply: Any) -> Tuple[Optional[str], str, bool]:
    """Read the UI's answer as ``(choice, note, answered)``.

    Three accepted shapes, so the picker could grow a note without breaking the
    old contract: ``None`` is a dismissal, a bare string is that option chosen
    (what the picker returned before the note existed), and a ``(choice, note)``
    pair is a submitted answer — with ``choice is None`` meaning the user took
    the escape row rather than any option.
    """
    if reply is None:
        return None, "", False
    if isinstance(reply, (list, tuple)):
        raw = reply[0] if reply else None
        note = normalize_note(reply[1] if len(reply) > 1 else "")
        choice = " ".join(str(raw).split()) if raw is not None else ""
        return (choice or None), note, True
    choice = " ".join(str(reply).split())
    # An empty string is nothing chosen and nothing said: a dismissal.
    return (choice or None), "", bool(choice)


def format_answer(choice: str, note: str = "") -> str:
    """The ToolMessage content for an answered question."""
    said = f"\nThey added: {note}" if note else ""
    return (
        f'The user chose: "{choice}"{said}\n\n'
        "Proceed on that basis. Don't re-ask or second-guess the choice."
    )


def format_discussion(note: str = "") -> str:
    """The ToolMessage content when the user refused all the offered options.

    Deliberately NOT the dismissed wording: dismissing means "decide for me",
    while taking the escape row means the opposite — the user wants to settle it
    in conversation, so proceeding on any option would override them.
    """
    if note:
        return (
            "The user picked none of the options and answered in their own "
            f"words instead:\n{note}\n\n"
            "That is the answer — don't proceed on any of the options you "
            "offered, and don't call this tool again for the same decision. "
            "Reply to what they wrote."
        )
    return (
        "The user picked none of the options and wants to talk it through. Don't "
        "proceed on any of them and don't call this tool again for the same "
        "decision. Reply in prose: say what the real trade-off is and ask what "
        "they'd prefer."
    )


def format_dismissed() -> str:
    """The ToolMessage content when the user closed the picker without choosing."""
    return (
        "The user dismissed the question without choosing. Do NOT ask again — "
        "continue with your own best judgment, and say which assumption you made."
    )


def format_unavailable(reason: str) -> str:
    """The ToolMessage content when there is no one to ask."""
    return (
        f"You cannot ask the user a question here ({reason}). Decide yourself "
        "using your best judgment and state the assumption you made."
    )


def ask(agent, question: Any, options: Any) -> str:
    """Put a multiple-choice question to the user; return the model-facing result.

    Blocks the calling (worker) thread while the picker is up, then hands the
    spinner back exactly as it was — the tool loop doesn't restart it on this
    client-side path, so without that the terminal would sit at a dead `>` for
    the rest of the turn.

    Refuses, rather than prompting, whenever nobody would see the prompt:
    - inside ANY sub-agent (``_spawn_depth``): its contract is to return a report
      to the parent, and a headless/background one has no terminal at all — a
      picker there would block a daemon thread on a prompt that never paints;
    - with no ``_question_ui`` hook (non-TTY, piped, tests): a plain full-screen
      dialog can't run there, and blocking a scripted run is worse than guessing.
    """
    text, opts, error = validate(question, options)
    if error:
        return error

    if getattr(agent, "_spawn_depth", 0) > 0 or agent._is_headless():
        return format_unavailable("you are a sub-agent with no direct user")

    ui = getattr(agent, "_question_ui", None)
    if ui is None:
        return format_unavailable("this session is not interactive")

    was_active, prev_label = agent._spinner_snapshot()
    agent._stop_spinner()
    try:
        reply = ui(text, opts)
    except Exception:
        # A dialog failure must not kill the turn — the model can carry on.
        reply = None
    finally:
        if was_active:
            agent._start_spinner(prev_label)

    choice, note, answered = normalize_reply(reply)
    if not answered:
        return format_dismissed()
    if choice is None:
        return format_discussion(note)
    return format_answer(choice, note)
