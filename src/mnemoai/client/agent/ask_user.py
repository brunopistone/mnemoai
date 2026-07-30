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
"""

from typing import Any, List, Optional, Tuple

# Keep the picker usable and the option labels renderable on one row.
MAX_OPTIONS = 8
MAX_LABEL_CHARS = 120
MAX_QUESTION_CHARS = 400


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


def format_answer(choice: str) -> str:
    """The ToolMessage content for an answered question."""
    return (
        f'The user chose: "{choice}"\n\n'
        "Proceed on that basis. Don't re-ask or second-guess the choice."
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
        choice = ui(text, opts)
    except Exception:
        # A dialog failure must not kill the turn — the model can carry on.
        choice = None
    finally:
        if was_active:
            agent._start_spinner(prev_label)

    if choice is None:
        return format_dismissed()
    return format_answer(str(choice))
