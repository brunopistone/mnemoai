"""Interactive configurator (first-run setup + /config, /model, /params).

Walks the user through creating/editing ``<app_home>/config.yaml`` from a
provider template. Edits are line-targeted (never a YAML round-trip) so the
templates' comments and prompt blocks are preserved.
"""

import getpass
import re
import sys
from pathlib import Path
from typing import Optional

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.widgets import Button, Dialog, Label, RadioList, TextArea

from mnemoai.models.provider_params import (
    providers,
    supported_keys,
    tunable_params,
)
from mnemoai.utils.config import Config
from mnemoai.utils.console import print_error
from mnemoai.utils.paths import config_path

# Importing readline (stdlib) is enough to give the built-in input() proper
# line-editing: arrow keys, history, and backspace work instead of leaking raw
# escape sequences (e.g. ^[[D) into the typed value. Guarded because readline
# is absent on some platforms (e.g. stock Windows).
try:
    import readline  # noqa: F401
except ImportError:
    pass

# Provider key -> (template filename, human label, default chat model)
_PROVIDERS = {
    "1": ("ollama", "config.yaml.example", "Ollama (local models)", "qwen3.5:4b"),
    "2": ("bedrock", "config.yaml.bedrock.example", "AWS Bedrock", "global.anthropic.claude-opus-4-8"),
    "3": ("mantle", "config.yaml.bedrock.mantle.example", "AWS Bedrock Mantle", "qwen.qwen3-32b"),
    # OpenAI / SageMaker / LiteLLM reuse the base template and transform its
    # model sections for the chosen provider (set TYPE, prune Ollama-only keys,
    # prompt provider-specific connection keys).
    "4": ("openai", "config.yaml.example", "OpenAI", "gpt-5-mini"),
    "5": ("anthropic", "config.yaml.example", "Anthropic (Claude API)", "claude-opus-4-8"),
    "6": ("sagemaker", "config.yaml.example", "Amazon SageMaker AI", "your-endpoint-name"),
    "7": ("litellm", "config.yaml.example", "LiteLLM (100+ providers)", "openai/your-model"),
}

# Human-facing menu label per provider TYPE (stored value is the canonical key).
_PROVIDER_LABELS = {
    "ollama": "ollama",
    "bedrock": "bedrock",
    "mantle": "bedrock-mantle",
    "openai": "openai",
    "anthropic": "anthropic",
    "sagemaker": "sagemaker",
    "litellm": "litellm",
}

# Mantle API protocol choice -> (value, description). Mirrors
# models.mantle_factory.VALID_PROTOCOLS.
_MANTLE_PROTOCOLS = {
    "1": ("chat_completions", "OpenAI Chat Completions (/v1) — most models"),
    "2": ("responses", "OpenAI Responses (/openai/v1) — e.g. openai.gpt-5.x"),
    "3": ("anthropic", "Anthropic Messages (/anthropic) — Claude models"),
}

def _templates_dir() -> Path:
    """Directory holding the packaged config templates (next to this module)."""
    return Path(__file__).resolve().parent


def config_exists() -> bool:
    """True if a config is already resolvable (so first-run setup is skipped)."""
    return Config._resolve_config_path() is not None


def _set_in_section(text: str, section: str, key: str, value: str) -> str:
    """Replace the first indented ``key`` inside top-level ``section`` in place."""
    out = []
    in_section = False
    done = False
    for line in text.splitlines():
        # A top-level key starts at column 0.
        if line and not line[0].isspace():
            in_section = line.split(":", 1)[0].strip() == section
        if in_section and not done:
            m = re.match(rf"(\s+){re.escape(key)}:(?:\s.*)?$", line)
            if m:
                out.append(f"{m.group(1)}{key}: {value}")
                done = True
                continue
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _set_or_add_in_section(text: str, section: str, key: str, value: str) -> str:
    """Set ``key`` inside ``section``, inserting it after the header if absent."""
    if _get_in_section(text, section, key) is not None:
        return _set_in_section(text, section, key, value)

    out = []
    in_section = False
    inserted = False
    child_indent = "  "
    for line in text.splitlines():
        is_top = bool(line) and not line[0].isspace()
        if is_top:
            in_section = line.split(":", 1)[0].strip() == section
            out.append(line)
            if in_section and not inserted:
                out.append(f"{child_indent}{key}: {value}")
                inserted = True
            continue
        # Track child indentation from the section's first indented line.
        if in_section and line and line[0].isspace():
            child_indent = line[: len(line) - len(line.lstrip())]
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _indent_of(line: str) -> int:
    """Number of leading spaces on a line."""
    return len(line) - len(line.lstrip())


def _find_section(lines: list, section: str) -> int:
    """Index of the first header line ``section:`` (no inline value) at any depth,
    or -1 — so nested sections like ``RAG.EMBED_MODEL_ID`` are reachable."""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == f"{section}:" or stripped.startswith(f"{section}:") and not stripped.split(":", 1)[1].strip():
            return i
    return -1


def _get_field(text: str, section: str, key: str) -> Optional[str]:
    """Read ``key`` from within ``section`` at any depth (comments stripped)."""
    lines = text.splitlines()
    idx = _find_section(lines, section)
    if idx < 0:
        return None
    header_indent = _indent_of(lines[idx])
    for line in lines[idx + 1:]:
        if line.strip() and _indent_of(line) <= header_indent:
            break  # left the section body
        m = re.match(rf"\s+{re.escape(key)}:\s*(.*)$", line)
        if m:
            return m.group(1).split(" #", 1)[0].strip() or None
    return None


def _set_field(text: str, section: str, key: str, value: str) -> str:
    """Set ``key`` within ``section`` at any depth, inserting it if absent."""
    lines = text.splitlines()
    idx = _find_section(lines, section)
    if idx < 0:
        return text  # section not present; nothing to do
    header_indent = _indent_of(lines[idx])
    body_indent = None  # the section's direct-child indent (first body line)
    for j in range(idx + 1, len(lines)):
        line = lines[j]
        if line.strip() and _indent_of(line) <= header_indent:
            break  # left the section body
        # Capture the body's indent from its FIRST indented line only — deeper
        # lines (e.g. a nested STOP: list's items) must not shift it.
        if body_indent is None and line.strip() and _indent_of(line) > header_indent:
            body_indent = _indent_of(line)
        m = re.match(rf"(\s+){re.escape(key)}:(?:\s.*)?$", line)
        if m:
            key_indent = len(m.group(1))
            # Drop a deeper-indented block value (e.g. a STOP list) so replacing
            # it with an inline value leaves no orphaned items.
            end = j + 1
            while end < len(lines):
                nxt = lines[end]
                if not nxt.strip():
                    break
                if _indent_of(nxt) <= key_indent:
                    break
                end += 1
            lines[j:end] = [f"{m.group(1)}{key}: {value}"]
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    # Not found in body -> insert just after the header at the body indent.
    indent = body_indent if body_indent is not None else header_indent + 2
    lines.insert(idx + 1, f"{' ' * indent}{key}: {value}")
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _remove_field(text: str, section: str, key: str) -> str:
    """Remove ``key`` (its line, block value, and leading comments) from
    ``section`` at any depth; no-op when absent."""
    lines = text.splitlines()
    idx = _find_section(lines, section)
    if idx < 0:
        return text
    header_indent = _indent_of(lines[idx])
    for j in range(idx + 1, len(lines)):
        line = lines[j]
        if line.strip() and _indent_of(line) <= header_indent:
            break  # left the section body
        m = re.match(rf"(\s+){re.escape(key)}:(?:\s.*)?$", line)
        if m:
            key_indent = len(m.group(1))
            start = j
            # Absorb this key's leading comment lines.
            while start - 1 > idx and lines[start - 1].strip().startswith("#"):
                start -= 1
            # Absorb the key line + deeper-indented continuation lines.
            end = j + 1
            while end < len(lines):
                nxt = lines[end]
                if not nxt.strip():
                    break
                if _indent_of(nxt) <= key_indent and not nxt.lstrip().startswith("#"):
                    break
                end += 1
            del lines[start:end]
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    return text


def _list_section_keys(text: str, section: str) -> list:
    """Return the direct child keys present in ``section`` (at any depth)."""
    lines = text.splitlines()
    idx = _find_section(lines, section)
    if idx < 0:
        return []
    header_indent = _indent_of(lines[idx])
    body_indent = None
    keys = []
    for line in lines[idx + 1:]:
        if line.strip() and _indent_of(line) <= header_indent:
            break  # left the section body
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if body_indent is None:
            body_indent = _indent_of(line)
        if _indent_of(line) != body_indent:
            continue  # nested deeper — not a direct key
        m = re.match(r"\s*([A-Za-z0-9_]+):", line)
        if m:
            keys.append(m.group(1))
    return keys


def _prune_unsupported_params(text: str, section: str, provider: str) -> str:
    """Drop keys the ``provider`` doesn't consume for ``section`` (per the
    registry); keeps NAME/TYPE; unknown provider prunes nothing."""
    allowed = supported_keys(section, provider)
    if allowed is None:
        return text  # unknown provider/section — don't touch anything
    keep = allowed | {"NAME", "TYPE"}
    for key in _list_section_keys(text, section):
        if key not in keep:
            text = _remove_field(text, section, key)
    return text


def _clear_inference_params(text: str, section: str, keep: set = None) -> str:
    """Remove model-specific inference params (temperature, penalties, reasoning,
    …) from ``section`` on a model change; connection/identity keys untouched.

    ``keep`` preserves named inference keys (e.g. MAX_TOKENS). Clears the union
    of every provider's tunable set so leftovers from another provider go too.
    """
    keep = keep or set()
    inference: set = set()
    for prov in providers(section):
        t = tunable_params(section, prov)
        if t:
            inference |= t
    present = set(_list_section_keys(text, section))
    for key in inference - keep:
        if key in present:
            text = _remove_field(text, section, key)
    return text


def _remove_top_section(text: str, section: str) -> str:
    """Remove a top-level ``section:`` block (header, body, and leading comments)
    — e.g. VISION_MODEL_ID when the user opts out."""
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(rf"{re.escape(section)}:\s*$", line) or (
            re.match(rf"{re.escape(section)}:", line)
            and not line.split(":", 1)[1].strip()
        ):
            # Drop this section's already-emitted leading comments.
            while out and out[-1].lstrip().startswith("#"):
                out.pop()
            i += 1
            # Skip the indented body.
            while i < len(lines) and (not lines[i].strip() or lines[i][0].isspace()):
                if lines[i].strip() and not lines[i][0].isspace():
                    break
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _set_top_level(text: str, key: str, value: str) -> str:
    """Replace the first top-level ``key: ...`` line (column 0)."""
    out = []
    done = False
    for line in text.splitlines():
        if not done and re.match(rf"{re.escape(key)}:(?:\s.*)?$", line):
            out.append(f"{key}: {value}")
            done = True
        else:
            out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _set_top_level_or_add(text: str, key: str, value: str) -> str:
    """Set a top-level ``key``, appending it if the config doesn't have it yet."""
    if _get_top_level(text, key) is not None:
        return _set_top_level(text, key, value)
    sep = "" if text.endswith("\n") or not text else "\n"
    return f"{text}{sep}{key}: {value}\n"


def _sync_doc_max_tokens(text: str) -> str:
    """Set ``DOC_MAX_TOKENS`` to 25% of ``MAX_CONVERSATION_TOKENS`` (so the doc
    read cap scales with the context window); no-op if the latter is unparseable."""
    raw = _get_top_level(text, "MAX_CONVERSATION_TOKENS")
    if raw is None:
        return text
    try:
        ctx = int(str(raw).strip())
    except (TypeError, ValueError):
        return text
    doc = max(1, ctx // 4)  # 25% of the context window
    return _set_top_level_or_add(text, "DOC_MAX_TOKENS", str(doc))


def _get_in_section(text: str, section: str, key: str) -> Optional[str]:
    """Read the first indented ``key`` inside top-level ``section`` (or None)."""
    in_section = False
    for line in text.splitlines():
        if line and not line[0].isspace():
            in_section = line.split(":", 1)[0].strip() == section
        if in_section:
            m = re.match(rf"\s+{re.escape(key)}:\s*(.*)$", line)
            if m:
                return m.group(1).split(" #", 1)[0].strip() or None
    return None


def _get_top_level(text: str, key: str) -> Optional[str]:
    """Read the value of the first top-level ``key``, stripped of trailing comments."""
    for line in text.splitlines():
        m = re.match(rf"{re.escape(key)}:\s*(.*)$", line)
        if m:
            return m.group(1).split(" #", 1)[0].strip() or None
    return None


# Sentinel raised to abort an interactive flow (Ctrl+C / Ctrl+D / EOF). The
# caller catches it and leaves the config untouched.
class _Cancelled(Exception):
    """User aborted the configurator step (Ctrl+C / Ctrl+D)."""


_CANCEL_HINT = "  (Press Ctrl+C or Ctrl+D at any prompt to cancel — nothing is saved.)"


def _is_tty() -> bool:
    """True on an interactive terminal (dialogs can render); else the ``_ask_*``
    helpers fall back to plain ``input()`` so scripted use never blocks."""
    return (
        hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
        and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    )


# Sentinel returned by the custom dialogs when the user cancels (Esc / Cancel),
# distinct from a legitimately-empty text entry.
_DIALOG_CANCEL = object()

# Sentinel returned when the user steps back (Back button / Ctrl+B), distinct
# from cancel; only shown when a previous wizard step exists.
_DIALOG_BACK = object()


class _GoBack(Exception):
    """User asked to return to the previous wizard step (Back / Ctrl+B)."""


def _dialog_hint(confirm: str, allow_back: bool) -> str:
    """Build a dialog key hint, inserting the Back cue only when available."""
    parts = [confirm]
    if allow_back:
        parts.append("Ctrl+B to go back")
    parts.append("Esc to cancel")
    return " · ".join(parts)


def _dialog_buttons(ok, cancel, back=None) -> list:
    """OK / [Back] / Cancel button row; Back inserted only when ``back`` given."""
    buttons = [Button(text="OK", handler=ok)]
    if back is not None:
        buttons.append(Button(text="Back", handler=back))
    buttons.append(Button(text="Cancel", handler=cancel))
    return buttons


def _dialog_input(
    title: str,
    default: Optional[str] = None,
    suggestion: Optional[str] = None,
    required: bool = False,
    allow_back: bool = False,
):
    """Centered text-input dialog; returns the text, ``_DIALOG_CANCEL``, or
    ``_DIALOG_BACK`` (only when ``allow_back``).

    ``default`` prefills the box (editable in place); ``suggestion`` is a
    display-only hint returned as a fallback on empty Enter (unless ``required``,
    which then blocks an empty confirm). Enter confirms, Esc cancels, Ctrl+B
    steps back.
    """
    field = TextArea(
        text=default or "",
        multiline=False,
        wrap_lines=False,
    )
    # Put the cursor at the end so a prefilled default is editable / appendable.
    field.buffer.cursor_position = len(field.text)

    base_hint = _dialog_hint("Enter to confirm", allow_back)
    if suggestion:
        base_hint = f"Suggestion: {suggestion}  ·  " + base_hint
    hint_label = Label(text=base_hint)

    def _resolve() -> str:
        # Empty box + suggestion → use the suggestion, unless required.
        text = field.text
        if not text.strip() and suggestion and not required:
            return suggestion
        return text

    def _ok() -> None:
        value = _resolve()
        if required and not value.strip():
            hint_label.text = "This field is required — please enter a value."
            return
        get_app().exit(result=value)

    def _cancel() -> None:
        get_app().exit(result=_DIALOG_CANCEL)

    def _back() -> None:
        get_app().exit(result=_DIALOG_BACK)

    dialog = Dialog(
        title=title,
        body=HSplit([hint_label, field], padding=1),
        buttons=_dialog_buttons(_ok, _cancel, _back if allow_back else None),
        with_background=True,
    )

    kb = KeyBindings()

    @kb.add("enter")
    def _(event) -> None:
        _ok()

    @kb.add("escape")
    @kb.add("c-c")
    def _(event) -> None:
        _cancel()

    if allow_back:
        @kb.add("c-b")
        def _(event) -> None:
            _back()

    app = Application(
        layout=Layout(dialog),
        key_bindings=kb,
        mouse_support=False,
        full_screen=True,
    )
    return app.run()


def _dialog_radio(
    title: str, options: list, default=None, info: str = "", allow_back: bool = False
):
    """Centered single-choice list (↑/↓ move, Enter confirms, Esc cancels);
    returns the chosen value, ``_DIALOG_CANCEL``, or ``_DIALOG_BACK``.

    ``options`` is ``[(value, label), …]``. ``info`` (optional) is shown above
    the list to surface the "Current setup" overview inside the dialog.
    """
    radio = RadioList(values=options, default=default)

    def _ok() -> None:
        get_app().exit(result=radio.current_value)

    def _cancel() -> None:
        get_app().exit(result=_DIALOG_CANCEL)

    def _back() -> None:
        get_app().exit(result=_DIALOG_BACK)

    # Bind Enter on the control itself (RadioList shadows a global Enter binding).
    radio.control.key_bindings.add("enter")(lambda event: _ok())

    body_items = []
    if info:
        body_items.append(Label(text=info))
    body_items.append(Label(text=_dialog_hint("↑/↓ move · Enter confirm", allow_back)))
    body_items.append(radio)

    dialog = Dialog(
        title=title,
        body=HSplit(body_items, padding=1),
        buttons=_dialog_buttons(_ok, _cancel, _back if allow_back else None),
        with_background=True,
    )

    kb = KeyBindings()

    @kb.add("escape")
    @kb.add("c-c")
    def _(event) -> None:
        _cancel()

    if allow_back:
        @kb.add("c-b")
        def _(event) -> None:
            _back()

    app = Application(
        layout=Layout(dialog),
        key_bindings=kb,
        mouse_support=False,
        full_screen=True,
    )
    return app.run()


def _dialog_yesno(title: str, default: bool, allow_back: bool = False):
    """Centered yes/no dialog (Y/N or ←/→ + Enter); True/False,
    ``_DIALOG_CANCEL`` on Esc (distinct from "No"), or ``_DIALOG_BACK``."""
    def _yes() -> None:
        get_app().exit(result=True)

    def _no() -> None:
        get_app().exit(result=False)

    def _cancel() -> None:
        get_app().exit(result=_DIALOG_CANCEL)

    def _back() -> None:
        get_app().exit(result=_DIALOG_BACK)

    yes_btn = Button(text="Yes", handler=_yes)
    no_btn = Button(text="No", handler=_no)
    buttons = [yes_btn, no_btn]
    if allow_back:
        buttons.append(Button(text="Back", handler=_back))

    dialog = Dialog(
        title=title,
        body=HSplit(
            [Label(text=_dialog_hint("Y/N or ←/→ + Enter", allow_back))],
            padding=1,
        ),
        buttons=buttons,
        with_background=True,
    )

    kb = KeyBindings()

    @kb.add("y")
    @kb.add("Y")
    def _(event) -> None:
        _yes()

    @kb.add("n")
    @kb.add("N")
    def _(event) -> None:
        _no()

    @kb.add("escape")
    @kb.add("c-c")
    def _(event) -> None:
        _cancel()

    if allow_back:
        @kb.add("c-b")
        def _(event) -> None:
            _back()

    app = Application(
        layout=Layout(dialog),
        key_bindings=kb,
        mouse_support=False,
        full_screen=True,
    )
    # Focus the default button so Tab/←/→ + Enter still works intuitively.
    app.layout.focus(yes_btn if default else no_btn)
    result = app.run()
    if result in (_DIALOG_CANCEL, _DIALOG_BACK):
        return result
    return bool(result)


def _ask(
    prompt: str,
    default: Optional[str] = None,
    suggestion: Optional[str] = None,
    required: bool = False,
    allow_back: bool = False,
) -> Optional[str]:
    """Prompt for a value, returning ``default`` on empty input/EOF/cancel.

    ``suggestion`` proposes a value without prefilling (empty Enter falls back to
    it, unless ``required`` — then it's display-only and an empty value re-asks).
    Cancelling raises ``_Cancelled``; stepping back raises ``_GoBack``.
    """
    if _is_tty():
        val = _dialog_input(
            prompt, default, suggestion=suggestion, required=required,
            allow_back=allow_back,
        )
        if val is _DIALOG_CANCEL:
            raise _Cancelled()
        if val is _DIALOG_BACK:
            raise _GoBack()
        val = val.strip()
        return val or default or (None if required else suggestion)

    # Non-TTY: show the suggestion in the bracket hint, but keep no prefill.
    hint = default or suggestion
    suffix = f" [{hint}]" if hint else ""
    while True:
        try:
            val = input(f"  {prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise _Cancelled()
        resolved = val or default or (None if required else suggestion)
        if required and not (resolved and str(resolved).strip()):
            print("  This field is required — please enter a value.")
            continue
        return resolved


def _ask_required(
    prompt: str, default: Optional[str] = None, allow_back: bool = False
) -> str:
    """Prompt for a value; empty input keeps ``default``, or (with no default)
    re-asks. Cancelling raises ``_Cancelled``; stepping back raises ``_GoBack``."""
    if _is_tty():
        val = _dialog_input(prompt, default, required=default is None,
                            allow_back=allow_back)
        if val is _DIALOG_CANCEL:
            raise _Cancelled()
        if val is _DIALOG_BACK:
            raise _GoBack()
        return val.strip() or (default or "")

    suffix = f" [{default}]" if default else ""
    while True:
        try:
            val = input(f"  {prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise _Cancelled()
        resolved = val or (default or "")
        if not resolved and default is None:
            print("  This field is required — please enter a value.")
            continue
        return resolved


def _ask_choice(
    prompt: str,
    valid: set,
    default: Optional[str] = None,
    labels: Optional[dict] = None,
    info: str = "",
    allow_back: bool = False,
) -> str:
    """Prompt for one of ``valid`` keys, re-asking on an invalid entry.

    On a TTY with ``labels`` (``{key: human_label}``) this is a single-choice
    dialog (``info`` shown inside it); else a plain ``input()`` re-ask loop.
    Cancelling raises ``_Cancelled``; stepping back raises ``_GoBack``.
    """
    if _is_tty() and labels:
        # Preserve labels' insertion order.
        options = [(k, labels.get(k, k)) for k in labels if k in valid]
        chosen = _dialog_radio(
            prompt, options, default=default if default in valid else None,
            info=info, allow_back=allow_back,
        )
        if chosen is _DIALOG_CANCEL:
            raise _Cancelled()
        if chosen is _DIALOG_BACK:
            raise _GoBack()
        return chosen

    while True:
        answer = _ask_required(prompt, default, allow_back=allow_back)
        if answer in valid:
            return answer
        print(f"  '{answer}' is not a valid choice; please pick one of: "
              f"{', '.join(sorted(valid))}.")


def _ask_number(
    prompt: str,
    default: Optional[str] = None,
    kind: str = "int",
    allow_none: bool = False,
    allow_back: bool = False,
) -> Optional[str]:
    """Prompt for an int/float, re-asking until it parses.

    Enter/default returns ``default``; "none" returns None when ``allow_none``;
    a non-numeric entry re-prompts. Cancelling raises ``_Cancelled``; stepping
    back raises ``_GoBack``.
    """
    while True:
        answer = _ask_required(prompt, default, allow_back=allow_back)
        if allow_none and answer.strip().lower() == "none":
            return None
        if default is not None and answer == default:
            return default
        try:
            if kind == "float":
                float(answer)
            else:
                int(answer)
            return answer
        except ValueError:
            label = "a number" if kind == "float" else "an integer"
            # No scrollback in dialog mode → surface the error in the next title.
            msg = f"'{answer}' is not {label}" + (
                " (or 'none')" if allow_none else ""
            )
            if _is_tty():
                prompt = f"{prompt.split(' — ')[0]} — {msg}"
            else:
                print(f"  {msg}; please try again.")


def _ask_bool(prompt: str, default: bool, allow_back: bool = False) -> bool:
    """Prompt yes/no, defaulting to ``default`` on empty input; cancelling raises
    ``_Cancelled``; stepping back raises ``_GoBack``."""
    if _is_tty():
        result = _dialog_yesno(prompt, default, allow_back=allow_back)
        if result is _DIALOG_CANCEL:
            raise _Cancelled()
        if result is _DIALOG_BACK:
            raise _GoBack()
        return result

    hint = "Y/n" if default else "y/N"
    try:
        val = input(f"  {prompt} ({hint}): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        raise _Cancelled()
    if not val:
        return default
    return val.startswith("y")


def _run_steps(text: str, steps: list) -> str:
    """Run a wizard's ordered steps with Back support, threading ``text`` through.

    Each step is ``fn(text, allow_back) -> text``: it prompts (passing
    ``allow_back`` to its ``_ask*`` call) and returns the patched text. Every
    step but the first is given ``allow_back=True``; a ``_GoBack`` from step *i*
    rewinds to step *i-1*, restoring the text as it was before that step ran so a
    re-answer replaces (not stacks on) the earlier one. ``_Cancelled`` propagates.
    """
    history = []  # text as it was BEFORE each executed step, indexed by step
    i = 0
    while i < len(steps):
        history = history[:i]
        history.append(text)
        try:
            text = steps[i](text, i > 0)
            i += 1
        except _GoBack:
            # Rewind to the previous step, restoring its pre-run text.
            i = max(0, i - 1)
            text = history[i]
    return text


def _set_bool(text: str, key: str, value: bool, section: Optional[str] = None) -> str:
    """Set a boolean toggle (``true``/``false``), top-level or within a section."""
    literal = "true" if value else "false"
    if section:
        return _set_in_section(text, section, key, literal)
    return _set_top_level(text, key, literal)


def _truthy(value: Optional[str]) -> bool:
    """Interpret a template's YAML scalar string as a bool (default True)."""
    return (value or "true").strip().lower() in ("true", "yes", "on", "1")


def _prompt_provider_type(section: str, current: str) -> str:
    """Prompt for a provider TYPE (from the registry), returning the chosen
    canonical key; ``current`` is the default."""
    options = list(providers(section))
    if current not in options:
        # Offer an unknown/legacy current value so it can be kept.
        options = [current] + options
    default_key = str(options.index(current) + 1)

    labels = {
        str(i): _PROVIDER_LABELS.get(prov, prov)
        for i, prov in enumerate(options, 1)
    }
    valid = set(labels)
    if not _is_tty():
        print(f"\n  Provider type for this model (current: {_PROVIDER_LABELS.get(current, current)})")
        for k, lbl in labels.items():
            print(f"    {k}) {lbl}")
    choice = _ask_choice(
        f"Provider type (current: {_PROVIDER_LABELS.get(current, current)})",
        valid,
        default_key,
        labels=labels,
    )
    return options[int(choice) - 1]


def _prompt_mantle_protocol(text: str, section: str, allow_back: bool = False) -> str:
    """Prompt for and set ``section``'s Mantle API protocol; leaves an absent
    ``chat_completions`` (the default) unwritten so Enter-through is a no-op."""
    existing = _get_field(text, section, "API_PROTOCOL")
    current = existing or "chat_completions"
    default_key = next((k for k, (v, _) in _MANTLE_PROTOCOLS.items() if v == current), "1")
    labels = {k: desc for k, (_, desc) in _MANTLE_PROTOCOLS.items()}
    if not _is_tty():
        print("  Mantle API protocol:")
        for k, lbl in labels.items():
            print(f"    {k}) {lbl}")
    choice = _ask_choice(
        "Mantle API protocol", set(_MANTLE_PROTOCOLS), default_key, labels=labels,
        allow_back=allow_back,
    )
    chosen = _MANTLE_PROTOCOLS[choice][0]
    if chosen != existing and not (existing is None and chosen == "chat_completions"):
        text = _set_field(text, section, "API_PROTOCOL", chosen)
    return text


def _conn_from_text(text: str, section: str) -> dict:
    """HOST/PORT/REGION currently set in ``section`` (values the vision section
    mirrors); read back from ``text`` so it survives step-based prompting."""
    conn = {}
    for k in ("HOST", "PORT", "REGION"):
        v = _get_field(text, section, k)
        if v is not None:
            conn[k] = v
    return conn


def _optional_field_step(section: str, key: str, prompt: str):
    """A step for an optional connection field: set it only when non-blank
    (blank keeps the provider's env/default). Returns ``fn(text, allow_back)``."""
    def _step(text: str, allow_back: bool) -> str:
        v = _ask(prompt, _get_field(text, section, key) or "", allow_back=allow_back)
        return _set_field(text, section, key, v) if v else text
    return _step


def _connection_steps(section: str, provider: str) -> list:
    """Back-able steps for the connection/auth keys ``provider`` needs (per the
    registry). Each is ``fn(text, allow_back) -> text``; shared by /config, /model."""
    allowed = supported_keys(section, provider) or set()
    steps = []

    if "HOST" in allowed:
        steps.append(lambda t, b: _set_field(
            t, section, "HOST",
            _ask("Ollama host", _get_field(t, section, "HOST") or "localhost", allow_back=b)))
    if "PORT" in allowed:
        steps.append(lambda t, b: _set_field(
            t, section, "PORT",
            _ask("Ollama port", _get_field(t, section, "PORT") or "11434", allow_back=b)))
    if "REGION" in allowed:
        steps.append(lambda t, b: _set_field(
            t, section, "REGION",
            _ask("AWS region", _get_field(t, section, "REGION") or "us-east-1", allow_back=b)))
    if "INPUT_FORMAT" in allowed:
        steps.append(lambda t, b: _set_field(
            t, section, "INPUT_FORMAT",
            _ask("Input format (openai_chat | huggingface)",
                 _get_field(t, section, "INPUT_FORMAT") or "openai_chat", allow_back=b)
            or "openai_chat"))
    if "API_PROTOCOL" in allowed and provider == "mantle":
        # Mantle's protocol is a required 3-way choice; OpenAI's is an advanced
        # hand-set opt-in the configurator doesn't prompt for.
        steps.append(lambda t, b: _prompt_mantle_protocol(t, section, allow_back=b))
    if provider == "litellm":
        if "API_BASE" in allowed:
            steps.append(_optional_field_step(section, "API_BASE", "LiteLLM API base URL (optional)"))
        if "API_KEY" in allowed:
            steps.append(_optional_field_step(section, "API_KEY", "LiteLLM API key (optional, or via the provider's env var)"))
    if provider == "anthropic":
        if "API_KEY" in allowed:
            steps.append(_optional_field_step(section, "API_KEY", "Anthropic API key (optional, or via ANTHROPIC_API_KEY env var)"))
        if "ENDPOINT_URL" in allowed:
            steps.append(_optional_field_step(section, "ENDPOINT_URL", "Anthropic base URL (optional, blank for api.anthropic.com)"))
    if provider == "openai" and "API_BASE" in allowed:
        # Optional OpenAI-compatible server; blank keeps api.openai.com.
        steps.append(_optional_field_step(section, "API_BASE", "OpenAI-compatible base URL (optional, blank for the OpenAI API)"))
        if "API_KEY" in allowed:
            steps.append(_optional_field_step(section, "API_KEY", "API key (optional; blank uses OPENAI_API_KEY, local servers need none)"))

    # Embeddings: optional vector-size override (fallback only).
    if section == "EMBED_MODEL_ID":
        steps.append(_optional_field_step(section, "DIMENSION", "Embedding dimension (optional; blank = auto-detect)"))

    return steps


def _prompt_provider_connection(text: str, section: str, provider: str):
    """Run the connection/auth steps for ``provider`` linearly (no Back) and
    print the credential note. Returns ``(text, conn)`` where ``conn`` holds the
    HOST/PORT/REGION the vision section can mirror. Thin wrapper over
    :func:`_connection_steps` kept for callers that want the one-shot form."""
    for step in _connection_steps(section, provider):
        text = step(text, False)
    _print_credential_note(provider)
    return text, _conn_from_text(text, section)


def _print_credential_note(provider: str) -> None:
    """Print the env-based auth note for a provider (the configurator can't set
    these keys itself)."""
    if provider == "bedrock":
        print("  Note: Bedrock uses your AWS credentials (`aws configure`) or a")
        print("  Bedrock API key (AWS_BEARER_TOKEN_BEDROCK env var).")
    elif provider == "mantle":
        print("  Note: Mantle uses your AWS credentials, or a Bedrock API key")
        print("  (BEDROCK_API_KEY env var / MODEL_ID.API_KEY).")
    elif provider == "sagemaker":
        print("  Note: SageMaker uses your AWS credentials; NAME is the endpoint name.")
    elif provider == "openai":
        print("  Note: OpenAI reads the OPENAI_API_KEY environment variable")
        print("  (set it in your shell or the config ENV section).")
    elif provider == "anthropic":
        print("  Note: Anthropic (Claude API) reads the ANTHROPIC_API_KEY env")
        print("  var, or MODEL_ID.API_KEY. This is the direct api.anthropic.com")
        print("  API — not Bedrock Mantle's 'anthropic' protocol.")


def _build_config(
    provider: str, default_model: str, template_text: str, template_file: str = ""
) -> str:
    """Prompt for the fields that vary and patch them into the template.

    Only commonly-changed fields are prompted; the rest keep the template's
    values (each default read from it, so Enter-through works). openai/anthropic/
    sagemaker/litellm reuse the Ollama-shaped base and transform their sections.
    """
    text = template_text
    transform_from_base = (
        template_file == "config.yaml.example" and provider != "ollama"
    )

    # Drop documentation-only STOP sequences from a generated config.
    text = _remove_field(text, "MODEL_ID", "STOP")
    text = _remove_field(text, "VISION_MODEL_ID", "STOP")

    # For providers reusing the Ollama-shaped base, set TYPE and prune unsupported
    # keys. OpenAI/Anthropic are multimodal, so mirror the vision section to them;
    # SageMaker/LiteLLM leave vision as Ollama.
    if transform_from_base:
        text = _set_in_section(text, "MODEL_ID", "TYPE", provider)
        text = _prune_unsupported_params(text, "MODEL_ID", provider)
        if provider in ("openai", "anthropic") and _find_section(text.splitlines(), "VISION_MODEL_ID") >= 0:
            text = _set_in_section(text, "VISION_MODEL_ID", "TYPE", provider)
            text = _prune_unsupported_params(text, "VISION_MODEL_ID", provider)

    # --- Chat model --- (name → connection → max_tokens → context, all Back-able)
    def _name_step(t, b):
        # default_model is an example (suggestion, not prefilled); name is mandatory.
        m = _ask("Chat model name", suggestion=default_model, required=True, allow_back=b)
        return _set_in_section(t, "MODEL_ID", "NAME", m) if m else t

    def _ctx_step(t, b):
        # Mandatory context window; defaults to the template value (or 65536).
        ctx = _ask_number(
            "Max context window",
            default=_get_top_level(t, "MAX_CONVERSATION_TOKENS") or "65536",
            kind="int", allow_back=b,
        )
        t = _set_top_level_or_add(t, "MAX_CONVERSATION_TOKENS", ctx or "65536")
        return _sync_doc_max_tokens(t)

    text = _run_steps(text, [
        _name_step,
        *_connection_steps("MODEL_ID", provider),
        lambda t, b: _prompt_max_tokens(t, "MODEL_ID", allow_back=b),
        _ctx_step,
    ])
    # HOST/PORT/REGION to mirror into the vision section.
    conn = _conn_from_text(text, "MODEL_ID")

    # --- Vision model (optional) ---
    if _ask_bool("Configure a vision model (for image description)?", True):
        vision = _ask("Vision model name", _get_in_section(text, "VISION_MODEL_ID", "NAME"))
        if vision:
            text = _set_in_section(text, "VISION_MODEL_ID", "NAME", vision)
        # Mirror the chat model's connection (same host/region, usually).
        for k, v in conn.items():
            if _get_in_section(text, "VISION_MODEL_ID", k) is not None:
                text = _set_in_section(text, "VISION_MODEL_ID", k, v)
        if (_get_in_section(text, "VISION_MODEL_ID", "TYPE") or "").lower() == "mantle":
            text = _prompt_mantle_protocol(text, "VISION_MODEL_ID")
        text = _prompt_max_tokens(text, "VISION_MODEL_ID")
    else:
        text = _remove_top_section(text, "VISION_MODEL_ID")

    # --- Profile ---
    profile = _ask("Profile name (isolates your data)", getpass.getuser() or "default")
    if profile:
        text = _set_in_section(text, "PROFILE", "NAME", profile)

    # --- Web search (Brave) ---
    brave = _ask("Brave Search API key (optional, press Enter to skip)", "")
    if brave:
        text = _set_top_level(text, "BRAVE_API_KEY", brave)
        text = _set_bool(text, "ENABLE_WEB_SEARCH", True)
    else:
        text = _set_bool(text, "ENABLE_WEB_SEARCH", False)

    # --- Other feature toggles (default from the template) ---
    text = _set_bool(text, "ENABLE_RAG", _ask_bool("Enable RAG (document indexing & search)?", _truthy(_get_top_level(text, "ENABLE_RAG"))))
    text = _set_bool(text, "ENABLE_EPISODIC_MEMORY", _ask_bool("Enable episodic memory (learn from past tasks)?", _truthy(_get_top_level(text, "ENABLE_EPISODIC_MEMORY"))))
    text = _set_bool(text, "ENABLE_PLAYBOOK", _ask_bool("Enable ACE playbook (learn strategies)?", _truthy(_get_top_level(text, "ENABLE_PLAYBOOK"))))
    text = _set_bool(text, "ENABLE_MEMORY", _ask_bool("Enable persistent memory (agent curates MEMORY.md)?", _truthy(_get_top_level(text, "ENABLE_MEMORY"))))
    text = _set_bool(text, "ENABLE_WEB_CRAWL", _ask_bool("Enable web crawler (fetch URLs)?", _truthy(_get_top_level(text, "ENABLE_WEB_CRAWL"))))

    routing = _ask_bool("Enable query routing (route queries to tool subsets)?", _truthy(_get_top_level(text, "ENABLE_ROUTING")))
    text = _set_bool(text, "ENABLE_ROUTING", routing)
    if routing:
        orchestration = _ask_bool("Enable orchestration (decompose complex tasks)?", _truthy(_get_top_level(text, "ENABLE_ORCHESTRATION")))
    else:
        orchestration = False
    text = _set_bool(text, "ENABLE_ORCHESTRATION", orchestration)

    text = _set_bool(text, "USE_PROFILING", _ask_bool("Enable user profiling (personalized responses)?", _truthy(_get_in_section(text, "PROFILE", "USE_PROFILING"))), section="PROFILE")

    text = _set_bool(text, "REQUIRE_BASH_CONFIRMATION", _ask_bool("Ask for confirmation before each shell command (execute_bash)?", _truthy(_get_top_level(text, "REQUIRE_BASH_CONFIRMATION"))))
    text = _set_bool(text, "REQUIRE_WRITE_CONFIRMATION", _ask_bool("Ask for confirmation before each file write (fs_write/file_edit)?", _truthy(_get_top_level(text, "REQUIRE_WRITE_CONFIRMATION"))))

    return text


def _run_configurator(dest: Path) -> Optional[Path]:
    """Pick a provider, fill the template, write to ``dest`` (caller handles
    first-run/overwrite gating); returns the Path, or None if a template's missing."""
    provider_labels = {k: label for k, (_, _, label, _) in _PROVIDERS.items()}
    if not _is_tty():
        print(_CANCEL_HINT)
        print("\n  Choose your LLM provider:")
        for k, lbl in provider_labels.items():
            print(f"    {k}) {lbl}")
    choice = _ask_choice(
        "Choose your LLM provider", set(_PROVIDERS), "1", labels=provider_labels
    )

    provider, template_file, label, default_model = _PROVIDERS[choice]
    template_path = _templates_dir() / template_file
    if not template_path.is_file():
        print_error(f"Template not found: {template_path}. Cannot continue setup.")
        return None

    template_text = template_path.read_text()
    config_text = _build_config(provider, default_model, template_text, template_file)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(config_text)

    print(f"\n  Config written to:\n    {dest}")
    # Env-based auth reminder survives to scrollback after the restart.
    _print_credential_note(provider)
    print(f"\n  This file lives in the config/ folder of your app home ({dest.parent.parent}),")
    print("  which also holds the rest of your runtime data (plans, tasks,")
    print("  conversations, RAG indexes, episodic memory, the ACE playbook, and")
    print("  an mcp/ folder for external MCP servers).")
    print("\n  Only the most common settings were configured here. The file")
    print("  contains many more options you can edit any time — per-model")
    print("  inference parameters (temperature, top_p, penalties, …), the")
    print("  embedding model, RAG / episodic-memory / playbook tuning, retry and")
    print("  compaction limits, and the routing / orchestrator / system prompts.")
    print("  See the README's 'Model Parameters' and 'Configuration' sections")
    print("  for the full list of arguments per provider.")
    print("=" * 64 + "\n")
    return dest


def run_first_run_setup() -> Optional[Path]:
    """Interactively create a first ``config.yaml``; returns the Path or None."""
    dest = config_path()

    # On a TTY the confirm dialog carries the framing; only the non-TTY fallback
    # prints the banner (a full-screen dialog would wipe it anyway).
    if not _is_tty():
        print()
        print("=" * 64)
        print("  Welcome to Mnemo AI — first-run setup")
        print("=" * 64)
        print(f"  No config found. Let's create one at:\n    {dest}\n")

    try:
        answer = _ask("Set up a config now? (Y/n)", "Y")
        if answer and answer.strip().lower().startswith("n"):
            print("  Skipped. Copy a template to that path manually to get started.")
            return None

        return _run_configurator(dest)
    except (KeyboardInterrupt, _Cancelled):
        print("\n  Setup cancelled. No config was written.")
        return None


def run_reconfigure() -> Optional[Path]:
    """Re-run the configurator over an existing config (``/config``), confirming
    the overwrite first; returns the written Path or None."""
    dest = config_path()

    # On a TTY the dialogs carry the caveat, so only the non-TTY fallback prints
    # the banner/WARNING here.
    if not _is_tty():
        print()
        print("=" * 64)
        print("  Reconfigure Mnemo AI")
        print("=" * 64)
        if dest.is_file():
            print(f"  WARNING: this will OVERWRITE your existing config at:\n    {dest}")
            print("  Your current settings will be replaced by the answers you give")
            print("  now. (Other runtime data — conversations, memory, etc. — is kept.)")
        else:
            print(f"  No config found yet; this will create one at:\n    {dest}")

    # Fold the overwrite caveat into the confirmation so a TTY still sees it.
    if dest.is_file():
        confirm_q = "Reconfigure now? This OVERWRITES your existing config (y/N)"
    else:
        confirm_q = "Create a new config now? (y/N)"

    try:
        answer = _ask(confirm_q, "N")
        if not (answer and answer.strip().lower().startswith("y")):
            print("  Cancelled. Existing config left untouched.")
            return None

        return _run_configurator(dest)
    except (KeyboardInterrupt, _Cancelled):
        print("\n  Reconfigure cancelled. Existing config left untouched.")
        return None


# /model-overridable sections: key -> (config section, label, is_chat_llm — the
# LLM also gets the context-window prompt).
_MODEL_SECTIONS = {
    "1": ("MODEL_ID", "Chat model (LLM)", True),
    "2": ("VISION_MODEL_ID", "Vision model", False),
    "3": ("EMBED_MODEL_ID", "Embeddings model", False),
}


def _section_summary(text: str, section: str) -> Optional[str]:
    """One-line summary of a model section (e.g. ``bedrock-mantle / … (us-east-1,
    anthropic)``), or None if not configured."""
    name = _get_field(text, section, "NAME")
    if not name:
        return None
    raw_type = _get_field(text, section, "TYPE") or "?"
    typ = _PROVIDER_LABELS.get(raw_type, raw_type)
    extras = []
    host = _get_field(text, section, "HOST")
    port = _get_field(text, section, "PORT")
    if host:
        extras.append(f"{host}:{port}" if port else host)
    region = _get_field(text, section, "REGION")
    if region:
        extras.append(region)
    protocol = _get_field(text, section, "API_PROTOCOL")
    if protocol:
        extras.append(protocol)
    suffix = f" ({', '.join(extras)})" if extras else ""
    return f"{typ} / {name}{suffix}"


def _prompt_max_tokens(text: str, section: str, allow_back: bool = False) -> str:
    """Prompt for the optional MAX_TOKENS of a section (Enter/'none' → remove the
    key = provider default; a number → set it)."""
    answer = _ask_number(
        "Max output tokens (number, or 'none' for provider default)",
        default="none",
        kind="int",
        allow_none=True,
        allow_back=allow_back,
    )
    if answer is None:
        return _remove_field(text, section, "MAX_TOKENS")
    return _set_field(text, section, "MAX_TOKENS", answer)


def _current_setup_text(text: str) -> str:
    """The "Current setup" overview (chat/vision/embeddings) as a string, so it
    can print to scrollback (non-TTY) or render inside the selection dialog."""
    vision = _section_summary(text, "VISION_MODEL_ID")
    embeddings = _section_summary(text, "EMBED_MODEL_ID")
    return (
        "Current setup:\n"
        f"  Chat (LLM):  {_section_summary(text, 'MODEL_ID') or '(not set)'}\n"
        f"  Vision:      {vision if vision else '(not configured)'}\n"
        f"  Embeddings:  {embeddings if embeddings else '(not configured)'}"
    )


def _print_current_setup(text: str) -> None:
    """Print the current chat/vision/embeddings models (indented for scrollback)."""
    for line in _current_setup_text(text).splitlines():
        print(f"  {line}")


def _prompt_model_section(text: str, section: str, is_llm: bool) -> str:
    """Prompt for one model section (provider type included, so it can switch
    providers) and patch ``text``; context window only for the chat LLM.

    The provider type is the anchor (chosen first, no Back into it — switching it
    invalidates the provider-dependent steps that follow); every step after it
    supports Back (Ctrl+B)."""
    cur_type = (_get_field(text, section, "TYPE") or "ollama").lower()
    new_type = _prompt_provider_type(section, cur_type)
    text = _set_field(text, section, "TYPE", new_type)

    # On a provider switch, strip keys the new provider doesn't consume. Clear
    # model-specific inference params on any change (a value tuned for one model
    # may be rejected by another); MAX_TOKENS is prompted below. Done before the
    # prompts so the connection steps re-add exactly the right keys.
    if new_type != cur_type:
        text = _prune_unsupported_params(text, section, new_type)
    text = _clear_inference_params(text, section, keep={"MAX_TOKENS"})

    def _name_step(t, b):
        name = _ask("Model name", _get_field(t, section, "NAME"), allow_back=b)
        return _set_field(t, section, "NAME", name) if name else t

    steps = [_name_step, *_connection_steps(section, new_type)]
    if section in ("MODEL_ID", "VISION_MODEL_ID"):
        steps.append(lambda t, b: _prompt_max_tokens(t, section, allow_back=b))
    if is_llm:
        # Context window (feeds num_ctx + compaction budget); defaults to 65536.
        def _ctx_step(t, b):
            ctx = _ask_number("Max context window", default="65536", kind="int",
                              allow_back=b)
            t = _set_top_level_or_add(t, "MAX_CONVERSATION_TOKENS", ctx or "65536")
            return _sync_doc_max_tokens(t)
        steps.append(_ctx_step)

    text = _run_steps(text, steps)
    _print_credential_note(new_type)
    return text


# --- /params: tune a model's inference parameters ---------------------------
# (kind, hint) per tunable key — kind drives validation, hint is the prompt.
# Which keys are offered comes from the registry's tunable set; this only says
# how to prompt/validate each. kind: float | int | bool | list | enum:a,b,c
_PARAM_META = {
    "TEMPERATURE": ("float", "sampling temperature, e.g. 0.7"),
    "TOP_P": ("float", "nucleus sampling, 0-1"),
    "TOP_K": ("int", "top-k sampling, e.g. 40"),
    "MAX_TOKENS": ("int", "max output tokens"),
    "PRESENCE_PENALTY": ("float", "e.g. 0.0-2.0"),
    "FREQUENCY_PENALTY": ("float", "e.g. 0.0-2.0"),
    "REPETITION_PENALTY": ("float", "e.g. 1.0-1.3"),
    "STOP": ("list", "comma-separated stop sequences"),
    "STREAM": ("bool", "stream tokens as they generate"),
    "REASONING": ("bool", "enable extended thinking"),
    "REASONING_EFFORT": (
        "enum:none,minimal,low,medium,high,xhigh,max",
        "reasoning effort (provider-dependent; e.g. low|medium|high)",
    ),
    "THINKING_TOKENS": ("int", "budget for thinking tokens"),
    "DIMENSION": ("int", "embedding vector size (match your embedder; for the fallback)"),
}

# Order params are prompted in (registry membership decides which appear).
_PARAM_ORDER = [
    "TEMPERATURE", "TOP_P", "TOP_K", "MAX_TOKENS",
    "PRESENCE_PENALTY", "FREQUENCY_PENALTY", "REPETITION_PENALTY",
    "STOP", "REASONING", "REASONING_EFFORT", "THINKING_TOKENS", "STREAM",
    "DIMENSION",
]

# /params-tunable sections: key -> (config section, label). Embeddings exposes
# only DIMENSION; chat/vision expose the generation knobs.
_PARAM_SECTIONS = {
    "1": ("MODEL_ID", "Chat model (LLM)"),
    "2": ("VISION_MODEL_ID", "Vision model"),
    "3": ("EMBED_MODEL_ID", "Embeddings model"),
}


def _validate_param(key: str, kind: str, raw: str) -> Optional[str]:
    """Return the YAML scalar to write for a param answer, or None if invalid
    for the kind (caller re-asks)."""
    raw = raw.strip()
    if kind == "float":
        try:
            float(raw)
            return raw
        except ValueError:
            return None
    if kind == "int":
        try:
            int(raw)
            return raw
        except ValueError:
            return None
    if kind == "bool":
        low = raw.lower()
        if low in ("true", "yes", "y", "on", "1"):
            return "true"
        if low in ("false", "no", "n", "off", "0"):
            return "false"
        return None
    if kind == "list":
        items = [p.strip() for p in raw.split(",") if p.strip()]
        if not items:
            return None
        return "[" + ", ".join('"' + it.replace('"', '\\"') + '"' for it in items) + "]"
    if kind.startswith("enum:"):
        allowed = kind.split(":", 1)[1].split(",")
        return raw if raw in allowed else None
    return raw


def _prompt_one_param(
    text: str, section: str, key: str, kind: str, hint: str, allow_back: bool = False
) -> str:
    """Prompt for one inference param (prefilled with the current value) and
    patch ``text``: unchanged/empty keeps it, ``none`` clears it, a valid value
    sets it (re-asks on invalid), Esc cancels, Ctrl+B steps back."""
    current = _get_field(text, section, key)
    base_hint = hint
    if kind.startswith("enum:"):
        base_hint = f"{hint} ({kind.split(':', 1)[1].replace(',', ' | ')})"
    elif kind == "bool":
        base_hint = f"{hint} (true/false)"

    prompt_hint = base_hint
    while True:
        answer = _ask(
            f"{key} [{prompt_hint}] ('none' to clear)", default=current,
            allow_back=allow_back,
        )
        if answer is None:
            return text
        answer = answer.strip()
        if answer == "" or answer == (current or ""):
            return text  # unchanged → keep
        if answer.lower() == "none":
            return _remove_field(text, section, key)
        value = _validate_param(key, kind, answer)
        if value is None:
            # Surface the error in the next prompt (a dialog hides any print()).
            err = f"'{answer}' is not a valid {kind.split(':', 1)[0]} value"
            if _is_tty():
                prompt_hint = f"{base_hint} — {err}, try again"
            else:
                print(f"    {err}; try again.")
            continue
        return _set_field(text, section, key, value)


def _prompt_inference_params(text: str, section: str) -> str:
    """Prompt every inference param the section's provider accepts (from the
    registry). Context window and connection keys are /model's job, not here."""
    provider = (_get_field(text, section, "TYPE") or "ollama").lower()
    tunable = tunable_params(section, provider)
    if not tunable:
        print(f"  Provider '{provider}' exposes no tunable inference parameters here.")
        return text

    def _step(key, kind, hint):
        return lambda t, back: _prompt_one_param(t, section, key, kind, hint, back)

    steps = []
    for key in _PARAM_ORDER:
        if key not in tunable:
            continue
        kind, hint = _PARAM_META.get(key, ("str", key))
        steps.append(_step(key, kind, hint))
    return _run_steps(text, steps)


def run_params_override() -> Optional[Path]:
    """Tune a configured model's inference parameters (``/params``); edits only
    those keys (use /model for provider/name/connection). Returns the Path, or
    None if cancelled or unchanged."""
    dest = config_path()
    if not dest.is_file():
        print_error("No config.yaml found. Run /config to create one first.")
        return None

    text = dest.read_text()

    available = {
        "1": _get_field(text, "MODEL_ID", "NAME") is not None,
        "2": _get_field(text, "VISION_MODEL_ID", "NAME") is not None,
        "3": _get_field(text, "EMBED_MODEL_ID", "NAME") is not None,
    }

    labels = {
        k: label for k, (_, label) in _PARAM_SECTIONS.items() if available.get(k)
    }
    # On a TTY the picker dialog carries the title + overview (info=), so only
    # the non-TTY fallback prints them here.
    if not _is_tty():
        print()
        print("=" * 64)
        print("  Tune inference parameters")
        print("=" * 64)
        _print_current_setup(text)
        print("\n  Which model's parameters do you want to tune? Only inference")
        print("  params are changed; provider, name, and connection stay as-is")
        print("  (use /model for those).\n")
        for k, lbl in labels.items():
            print(f"    {k}) {lbl}")
        print(_CANCEL_HINT)
    try:
        valid = set(labels)
        choice = _ask_choice(
            "Which model's parameters to tune?",
            valid,
            "1",
            labels=labels,
            info=_current_setup_text(text),
        )
        section, label = _PARAM_SECTIONS[choice]
        new_text = _prompt_inference_params(text, section)
    except (KeyboardInterrupt, _Cancelled):
        print("\n  Cancelled. Config left untouched.")
        return None

    # Re-sync the doc read cap (corrects a hand-edited value too).
    new_text = _sync_doc_max_tokens(new_text)

    if new_text == text:
        print("  No changes made.")
        return None

    dest.write_text(new_text)
    print(f"\n  Updated {label} parameters in:\n    {dest}")
    print("  Reload to apply: the change takes effect on the next config reload.")
    print("=" * 64 + "\n")
    return dest


def run_model_override() -> Optional[Path]:
    """Override one model section in place (``/model``), preserving the rest;
    returns the written Path, or None if cancelled or there's no config."""
    dest = config_path()
    if not dest.is_file():
        print_error("No config.yaml found. Run /config to create one first.")
        return None

    text = dest.read_text()

    # Vision/embeddings are optional; only offer them when present (LLM always is).
    available = {"1": True}
    available["2"] = _get_field(text, "VISION_MODEL_ID", "NAME") is not None
    available["3"] = _get_field(text, "EMBED_MODEL_ID", "NAME") is not None

    labels = {
        k: label for k, (_, label, _) in _MODEL_SECTIONS.items() if available.get(k)
    }
    # On a TTY the picker dialog carries the title + overview (info=); print here
    # only for the non-TTY fallback.
    if not _is_tty():
        print()
        print("=" * 64)
        print("  Override a model")
        print("=" * 64)
        _print_current_setup(text)
        print("\n  Which model do you want to change? Only that section is edited;")
        print("  everything else in your config is left as-is.\n")
        for k, lbl in labels.items():
            print(f"    {k}) {lbl}")
        print(_CANCEL_HINT)
    try:
        valid = set(labels)
        choice = _ask_choice(
            "Which model to change?",
            valid,
            "1",
            labels=labels,
            info=_current_setup_text(text),
        )
        section, label, is_llm = _MODEL_SECTIONS[choice]
        new_text = _prompt_model_section(text, section, is_llm)
    except (KeyboardInterrupt, _Cancelled):
        print("\n  Cancelled. Config left untouched.")
        return None

    if new_text == text:
        print("  No changes made.")
        return None

    dest.write_text(new_text)
    print(f"\n  Updated {label} in:\n    {dest}")
    print("  Inference parameters were reset to model defaults for this change;")
    print("  use /params to tune them. For the full per-provider parameter list,")
    print("  see the README's 'Model Parameters' section.")
    print("=" * 64 + "\n")
    return dest
