"""First-run interactive configurator.

When no ``config.yaml`` can be resolved, walk the user through creating one at
``<app_home>/config.yaml`` by picking a provider template and filling in a few
fields. The rich prompt blocks and comments in the templates are preserved:
edits are line-targeted, never a YAML round-trip (which would drop comments).

Entry points:
- ``config_exists()`` — True if a config is already resolvable (skip setup)
- ``run_first_run_setup()`` — interactive flow; returns the written Path or None
"""

import getpass
import re
from pathlib import Path
from typing import Optional

from utils.paths import config_path

# Provider key -> (template filename, human label, default chat model)
_PROVIDERS = {
    "1": ("ollama", "config.yaml.example", "Ollama (local models)", "qwen3.5:4b"),
    "2": ("bedrock", "config.yaml.bedrock.example", "AWS Bedrock", "global.anthropic.claude-opus-4-8"),
    "3": ("mantle", "config.yaml.bedrock.mantle.example", "AWS Bedrock Mantle", "qwen.qwen3-32b"),
}

# Mantle API protocol choice -> (value, description). Mirrors
# models.mantle_factory.VALID_PROTOCOLS.
_MANTLE_PROTOCOLS = {
    "1": ("chat_completions", "OpenAI Chat Completions (/v1) — most models"),
    "2": ("responses", "OpenAI Responses (/openai/v1) — e.g. openai.gpt-5.x"),
    "3": ("anthropic", "Anthropic Messages (/anthropic) — Claude models"),
}


def _templates_dir() -> Path:
    """Directory holding the packaged config templates (ships next to this module)."""
    return Path(__file__).resolve().parent


def config_exists() -> bool:
    """True if a config is already resolvable, so first-run setup should be skipped.

    Delegates to the same resolution the app uses ($PERSONAL_AI_ASSISTANT_CONFIG
    -> <app_home>/config.yaml -> repo fallback), so any of those existing counts.
    """
    from utils.config import Config

    return Config._resolve_config_path() is not None


def _set_in_section(text: str, section: str, key: str, value: str) -> str:
    """Replace ``  key: ...`` for the first indented ``key`` inside top-level ``section``.

    Only the first occurrence within the section is changed; the rest of the
    file (including identically-named keys in other sections) is untouched.
    """
    out = []
    in_section = False
    done = False
    for line in text.splitlines():
        # A top-level key starts at column 0 and is "NAME:" or "NAME: value".
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
    """Set ``key`` inside ``section``, inserting it if absent.

    Like ``_set_in_section`` when the key already exists; otherwise the key is
    added right after the section header, indented to match the section's
    other children (defaults to two spaces). Only the first matching top-level
    section is touched.
    """
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
        # Track the section's child indentation from its first indented line.
        if in_section and line and line[0].isspace():
            child_indent = line[: len(line) - len(line.lstrip())]
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _indent_of(line: str) -> int:
    """Number of leading spaces on a line."""
    return len(line) - len(line.lstrip())


def _find_section(lines: list, section: str) -> int:
    """Return the index of the first header line ``<indent>section:`` (no inline
    value), at any depth, or -1. Used by the depth-agnostic field editors so
    nested sections like ``RAG.EMBED_MODEL_ID`` are reachable.
    """
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == f"{section}:" or stripped.startswith(f"{section}:") and not stripped.split(":", 1)[1].strip():
            return i
    return -1


def _get_field(text: str, section: str, key: str) -> Optional[str]:
    """Read ``key`` from within ``section`` at any depth (stripped of comments).

    Scans the section body — lines indented deeper than the header — stopping
    when indentation returns to the header's level or shallower.
    """
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
    """Set ``key`` within ``section`` at any depth, inserting it if absent.

    Replaces the existing line in place, or inserts right after the header at
    the body's indentation. Only the first matching section is touched.
    """
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
            lines[j] = f"{m.group(1)}{key}: {value}"
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    # Not found in body -> insert just after the header at the body indent.
    indent = body_indent if body_indent is not None else header_indent + 2
    lines.insert(idx + 1, f"{' ' * indent}{key}: {value}")
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _remove_field(text: str, section: str, key: str) -> str:
    """Remove ``key`` from within ``section`` at any depth, if present.

    No-op when the section or key is absent. Used to make an optional field
    (e.g. MAX_TOKENS) fall back to the provider default by dropping the line.
    """
    lines = text.splitlines()
    idx = _find_section(lines, section)
    if idx < 0:
        return text
    header_indent = _indent_of(lines[idx])
    for j in range(idx + 1, len(lines)):
        line = lines[j]
        if line.strip() and _indent_of(line) <= header_indent:
            break  # left the section body
        if re.match(rf"\s+{re.escape(key)}:(?:\s.*)?$", line):
            del lines[j]
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    return text


def _remove_top_section(text: str, section: str) -> str:
    """Remove a top-level ``section:`` block (header + its indented body).

    Also drops immediately-preceding comment lines that belong to the section.
    Used to drop an optional section (e.g. VISION_MODEL_ID) when the user opts
    out. Only top-level sections are handled.
    """
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(rf"{re.escape(section)}:\s*$", line) or (
            re.match(rf"{re.escape(section)}:", line)
            and not line.split(":", 1)[1].strip()
        ):
            # Drop any contiguous comment lines we already emitted for it.
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


def _get_in_section(text: str, section: str, key: str) -> Optional[str]:
    """Read the value of the first indented ``key`` inside top-level ``section``.

    Returns the inline value (stripped of trailing comments), or None if the
    key isn't present. Mirrors ``_set_in_section``'s targeting.
    """
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


def _ask(prompt: str, default: Optional[str] = None) -> Optional[str]:
    """Prompt for a value, returning ``default`` on empty input or EOF."""
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"  {prompt}{suffix}: ").strip()
    except EOFError:
        print()
        return default
    return val or default


def _ask_bool(prompt: str, default: bool) -> bool:
    """Prompt for a yes/no answer, defaulting to ``default`` on empty input/EOF."""
    hint = "Y/n" if default else "y/N"
    try:
        val = input(f"  {prompt} ({hint}): ").strip().lower()
    except EOFError:
        print()
        return default
    if not val:
        return default
    return val.startswith("y")


def _set_bool(text: str, key: str, value: bool, section: Optional[str] = None) -> str:
    """Set a boolean toggle (``true``/``false``), top-level or within a section."""
    literal = "true" if value else "false"
    if section:
        return _set_in_section(text, section, key, literal)
    return _set_top_level(text, key, literal)


def _truthy(value: Optional[str]) -> bool:
    """Interpret a template's YAML scalar string as a bool (default True)."""
    return (value or "true").strip().lower() in ("true", "yes", "on", "1")


def _build_config(provider: str, default_model: str, template_text: str) -> str:
    """Prompt for the fields that vary and patch them into the template text.

    Only fields a typical user needs to change are prompted; everything else
    (RAG/episodic/playbook tuning, the large prompt blocks, etc.) keeps the
    template's values. Each prompt's default is read from the template so a
    user can press Enter through the whole flow.
    """
    text = template_text

    # --- Chat model ---
    print("\n  -- Chat model --")
    model = _ask("Chat model name", default_model)
    if model:
        text = _set_in_section(text, "MODEL_ID", "NAME", model)

    # Connection details (reused for the vision model, which usually runs on
    # the same host/region).
    conn = {}
    if provider == "ollama":
        conn["HOST"] = _ask("Ollama host", _get_in_section(text, "MODEL_ID", "HOST") or "localhost")
        conn["PORT"] = _ask("Ollama port", _get_in_section(text, "MODEL_ID", "PORT") or "11434")
        for k, v in conn.items():
            text = _set_in_section(text, "MODEL_ID", k, v)
    elif provider == "mantle":
        conn["REGION"] = _ask("AWS region", _get_in_section(text, "MODEL_ID", "REGION") or "us-east-1")
        text = _set_in_section(text, "MODEL_ID", "REGION", conn["REGION"])

        # API protocol (which wire format Mantle serves the model under).
        current = _get_in_section(text, "MODEL_ID", "API_PROTOCOL") or "chat_completions"
        default_key = next((k for k, (v, _) in _MANTLE_PROTOCOLS.items() if v == current), "1")
        print("  Mantle API protocol:")
        for k, (_, desc) in _MANTLE_PROTOCOLS.items():
            print(f"    {k}) {desc}")
        pchoice = _ask("Protocol", default_key) or default_key
        if pchoice not in _MANTLE_PROTOCOLS:
            print(f"  '{pchoice}' is not valid; defaulting to chat_completions.")
            pchoice = "1"
        # Protocol is per-model (the vision model may differ), so it's set on
        # MODEL_ID only — not mirrored into the vision section via `conn`.
        text = _set_or_add_in_section(
            text, "MODEL_ID", "API_PROTOCOL", _MANTLE_PROTOCOLS[pchoice][0]
        )
    elif provider == "bedrock":
        print("  Note: Bedrock uses your AWS credentials (run `aws configure`).")

    # MAX_TOKENS (output tokens) is optional — 'none' / blank drops it so the
    # provider default applies.
    text = _prompt_max_tokens(text, "MODEL_ID")

    # Max context window (top-level MAX_CONVERSATION_TOKENS) is mandatory; always
    # written, defaulting to the template's value (or 65536 if missing).
    ctx = _ask(
        "Max context window",
        _get_top_level(text, "MAX_CONVERSATION_TOKENS") or "65536",
    )
    text = _set_top_level_or_add(text, "MAX_CONVERSATION_TOKENS", ctx or "65536")

    # --- Vision model (optional) ---
    print("\n  -- Vision model (image description) --")
    if _ask_bool("Configure a vision model (for image description)?", True):
        vision = _ask("Vision model name", _get_in_section(text, "VISION_MODEL_ID", "NAME"))
        if vision:
            text = _set_in_section(text, "VISION_MODEL_ID", "NAME", vision)
        # Mirror the chat model's connection (host/port or region) into the
        # vision section — it usually runs on the same host/region.
        for k, v in conn.items():
            if _get_in_section(text, "VISION_MODEL_ID", k) is not None:
                text = _set_in_section(text, "VISION_MODEL_ID", k, v)
        # Mantle vision protocol is per-model too, so ask for it separately.
        if provider == "mantle":
            current = _get_in_section(text, "VISION_MODEL_ID", "API_PROTOCOL") or "chat_completions"
            default_key = next((k for k, (val, _) in _MANTLE_PROTOCOLS.items() if val == current), "1")
            print("  Vision API protocol:")
            for k, (_, desc) in _MANTLE_PROTOCOLS.items():
                print(f"    {k}) {desc}")
            vchoice = _ask("Protocol", default_key) or default_key
            if vchoice not in _MANTLE_PROTOCOLS:
                print(f"  '{vchoice}' is not valid; defaulting to chat_completions.")
                vchoice = "1"
            text = _set_or_add_in_section(
                text, "VISION_MODEL_ID", "API_PROTOCOL", _MANTLE_PROTOCOLS[vchoice][0]
            )
        text = _prompt_max_tokens(text, "VISION_MODEL_ID")
    else:
        # Drop the vision section entirely; image description stays disabled
        # until the user adds VISION_MODEL_ID back (e.g. via /config or /model).
        text = _remove_top_section(text, "VISION_MODEL_ID")
        print("  -> Vision disabled (no VISION_MODEL_ID).")

    # --- Profile ---
    print("\n  -- Profile --")
    profile = _ask("Profile name (isolates your data)", getpass.getuser() or "default")
    if profile:
        text = _set_in_section(text, "PROFILE", "NAME", profile)

    # --- Web search (Brave) ---
    print("\n  -- Features --")
    brave = _ask("Brave Search API key (optional, press Enter to skip)", "")
    if brave:
        text = _set_top_level(text, "BRAVE_API_KEY", brave)
        text = _set_bool(text, "ENABLE_WEB_SEARCH", True)
    else:
        text = _set_bool(text, "ENABLE_WEB_SEARCH", False)
        print("  -> Web search disabled (no API key). Set BRAVE_API_KEY and")
        print("     ENABLE_WEB_SEARCH: true later to enable it.")

    # --- Other feature toggles (default from the template) ---
    text = _set_bool(text, "ENABLE_RAG", _ask_bool("Enable RAG (document indexing & search)?", _truthy(_get_top_level(text, "ENABLE_RAG"))))
    text = _set_bool(text, "ENABLE_EPISODIC_MEMORY", _ask_bool("Enable episodic memory (learn from past tasks)?", _truthy(_get_top_level(text, "ENABLE_EPISODIC_MEMORY"))))
    text = _set_bool(text, "ENABLE_PLAYBOOK", _ask_bool("Enable ACE playbook (learn strategies)?", _truthy(_get_top_level(text, "ENABLE_PLAYBOOK"))))
    text = _set_bool(text, "ENABLE_WEB_CRAWL", _ask_bool("Enable web crawler (fetch URLs)?", _truthy(_get_top_level(text, "ENABLE_WEB_CRAWL"))))

    routing = _ask_bool("Enable query routing (route queries to tool subsets)?", _truthy(_get_top_level(text, "ENABLE_ROUTING")))
    text = _set_bool(text, "ENABLE_ROUTING", routing)
    if routing:
        orchestration = _ask_bool("Enable orchestration (decompose complex tasks)?", _truthy(_get_top_level(text, "ENABLE_ORCHESTRATION")))
    else:
        orchestration = False
        print("  -> Orchestration disabled (it requires query routing).")
    text = _set_bool(text, "ENABLE_ORCHESTRATION", orchestration)

    text = _set_bool(text, "USE_PROFILING", _ask_bool("Enable user profiling (personalized responses)?", _truthy(_get_in_section(text, "PROFILE", "USE_PROFILING"))), section="PROFILE")

    return text


def _run_configurator(dest: Path) -> Optional[Path]:
    """Shared flow: pick a provider, fill the template, write to ``dest``.

    The caller handles any "first run?" / "overwrite?" gating and confirmation
    before calling this. Returns the written Path, or None if a template was
    missing.
    """
    print("\n  Choose your LLM provider:")
    for k, (_, _, label, _) in _PROVIDERS.items():
        print(f"    {k}) {label}")
    choice = _ask("Provider", "1") or "1"
    if choice not in _PROVIDERS:
        print(f"  '{choice}' is not a valid choice; defaulting to Ollama.")
        choice = "1"

    provider, template_file, label, default_model = _PROVIDERS[choice]
    template_path = _templates_dir() / template_file
    if not template_path.is_file():
        print(f"  Template not found: {template_path}. Cannot continue setup.")
        return None

    print(f"\n  Configuring for: {label}\n")
    template_text = template_path.read_text()
    config_text = _build_config(provider, default_model, template_text)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(config_text)

    print(f"\n  Config written to:\n    {dest}")
    print(f"\n  This file lives in your app home ({dest.parent}), which also")
    print("  holds the rest of your runtime data (plans, tasks, conversations,")
    print("  RAG indexes, episodic memory, and the ACE playbook).")
    print("\n  Only the most common settings were configured here. The file")
    print("  contains many more options you can edit any time — embedding model,")
    print("  RAG / episodic-memory / playbook tuning, retry and compaction limits,")
    print("  and the routing / orchestrator / system prompts. See the README's")
    print("  Configuration section for the full reference.")
    print("=" * 64 + "\n")
    return dest


def run_first_run_setup() -> Optional[Path]:
    """Interactively create a first ``config.yaml``.

    Returns the written Path, or None if the user declined or a template was
    missing. Safe to call only when stdin is interactive (caller's check).
    """
    dest = config_path()

    print()
    print("=" * 64)
    print("  Welcome to personal-ai-assistant — first-run setup")
    print("=" * 64)
    print(f"  No config found. Let's create one at:\n    {dest}\n")

    try:
        answer = _ask("Set up a config now? (Y/n)", "Y")
        if answer and answer.strip().lower().startswith("n"):
            print("  Skipped. Copy a template to that path manually to get started.")
            return None

        return _run_configurator(dest)
    except KeyboardInterrupt:
        print("\n  Setup cancelled. No config was written.")
        return None


def run_reconfigure() -> Optional[Path]:
    """Re-run the configurator over an existing config (the ``/config`` command).

    Warns that the current ``config.yaml`` will be overwritten and asks for
    confirmation before restarting the setup flow. Returns the written Path,
    or None if the user declined or a template was missing.
    """
    dest = config_path()

    print()
    print("=" * 64)
    print("  Reconfigure personal-ai-assistant")
    print("=" * 64)
    if dest.is_file():
        print(f"  WARNING: this will OVERWRITE your existing config at:\n    {dest}")
        print("  Your current settings will be replaced by the answers you give")
        print("  now. (Other runtime data — conversations, memory, etc. — is kept.)")
    else:
        print(f"  No config found yet; this will create one at:\n    {dest}")

    try:
        answer = _ask("Reconfigure now? (y/N)", "N")
        if not (answer and answer.strip().lower().startswith("y")):
            print("  Cancelled. Existing config left untouched.")
            return None

        return _run_configurator(dest)
    except KeyboardInterrupt:
        print("\n  Reconfigure cancelled. Existing config left untouched.")
        return None


# Model sections that /model can override: key -> (config section, label,
# whether it's the chat LLM — which gets the context-window prompt).
# Embeddings lives nested under RAG.
_MODEL_SECTIONS = {
    "1": ("MODEL_ID", "Chat model (LLM)", True),
    "2": ("VISION_MODEL_ID", "Vision model", False),
    "3": ("EMBED_MODEL_ID", "Embeddings model", False),
}


def _section_summary(text: str, section: str) -> Optional[str]:
    """One-line summary of a model section, or None if it isn't configured.

    Example: ``mantle / anthropic.claude-haiku-4-5 (us-east-1, anthropic)``.
    """
    name = _get_field(text, section, "NAME")
    if not name:
        return None
    typ = _get_field(text, section, "TYPE") or "?"
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


def _prompt_max_tokens(text: str, section: str) -> str:
    """Prompt for the optional MAX_TOKENS of a model section.

    MAX_TOKENS (max output tokens) is optional — when unset the provider's own
    default applies. Convention:
      * Enter  -> keep the current value (or stay unset if there is none)
      * a number -> set it
      * 'none' -> remove the key (fall back to the provider default)
    """
    current = _get_field(text, section, "MAX_TOKENS")
    default = current if current else "none"
    answer = _ask("Max output tokens (number, or 'none' for provider default)", default)
    if answer is None:
        return text
    answer = answer.strip().lower()
    if answer in ("none", ""):
        return _remove_field(text, section, "MAX_TOKENS")
    return _set_field(text, section, "MAX_TOKENS", answer)


def _print_current_setup(text: str) -> None:
    """Print the current chat/vision/embeddings models. Vision and embeddings
    are optional and only shown when present in the config.
    """
    print("  Current setup:")
    print(f"    Chat (LLM):  {_section_summary(text, 'MODEL_ID') or '(not set)'}")
    vision = _section_summary(text, "VISION_MODEL_ID")
    print(f"    Vision:      {vision if vision else '(not configured)'}")
    embeddings = _section_summary(text, "EMBED_MODEL_ID")
    print(f"    Embeddings:  {embeddings if embeddings else '(not configured)'}")


def _prompt_model_section(text: str, section: str, is_llm: bool) -> str:
    """Prompt for one model section's fields and patch them into ``text``.

    Defaults are read from the current config so the user can press Enter to
    keep a value. The provider type itself is editable, so a section can be
    switched between providers (e.g. ollama -> bedrock). Connection prompts
    follow the chosen type. MAX_TOKENS is prompted for the chat (LLM) and
    vision models; the context window (MAX_CONVERSATION_TOKENS) only for the
    chat model (``is_llm``). Embeddings has neither.
    """
    cur_type = (_get_field(text, section, "TYPE") or "ollama").lower()
    print(f"\n  Provider type for this model (current: {cur_type})")
    print("    options: ollama | bedrock | mantle | openai | sagemaker | litellm")
    new_type = (_ask("Type", cur_type) or cur_type).lower()
    text = _set_field(text, section, "TYPE", new_type)

    name = _ask("Model name", _get_field(text, section, "NAME"))
    if name:
        text = _set_field(text, section, "NAME", name)

    if new_type == "ollama":
        host = _ask("Ollama host", _get_field(text, section, "HOST") or "localhost")
        port = _ask("Ollama port", _get_field(text, section, "PORT") or "11434")
        text = _set_field(text, section, "HOST", host)
        text = _set_field(text, section, "PORT", port)
    elif new_type in ("bedrock", "mantle", "sagemaker"):
        region = _ask("AWS region", _get_field(text, section, "REGION") or "us-east-1")
        text = _set_field(text, section, "REGION", region)
        if new_type == "mantle":
            existing = _get_field(text, section, "API_PROTOCOL")
            current = existing or "chat_completions"
            default_key = next((k for k, (v, _) in _MANTLE_PROTOCOLS.items() if v == current), "1")
            print("  Mantle API protocol:")
            for k, (_, desc) in _MANTLE_PROTOCOLS.items():
                print(f"    {k}) {desc}")
            pchoice = _ask("Protocol", default_key) or default_key
            if pchoice not in _MANTLE_PROTOCOLS:
                print(f"  '{pchoice}' is not valid; defaulting to chat_completions.")
                pchoice = "1"
            chosen = _MANTLE_PROTOCOLS[pchoice][0]
            # Only write when it changes — avoids inserting an explicit
            # chat_completions line (the implicit default) when the user just
            # Enters through, so "no input" stays a true no-op.
            if chosen != existing and not (existing is None and chosen == "chat_completions"):
                text = _set_field(text, section, "API_PROTOCOL", chosen)

    # MAX_TOKENS (output tokens) is optional, for chat and vision (not embeddings).
    if section in ("MODEL_ID", "VISION_MODEL_ID"):
        text = _prompt_max_tokens(text, section)

    if is_llm:
        # Max context window is the top-level MAX_CONVERSATION_TOKENS (feeds
        # Ollama num_ctx and the compaction budget). It is mandatory, so it's
        # always written; default to the current value (or 65536 if unset).
        ctx = _ask(
            "Max context window",
            _get_top_level(text, "MAX_CONVERSATION_TOKENS") or "65536",
        )
        text = _set_top_level_or_add(text, "MAX_CONVERSATION_TOKENS", ctx or "65536")

    return text


def run_model_override() -> Optional[Path]:
    """Override just one model section in the existing config (``/model``).

    Asks which model to change — chat (LLM), vision, or embeddings (the last
    only when RAG/embeddings is configured) — then edits only that section in
    place, preserving everything else. Returns the written Path, or None if the
    user cancelled or there's no config to edit.
    """
    from utils.config import config

    dest = config_path()
    if not dest.is_file():
        print("  No config.yaml found. Run /config to create one first.")
        return None

    text = dest.read_text()

    # Vision and embeddings are optional; only offer them when present in the
    # config (chat/LLM is always present). Track which choices are available.
    available = {"1": True}
    available["2"] = _get_field(text, "VISION_MODEL_ID", "NAME") is not None
    available["3"] = _get_field(text, "EMBED_MODEL_ID", "NAME") is not None

    print()
    print("=" * 64)
    print("  Override a model")
    print("=" * 64)
    _print_current_setup(text)
    print("\n  Which model do you want to change? Only that section is edited;")
    print("  everything else in your config is left as-is.\n")
    for k, (_, label, _) in _MODEL_SECTIONS.items():
        if not available.get(k):
            continue
        print(f"    {k}) {label}")

    try:
        choice = _ask("Model", "1") or "1"
        if choice not in _MODEL_SECTIONS:
            print(f"  '{choice}' is not a valid choice. Cancelled.")
            return None
        if not available.get(choice):
            section_label = _MODEL_SECTIONS[choice][1]
            print(f"  {section_label} is not configured. Cancelled.")
            return None

        section, label, is_llm = _MODEL_SECTIONS[choice]
        print(f"\n  -- {label} --")
        new_text = _prompt_model_section(text, section, is_llm)
    except KeyboardInterrupt:
        print("\n  Cancelled. Config left untouched.")
        return None

    if new_text == text:
        print("  No changes made.")
        return None

    dest.write_text(new_text)
    print(f"\n  Updated {label} in:\n    {dest}")
    print("=" * 64 + "\n")
    return dest
