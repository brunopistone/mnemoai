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

    max_tokens = _ask("Max output tokens", _get_in_section(text, "MODEL_ID", "MAX_TOKENS"))
    if max_tokens:
        text = _set_in_section(text, "MODEL_ID", "MAX_TOKENS", max_tokens)

    # --- Vision model ---
    print("\n  -- Vision model (image description) --")
    vision = _ask("Vision model name", _get_in_section(text, "VISION_MODEL_ID", "NAME"))
    if vision:
        text = _set_in_section(text, "VISION_MODEL_ID", "NAME", vision)
    # Mirror the chat model's connection (host/port or region) into the vision
    # section — the vision model usually runs on the same host/region.
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
