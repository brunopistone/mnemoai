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

from mnemoai.utils.console import print_error
from mnemoai.utils.paths import config_path

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

# Human-facing label for each provider TYPE. The stored config value is the
# canonical key (left); the label (right) is what the menus show — e.g. Mantle
# is a Bedrock access path, so it reads "bedrock-mantle". Unlisted types fall
# back to their key.
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

# Which config keys each provider actually consumes is owned by the model layer
# (models/provider_params.py), derived from the controller init methods. The
# configurator imports it so the two never drift; see _prune_unsupported_params.
# MAX_CONVERSATION_TOKENS is a top-level key (not part of a model section), so
# it's never pruned by the section-level logic.


def _templates_dir() -> Path:
    """Directory holding the packaged config templates (ships next to this module)."""
    return Path(__file__).resolve().parent


def config_exists() -> bool:
    """True if a config is already resolvable, so first-run setup should be skipped.

    Delegates to the same resolution the app uses ($MNEMOAI_CONFIG
    -> <app_home>/config.yaml -> repo fallback), so any of those existing counts.
    """
    from mnemoai.utils.config import Config

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
            key_indent = len(m.group(1))
            # If the old value was a block (list/mapping), its lines are indented
            # deeper than the key — drop them so replacing e.g. a multi-line
            # STOP list with an inline value doesn't leave orphaned items behind.
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
    """Remove ``key`` (and its block) from within ``section`` at any depth.

    Drops the ``key:`` line plus any deeper-indented continuation lines (a
    list like ``STOP:`` or a nested mapping) and any immediately-preceding
    comment lines describing it. No-op when the section or key is absent. Used
    to make an optional field fall back to the provider default, or to strip
    provider-specific params when switching providers.
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
        m = re.match(rf"(\s+){re.escape(key)}:(?:\s.*)?$", line)
        if m:
            key_indent = len(m.group(1))
            start = j
            # Absorb preceding comment lines that belong to this key.
            while start - 1 > idx and lines[start - 1].strip().startswith("#"):
                start -= 1
            # Absorb the key line + any deeper-indented continuation lines.
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
            continue  # nested deeper (e.g. a list item) — not a direct key
        m = re.match(r"\s*([A-Za-z0-9_]+):", line)
        if m:
            keys.append(m.group(1))
    return keys


def _prune_unsupported_params(text: str, section: str, provider: str) -> str:
    """Drop keys the ``provider`` doesn't consume for ``section``.

    The supported-key registry lives in ``models.provider_params`` (derived
    from the controller init methods), so this prunes any stale parameter left
    over from a previous provider — connection, auth, and inference alike.
    ``NAME``/``TYPE`` are always kept; an unknown provider prunes nothing.
    """
    from mnemoai.models.provider_params import supported_keys

    allowed = supported_keys(section, provider)
    if allowed is None:
        return text  # unknown provider/section — don't touch anything
    keep = allowed | {"NAME", "TYPE"}
    for key in _list_section_keys(text, section):
        if key not in keep:
            text = _remove_field(text, section, key)
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


def _prompt_provider_type(section: str, current: str) -> str:
    """Prompt for a provider TYPE as a numbered menu, returning the chosen key.

    Options come from the registry (so embeddings, which has no Mantle path,
    only offers its own providers) and are shown with their human label
    (``bedrock-mantle`` for ``mantle``). The current type is the default; an
    invalid choice keeps it. Returns the canonical config value (e.g. ``mantle``).
    """
    from mnemoai.models.provider_params import providers

    options = list(providers(section))
    if current not in options:
        # Unknown/legacy current value — still offer it so it can be kept.
        options = [current] + options
    default_key = str(options.index(current) + 1)

    print(f"\n  Provider type for this model (current: {_PROVIDER_LABELS.get(current, current)})")
    for i, prov in enumerate(options, 1):
        print(f"    {i}) {_PROVIDER_LABELS.get(prov, prov)}")
    choice = _ask("Provider", default_key) or default_key
    if not choice.isdigit() or not (1 <= int(choice) <= len(options)):
        print(f"  '{choice}' is not a valid choice; keeping {_PROVIDER_LABELS.get(current, current)}.")
        return current
    return options[int(choice) - 1]


def _prompt_mantle_protocol(text: str, section: str) -> str:
    """Prompt for the Mantle API protocol of ``section`` and set it.

    Skips writing an explicit ``chat_completions`` (the implicit default) when
    it wasn't already present, so an Enter-through stays a no-op.
    """
    existing = _get_field(text, section, "API_PROTOCOL")
    current = existing or "chat_completions"
    default_key = next((k for k, (v, _) in _MANTLE_PROTOCOLS.items() if v == current), "1")
    print("  Mantle API protocol:")
    for k, (_, desc) in _MANTLE_PROTOCOLS.items():
        print(f"    {k}) {desc}")
    choice = _ask("Protocol", default_key) or default_key
    if choice not in _MANTLE_PROTOCOLS:
        print(f"  '{choice}' is not valid; defaulting to chat_completions.")
        choice = "1"
    chosen = _MANTLE_PROTOCOLS[choice][0]
    if chosen != existing and not (existing is None and chosen == "chat_completions"):
        text = _set_field(text, section, "API_PROTOCOL", chosen)
    return text


def _prompt_provider_connection(text: str, section: str, provider: str):
    """Prompt the connection/auth keys ``provider`` needs for ``section``.

    Section-aware via the ``provider_params`` registry: a key is only prompted
    when the provider actually consumes it for this section (e.g. ``INPUT_FORMAT``
    for SageMaker chat but not embeddings; ``HOST``/``PORT`` only for Ollama).
    Used by BOTH ``/config`` and ``/model`` so the two ask the same things.

    Returns ``(text, conn)`` where ``conn`` holds the mirrorable connection
    values (HOST/PORT/REGION) the vision section can reuse.
    """
    from mnemoai.models.provider_params import supported_keys

    allowed = supported_keys(section, provider) or set()
    conn = {}

    if "HOST" in allowed:
        host = _ask("Ollama host", _get_field(text, section, "HOST") or "localhost")
        text = _set_field(text, section, "HOST", host)
        conn["HOST"] = host
    if "PORT" in allowed:
        port = _ask("Ollama port", _get_field(text, section, "PORT") or "11434")
        text = _set_field(text, section, "PORT", port)
        conn["PORT"] = port
    if "REGION" in allowed:
        region = _ask("AWS region", _get_field(text, section, "REGION") or "us-east-1")
        text = _set_field(text, section, "REGION", region)
        conn["REGION"] = region
    if "INPUT_FORMAT" in allowed:
        fmt = _ask(
            "Input format (openai_chat | huggingface)",
            _get_field(text, section, "INPUT_FORMAT") or "openai_chat",
        ) or "openai_chat"
        text = _set_field(text, section, "INPUT_FORMAT", fmt)
    if "API_PROTOCOL" in allowed:
        text = _prompt_mantle_protocol(text, section)
    if provider == "litellm":
        if "API_BASE" in allowed:
            v = _ask("LiteLLM API base URL (optional)", _get_field(text, section, "API_BASE") or "")
            if v:
                text = _set_field(text, section, "API_BASE", v)
        if "API_KEY" in allowed:
            v = _ask("LiteLLM API key (optional, or via the provider's env var)", _get_field(text, section, "API_KEY") or "")
            if v:
                text = _set_field(text, section, "API_KEY", v)
    if provider == "anthropic":
        if "API_KEY" in allowed:
            v = _ask("Anthropic API key (optional, or via ANTHROPIC_API_KEY env var)", _get_field(text, section, "API_KEY") or "")
            if v:
                text = _set_field(text, section, "API_KEY", v)
        if "ENDPOINT_URL" in allowed:
            v = _ask("Anthropic base URL (optional, blank for api.anthropic.com)", _get_field(text, section, "ENDPOINT_URL") or "")
            if v:
                text = _set_field(text, section, "ENDPOINT_URL", v)

    # Credential notes (env-based auth the configurator can't set for the user).
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

    return text, conn


def _build_config(
    provider: str, default_model: str, template_text: str, template_file: str = ""
) -> str:
    """Prompt for the fields that vary and patch them into the template text.

    Only fields a typical user needs to change are prompted; everything else
    (RAG/episodic/playbook tuning, the large prompt blocks, etc.) keeps the
    template's values. Each prompt's default is read from the template so a
    user can press Enter through the whole flow.

    ``template_file`` identifies the source template; openai/sagemaker/litellm
    reuse the Ollama-shaped base (``config.yaml.example``) and have their model
    sections transformed for the chosen provider.
    """
    text = template_text
    transform_from_base = (
        template_file == "config.yaml.example" and provider != "ollama"
    )

    # STOP sequences are kept in the example template for documentation, but
    # not written into a generated config (they're easy to get wrong and only
    # apply to some models). Drop them from both model sections up front.
    text = _remove_field(text, "MODEL_ID", "STOP")
    text = _remove_field(text, "VISION_MODEL_ID", "STOP")

    # The base template (config.yaml.example) is Ollama-shaped. For providers
    # that reuse it (openai/anthropic/sagemaker/litellm), set TYPE and prune the
    # keys the new provider doesn't consume so we start from a clean section.
    # The vision section: OpenAI and Anthropic (Claude) are multimodal, so set
    # vision to the same provider; SageMaker/LiteLLM have no first-class vision
    # path, so leave vision as Ollama (the user can keep or drop it below).
    if transform_from_base:
        text = _set_in_section(text, "MODEL_ID", "TYPE", provider)
        text = _prune_unsupported_params(text, "MODEL_ID", provider)
        if provider in ("openai", "anthropic") and _find_section(text.splitlines(), "VISION_MODEL_ID") >= 0:
            text = _set_in_section(text, "VISION_MODEL_ID", "TYPE", provider)
            text = _prune_unsupported_params(text, "VISION_MODEL_ID", provider)

    # --- Chat model ---
    print("\n  -- Chat model --")
    model = _ask("Chat model name", default_model)
    if model:
        text = _set_in_section(text, "MODEL_ID", "NAME", model)

    # Connection/auth prompts for the provider (shared with /model), plus the
    # mirrorable connection values (HOST/PORT/REGION) for the vision section.
    text, conn = _prompt_provider_connection(text, "MODEL_ID", provider)

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
        # If the vision section is Mantle, its protocol is per-model — ask for it.
        if (_get_in_section(text, "VISION_MODEL_ID", "TYPE") or "").lower() == "mantle":
            text = _prompt_mantle_protocol(text, "VISION_MODEL_ID")
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

    text = _set_bool(text, "REQUIRE_BASH_CONFIRMATION", _ask_bool("Ask for confirmation before each shell command (execute_bash)?", _truthy(_get_top_level(text, "REQUIRE_BASH_CONFIRMATION"))))
    text = _set_bool(text, "REQUIRE_WRITE_CONFIRMATION", _ask_bool("Ask for confirmation before each file write (fs_write/file_edit)?", _truthy(_get_top_level(text, "REQUIRE_WRITE_CONFIRMATION"))))

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
        print_error(f"Template not found: {template_path}. Cannot continue setup.")
        return None

    print(f"\n  Configuring for: {label}\n")
    template_text = template_path.read_text()
    config_text = _build_config(provider, default_model, template_text, template_file)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(config_text)

    print(f"\n  Config written to:\n    {dest}")
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
    """Interactively create a first ``config.yaml``.

    Returns the written Path, or None if the user declined or a template was
    missing. Safe to call only when stdin is interactive (caller's check).
    """
    dest = config_path()

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
    print("  Reconfigure Mnemo AI")
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


def _prompt_max_tokens(text: str, section: str) -> str:
    """Prompt for the optional MAX_TOKENS of a model section.

    MAX_TOKENS (max output tokens) is optional and model-specific, so it
    defaults to ``none`` (the provider default) rather than carrying over the
    previous model's value. Convention:
      * Enter / 'none' -> no MAX_TOKENS (provider default; key removed)
      * a number       -> set it
    """
    answer = _ask("Max output tokens (number, or 'none' for provider default)", "none")
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
    new_type = _prompt_provider_type(section, cur_type)
    text = _set_field(text, section, "TYPE", new_type)

    name = _ask("Model name", _get_field(text, section, "NAME"))
    if name:
        text = _set_field(text, section, "NAME", name)

    # Prompt the provider's connection/auth keys (shared with /config), so
    # /model asks the same mandatory params — region, Mantle protocol,
    # SageMaker input format, LiteLLM API base/key, etc.
    text, _ = _prompt_provider_connection(text, section, new_type)

    # On a provider switch, strip every parameter the new provider doesn't
    # consume (connection, auth, and inference alike), per the model layer's
    # supported-key registry — so no stale, unsupported keys are left behind
    # (e.g. REGION/API_PROTOCOL after mantle -> ollama, HOST/PORT/penalties
    # after ollama -> bedrock). The keys just written for the new provider are
    # in its allowed set, so they survive.
    if new_type != cur_type:
        text = _prune_unsupported_params(text, section, new_type)

    if new_type != "ollama":
        print(
            f"  Note: provider '{_PROVIDER_LABELS.get(new_type, new_type)}' may accept different inference "
            "parameters\n  (temperature, penalties, etc.). Edit config.yaml "
            f"directly to tune the\n  {section} section — see the README's "
            "'Model Parameters' section for the\n  full list of supported "
            "parameters per provider."
        )

    # MAX_TOKENS (output tokens) is optional, for chat and vision (not embeddings).
    if section in ("MODEL_ID", "VISION_MODEL_ID"):
        text = _prompt_max_tokens(text, section)

    if is_llm:
        # Max context window is the top-level MAX_CONVERSATION_TOKENS (feeds
        # Ollama num_ctx and the compaction budget). It is mandatory and
        # model-specific, so it defaults to 65536 rather than carrying over the
        # previous model's value.
        ctx = _ask("Max context window", "65536")
        text = _set_top_level_or_add(text, "MAX_CONVERSATION_TOKENS", ctx or "65536")

    return text


# --- /params: tune the inference/generation parameters of a model -----------
# Metadata for every tunable key the registry may report (provider_params.
# tunable_params). Each entry: (kind, hint) where kind drives validation/parsing
# and hint is shown in the prompt. The *set* of keys offered for a given model
# is the provider's tunable set from the registry (so we never prompt a key the
# provider ignores); this table only supplies how to prompt/validate each.
#   kind: "float" | "int" | "bool" | "list" | one of an enum's allowed values
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
    "REASONING_EFFORT": ("enum:minimal,low,medium,high", "reasoning effort"),
    "THINKING_TOKENS": ("int", "budget for thinking tokens"),
}

# Order params are prompted in (registry membership decides which appear).
_PARAM_ORDER = [
    "TEMPERATURE", "TOP_P", "TOP_K", "MAX_TOKENS",
    "PRESENCE_PENALTY", "FREQUENCY_PENALTY", "REPETITION_PENALTY",
    "STOP", "REASONING", "REASONING_EFFORT", "THINKING_TOKENS", "STREAM",
]

# /params can tune the chat and vision models (embeddings take no inference
# params). key -> (config section, label).
_PARAM_SECTIONS = {
    "1": ("MODEL_ID", "Chat model (LLM)"),
    "2": ("VISION_MODEL_ID", "Vision model"),
}


def _validate_param(key: str, kind: str, raw: str) -> Optional[str]:
    """Validate/normalize a raw answer for a param, or return None if invalid.

    Returns the YAML scalar string to write (e.g. "0.7", "true",
    "[\"</s>\"]"), or None when the input doesn't fit the kind (caller re-asks).
    """
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
        # YAML flow sequence with double-quoted items.
        return "[" + ", ".join('"' + it.replace('"', '\\"') + '"' for it in items) + "]"
    if kind.startswith("enum:"):
        allowed = kind.split(":", 1)[1].split(",")
        return raw if raw in allowed else None
    return raw


def _prompt_one_param(text: str, section: str, key: str, kind: str, hint: str) -> str:
    """Prompt for a single inference param and patch ``text`` accordingly.

    Convention:
      * Enter        -> keep the current value (no change)
      * 'none'       -> clear it (remove the key; provider default applies)
      * a valid value-> set it (re-asks on invalid input)
    """
    current = _get_field(text, section, key)
    cur_disp = current if current is not None else "provider default"
    if kind.startswith("enum:"):
        hint = f"{hint} ({kind.split(':', 1)[1].replace(',', ' | ')})"
    elif kind == "bool":
        hint = f"{hint} (true/false)"
    while True:
        answer = _ask(f"{key} [{hint}] (current: {cur_disp}; 'none' to clear)", "")
        if answer is None or answer.strip() == "":
            return text  # keep current
        if answer.strip().lower() == "none":
            return _remove_field(text, section, key)
        value = _validate_param(key, kind, answer)
        if value is None:
            print(f"    '{answer}' is not a valid {kind.split(':', 1)[0]} value; try again.")
            continue
        return _set_field(text, section, key, value)


def _prompt_inference_params(text: str, section: str) -> str:
    """Prompt every inference param the section's provider accepts.

    The provider TYPE is read from the section; the set of tunable keys comes
    from the registry (``tunable_params``) so only params the provider actually
    consumes are offered. MAX_CONVERSATION_TOKENS (context window) and the
    connection/auth keys are out of scope here — those live in /model.
    """
    from mnemoai.models.provider_params import tunable_params

    provider = (_get_field(text, section, "TYPE") or "ollama").lower()
    tunable = tunable_params(section, provider)
    if not tunable:
        print(f"  Provider '{provider}' exposes no tunable inference parameters here.")
        return text

    print(f"\n  Provider: {provider}. Press Enter to keep a value, type 'none' to")
    print("  clear it (provider default), or enter a new value.\n")
    for key in _PARAM_ORDER:
        if key not in tunable:
            continue
        kind, hint = _PARAM_META.get(key, ("str", key))
        text = _prompt_one_param(text, section, key, kind, hint)
    return text


def run_params_override() -> Optional[Path]:
    """Tune the inference parameters of a configured model (the ``/params`` command).

    Asks which model to tune — chat (LLM) or vision (whichever are configured) —
    then walks the generation params that model's provider accepts (temperature,
    top_p, penalties, reasoning, stop, stream, …), editing only those keys in
    place. Provider/name/connection are unchanged here — use /model for those.
    Returns the written Path, or None if cancelled or nothing changed.
    """
    dest = config_path()
    if not dest.is_file():
        print_error("No config.yaml found. Run /config to create one first.")
        return None

    text = dest.read_text()

    available = {
        "1": _get_field(text, "MODEL_ID", "NAME") is not None,
        "2": _get_field(text, "VISION_MODEL_ID", "NAME") is not None,
    }

    print()
    print("=" * 64)
    print("  Tune inference parameters")
    print("=" * 64)
    _print_current_setup(text)
    print("\n  Which model's parameters do you want to tune? Only inference")
    print("  params are changed; provider, name, and connection stay as-is")
    print("  (use /model for those).\n")
    for k, (_, label) in _PARAM_SECTIONS.items():
        if available.get(k):
            print(f"    {k}) {label}")

    try:
        choice = _ask("Model", "1") or "1"
        if choice not in _PARAM_SECTIONS:
            print(f"  '{choice}' is not a valid choice. Cancelled.")
            return None
        if not available.get(choice):
            print(f"  {_PARAM_SECTIONS[choice][1]} is not configured. Cancelled.")
            return None

        section, label = _PARAM_SECTIONS[choice]
        print(f"\n  -- {label} parameters --")
        new_text = _prompt_inference_params(text, section)
    except KeyboardInterrupt:
        print("\n  Cancelled. Config left untouched.")
        return None

    if new_text == text:
        print("  No changes made.")
        return None

    dest.write_text(new_text)
    print(f"\n  Updated {label} parameters in:\n    {dest}")
    print("  Reload to apply: the change takes effect on the next config reload.")
    print("=" * 64 + "\n")
    return dest


def run_model_override() -> Optional[Path]:
    """Override just one model section in the existing config (``/model``).

    Asks which model to change — chat (LLM), vision, or embeddings (the last
    only when RAG/embeddings is configured) — then edits only that section in
    place, preserving everything else. Returns the written Path, or None if the
    user cancelled or there's no config to edit.
    """
    dest = config_path()
    if not dest.is_file():
        print_error("No config.yaml found. Run /config to create one first.")
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
    print("  For the full list of parameters you can set per provider, see the")
    print("  README's 'Model Parameters' section.")
    print("=" * 64 + "\n")
    return dest
