"""Centralized filesystem paths for the assistant.

All persistent state lives under a single app-home directory so it's easy to
find, back up, or relocate:

    ~/.mnemoai/
    ├── config/                             # config.yaml + provider examples
    │   ├── config.yaml                     # user config (installed CLI)
    │   ├── config.yaml*.example            # bundled examples (copied here to read)
    │   └── prompt.yaml                     # application prompts
    ├── mcp/                                # external MCP servers
    │   ├── mcp.json                        # optional, user-created
    │   └── mcp.json.example                # bundled example (copied here to read)
    ├── plans/plan_<ts>.md                  # approved plan-mode plans
    ├── skills                              # skills folder
    ├── STEERING.md                         # user-authored always-on instructions
    ├── tasks/                              # background-task output
    └── {profile}/                          # per-user-profile data
        ├── conversations/  todos/  rag_*  chunk_cache_*  profile JSON
        └── models/{model}/                # per-chat-model memory
            ├── episodic_memory/
            └── playbook/

Override the root with ``$MNEMOAI_HOME``. The config file location
can additionally be overridden with ``$MNEMOAI_CONFIG``.
"""

import os
import re
import shutil
from pathlib import Path

DEFAULT_HOME_DIRNAME = ".mnemoai"


def app_home() -> Path:
    """Return the root app-home directory (created), honoring the env override.

    ``$MNEMOAI_HOME`` overrides the default ``~/.mnemoai``.
    """
    env_home = os.environ.get("MNEMOAI_HOME")
    home = Path(env_home).expanduser() if env_home else Path.home() / DEFAULT_HOME_DIRNAME
    home.mkdir(parents=True, exist_ok=True)
    return home


def config_dir() -> Path:
    """Directory holding config.yaml and the bundled config examples (created)."""
    d = app_home() / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mcp_dir() -> Path:
    """Directory holding mcp.json and the bundled mcp example (created)."""
    d = app_home() / "mcp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_path() -> Path:
    """Default config.yaml location: ``<app_home>/config/config.yaml`` (not auto-created)."""
    return config_dir() / "config.yaml"


def legacy_config_path() -> Path:
    """Pre-subfolder config location (``<app_home>/config.yaml``), read-only fallback.

    Kept so installs created before the ``config/`` subfolder still load without
    re-running setup. New configs are always written to :func:`config_path`.
    """
    return app_home() / "config.yaml"


def prompts_path() -> Path:
    """Location of the LLM prompts file: ``<app_home>/config/prompts.yaml``.

    All model-facing prompts (system, routing, orchestrator, aggregator, and the
    compaction summary prompts) live here, separate from ``config.yaml`` which
    holds only configuration. Seeded from the bundled template on first run.
    """
    return config_dir() / "prompts.yaml"


def mcp_config_path() -> Path:
    """Location of the external MCP servers config: ``<app_home>/mcp/mcp.json``.

    Holds extra MCP servers to launch alongside mnemoai's built-in server, in
    the same ``{"mcpServers": {...}}`` schema. ``$MNEMOAI_HOME`` moves it with 
    the rest of the app home. Not auto-created.
    """
    return mcp_dir() / "mcp.json"


def legacy_mcp_config_path() -> Path:
    """Pre-subfolder mcp.json location (``<app_home>/mcp.json``), read-only fallback."""
    return app_home() / "mcp.json"


def seed_example_files() -> None:
    """Copy the package's bundled ``*.example`` templates into the app home.

    Gives users browsable examples right next to their live files:
    ``config/`` gets the ``config.yaml*.example`` templates and ``mcp/`` gets
    ``mcp.json.example``. Idempotent and non-destructive — only copies an
    example that isn't already present, and never touches ``config.yaml`` /
    ``mcp.json``. The configurator still reads the canonical templates from the
    package, so these copies are purely for the user to read.
    """
    pkg_templates = Path(__file__).resolve().parent  # mnemoai/utils/
    try:
        for example in pkg_templates.glob("config.yaml*.example"):
            dest = config_dir() / example.name
            if not dest.exists():
                shutil.copyfile(example, dest)
        # prompts.yaml is the live prompts file (not a *.example): seed the
        # actual file so the app has prompts out of the box. Never overwrite.
        prompts_template = pkg_templates / "prompts.yaml"
        if prompts_template.is_file():
            dest = prompts_path()
            if not dest.exists():
                shutil.copyfile(prompts_template, dest)
        mcp_example = pkg_templates / "mcp.json.example"
        if mcp_example.is_file():
            dest = mcp_dir() / mcp_example.name
            if not dest.exists():
                shutil.copyfile(mcp_example, dest)
        # Seed the bundled example skill(s) into the skills dir so the feature is
        # discoverable out of the box. Per-skill (like the config *.example files
        # above): copy any bundled skill whose directory doesn't exist yet, so a
        # NEW bundled skill also reaches an EXISTING install on upgrade. Never
        # overwrites a user's own skills. Trade-off: a bundled skill the user
        # deleted reappears on upgrade — acceptable for a refreshed example.
        skills_template_root = pkg_templates / "skills_example"
        if skills_template_root.is_dir():
            dest_root = skills_dir()
            for skill_dir in skills_template_root.iterdir():
                if skill_dir.is_dir():
                    dest = dest_root / skill_dir.name
                    if not dest.exists():
                        shutil.copytree(skill_dir, dest)
    except OSError:
        # Seeding examples is a convenience; never let it block startup.
        pass


def plans_dir() -> Path:
    """Directory for plan-mode state (created)."""
    d = app_home() / "plans"
    d.mkdir(parents=True, exist_ok=True)
    return d


def skills_dir() -> Path:
    """Directory holding agent skills, one ``<name>/SKILL.md`` per skill (created).

    Seeded with a bundled example on first run by :func:`seed_example_files`.
    """
    d = app_home() / "skills"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tasks_dir() -> Path:
    """Directory for background-task output (created)."""
    d = app_home() / "tasks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    """Directory for app log files (created)."""
    d = app_home() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mcp_log_path() -> Path:
    """Log file capturing MCP server subprocess stderr, so their noise (e.g.
    ``npm notice`` from an ``npx`` server) stays out of the terminal but remains
    available for debugging a server that fails to start."""
    return logs_dir() / "mcp.log"


def _profile_name() -> str:
    """Resolve the active profile name from config (lazy import to avoid cycles)."""
    from mnemoai.utils.config import config

    return config.get("PROFILE", {}).get("NAME", "default")


def profile_dir(profile: str = None) -> Path:
    """Per-profile data directory (created).

    Args:
        profile: Profile name; resolved from config when omitted.
    """
    name = profile or _profile_name()
    d = app_home() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def conversations_dir(profile: str = None) -> Path:
    """Per-profile directory for saved conversations (created).

    ``/save`` writes here and ``/load`` lists from here. Lives under the profile
    dir as ``conversations/`` (see the app-home layout), keeping saved chats out
    of the profile root.

    Args:
        profile: Profile name; resolved from config when omitted.
    """
    d = profile_dir(profile) / "conversations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def memory_file_path(profile: str = None) -> Path:
    """Path to the curated ``MEMORY.md`` (profile-scoped, not auto-created).

    A small, bounded markdown file the agent maintains itself (Hermes-style) and
    that is injected whole into the system prompt at session start. Profile-
    scoped — shared across chat models — since it holds user/environment facts,
    not model-specific learnings (those live under :func:`model_dir`).
    """
    return profile_dir(profile) / "MEMORY.md"


def global_steering_path() -> Path:
    """The user-level ``STEERING.md`` at the app-home root (not auto-created)."""
    return app_home() / "STEERING.md"


def steering_files(cwd: Path = None) -> list:
    """Discover STEERING.md files in precedence order (low → high priority).

    User-authored, always-on instructions. Resolution, broadest to most
    specific:

      1. ``<app_home>/STEERING.md`` — global/user (applies everywhere)
      2. ``./STEERING.md`` walking from ``cwd`` UP to the project root (the first
         ancestor containing ``.git``, else the filesystem root), collected
         root-first so a deeper (more specific) file is applied last.

    Returns only existing files, de-duplicated, in apply order. Tolerant: any
    resolution error yields an empty list rather than raising.
    """
    found: list = []
    try:
        g = global_steering_path()
        if g.is_file():
            found.append(g)

        start = Path(cwd).expanduser() if cwd else Path.cwd()
        start = start.resolve()
        # Walk up collecting dirs until a .git root (inclusive) or the fs root.
        chain = [start] + list(start.parents)
        project_root_idx = None
        for i, d in enumerate(chain):
            if (d / ".git").exists():
                project_root_idx = i
                break
        ancestors = chain[: project_root_idx + 1] if project_root_idx is not None else chain
        # Apply root-first (broadest) → cwd-last (most specific).
        for d in reversed(ancestors):
            f = d / "STEERING.md"
            if f.is_file() and f not in found:
                found.append(f)
    except Exception:
        return found
    return found


def sanitize_model_name(name: str) -> str:
    """Make a model id safe to use as a directory name.

    Model ids contain characters that are awkward or illegal in paths
    (``/``, ``:``, spaces, etc.). Collapse anything outside ``[A-Za-z0-9._-]``
    to ``_``.
    """
    if not name:
        return "default"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return safe or "default"


def model_dir(model_name: str, profile: str = None) -> Path:
    """Per-(profile, chat-model) directory for episodic memory + playbook (created).

    Scoping memory by model keeps a store built with one model from
    contaminating another.
    """
    d = profile_dir(profile) / "models" / sanitize_model_name(model_name)
    d.mkdir(parents=True, exist_ok=True)
    return d
