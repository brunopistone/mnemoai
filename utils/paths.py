"""Centralized filesystem paths for the assistant.

All persistent state lives under a single app-home directory so it's easy to
find, back up, or relocate (similar to Claude Code's ``~/.claude``):

    ~/.personal-ai-assistant/
    ├── config.yaml                         # user config (installed CLI)
    ├── plans/current_plan.json             # plan-mode state
    ├── tasks/                              # background-task output
    └── {profile}/                          # per-user-profile data
        ├── conversations/  todos/  rag_*  chunk_cache_*  profile JSON
        └── models/{model}/                # per-chat-model memory
            ├── episodic_memory/
            └── playbook/

Override the root with ``$PERSONAL_AI_ASSISTANT_HOME``. The config file location
can additionally be overridden with ``$PERSONAL_AI_ASSISTANT_CONFIG``.
"""

import os
import re
from pathlib import Path


DEFAULT_HOME_DIRNAME = ".personal-ai-assistant"


def app_home() -> Path:
    """Return the root app-home directory (created), honoring the env override.

    ``$PERSONAL_AI_ASSISTANT_HOME`` overrides the default ``~/.personal-ai-assistant``.
    """
    env_home = os.environ.get("PERSONAL_AI_ASSISTANT_HOME")
    home = Path(env_home).expanduser() if env_home else Path.home() / DEFAULT_HOME_DIRNAME
    home.mkdir(parents=True, exist_ok=True)
    return home


def config_path() -> Path:
    """Default config.yaml location inside the app home (not auto-created)."""
    return app_home() / "config.yaml"


def plans_dir() -> Path:
    """Directory for plan-mode state (created)."""
    d = app_home() / "plans"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tasks_dir() -> Path:
    """Directory for background-task output (created)."""
    d = app_home() / "tasks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _profile_name() -> str:
    """Resolve the active profile name from config (lazy import to avoid cycles)."""
    from utils.config import config

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
