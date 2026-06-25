"""Configuration management for the AI application."""

import os
from pathlib import Path
from typing import Any

import yaml

from mnemoai.utils.logger import logger


class PromptError(Exception):
    """Raised when a required prompt is missing from prompts.yaml.

    Prompts are read ONLY from prompts.yaml — there are no in-code fallbacks and
    config.yaml is never consulted for prompts. A missing required prompt is a
    hard failure surfaced at startup.
    """


class Config:
    """Configuration manager for the application."""

    _instance = None

    def __new__(cls) -> "Config":
        """Singleton pattern implementation.

        Args:
            cls: Class reference

        Returns:
            Singleton instance
        """
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the configuration."""
        if getattr(self, "_initialized", False):
            return

        self._config_data = {}
        self._prompts_data = {}
        self._load_config()
        self._load_prompts()
        self._initialized = True

    @staticmethod
    def _resolve_config_path() -> "Path | None":
        """Find the config.yaml to load, checking locations in priority order.

        Resolution order (first existing wins):
          1. ``$MNEMOAI_CONFIG`` — explicit override (handy for
             switching between provider configs, e.g. ollama vs bedrock vs mantle)
          2. ``<app_home>/config/config.yaml`` — the user config used by the
             installed CLI; app home defaults to ``~/.mnemoai`` and honors
             ``$MNEMOAI_HOME``
          3. ``<app_home>/config.yaml`` — legacy pre-subfolder location, so
             installs created before the ``config/`` subfolder still load
          4. ``<package>/utils/config.yaml`` — package-relative fallback, so
             running from a checkout (``python main.py``) keeps working unchanged

        Returns:
            The resolved Path, or None if no config exists in any location.
        """
        env_path = os.environ.get("MNEMOAI_CONFIG")
        if env_path:
            return Path(env_path).expanduser()

        from mnemoai.utils.paths import (
            config_path as home_config_path,
        )
        from mnemoai.utils.paths import (
            legacy_config_path,
        )

        user_config = home_config_path()
        if user_config.is_file():
            return user_config

        legacy = legacy_config_path()
        if legacy.is_file():
            return legacy

        repo_config = Path(os.path.dirname(__file__)) / "config.yaml"
        if repo_config.is_file():
            return repo_config

        return None

    def _load_config(self) -> None:
        """Load configuration from file and environment variables.

        Returns:
            None
        """
        config_path = self._resolve_config_path()
        if config_path is None:
            print(
                "No config.yaml found. Create one at "
                "~/.mnemoai/config/config.yaml (or set "
                "MNEMOAI_CONFIG). An example is copied to "
                "~/.mnemoai/config/config.yaml.example on first run — "
                "copy it to config.yaml and edit, e.g.:\n"
                "  cp ~/.mnemoai/config/config.yaml.example "
                "~/.mnemoai/config/config.yaml"
            )
            self._config_data = {}
            return

        try:
            with open(config_path, "r") as f:
                self._config_data = yaml.safe_load(f) or {}
            logger.debug(f"Loaded config from {config_path}")
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Error loading config file ({config_path}): {e}")
            self._config_data = {}

        # Set environment variables (only log once)
        if "ENV" in self._config_data and not hasattr(self, "_env_vars_set"):
            for key, value in self._config_data["ENV"].items():
                os.environ[key] = str(value)
            self._env_vars_set = True

    # Prompt keys that now live in prompts.yaml, not config.yaml. If one is
    # found in config.yaml we warn once and ignore it (prompts.yaml is the
    # single source of truth for prompts).
    _PROMPT_KEYS = (
        "SYSTEM_PROMPT",
        "ROUTING_PROMPT",
        "ORCHESTRATOR_PROMPT",
        "AGGREGATOR_PROMPT",
    )

    @staticmethod
    def _resolve_prompts_path() -> "Path | None":
        """Find the prompts.yaml to load (mirrors :meth:`_resolve_config_path`).

        Order: ``$MNEMOAI_PROMPTS`` -> ``<app_home>/config/prompts.yaml`` ->
        ``<package>/utils/prompts.yaml`` (bundled defaults). Returns None if none
        exist (the app then uses the bundled package copy via the fallback).
        """
        env_path = os.environ.get("MNEMOAI_PROMPTS")
        if env_path:
            return Path(env_path).expanduser()

        from mnemoai.utils.paths import prompts_path as home_prompts_path

        user_prompts = home_prompts_path()
        if user_prompts.is_file():
            return user_prompts

        pkg_prompts = Path(os.path.dirname(__file__)) / "prompts.yaml"
        if pkg_prompts.is_file():
            return pkg_prompts

        return None

    def _load_prompts(self) -> None:
        """Load LLM prompts from prompts.yaml into ``_prompts_data``.

        Prompts are kept separate from configuration: ``config.yaml`` holds
        settings, ``prompts.yaml`` holds every model-facing prompt. If a legacy
        ``config.yaml`` still carries prompt keys, warn once — they're ignored.
        """
        prompts_path = self._resolve_prompts_path()
        if prompts_path is not None:
            try:
                with open(prompts_path, "r") as f:
                    self._prompts_data = yaml.safe_load(f) or {}
                logger.debug(f"Loaded prompts from {prompts_path}")
            except (FileNotFoundError, yaml.YAMLError) as e:
                print(f"Error loading prompts file ({prompts_path}): {e}")
                self._prompts_data = {}

        # One-time migration nudge: prompt keys in config.yaml are now ignored.
        stale = [k for k in self._PROMPT_KEYS if k in self._config_data]
        if stale and not getattr(self, "_prompt_migration_warned", False):
            logger.warning(
                "Prompt keys in config.yaml are no longer read (%s); prompts now "
                "live in prompts.yaml. Move any customizations there.",
                ", ".join(stale),
            )
            self._prompt_migration_warned = True

    def reload(self) -> None:
        """Re-read config and prompts from disk into the existing singleton.

        Used after first-run setup writes a fresh config.yaml: every module
        holds a reference to this one object, so we mutate it in place rather
        than swap the instance.
        """
        self._config_data = {}
        self._prompts_data = {}
        # Allow ENV vars to be re-applied for the newly resolved config.
        if hasattr(self, "_env_vars_set"):
            del self._env_vars_set
        self._load_config()
        self._load_prompts()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: The configuration key
            default: Default value if key is not found

        Returns:
            The configuration value or default
        """
        return self._config_data.get(key, default)

    def prompt(self, key: str, default: Any = None) -> Any:
        """Get an LLM prompt from prompts.yaml.

        Prompts live in prompts.yaml, separate from configuration. Use this
        (not :meth:`get`) for every model-facing prompt. Prefer
        :meth:`require_prompt` for prompts the app cannot run without.

        Args:
            key: The prompt key (e.g. ``SYSTEM_PROMPT``, ``ROUTING_PROMPT``).
            default: Returned when the prompt isn't defined.

        Returns:
            The prompt string, or ``default``.
        """
        return self._prompts_data.get(key, default)

    def require_prompt(self, key: str) -> str:
        """Get a required prompt from prompts.yaml, or raise.

        There are no in-code prompt fallbacks: a prompt the app needs MUST be
        defined (and non-empty) in prompts.yaml. A missing one is a hard,
        explicit failure rather than a silent default.

        Args:
            key: The prompt key.

        Returns:
            The prompt string (guaranteed non-empty).

        Raises:
            PromptError: if the key is missing or empty.
        """
        value = self._prompts_data.get(key)
        if not (value and str(value).strip()):
            raise PromptError(
                f"Required prompt '{key}' is missing from prompts.yaml. "
                f"Prompts are read only from prompts.yaml — add '{key}' there "
                "(see the bundled prompts.yaml for the default)."
            )
        return value

    def validate_prompts(self, *, routing: bool, orchestration: bool) -> None:
        """Fail fast if any required prompt is absent from prompts.yaml.

        Mandatory prompts are always required. Conditional prompts are required
        only when their feature is enabled (so a config that doesn't use routing
        needn't define ROUTING_PROMPT).

        Args:
            routing: Whether ENABLE_ROUTING is on (requires ROUTING_PROMPT).
            orchestration: Whether ENABLE_ORCHESTRATION is on (requires
                ORCHESTRATOR_PROMPT and AGGREGATOR_PROMPT).

        Raises:
            PromptError: listing every required prompt that's missing.
        """
        required = ["SYSTEM_PROMPT", "SUMMARY_SYSTEM_PROMPT", "SUMMARY_TASK_PROMPT"]
        if routing:
            required.append("ROUTING_PROMPT")
        if orchestration:
            required += ["ORCHESTRATOR_PROMPT", "AGGREGATOR_PROMPT"]

        missing = [
            k for k in required
            if not (self._prompts_data.get(k) and str(self._prompts_data[k]).strip())
        ]
        if missing:
            raise PromptError(
                "prompts.yaml is missing required prompt(s): "
                + ", ".join(missing)
                + ". The app reads prompts only from prompts.yaml (never from "
                "config.yaml or hardcoded defaults). Add them — the bundled "
                "prompts.yaml has the defaults you can copy."
            )

    @property
    def system_prompt(self) -> str:
        """Get the system prompt (from prompts.yaml).

        Returns:
            System prompt string.

        Raises:
            PromptError: if SYSTEM_PROMPT is missing/empty.
        """
        return self.require_prompt("SYSTEM_PROMPT")


# Create a singleton instance
config = Config()
