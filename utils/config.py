"""Configuration management for the AI application."""

import os
import yaml
from pathlib import Path
from typing import Any
from utils.logger import logger


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
        self._load_config()
        self._initialized = True

    @staticmethod
    def _resolve_config_path() -> "Path | None":
        """Find the config.yaml to load, checking locations in priority order.

        Resolution order (first existing wins):
          1. ``$PERSONAL_AI_ASSISTANT_CONFIG`` — explicit override (handy for
             switching between provider configs, e.g. ollama vs bedrock vs mantle)
          2. ``<app_home>/config.yaml`` — the user config used by the installed
             CLI; app home defaults to ``~/.personal-ai-assistant`` and honors
             ``$PERSONAL_AI_ASSISTANT_HOME``
          3. ``<repo>/utils/config.yaml`` — repo-relative fallback, so running
             from a checkout (``python main.py``) keeps working unchanged

        Returns:
            The resolved Path, or None if no config exists in any location.
        """
        env_path = os.environ.get("PERSONAL_AI_ASSISTANT_CONFIG")
        if env_path:
            return Path(env_path).expanduser()

        from utils.paths import config_path as home_config_path

        user_config = home_config_path()
        if user_config.is_file():
            return user_config

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
                "~/.personal-ai-assistant/config.yaml (or set "
                "PERSONAL_AI_ASSISTANT_CONFIG). Copy a template to start, e.g.:\n"
                "  mkdir -p ~/.personal-ai-assistant && \\\n"
                "  cp utils/config.yaml.example "
                "~/.personal-ai-assistant/config.yaml"
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

    def reload(self) -> None:
        """Re-read config from disk into the existing singleton instance.

        Used after first-run setup writes a fresh config.yaml: every module
        holds a reference to this one object, so we mutate it in place rather
        than swap the instance.
        """
        self._config_data = {}
        # Allow ENV vars to be re-applied for the newly resolved config.
        if hasattr(self, "_env_vars_set"):
            del self._env_vars_set
        self._load_config()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: The configuration key
            default: Default value if key is not found

        Returns:
            The configuration value or default
        """
        return self._config_data.get(key, default)

    @property
    def system_prompt(self) -> str:
        """Get the system prompt.

        Returns:
            System prompt string
        """
        return self._config_data.get(
            "SYSTEM_PROMPT",
            None,
        )


# Create a singleton instance
config = Config()
