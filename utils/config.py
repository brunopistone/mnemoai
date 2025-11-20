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

    def _load_config(self) -> None:
        """Load configuration from file and environment variables.

        Returns:
            None
        """
        # Load from config file
        config_path = Path(os.path.dirname(__file__)) / "config.yaml"
        try:
            with open(config_path, "r") as f:
                self._config_data = yaml.safe_load(f) or {}
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Error loading config file: {e}")
            self._config_data = {}

        # Set environment variables (only log once)
        if "ENV" in self._config_data and not hasattr(self, "_env_vars_set"):
            for key, value in self._config_data["ENV"].items():
                os.environ[key] = str(value)
            self._env_vars_set = True

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
