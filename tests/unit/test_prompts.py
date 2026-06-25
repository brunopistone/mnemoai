"""Unit tests for prompt loading & fail-fast validation (utils/config.py).

Prompts are read ONLY from prompts.yaml — never from config.yaml, never from
in-code fallbacks. Required prompts must be present or the app fails loudly.
"""

import pytest
import yaml

from mnemoai.utils.config import Config, PromptError


def _config_with_prompts(prompts: dict, config_data: dict = None) -> Config:
    """A throwaway Config-like instance with controlled prompts/config dicts.

    Uses ``object.__new__`` (NOT ``Config()`` / ``Config.__new__``) so we do not
    touch the Config SINGLETON — otherwise we'd clobber the shared
    ``_prompts_data`` and pollute other tests that read prompts.
    """
    c = object.__new__(Config)
    c._config_data = config_data or {}
    c._prompts_data = prompts
    return c


# All six prompts, loaded from the bundled package template (the source of
# defaults that seeds a user's prompts.yaml).
def _bundled_prompts() -> dict:
    import os

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(here, "src", "mnemoai", "utils", "prompts.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


class TestRequirePrompt:
    def test_returns_present_prompt(self):
        c = _config_with_prompts({"SYSTEM_PROMPT": "you are an agent"})
        assert c.require_prompt("SYSTEM_PROMPT") == "you are an agent"

    def test_missing_prompt_raises(self):
        c = _config_with_prompts({})
        with pytest.raises(PromptError):
            c.require_prompt("SYSTEM_PROMPT")

    def test_empty_prompt_raises(self):
        c = _config_with_prompts({"SYSTEM_PROMPT": "   "})
        with pytest.raises(PromptError):
            c.require_prompt("SYSTEM_PROMPT")


class TestValidatePrompts:
    def test_bundled_prompts_pass_full_validation(self):
        c = _config_with_prompts(_bundled_prompts())
        # Should not raise with everything enabled.
        c.validate_prompts(routing=True, orchestration=True)

    def test_missing_mandatory_fails(self):
        prompts = _bundled_prompts()
        del prompts["SUMMARY_TASK_PROMPT"]
        c = _config_with_prompts(prompts)
        with pytest.raises(PromptError) as e:
            c.validate_prompts(routing=False, orchestration=False)
        assert "SUMMARY_TASK_PROMPT" in str(e.value)

    def test_routing_prompt_only_required_when_routing_enabled(self):
        prompts = _bundled_prompts()
        del prompts["ROUTING_PROMPT"]
        c = _config_with_prompts(prompts)
        # routing off -> fine
        c.validate_prompts(routing=False, orchestration=False)
        # routing on -> required
        with pytest.raises(PromptError) as e:
            c.validate_prompts(routing=True, orchestration=False)
        assert "ROUTING_PROMPT" in str(e.value)

    def test_orchestration_prompts_only_required_when_enabled(self):
        prompts = _bundled_prompts()
        del prompts["ORCHESTRATOR_PROMPT"]
        del prompts["AGGREGATOR_PROMPT"]
        c = _config_with_prompts(prompts)
        c.validate_prompts(routing=False, orchestration=False)  # fine
        with pytest.raises(PromptError) as e:
            c.validate_prompts(routing=False, orchestration=True)
        msg = str(e.value)
        assert "ORCHESTRATOR_PROMPT" in msg and "AGGREGATOR_PROMPT" in msg

    def test_empty_prompts_file_fails_on_mandatory(self):
        c = _config_with_prompts({})
        with pytest.raises(PromptError):
            c.validate_prompts(routing=False, orchestration=False)


class TestNoConfigYamlFallback:
    def test_prompt_in_config_yaml_is_ignored(self):
        # A prompt placed in config.yaml must NOT satisfy a prompts.yaml lookup.
        c = _config_with_prompts(
            prompts={}, config_data={"SYSTEM_PROMPT": "from config.yaml"}
        )
        with pytest.raises(PromptError):
            c.require_prompt("SYSTEM_PROMPT")
