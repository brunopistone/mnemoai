"""Unit tests for provider prompt-cache breakpoints (models/prompt_cache.py).

Caching is on by default wherever it works, so the gating is what needs pinning:
a marker sent to a provider or model family that doesn't take one is a 400 on
every call, i.e. a broken session rather than a missed saving. These tests cover
the policy decision, the two request shapes it produces, and the agent
chokepoints every binding and system message goes through.

Pure logic: no LLM, no config.yaml, no network.
"""

import ast
import pathlib
import re

from langchain_core.messages import SystemMessage

from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.models import prompt_cache


def _bedrock(name="us.anthropic.claude-sonnet-4-5-20250929-v1:0", **extra):
    return {"TYPE": "bedrock", "NAME": name, "REGION": "us-east-1", **extra}


def _mantle(protocol="anthropic", name="claude-opus-5", **extra):
    return {"TYPE": "mantle", "NAME": name, "API_PROTOCOL": protocol, **extra}


class _Model:
    """Chat-model stand-in recording the kwargs bound onto it."""

    def __init__(self, model_id=""):
        self.model_id = model_id
        self.bound_kwargs = None

    def bind(self, **kwargs):
        bound = _Model(self.model_id)
        bound.bound_kwargs = kwargs
        return bound

    def bind_tools(self, tools):
        bound = _Model(self.model_id)
        bound.tools = tools
        return bound


class _Tool:
    def __init__(self, name):
        self.name = name


class TestPolicyProviderGating:
    def test_bedrock_and_anthropic_cache_by_default(self):
        # No config edit needed: an existing install gets caching on upgrade.
        assert prompt_cache.policy(_bedrock()).enabled
        assert prompt_cache.policy(
            {"TYPE": "anthropic", "NAME": "claude-opus-4-20250514"}
        ).enabled

    def test_providers_without_the_kwarg_stay_off(self):
        # ChatOpenAI/ChatOllama forward an unknown kwarg into the request body —
        # a 400 on every call, so these must never be marked.
        for section in (
            {"TYPE": "openai", "NAME": "gpt-4o"},
            {"TYPE": "ollama", "NAME": "qwen3:32b"},
            {"TYPE": "litellm", "NAME": "anthropic/claude-opus-4"},
            {"TYPE": "sagemaker", "NAME": "claude-endpoint"},
            {"TYPE": "", "NAME": "claude"},
            {},
        ):
            assert prompt_cache.policy(section) is prompt_cache.OFF, section

    def test_mantle_only_on_the_anthropic_protocol(self):
        # The OpenAI-shaped Mantle protocols would put the kwarg in the body.
        assert prompt_cache.policy(_mantle("anthropic")).enabled
        assert not prompt_cache.policy(_mantle("responses")).enabled
        assert not prompt_cache.policy(_mantle("chat_completions")).enabled
        assert not prompt_cache.policy(_mantle("")).enabled

    def test_only_cacheable_model_families(self):
        # Bedrock REJECTS a cachePoint for a family that can't cache.
        assert prompt_cache.policy(_bedrock("amazon.nova-pro-v1:0")).enabled
        assert prompt_cache.policy(_bedrock("eu.anthropic.claude-haiku-4-5")).enabled
        assert not prompt_cache.policy(_bedrock("meta.llama3-3-70b-instruct-v1:0")).enabled
        assert not prompt_cache.policy(_bedrock("mistral.mistral-large-2407-v1:0")).enabled
        assert not prompt_cache.policy(_bedrock("")).enabled

    def test_missing_section_is_off(self):
        assert prompt_cache.policy(None) is prompt_cache.OFF


class TestPolicyOptOut:
    def test_falsey_values_disable_it(self):
        # Hand-edited YAML: a quoted "false" must disable it like a bare false.
        for value in (False, "false", "False", " NO ", "off", "0", 0):
            assert not prompt_cache.policy(_bedrock(PROMPT_CACHE=value)).enabled, value

    def test_true_and_absent_keep_it_on(self):
        assert prompt_cache.policy(_bedrock(PROMPT_CACHE=True)).enabled
        assert prompt_cache.policy(_bedrock(PROMPT_CACHE="true")).enabled
        assert prompt_cache.policy(_bedrock()).enabled

    def test_a_valueless_key_is_not_an_opt_out(self):
        # `PROMPT_CACHE:` with nothing after it parses as None — no opinion, so
        # the default (on) stands; an empty string is treated the same way.
        assert prompt_cache.policy(_bedrock(PROMPT_CACHE=None)).enabled
        assert prompt_cache.policy(_bedrock(PROMPT_CACHE="")).enabled

    def test_true_cannot_force_an_unsupported_provider(self):
        # The kwarg would be rejected, so an explicit opt-IN can't override gating.
        assert not prompt_cache.policy(
            {"TYPE": "ollama", "NAME": "claude-ish", "PROMPT_CACHE": True}
        ).enabled
        assert not prompt_cache.policy(_mantle("responses", PROMPT_CACHE=True)).enabled


class TestTTL:
    def test_defaults_to_five_minutes(self):
        assert prompt_cache.policy(_bedrock()).control["ttl"] == "5m"

    def test_one_hour_is_honored(self):
        assert prompt_cache.policy(_bedrock(PROMPT_CACHE_TTL="1h")).control["ttl"] == "1h"
        assert prompt_cache.ttl({"PROMPT_CACHE_TTL": " 1H "}) == "1h"

    def test_an_unrecognized_ttl_lands_on_the_cheap_default(self):
        # 1h costs more to WRITE, so a typo must not silently upgrade the tier.
        for value in ("10m", "1 hour", "forever", 3600, None, ""):
            assert prompt_cache.ttl({"PROMPT_CACHE_TTL": value}) == "5m", value

    def test_control_carries_both_halves(self):
        # Anthropic requires `type`; Bedrock ignores it and reads only a non-default
        # `ttl` — one dict serves both expansions.
        assert prompt_cache.policy(_bedrock()).control == {
            "type": "ephemeral", "ttl": "5m"
        }


class TestSystemMarkerAsymmetry:
    def test_bedrock_does_not_mark_the_system_prompt(self):
        # langchain-aws already emits a cachePoint after the system blocks, and a
        # Converse content block rejects an Anthropic-style cache_control key.
        assert prompt_cache.policy(_bedrock()).mark_system is False

    def test_anthropic_transports_mark_it(self):
        assert prompt_cache.policy(_mantle()).mark_system is True
        assert prompt_cache.policy(
            {"TYPE": "anthropic", "NAME": "claude-opus-4-20250514"}
        ).mark_system is True

    def test_off_marks_nothing(self):
        assert prompt_cache.OFF.mark_system is False
        assert prompt_cache.OFF.enabled is False


class TestBind:
    def test_attaches_the_kwarg(self):
        policy = prompt_cache.policy(_bedrock())
        bound = prompt_cache.bind(_Model("claude-sonnet-4-5"), policy)
        assert bound.bound_kwargs == {
            "cache_control": {"type": "ephemeral", "ttl": "5m"}
        }

    def test_passes_the_model_through_when_off(self):
        model = _Model("qwen3:32b")
        assert prompt_cache.bind(model, prompt_cache.OFF) is model
        assert prompt_cache.bind(model) is model

    def test_a_model_level_family_override_vetoes_the_marker(self):
        # A custom sub-agent type can swap the model NAME on the same provider; a
        # non-cacheable family must not inherit the parent's marker.
        policy = prompt_cache.policy(_bedrock())
        model = _Model("meta.llama3-3-70b-instruct-v1:0")
        assert prompt_cache.bind(model, policy) is model

    def test_an_unreadable_model_id_trusts_the_policy(self):
        policy = prompt_cache.policy(_bedrock())
        bound = prompt_cache.bind(_Model(""), policy)
        assert bound.bound_kwargs["cache_control"]["ttl"] == "5m"

    def test_a_model_without_bind_is_returned_unchanged(self):
        # Losing the marker costs money; losing the model breaks the turn.
        class _Exotic:
            model_id = "claude-opus-5"

        model = _Exotic()
        assert prompt_cache.bind(model, prompt_cache.policy(_bedrock())) is model

    def test_the_control_dict_is_copied_per_binding(self):
        # A shared dict a client library mutated would leak across models.
        policy = prompt_cache.policy(_bedrock())
        bound = prompt_cache.bind(_Model("claude"), policy)
        bound.bound_kwargs["cache_control"]["ttl"] = "1h"
        assert policy.control["ttl"] == "5m"


class TestSystemMessage:
    def test_plain_when_caching_is_off(self):
        msg = prompt_cache.system_message("You are an agent.", prompt_cache.OFF)
        assert isinstance(msg, SystemMessage)
        assert msg.content == "You are an agent."

    def test_plain_on_bedrock(self):
        msg = prompt_cache.system_message("You are an agent.", prompt_cache.policy(_bedrock()))
        assert msg.content == "You are an agent."

    def test_marked_on_the_anthropic_transports(self):
        msg = prompt_cache.system_message("You are an agent.", prompt_cache.policy(_mantle()))
        assert msg.content == [
            {
                "type": "text",
                "text": "You are an agent.",
                "cache_control": {"type": "ephemeral", "ttl": "5m"},
            }
        ]

    def test_empty_text_stays_a_plain_message(self):
        # A marked empty block is a request error, not an empty cache entry.
        assert prompt_cache.system_message("", prompt_cache.policy(_mantle())).content == ""


class TestAgentChokepoints:
    """Every binding and system message the agent builds goes through these two."""

    def _agent(self, policy=None):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a.system_prompt = "SYSTEM"
        if policy is not None:
            a._cache_policy = policy
        return a

    def test_bind_tools_binds_the_subset_then_the_marker(self):
        a = self._agent(prompt_cache.policy(_bedrock()))
        tools = [_Tool("fs_read")]
        bound = a._bind_tools(_Model("claude-sonnet-4-5"), tools)
        assert bound.bound_kwargs["cache_control"]["ttl"] == "5m"

    def test_bind_tools_without_tools_still_marks(self):
        # A route with an empty tool subset (simple_qa) uses the bare model.
        a = self._agent(prompt_cache.policy(_bedrock()))
        bound = a._bind_tools(_Model("claude-sonnet-4-5"), [])
        assert bound.bound_kwargs["cache_control"]["type"] == "ephemeral"

    def test_a_stub_agent_without_a_policy_gets_the_plain_binding(self):
        # Tests build agents via __new__; a missing policy must not raise.
        a = self._agent()
        model = _Model("claude")
        assert a._bind_tools(model, None) is model
        assert a._system_message().content == "SYSTEM"

    def test_system_message_uses_the_agent_prompt_by_default(self):
        a = self._agent(prompt_cache.policy(_mantle()))
        assert a._system_message().content[0]["text"] == "SYSTEM"
        # An explicit prompt (worker loop / sub-agent) is marked the same way.
        assert a._system_message("WORKER").content[0]["text"] == "WORKER"


class TestNoBypass:
    """Source guards: a new call site must not skip the chokepoints."""

    def _agent_sources(self):
        agent_dir = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src" / "mnemoai" / "client" / "agent"
        )
        return sorted(agent_dir.glob("*.py"))

    def test_bind_tools_is_only_called_inside_the_chokepoint(self):
        # Five call sites need the marker (main loop, per-route models, rebind,
        # orchestrator workers, sub-agents); each goes through _bind_tools.
        offenders = []
        for path in self._agent_sources():
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name == "_bind_tools":
                    continue
                for inner in ast.walk(node):
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr == "bind_tools"
                    ):
                        offenders.append(f"{path.name}:{inner.lineno} ({node.name})")
        assert not offenders, (
            "bind_tools called outside _bind_tools — these bindings would lose the "
            f"prompt-cache breakpoint: {offenders}"
        )

    def test_the_system_prompt_is_never_wrapped_directly(self):
        # SystemMessage(content=…) for the conversation prefix must come from
        # _system_message(); the orchestrator/aggregator one-offs are unaffected.
        src = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src" / "mnemoai" / "client" / "agent" / "agent.py"
        ).read_text()
        assert not re.findall(r"SystemMessage\(\s*content=[^)]*sys(?:tem)?_prompt", src)
