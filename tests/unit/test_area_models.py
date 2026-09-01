"""Unit tests for per-area model overrides (``AREA_MODELS``).

Three layers, deliberately separate: the pure config reader, the controller's
variant builder (does an override actually reach the provider dispatch, and is
the *whole* param snapshot re-applied), and the client wiring (does each area get
its model, does an absent section change nothing, does a build failure degrade to
the main model rather than breaking the turn).

The invariant most worth pinning is the last one: these are optional side models
for calls the user never sees, so every failure mode must end in a working turn.
"""

import pytest

from mnemoai.client.client import LangGraphClient
from mnemoai.models import area_models
from mnemoai.models.controllers.llm_controller import LangChainLLMController


class TestReadingTheConfig:
    """overrides_for(): tolerant, because most installs never write this."""

    def _cfg(self, monkeypatch, section):
        monkeypatch.setattr(
            area_models.config,
            "get",
            lambda k, d=None: section if k == area_models.CONFIG_SECTION else d,
        )

    def test_no_section_means_no_overrides(self, monkeypatch):
        self._cfg(monkeypatch, {})
        for area in area_models.AREAS:
            assert area_models.overrides_for(area) == {}
        assert area_models.configured() == {}

    def test_a_bare_string_is_shorthand_for_the_model_name(self, monkeypatch):
        self._cfg(monkeypatch, {"ROUTER": "qwen3.5:1.7b"})
        assert area_models.overrides_for("ROUTER") == {"NAME": "qwen3.5:1.7b"}

    def test_a_dict_is_taken_verbatim(self, monkeypatch):
        self._cfg(monkeypatch, {"SUMMARY": {"NAME": "small", "TEMPERATURE": 0.3}})
        assert area_models.overrides_for("SUMMARY") == {
            "NAME": "small",
            "TEMPERATURE": 0.3,
        }

    def test_an_area_may_switch_provider(self, monkeypatch):
        self._cfg(monkeypatch, {"ROUTER": {"TYPE": "ollama", "NAME": "qwen3.5:1.7b"}})
        assert area_models.overrides_for("ROUTER")["TYPE"] == "ollama"

    def test_the_area_key_is_case_and_space_insensitive(self, monkeypatch):
        self._cfg(monkeypatch, {" router ": "small"})
        assert area_models.overrides_for("ROUTER") == {"NAME": "small"}

    def test_junk_degrades_to_no_override_rather_than_raising(self, monkeypatch):
        for section in ("not a mapping", 7, None, ["ROUTER"]):
            self._cfg(monkeypatch, section)
            assert area_models.overrides_for("ROUTER") == {}
            assert area_models.configured() == {}
            assert area_models.unknown_keys() == ()

    def test_an_unusable_value_is_no_override(self, monkeypatch):
        for value in (None, "", "   ", 3, []):
            self._cfg(monkeypatch, {"ROUTER": value})
            assert area_models.overrides_for("ROUTER") == {}

    def test_an_empty_value_cannot_blank_a_real_param(self, monkeypatch):
        # A commented-out or emptied key must not be sent as an empty NAME.
        self._cfg(monkeypatch, {"ROUTER": {"NAME": "small", "TEMPERATURE": None}})
        assert area_models.overrides_for("ROUTER") == {"NAME": "small"}

    def test_configured_lists_only_the_areas_with_an_override(self, monkeypatch):
        self._cfg(monkeypatch, {"ROUTER": "small", "SUMMARY": {}})
        assert list(area_models.configured()) == ["ROUTER"]

    def test_a_misspelled_area_is_reported_not_silently_ignored(self, monkeypatch):
        self._cfg(monkeypatch, {"ROUTERS": "small", "ROUTER": "small"})
        assert area_models.unknown_keys() == ("ROUTERS",)

    def test_every_area_has_a_description(self):
        for area in area_models.AREAS:
            assert area_models.DESCRIPTIONS[area]

    def test_label_fills_either_half_from_the_main_model(self):
        assert area_models.label({"NAME": "small"}, "big", "bedrock") == "small (bedrock)"
        assert area_models.label({"TYPE": "ollama"}, "big") == "big (ollama)"
        assert area_models.label({}, "big") == "big"


class TestBuildingTheVariant:
    """build_model_variant(): the override reaches dispatch with a full snapshot."""

    def _controller(self, monkeypatch, model_id):
        monkeypatch.setattr(
            "mnemoai.models.controllers.llm_controller.config.get",
            lambda k, d=None: model_id if k == "MODEL_ID" else d,
        )
        return LangChainLLMController(verbose=False)

    def _capture(self, monkeypatch):
        """Replace provider dispatch with a recorder of the peer's snapshot."""
        seen = {}

        def _fake(self, provider, *args, **kwargs):
            seen.update(
                provider=provider,
                name=self.model_name,
                type=self.model_type,
                temperature=self.temperature,
                reasoning=self.reasoning_model,
                effort=self.reasoning_effort,
                verbose=self.verbose_mode,
                callbacks=args[0] if args else kwargs.get("callbacks"),
            )
            self.model = "BUILT"

        monkeypatch.setattr(LangChainLLMController, "_dispatch_provider", _fake)
        return seen

    def test_no_overrides_builds_the_configured_model(self, monkeypatch):
        ctrl = self._controller(monkeypatch, {"NAME": "big", "TYPE": "ollama"})
        seen = self._capture(monkeypatch)
        assert ctrl.build_model_variant() == "BUILT"
        assert (seen["name"], seen["type"]) == ("big", "ollama")

    def test_a_name_override_swaps_only_the_name(self, monkeypatch):
        ctrl = self._controller(
            monkeypatch, {"NAME": "big", "TYPE": "ollama", "TEMPERATURE": 0.6}
        )
        seen = self._capture(monkeypatch)
        ctrl.build_model_variant({"NAME": "small"})
        assert seen["name"] == "small"
        assert seen["type"] == "ollama"
        assert seen["temperature"] == 0.6

    def test_a_type_override_dispatches_to_the_other_provider(self, monkeypatch):
        ctrl = self._controller(monkeypatch, {"NAME": "big", "TYPE": "bedrock"})
        seen = self._capture(monkeypatch)
        ctrl.build_model_variant({"TYPE": "ollama", "NAME": "small"})
        assert seen["provider"] == "ollama"

    def test_every_param_is_re_snapshotted_not_just_the_name(self, monkeypatch):
        # The bug this guards: patching model_id alone leaves the derived
        # attributes (which the _initialize_* paths actually read) stale.
        ctrl = self._controller(
            monkeypatch,
            {"NAME": "big", "TYPE": "ollama", "TEMPERATURE": 0.6, "REASONING": True},
        )
        seen = self._capture(monkeypatch)
        ctrl.build_model_variant({"TEMPERATURE": 0.1, "REASONING": False})
        assert seen["temperature"] == 0.1
        assert seen["reasoning"] is False

    def test_the_variant_never_touches_this_controller(self, monkeypatch):
        ctrl = self._controller(monkeypatch, {"NAME": "big", "TYPE": "ollama"})
        ctrl.model = "ORIGINAL"
        self._capture(monkeypatch)
        ctrl.build_model_variant({"NAME": "small"})
        assert ctrl.model == "ORIGINAL"
        assert ctrl.model_name == "big"
        assert ctrl.model_id["NAME"] == "big"

    def test_non_reasoning_clears_thinking_and_verbosity(self, monkeypatch):
        ctrl = self._controller(
            monkeypatch,
            {
                "NAME": "big",
                "TYPE": "ollama",
                "REASONING": True,
                "REASONING_EFFORT": "high",
            },
        )
        ctrl.verbose_mode = True
        seen = self._capture(monkeypatch)
        ctrl.build_non_reasoning_model()
        assert seen["reasoning"] is False
        assert seen["effort"] is None
        assert seen["verbose"] is False

    def test_the_summary_variant_can_also_swap_the_model(self, monkeypatch):
        ctrl = self._controller(
            monkeypatch, {"NAME": "big", "TYPE": "ollama", "REASONING": True}
        )
        seen = self._capture(monkeypatch)
        ctrl.build_non_reasoning_model(overrides={"NAME": "small"})
        assert seen["name"] == "small"
        assert seen["reasoning"] is False

    def test_an_ordinary_variant_inherits_verbosity(self, monkeypatch):
        ctrl = self._controller(monkeypatch, {"NAME": "big", "TYPE": "ollama"})
        ctrl.verbose_mode = True
        seen = self._capture(monkeypatch)
        ctrl.build_model_variant({"NAME": "small"})
        assert seen["verbose"] is True

    def test_callbacks_are_passed_through(self, monkeypatch):
        ctrl = self._controller(monkeypatch, {"NAME": "big", "TYPE": "ollama"})
        seen = self._capture(monkeypatch)
        handler = object()
        ctrl.build_model_variant({"NAME": "small"}, callbacks=[handler])
        assert seen["callbacks"] == [handler]


class _Ctrl:
    """A controller stand-in that records what variants were asked for."""

    model_name = "MAIN"

    def __init__(self, fail=False):
        self.asked = []
        self.fail = fail

    def build_model_variant(self, overrides=None, callbacks=None, non_reasoning=False):
        self.asked.append(overrides)
        if self.fail:
            raise RuntimeError("provider unreachable")
        return f"MODEL:{(overrides or {}).get('NAME', 'MAIN')}"

    def build_non_reasoning_model(self, callbacks=None, overrides=None):
        return self.build_model_variant(overrides, callbacks, non_reasoning=True)


class TestClientWiring:
    """_area_model(): one build per area, cached, never fatal."""

    def _client(self, monkeypatch, section, fail=False):
        c = LangGraphClient.__new__(LangGraphClient)
        c._area_model_cache = {}
        c.agent = None
        c.llm_controller = _Ctrl(fail=fail)
        monkeypatch.setattr(
            area_models.config,
            "get",
            lambda k, d=None: section if k == area_models.CONFIG_SECTION else d,
        )
        return c

    def test_an_unconfigured_area_returns_none_and_builds_nothing(self, monkeypatch):
        c = self._client(monkeypatch, {})
        assert c._area_model("ROUTER") is None
        assert c.llm_controller.asked == []

    def test_a_configured_area_gets_its_own_model(self, monkeypatch):
        c = self._client(monkeypatch, {"ROUTER": "small"})
        assert c._area_model("ROUTER") == "MODEL:small"
        assert c.llm_controller.asked == [{"NAME": "small"}]

    def test_the_model_is_built_once_per_area(self, monkeypatch):
        c = self._client(monkeypatch, {"ROUTER": "small"})
        for _ in range(3):
            c._area_model("ROUTER")
        assert len(c.llm_controller.asked) == 1

    def test_a_failed_build_falls_back_to_the_main_model(self, monkeypatch):
        c = self._client(monkeypatch, {"ROUTER": "small"}, fail=True)
        assert c._area_model("ROUTER") is None

    def test_a_failed_build_is_not_retried_every_turn(self, monkeypatch):
        c = self._client(monkeypatch, {"ROUTER": "small"}, fail=True)
        c._area_model("ROUTER")
        c._area_model("ROUTER")
        assert len(c.llm_controller.asked) == 1

    def test_areas_are_independent(self, monkeypatch):
        c = self._client(monkeypatch, {"ROUTER": "small"})
        assert c._area_model("ROUTER") == "MODEL:small"
        assert c._area_model("ORCHESTRATOR") is None
        assert c._area_model("SUMMARY") is None

    def test_usage_is_attributed_to_the_area_model(self, monkeypatch):
        c = self._client(monkeypatch, {"ROUTER": "small"})
        c.agent = type("A", (), {"usage_model_name": "big"})()
        assert c._area_usage_name("ROUTER") == "small"

    def test_usage_falls_back_to_the_main_name_without_an_override(self, monkeypatch):
        c = self._client(monkeypatch, {})
        c.agent = type("A", (), {"usage_model_name": "big"})()
        assert c._area_usage_name("ROUTER") == "big"

    def test_usage_falls_back_to_the_main_name_when_the_build_failed(self, monkeypatch):
        c = self._client(monkeypatch, {"ROUTER": "small"}, fail=True)
        c.agent = type("A", (), {"usage_model_name": "big"})()
        assert c._area_usage_name("ROUTER") == "big"


class TestTheSummaryArea:
    """_summary_model(): the override rides on top of the reasoning-off treatment."""

    def _client(self, monkeypatch, section, think=False):
        c = LangGraphClient.__new__(LangGraphClient)
        c._area_model_cache = {}
        c.model = "MAIN"
        c.llm_controller = _Ctrl()

        # One lambda for both keys: `config` is a singleton, so client.py and
        # area_models read the SAME object — two patches would shadow each other.
        def _get(key, default=None):
            if key == "LLM":
                return {"SUMMARIZATION_THINK": think}
            if key == area_models.CONFIG_SECTION:
                return section
            return default

        monkeypatch.setattr(area_models.config, "get", _get)
        monkeypatch.setattr("mnemoai.client.client.config.get", _get)
        return c

    def test_without_an_override_the_main_model_is_used_non_reasoning(self, monkeypatch):
        c = self._client(monkeypatch, {})
        assert c._summary_model() == "MODEL:MAIN"
        assert c.llm_controller.asked == [None]

    def test_the_override_reaches_the_summary_build(self, monkeypatch):
        c = self._client(monkeypatch, {"SUMMARY": "small"})
        assert c._summary_model() == "MODEL:small"
        assert c.llm_controller.asked == [{"NAME": "small"}]

    def test_summarization_think_keeps_the_main_model_and_ignores_the_area(
        self, monkeypatch
    ):
        # The flag means "summarize with the full model, thinking and all", so an
        # area override would contradict it.
        c = self._client(monkeypatch, {"SUMMARY": "small"}, think=True)
        assert c._summary_model() == "MAIN"
        assert c.llm_controller.asked == []


class TestReloadRederivesThem:
    """/params re-reads config, so the area models must be rebuilt from it too."""

    def _client(self, monkeypatch, section):
        c = LangGraphClient.__new__(LangGraphClient)
        c.verbose_mode = False
        c.callback_handler = None
        c.model = "OLD"
        c.llm_controller = _Ctrl()
        c._subagent_model_cache = {"x": "STALE"}
        c._area_model_cache = {}
        c._summary_model_cached = "STALE"
        c.agent = type(
            "A",
            (),
            {
                "rebind_model": lambda self, m: setattr(self, "model", m),
                "router": type("R", (), {"model": "OLD", "usage_model_name": "old"})(),
                "usage_model_name": "big",
                "orchestrator_model": None,
            },
        )()

        def _get(key, default=None):
            return section if key == area_models.CONFIG_SECTION else default

        monkeypatch.setattr(area_models.config, "get", _get)
        monkeypatch.setattr("mnemoai.client.client.config.get", _get)
        monkeypatch.setattr("mnemoai.client.client.config.reload", lambda: None)
        monkeypatch.setattr(
            "mnemoai.client.client.LangChainLLMController",
            lambda verbose=False: _NewCtrl(),
        )
        return c

    def test_the_area_models_are_rebuilt_off_the_new_config(self, monkeypatch):
        c = self._client(monkeypatch, {"ROUTER": "small", "ORCHESTRATOR": "mid"})
        assert c.reload_inference_params() is True
        # Rebuilt by the NEW controller, so a reloaded param actually lands.
        assert c.agent.router.model == "NEW:small"
        assert c.agent.orchestrator_model == "NEW:mid"

    def test_a_stale_cache_cannot_survive_the_reload(self, monkeypatch):
        c = self._client(monkeypatch, {"ROUTER": "small"})
        c._area_model_cache["ROUTER"] = "STALE"
        c.reload_inference_params()
        assert c.agent.router.model == "NEW:small"
        assert c._subagent_model_cache == {}
        assert not hasattr(c, "_summary_model_cached")

    def test_removing_an_override_returns_the_area_to_the_main_model(self, monkeypatch):
        c = self._client(monkeypatch, {})
        c._area_model_cache["ROUTER"] = "STALE"
        c.reload_inference_params()
        assert c.agent.router.model == "NEW-MAIN"
        assert c.agent.orchestrator_model is None
        assert c.agent.router.usage_model_name == "big"


class _NewCtrl(_Ctrl):
    """The post-reload controller: its variants are visibly not the old ones."""

    def initialize_model(self, callbacks=None):
        self.model = "NEW-MAIN"

    def get_model(self):
        return self.model

    def build_model_variant(self, overrides=None, callbacks=None, non_reasoning=False):
        self.asked.append(overrides)
        return f"NEW:{(overrides or {}).get('NAME', 'MAIN')}"


class TestDoctorReportsIt:
    """The feature is invisible at runtime, so /doctor has to name it."""

    def _cfg(self, monkeypatch, section):
        from mnemoai.client import doctor

        monkeypatch.setattr(
            doctor.area_models.config,
            "get",
            lambda k, d=None: section if k == area_models.CONFIG_SECTION else d,
        )
        return doctor

    def test_a_configured_area_is_listed(self, monkeypatch):
        doctor = self._cfg(monkeypatch, {"ROUTER": "small"})
        rows = doctor._area_model_checks({"NAME": "big", "TYPE": "bedrock"})
        assert len(rows) == 1
        assert rows[0].name == area_models.DESCRIPTIONS["ROUTER"]
        assert "small (bedrock)" in rows[0].detail

    def test_nothing_is_reported_when_nothing_is_configured(self, monkeypatch):
        doctor = self._cfg(monkeypatch, {})
        assert doctor._area_model_checks({"NAME": "big", "TYPE": "bedrock"}) == []

    def test_a_misspelled_area_warns(self, monkeypatch):
        doctor = self._cfg(monkeypatch, {"ROUTERS": "small"})
        rows = doctor._area_model_checks({"NAME": "big", "TYPE": "bedrock"})
        assert [r.status for r in rows] == [doctor.WARN]
        assert "ROUTERS" in rows[0].detail


class TestDecompositionUsesTheAreaModel:
    """The orchestrator's model reaches the decomposition call, not just config."""

    def _agent(self, orchestrator_model):
        from mnemoai.client.agent.agent import LangGraphAgent

        a = LangGraphAgent.__new__(LangGraphAgent)
        a.orchestrator_model = orchestrator_model
        return a

    class _M:
        callbacks = None

        def __init__(self, tag):
            self.tag = tag
            self.calls = 0

        def invoke(self, messages, config=None):
            from langchain_core.messages import AIMessage

            self.calls += 1
            return AIMessage(
                content=f'[{{"description": "{self.tag}", "category": "full"}}]'
            )

    def test_the_area_model_runs_the_decomposition(self):
        area = self._M("area")
        main = self._M("main")
        a = self._agent(area)
        a.model = main
        out = a._decompose_task("q", "decompose", {"full"})
        assert [s["description"] for s in out] == ["area"]
        assert main.calls == 0

    def test_without_one_the_main_model_still_runs_it(self):
        main = self._M("main")
        a = self._agent(None)
        a.model = main
        out = a._decompose_task("q", "decompose", {"full"})
        assert [s["description"] for s in out] == ["main"]

    def test_the_in_place_fallback_mutates_the_model_it_invokes(self):
        area = self._M("area")
        main = self._M("main")
        a = self._agent(area)
        a.model = main
        touched = []
        a._non_reasoning = lambda model=None: None  # force the fallback branch
        a._disable_reasoning = lambda model=None: touched.append(model) or {"x": 1}
        a._restore_reasoning = lambda saved, model=None: touched.append(("restore", model))
        a._decompose_task("q", "decompose", {"full"})
        assert touched == [area, ("restore", area)]
        assert main.calls == 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
