"""Provider dispatch + shared config init on BaseModelController.

The LLM, vision, and embeddings controllers each had their own ``if/elif`` chain
over the configured provider ``TYPE``. They now share one table-driven dispatch,
so these tests exercise it directly (no live provider needed) and pin that all
three controllers actually route through it.
"""

import pytest

from mnemoai.models.controllers.base_model_controller import BaseModelController


class _Fake(BaseModelController):
    PROVIDERS = ("alpha", "beta")
    PROVIDER_METHOD_TEMPLATE = "_initialize_{provider}_model"
    PROVIDER_LABEL = "test model"

    def __init__(self):
        self.calls = []

    def _initialize_alpha_model(self, *args, **kwargs):
        self.calls.append(("alpha", args, kwargs))
        return "alpha-result"

    def _initialize_beta_model(self, *args, **kwargs):
        self.calls.append(("beta", args, kwargs))
        return "beta-result"


class TestDispatch:
    def test_routes_to_the_declared_provider(self):
        c = _Fake()
        assert c._dispatch_provider("alpha") == "alpha-result"
        assert c.calls[0][0] == "alpha"

    def test_forwards_positional_and_keyword_args(self):
        c = _Fake()
        c._dispatch_provider("beta", ["cb"], flag=True)
        _, args, kwargs = c.calls[0]
        assert args == (["cb"],)
        assert kwargs == {"flag": True}

    def test_unknown_provider_raises_with_the_label(self):
        with pytest.raises(ValueError, match="Unsupported test model type: nope"):
            _Fake()._dispatch_provider("nope")

    def test_declared_but_unimplemented_provider_is_a_distinct_error(self):
        """A wiring bug must not look like a user config typo."""

        class _Broken(BaseModelController):
            PROVIDERS = ("ghost",)

        with pytest.raises(ValueError, match="has no _initialize_ghost_model"):
            _Broken()._dispatch_provider("ghost")

    def test_empty_providers_rejects_everything(self):
        with pytest.raises(ValueError, match="Unsupported model type"):
            BaseModelController()._dispatch_provider("bedrock")


class TestRealControllersUseTheTable:
    def test_llm_controller_declares_all_seven_providers(self):
        from mnemoai.models.controllers.llm_controller import LangChainLLMController

        assert set(LangChainLLMController.PROVIDERS) == {
            "bedrock",
            "mantle",
            "ollama",
            "openai",
            "anthropic",
            "sagemaker",
            "litellm",
        }

    def test_vision_controller_declares_all_seven_providers(self):
        from mnemoai.models.controllers.vision_model_controller import (
            VisionModelController,
        )

        assert set(VisionModelController.PROVIDERS) == {
            "bedrock",
            "mantle",
            "ollama",
            "openai",
            "anthropic",
            "sagemaker",
            "litellm",
        }

    def test_embeddings_controller_declares_its_five_providers(self):
        from mnemoai.models.controllers.embeddings_controller import (
            EmbeddingsController,
        )

        assert set(EmbeddingsController.PROVIDERS) == {
            "ollama",
            "bedrock",
            "openai",
            "sagemaker",
            "litellm",
        }

    @pytest.mark.parametrize(
        "controller_path,template",
        [
            (
                "mnemoai.models.controllers.llm_controller.LangChainLLMController",
                "_initialize_{provider}_model",
            ),
            (
                "mnemoai.models.controllers.vision_model_controller.VisionModelController",
                "_initialize_{provider}_model",
            ),
            (
                "mnemoai.models.controllers.embeddings_controller.EmbeddingsController",
                "_embed_{provider}",
            ),
        ],
    )
    def test_every_declared_provider_has_an_implementing_method(
        self, controller_path, template
    ):
        """The failure this guards: declaring a provider without wiring it."""
        module_path, cls_name = controller_path.rsplit(".", 1)
        cls = getattr(__import__(module_path, fromlist=[cls_name]), cls_name)
        for provider in cls.PROVIDERS:
            name = template.format(provider=provider)
            assert callable(getattr(cls, name, None)), f"{cls_name} missing {name}"

    @pytest.mark.parametrize(
        "module_path",
        [
            "mnemoai.models.controllers.llm_controller",
            "mnemoai.models.controllers.vision_model_controller",
            "mnemoai.models.controllers.embeddings_controller",
        ],
    )
    def test_the_if_elif_chain_is_gone(self, module_path):
        source = open(
            __import__(module_path, fromlist=["_"]).__file__, encoding="utf-8"
        ).read()
        assert "_dispatch_provider" in source
        assert 'elif self.model_type ==' not in source
        assert 'elif self.embed_model_type ==' not in source


class TestSharedConfigInit:
    def test_reads_the_common_keys_from_the_passed_block(self):
        class _C(BaseModelController):
            pass

        c = _C()
        c._init_model_config(
            {
                "NAME": "some-model",
                "TYPE": "bedrock",
                "REGION": "eu-west-1",
                "MAX_TOKENS": 4096,
                "TOP_P": 0.9,
            },
            8192,
        )
        assert c.model_name == "some-model"
        assert c.model_type == "bedrock"
        assert c.region == "eu-west-1"
        assert c.max_tokens == 4096
        assert c.top_p == 0.9
        assert c.max_conversation_tokens == 8192

    def test_unset_optional_params_stay_none(self):
        """None is load-bearing: build_kwargs omits None so it's never sent."""

        class _C(BaseModelController):
            pass

        c = _C()
        c._init_model_config({"NAME": "m", "TYPE": "ollama"}, 8192)
        assert c.temperature is None
        assert c.top_p is None
        assert c.top_k is None
        assert c.max_tokens is None
        assert c.stop is None
        # Non-None defaults that DO apply.
        assert c.region == "us-east-1"
        assert c.stream is True

    def test_base_does_not_read_the_config_singleton(self):
        """The base must stay a pure function of its arguments.

        Reading ``config`` inside the base broke the vision controller tests,
        which patch ``config`` in the module they exercise. Keeping the read at
        the call site is what makes that convention keep working. Asserted on the
        import rather than the call, since the module can't reach the singleton
        without importing it.
        """
        from mnemoai.models.controllers import base_model_controller as bmc

        assert not hasattr(bmc, "config")
