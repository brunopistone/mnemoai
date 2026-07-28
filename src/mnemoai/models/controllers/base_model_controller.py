"""Shared base for the LLM, vision, and embeddings controllers.

All three controllers had the same two things written out by hand:

* an ``if/elif`` chain over ``TYPE`` calling a per-provider method — 7 branches
  in the LLM controller, 7 in the vision controller, 5 in the embeddings
  controller. Adding a provider meant editing three chains, and forgetting one
  produced a confusing "Unsupported ..." error rather than a missing feature.
* the same block of ``self.x = self.model_id.get("X", default)`` reads, differing
  only in which config section they came from.

This base owns both. Per-provider *inference parameters* still live in
``models.provider_params`` (the single source of truth, consumed via
``build_kwargs``) — this class handles dispatch and the common config reads, not
parameter mapping.

Dispatch is by naming convention (``_initialize_<provider>_model`` /
``_embed_<provider>``), validated against the subclass's declared ``PROVIDERS``.
That is deliberately a *declared* tuple rather than "call whatever method
exists": an unlisted provider must fail with the same clear error as before, and
the tuple doubles as the readable list of what a controller supports.
"""

from typing import Any, Dict, Tuple


class BaseModelController:
    """Provider dispatch + common config init for the model controllers.

    Subclasses declare:
        PROVIDERS: the provider ``TYPE`` values this controller supports.
        PROVIDER_METHOD_TEMPLATE: how a provider maps to a method name.
        PROVIDER_LABEL: what to call the thing in the "unsupported" error.
    """

    PROVIDERS: Tuple[str, ...] = ()
    PROVIDER_METHOD_TEMPLATE: str = "_initialize_{provider}_model"
    PROVIDER_LABEL: str = "model"

    def _init_model_config(
        self, model_id: Dict[str, Any], max_conversation_tokens: int
    ) -> Dict[str, Any]:
        """Set the attributes every model controller needs from its config block.

        Takes the already-read config dict rather than reading the ``config``
        singleton itself, for two reasons: the base stays a pure function of its
        input (no hidden global), and each controller keeps its own
        ``config.get(...)`` call so a test can patch ``config`` in the module it
        is exercising — the project's documented patch-where-it's-looked-up
        convention. Reading it in here instead silently broke those patches.

        Only the genuinely common keys are set; a controller's own extras (the
        LLM's reasoning/penalty knobs, the vision controller's
        ``INPUT_FORMAT``/``API_BASE``) stay in its ``__init__``.

        The ``None`` defaults are load-bearing, not laziness:
        ``provider_params.build_kwargs`` omits any param that is ``None``, so an
        unset knob is never sent to the provider. That matters for
        ``temperature`` in particular — newer Bedrock Claude models reject it as
        deprecated, so it must only appear when explicitly configured.

        Args:
            model_id: The model config block (``MODEL_ID``/``VISION_MODEL_ID``).
            max_conversation_tokens: Root-level context budget.

        Returns:
            The same dict, for the caller's own further reads.
        """
        model_id = model_id or {}

        self.model_id = model_id
        self.model_name = model_id["NAME"]
        self.model_type = model_id["TYPE"]
        # Optional custom Bedrock endpoint (e.g. Bedrock Mantle); routes SigV4
        # Converse calls there when set.
        self.endpoint_url = model_id.get("ENDPOINT_URL", None)
        self.region = model_id.get("REGION", "us-east-1")
        self.max_tokens = model_id.get("MAX_TOKENS", None)
        self.max_conversation_tokens = max_conversation_tokens
        self.temperature = model_id.get("TEMPERATURE", None)
        self.top_p = model_id.get("TOP_P", None)
        self.top_k = model_id.get("TOP_K", None)
        self.stop = model_id.get("STOP", None)
        self.stream = model_id.get("STREAM", True)

        return model_id

    def _provider_method(self, provider: str):
        """Resolve the bound per-provider method, or raise the standard error.

        Raises:
            ValueError: if ``provider`` is not in ``PROVIDERS`` (unsupported), or
                is declared but has no implementing method (a wiring bug — worth
                a distinct message so it isn't mistaken for a config typo).
        """
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported {self.PROVIDER_LABEL} type: {provider}")

        name = self.PROVIDER_METHOD_TEMPLATE.format(provider=provider)
        method = getattr(self, name, None)
        if method is None or not callable(method):
            raise ValueError(
                f"{type(self).__name__} declares {self.PROVIDER_LABEL} provider "
                f"'{provider}' but has no {name}() method."
            )
        return method

    def _dispatch_provider(self, provider: str, *args, **kwargs):
        """Call the per-provider method for ``provider``."""
        return self._provider_method(provider)(*args, **kwargs)
