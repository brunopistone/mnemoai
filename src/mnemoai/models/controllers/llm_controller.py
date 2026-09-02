"""LangChain-based LLM controller for multi-provider support."""

from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel

from mnemoai.models.chat_models.bedrock_stream_compat import harden_converse_stream
from mnemoai.models.chat_models.chat_ollama_wrapper import ChatOllamaWrapper
from mnemoai.models.chat_models.sagemaker_chat import ChatSageMaker
from mnemoai.models.controllers.base_model_controller import BaseModelController
from mnemoai.models.provider_params import build_kwargs, extra_params
from mnemoai.utils.config import config
from mnemoai.utils.logger import logger


class LangChainLLMController(BaseModelController):
    """LLM Controller using LangChain model abstractions."""

    CONFIG_SECTION = "MODEL_ID"
    PROVIDERS = (
        "bedrock",
        "mantle",
        "ollama",
        "openai",
        "anthropic",
        "sagemaker",
        "litellm",
        "mlx",
    )
    PROVIDER_LABEL = "model"

    def __init__(self, verbose: bool = False) -> None:
        self.verbose_mode = verbose
        # The config.get calls stay HERE so tests can patch this module's config.
        self._apply_model_id(
            config.get("MODEL_ID"),
            config.get("MAX_CONVERSATION_TOKENS", 1024 * 8),
        )

        self.model: Optional[BaseChatModel] = None

    def _apply_model_id(self, model_id: dict, max_conversation_tokens: int) -> None:
        """Snapshot every inference param this controller reads from a model block.

        Separate from ``__init__`` because a model block is also applied *after*
        construction, by :meth:`build_model_variant` — every ``_initialize_*``
        path reads these attributes, never ``self.model_id``, so a variant that
        patched only the dict would build the new model with the old params.
        """
        # Shared reads (model_id/name/type, region, endpoint_url, max_tokens,
        # max_conversation_tokens, temperature, top_p, top_k, stop, stream).
        self._init_model_config(model_id, max_conversation_tokens)
        # LLM-only knobs.
        self.frequency_penalty = self.model_id.get("FREQUENCY_PENALTY", None)
        self.min_p = self.model_id.get("MIN_P", None)
        self.presence_penalty = self.model_id.get("PRESENCE_PENALTY", None)
        self.reasoning_effort = self.model_id.get("REASONING_EFFORT", None)
        self.reasoning_model = self.model_id.get("REASONING", False)
        self.repetition_penalty = self.model_id.get("REPETITION_PENALTY", None)
        self.thinking_tokens = self.model_id.get("THINKING_TOKENS", 1024 * 2)

    def initialize_model(self, callbacks: list[BaseCallbackHandler] = None) -> None:
        """Initialize the LLM model based on the configured ``TYPE``."""
        self._dispatch_provider(self.model_type, callbacks)

    def _boto_config(self):
        """botocore Config for the Bedrock client (see ``_boto_request_config``).

        The ``config.get`` stays here, not in the base, so a test can patch this
        module's ``config``.
        """
        return self._boto_request_config(config.get("LLM", {}))

    def _initialize_bedrock_model(self, callbacks: list = None) -> None:
        """Initialize AWS Bedrock model using LangChain Converse API."""
        from langchain_aws import ChatBedrockConverse

        logger.info("Initializing Bedrock model via LangChain...")

        passthrough, _ = build_kwargs("MODEL_ID", "bedrock", self)
        kwargs = {
            "model": self.model_name,
            "region_name": self.region,
            "callbacks": callbacks,
            "config": self._boto_config(),
            **passthrough,
        }

        # Route to a custom Bedrock endpoint (e.g. Bedrock Mantle) when set.
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
            logger.info(f"Using custom Bedrock endpoint: {self.endpoint_url}")

        # Honor STREAM on Converse. ChatBedrockConverse auto-derives
        # `disable_streaming` from a HARDCODED model-id allowlist that lags new
        # releases (langchain-aws 1.6.0 matches claude-3/-sonnet-4/-opus-4/-haiku-4
        # but NOT claude-opus-5 / claude-sonnet-5), so a newer Claude silently fell
        # back to one non-streaming call — no token-by-token output, no error. Its
        # validator only auto-sets the flag when the key is ABSENT, so setting it
        # explicitly wins. Verified against live Bedrock: opus-5 streams fine
        # (incl. with tools bound + adaptive thinking). EXTRA_PARAMS is applied
        # after this, so an explicit user override still takes precedence.
        kwargs["disable_streaming"] = not self.stream

        # Extended thinking (REASONING or REASONING_EFFORT). The Converse endpoint
        # fans in EVERY Bedrock family (nova/mistral/llama/deepseek/qwen/…), so
        # gate the Anthropic-only `thinking` fields on the model actually being
        # Claude. A non-Claude with reasoning enabled gets NO injection here — many
        # families reason automatically (DeepSeek-R1) or don't take an effort field
        # on Converse; EXTRA_PARAMS (applied below) stays the escape hatch for a
        # deliberate provider-specific reasoning field.
        if self.reasoning_model or self.reasoning_effort:
            from mnemoai.models.mantle_factory import (
                _EFFORT_TO_TOKENS,
                _anthropic_thinking_kwargs,
                is_anthropic_model,
            )

            if is_anthropic_model(self.model_name):
                budget = (
                    _EFFORT_TO_TOKENS.get(self.reasoning_effort, self.thinking_tokens)
                    if self.reasoning_effort
                    else self.thinking_tokens
                )
                kwargs["additional_model_request_fields"] = _anthropic_thinking_kwargs(
                    self.model_name, self.reasoning_effort, budget
                )
                # Older Claude requires temperature=1 with thinking (newer rejects).
                if self.temperature is not None:
                    kwargs["temperature"] = 1.0
            else:
                logger.debug(
                    "Skipping Anthropic thinking fields for non-Claude Bedrock "
                    "model '%s'; reasoning is provider-specific (many families "
                    "reason automatically). Use EXTRA_PARAMS to hand-inject "
                    "provider reasoning fields.",
                    self.model_name,
                )

        kwargs.update(extra_params(self.model_id))

        # Same class of problem as `disable_streaming` above: langchain-aws'
        # stream parser indexes its own converter's result unguarded, so ONE
        # block type it has no branch for (a GPT model's encrypted reasoning)
        # kills the turn with an IndexError. See `bedrock_stream_compat`.
        self.model = harden_converse_stream(ChatBedrockConverse(**kwargs))

    def _initialize_mantle_model(self, callbacks: list = None) -> None:
        """Initialize an AWS Bedrock Mantle model (see ``models.mantle_factory``;
        protocol chosen via ``API_PROTOCOL``)."""
        from mnemoai.models.mantle_factory import build_mantle_model

        self.model = build_mantle_model(
            self.model_id,
            callbacks=callbacks,
            streaming=self.stream,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            reasoning_effort=self.reasoning_effort,
            thinking_tokens=self.thinking_tokens,
            reasoning_model=self.reasoning_model,
            extra_params=extra_params(self.model_id),
        )

    def _initialize_litellm_model(self, callbacks: list = None) -> None:
        """Initialize LiteLLM model using langchain-litellm."""
        # Deferred: langchain_litellm pulls litellm→transformers→torch (~1.5s),
        # dead weight for every non-litellm provider (kept off the startup path).
        from langchain_litellm import ChatLiteLLM

        logger.info("Initializing LiteLLM model...")

        passthrough, model_kwargs = build_kwargs("MODEL_ID", "litellm", self)
        kwargs = {
            "model": self.model_name,
            "callbacks": callbacks,
            "streaming": self.stream,
            **passthrough,
        }

        if self.model_id.get("API_BASE"):
            kwargs["api_base"] = self.model_id["API_BASE"]
        if self.model_id.get("API_KEY"):
            kwargs["api_key"] = self.model_id["API_KEY"]

        model_kwargs.update(extra_params(self.model_id))

        self.model = ChatLiteLLM(model_kwargs=model_kwargs, **kwargs)

    def _initialize_ollama_model(self, callbacks: list = None) -> None:
        """Initialize Ollama model using LangChain."""
        logger.info("Initializing Ollama model...")

        host = self.model_id.get("HOST", "localhost")
        port = self.model_id.get("PORT", 11434)
        base_url = f"http://{host}:{port}"

        passthrough, _ = build_kwargs("MODEL_ID", "ollama", self)
        kwargs = {
            "model": self.model_name,
            "base_url": base_url,
            "callbacks": callbacks,
            **passthrough,
        }
        kwargs["num_ctx"] = self.max_conversation_tokens

        # Surface reasoning output for thinking models when verbose.
        if self.verbose_mode:
            kwargs["reasoning"] = True

        kwargs.update(extra_params(self.model_id))

        self.model = ChatOllamaWrapper(**kwargs)

    def _initialize_mlx_model(self, callbacks: list = None) -> None:
        """Initialize a model served by a local MLX server (Apple Silicon).

        The server speaks the OpenAI protocol, so this reuses ``ChatOpenAI``
        rather than a bespoke client. Two things make it its own provider instead
        of "openai with an API_BASE":

        * connection is ``HOST``/``PORT`` (default ``127.0.0.1:8000``) like the
          other local runner, so no one has to hand-write a base URL with the
          ``/v1`` suffix;
        * ``KEEP_ALIVE`` is passed per request — the MLX server treats it as how
          long to keep the model resident afterwards (``30m``, ``0`` to unload at
          once, ``-1`` to pin), which is what makes on-demand model swapping
          usable from here.

        ``TOP_K``/``MIN_P``/``REPETITION_PENALTY``/``KEEP_ALIVE`` are not part of
        the OpenAI API, so they travel in ``extra_body`` — the openai SDK's
        ``create()`` is typed and rejects unknown top-level keys, which is what
        ``model_kwargs`` would produce (``TypeError: unexpected keyword argument
        'top_k'``, raised client-side before any request goes out).

        The subclass is what keeps a thinking model's reasoning visible: the
        server's reasoning parser reports it in a non-standard field that plain
        ``ChatOpenAI`` discards (see ``chat_openai_reasoning``).
        """
        from mnemoai.models.chat_models.chat_openai_reasoning import ChatOpenAIReasoning

        logger.info("Initializing MLX model...")

        passthrough, extra_body = build_kwargs("MODEL_ID", "mlx", self)
        kwargs = {
            "model": self.model_name,
            "callbacks": callbacks,
            "streaming": self.stream,
            **passthrough,
        }

        # API_BASE wins when set (non-default mount point); otherwise HOST/PORT.
        base_url = self.model_id.get("API_BASE") or self.endpoint_url
        if not base_url:
            host = self.model_id.get("HOST", "127.0.0.1")
            port = self.model_id.get("PORT", 8000)
            base_url = f"http://{host}:{port}/v1"
        kwargs["base_url"] = base_url
        logger.info(f"Using MLX server endpoint: {base_url}")

        # The server ignores auth; a placeholder lets the client construct
        # without OPENAI_API_KEY leaking in from the environment.
        kwargs["api_key"] = self.model_id.get("API_KEY") or "sk-local"

        keep_alive = self.model_id.get("KEEP_ALIVE")
        if keep_alive is not None:
            extra_body["keep_alive"] = keep_alive

        # EXTRA_PARAMS is a body passthrough here (the MLX server accepts extra
        # fields), so it merges into extra_body rather than the client kwargs.
        extra_body.update(extra_params(self.model_id))
        if extra_body:
            kwargs["extra_body"] = extra_body

        self.model = ChatOpenAIReasoning(**kwargs)

    def _initialize_openai_model(self, callbacks: list = None) -> None:
        """Initialize an OpenAI-compatible model.

        ``API_BASE`` (alias ``ENDPOINT_URL``) points at any OpenAI-compatible
        server (local llama-server / LM Studio / vLLM); a placeholder key is used
        when a custom base URL is set. ``API_PROTOCOL`` selects chat_completions
        (default) or responses — on responses, ``REASONING_EFFORT`` is sent as
        ``reasoning={effort, summary:"auto"}`` so the summary is visible.

        On chat_completions the subclass recovers a local server's reasoning
        field (see ``chat_openai_reasoning``); it is inert against real OpenAI
        and on the responses protocol, which has its own converters.
        """
        from mnemoai.models.chat_models.chat_openai_reasoning import ChatOpenAIReasoning

        logger.info("Initializing OpenAI model...")

        protocol = self.model_id.get("API_PROTOCOL", "chat_completions")
        passthrough, model_kwargs = build_kwargs("MODEL_ID", "openai", self)
        kwargs = {
            "model": self.model_name,
            "callbacks": callbacks,
            "streaming": self.stream,
            **passthrough,
        }

        # API_BASE (canonical) or ENDPOINT_URL (alias) → OpenAI-compatible server.
        base_url = self.model_id.get("API_BASE") or self.endpoint_url
        if base_url:
            kwargs["base_url"] = base_url
            logger.info(f"Using OpenAI-compatible endpoint: {base_url}")
        # Local servers ignore auth; a placeholder key lets the client construct
        # without OPENAI_API_KEY. An explicit API_KEY (or env var) still wins.
        if self.model_id.get("API_KEY"):
            kwargs["api_key"] = self.model_id["API_KEY"]
        elif base_url:
            kwargs["api_key"] = "sk-local"

        extra = extra_params(self.model_id)
        if protocol == "responses":
            kwargs["use_responses_api"] = True
            # Upgrade the bare reasoning_effort to a reasoning={effort, summary}
            # object (so the summary is visible), unless the user set their own.
            model_kwargs.pop("reasoning_effort", None)
            if (
                self.reasoning_effort
                and "reasoning" not in extra
                and "reasoning_effort" not in extra
            ):
                kwargs["reasoning"] = {
                    "effort": self.reasoning_effort,
                    "summary": "auto",
                }
            # First-class args; lift from EXTRA_PARAMS to avoid a model_kwargs clash.
            if "reasoning" in extra:
                kwargs["reasoning"] = extra.pop("reasoning")
            if "reasoning_effort" in extra:
                kwargs["reasoning_effort"] = extra.pop("reasoning_effort")

        # chat_completions reasoning_effort stays in model_kwargs; merge EXTRA_PARAMS.
        model_kwargs.update(extra)
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs

        self.model = ChatOpenAIReasoning(**kwargs)

    def _initialize_anthropic_model(self, callbacks: list = None) -> None:
        """Initialize a direct Anthropic API model (``api.anthropic.com`` or a
        custom ``ENDPOINT_URL``), distinct from the Mantle anthropic protocol."""
        from langchain_anthropic import ChatAnthropic

        logger.info("Initializing Anthropic model via LangChain...")

        passthrough, _ = build_kwargs("MODEL_ID", "anthropic", self)
        # ChatAnthropic requires max_tokens; default when not configured.
        passthrough.setdefault("max_tokens", self.max_tokens or 4096)
        kwargs = {
            "model": self.model_name,
            "callbacks": callbacks,
            "streaming": self.stream,
            **passthrough,
        }

        if self.model_id.get("API_KEY"):
            kwargs["api_key"] = self.model_id["API_KEY"]
        if self.endpoint_url:
            kwargs["base_url"] = self.endpoint_url
            logger.info(f"Using custom Anthropic endpoint: {self.endpoint_url}")

        # Extended thinking (REASONING or REASONING_EFFORT). Budget must be
        # < max_tokens (bump if needed) and temperature/top_p/top_k are dropped.
        # _anthropic_thinking_kwargs picks the version-specific request form.
        # TYPE=anthropic only reaches Claude, but gate defensively for symmetry so
        # a non-Claude id behind an OpenAI-shaped proxy isn't sent Anthropic fields
        # (EXTRA_PARAMS below stays the escape hatch).
        if self.reasoning_model or self.reasoning_effort:
            from mnemoai.models.mantle_factory import (
                _EFFORT_TO_TOKENS,
                _anthropic_thinking_kwargs,
                is_anthropic_model,
            )

            if is_anthropic_model(self.model_name):
                budget = (
                    _EFFORT_TO_TOKENS.get(self.reasoning_effort, self.thinking_tokens)
                    if self.reasoning_effort
                    else self.thinking_tokens
                )
                if kwargs["max_tokens"] <= budget:
                    kwargs["max_tokens"] = budget + 1024
                kwargs.update(
                    _anthropic_thinking_kwargs(
                        self.model_name, self.reasoning_effort, budget
                    )
                )
                kwargs.pop("temperature", None)
                kwargs.pop("top_p", None)
                kwargs.pop("top_k", None)
            else:
                logger.debug(
                    "Skipping Anthropic thinking fields for non-Claude model '%s' "
                    "on the direct-anthropic path.",
                    self.model_name,
                )

        # EXTRA_PARAMS applied last so an explicit override wins.
        kwargs.update(extra_params(self.model_id))

        self.model = ChatAnthropic(**kwargs)

    def _initialize_sagemaker_model(self, callbacks: list = None) -> None:
        """Initialize SageMaker model using ChatSageMaker wrapper."""

        logger.info("Initializing SageMaker model...")

        endpoint_name = self.model_name
        input_format = self.model_id.get("INPUT_FORMAT", "openai_chat")

        passthrough, _ = build_kwargs("MODEL_ID", "sagemaker", self)
        kwargs = {
            "endpoint_name": endpoint_name,
            "region_name": self.region,
            "input_format": input_format,
            "callbacks": callbacks,
            **passthrough,
        }

        kwargs.update(extra_params(self.model_id))

        self.model = ChatSageMaker(**kwargs)

    def get_model(self) -> BaseChatModel:
        """The initialized LLM model, initializing it if needed."""
        if self.model is None:
            self.initialize_model()
        return self.model

    def build_model_variant(
        self,
        overrides: Optional[dict] = None,
        callbacks: list[BaseCallbackHandler] = None,
        non_reasoning: bool = False,
    ) -> BaseChatModel:
        """Build an independent model from this config, with ``overrides`` merged in.

        The one way to get a model that is *nearly* the configured one: a peer
        controller is constructed (which re-reads the config, so a ``/params``
        edit is picked up), ``overrides`` are merged over its ``MODEL_ID`` block,
        and the whole snapshot is re-applied through :meth:`_apply_model_id` before
        dispatch. Going through the ordinary provider path is what makes an
        override of ``TYPE`` work — the variant may be a different provider
        entirely, not just a different model name.

        Args:
            overrides: Partial ``MODEL_ID`` merged over the configured one.
            callbacks: LangChain callbacks for the new instance.
            non_reasoning: Disable extended thinking on the variant.

        Returns:
            A new model instance. Never mutates this controller or its model.
        """
        peer = LangChainLLMController(verbose=self.verbose_mode and not non_reasoning)
        if overrides:
            peer._apply_model_id(
                {**peer.model_id, **overrides}, peer.max_conversation_tokens
            )
        if non_reasoning:
            peer.reasoning_effort = None
            peer.reasoning_model = False
        peer.initialize_model(callbacks=callbacks)
        return peer.get_model()

    def build_non_reasoning_model(
        self,
        callbacks: list[BaseCallbackHandler] = None,
        overrides: Optional[dict] = None,
    ) -> BaseChatModel:
        """Build a fresh model instance with extended thinking/reasoning DISABLED.

        Provider-agnostic: most ``_initialize_*`` paths gate thinking on
        ``REASONING`` / ``REASONING_EFFORT`` (self.reasoning_model /
        self.reasoning_effort), so a peer controller with both cleared yields the
        SAME model with reasoning off — no crash for providers that never had it.
        The Ollama path instead surfaces reasoning via ``verbose_mode`` (sets
        ``reasoning=True``), so the peer is also built non-verbose to suppress it
        there. Used for compaction summaries, which don't benefit from a slow
        reasoning pass. Returns an independent instance; never mutates this one.
        """
        return self.build_model_variant(
            overrides=overrides, callbacks=callbacks, non_reasoning=True
        )

    def get_model_type(self) -> str:
        """The model type string (bedrock, ollama, openai, …)."""
        return self.model_type
