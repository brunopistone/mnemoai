"""LangChain-based LLM controller for multi-provider support."""

from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_litellm import ChatLiteLLM

from mnemoai.models.chat_models.chat_ollama_wrapper import ChatOllamaWrapper
from mnemoai.models.chat_models.sagemaker_chat import ChatSageMaker
from mnemoai.models.controllers.base_model_controller import BaseModelController
from mnemoai.models.provider_params import build_kwargs, extra_params
from mnemoai.utils.config import config
from mnemoai.utils.logger import logger


class LangChainLLMController(BaseModelController):
    """LLM Controller using LangChain model abstractions."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose_mode = verbose
        self.model_id = config.get("MODEL_ID")
        self.model_name = self.model_id["NAME"]
        self.model_type = self.model_id["TYPE"]
        self.region = self.model_id.get("REGION", "us-east-1")
        # Optional custom Bedrock endpoint (e.g. Mantle); routes SigV4 Converse
        # calls there when set.
        self.endpoint_url = self.model_id.get("ENDPOINT_URL", None)
        self.frequency_penalty = self.model_id.get("FREQUENCY_PENALTY", None)
        self.max_conversation_tokens = config.get("MAX_CONVERSATION_TOKENS", 1024 * 8)
        self.max_tokens = self.model_id.get("MAX_TOKENS", None)
        self.min_p = self.model_id.get("MIN_P", None)
        self.presence_penalty = self.model_id.get("PRESENCE_PENALTY", None)
        self.reasoning_effort = self.model_id.get("REASONING_EFFORT", None)
        self.reasoning_model = self.model_id.get("REASONING", False)
        self.repetition_penalty = self.model_id.get("REPETITION_PENALTY", None)
        self.stop = self.model_id.get("STOP", None)
        self.stream = self.model_id.get("STREAM", True)
        self.temperature = self.model_id.get("TEMPERATURE", None)
        self.thinking_tokens = self.model_id.get("THINKING_TOKENS", 1024 * 2)
        self.top_k = self.model_id.get("TOP_K", None)
        self.top_p = self.model_id.get("TOP_P", None)

        self.model: Optional[BaseChatModel] = None

    def initialize_model(self, callbacks: list[BaseCallbackHandler] = None) -> None:
        """Initialize the LLM model based on the configured ``TYPE``."""
        if self.model_type == "bedrock":
            self._initialize_bedrock_model(callbacks)
        elif self.model_type == "mantle":
            self._initialize_mantle_model(callbacks)
        elif self.model_type == "ollama":
            self._initialize_ollama_model(callbacks)
        elif self.model_type == "openai":
            self._initialize_openai_model(callbacks)
        elif self.model_type == "anthropic":
            self._initialize_anthropic_model(callbacks)
        elif self.model_type == "sagemaker":
            self._initialize_sagemaker_model(callbacks)
        elif self.model_type == "litellm":
            self._initialize_litellm_model(callbacks)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _initialize_bedrock_model(self, callbacks: list = None) -> None:
        """Initialize AWS Bedrock model using LangChain Converse API."""
        from langchain_aws import ChatBedrockConverse

        logger.info("Initializing Bedrock model via LangChain...")

        passthrough, _ = build_kwargs("MODEL_ID", "bedrock", self)
        kwargs = {
            "model": self.model_name,
            "region_name": self.region,
            "callbacks": callbacks,
            **passthrough,
        }

        # Route to a custom Bedrock endpoint (e.g. Bedrock Mantle) when set.
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
            logger.info(f"Using custom Bedrock endpoint: {self.endpoint_url}")

        # Extended thinking (REASONING or REASONING_EFFORT). Newer Claude rejects
        # the old {"type":"enabled",…} form, so use the version-aware builder.
        if self.reasoning_model or self.reasoning_effort:
            from mnemoai.models.mantle_factory import _anthropic_thinking_kwargs

            effort_to_tokens = {
                "low": 1024,
                "medium": 8192,
                "high": 16384,
                "max": 32768,
            }
            budget = (
                effort_to_tokens.get(self.reasoning_effort, self.thinking_tokens)
                if self.reasoning_effort
                else self.thinking_tokens
            )
            kwargs["additional_model_request_fields"] = _anthropic_thinking_kwargs(
                self.model_name, self.reasoning_effort, budget
            )
            # Older Claude requires temperature=1 with thinking (newer rejects it).
            if self.temperature is not None:
                kwargs["temperature"] = 1.0

        kwargs.update(extra_params(self.model_id))

        self.model = ChatBedrockConverse(**kwargs)

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

    def _initialize_openai_model(self, callbacks: list = None) -> None:
        """Initialize an OpenAI-compatible model.

        ``API_BASE`` (alias ``ENDPOINT_URL``) points at any OpenAI-compatible
        server (local llama-server / LM Studio / vLLM); a placeholder key is used
        when a custom base URL is set. ``API_PROTOCOL`` selects chat_completions
        (default) or responses — on responses, ``REASONING_EFFORT`` is sent as
        ``reasoning={effort, summary:"auto"}`` so the summary is visible.
        """
        from langchain_openai import ChatOpenAI

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

        self.model = ChatOpenAI(**kwargs)

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
        if self.reasoning_model or self.reasoning_effort:
            from mnemoai.models.mantle_factory import _anthropic_thinking_kwargs

            effort_to_tokens = {
                "low": 1024,
                "medium": 8192,
                "high": 16384,
                "max": 32768,
            }
            budget = (
                effort_to_tokens.get(self.reasoning_effort, self.thinking_tokens)
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

    def build_non_reasoning_model(
        self, callbacks: list[BaseCallbackHandler] = None
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
        peer = LangChainLLMController(verbose=False)
        peer.reasoning_effort = None
        peer.reasoning_model = False
        peer.initialize_model(callbacks=callbacks)
        return peer.get_model()

    def get_model_type(self) -> str:
        """The model type string (bedrock, ollama, openai, …)."""
        return self.model_type
