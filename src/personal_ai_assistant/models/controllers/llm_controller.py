"""LangChain-based LLM controller for multi-provider support."""

from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_litellm import ChatLiteLLM
from personal_ai_assistant.models.controllers.base_model_controller import BaseModelController
from personal_ai_assistant.models.chat_models.chat_ollama_wrapper import ChatOllamaWrapper
from personal_ai_assistant.models.chat_models.sagemaker_chat import ChatSageMaker
from personal_ai_assistant.models.provider_params import build_kwargs
from personal_ai_assistant.utils.config import config
from personal_ai_assistant.utils.logger import logger


class LangChainLLMController(BaseModelController):
    """LLM Controller using LangChain model abstractions."""

    def __init__(self, verbose: bool = False) -> None:
        """Initialize LLM controller.

        Args:
            verbose: Enable verbose mode to show thinking process
        """
        self.verbose_mode = verbose
        self.model_id = config.get("MODEL_ID")
        self.model_name = self.model_id["NAME"]
        self.model_type = self.model_id["TYPE"]
        self.region = self.model_id.get("REGION", "us-east-1")
        # Optional custom Bedrock endpoint (e.g. Bedrock Mantle:
        # https://bedrock-mantle.<region>.api.aws). Routes standard SigV4
        # Converse calls to that endpoint when set; default endpoint otherwise.
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
        """Initialize the LLM model based on configured type.

        Args:
            callbacks: Optional list of callback handlers for streaming
        """
        if self.model_type == "bedrock":
            self._initialize_bedrock_model(callbacks)
        elif self.model_type == "mantle":
            self._initialize_mantle_model(callbacks)
        elif self.model_type == "ollama":
            self._initialize_ollama_model(callbacks)
        elif self.model_type == "openai":
            self._initialize_openai_model(callbacks)
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

        # Enable thinking/reasoning for Claude models
        if self.reasoning_model:
            # Map REASONING_EFFORT to budget_tokens when set
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

            kwargs["additional_model_request_fields"] = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": budget,
                }
            }
            # Older Claude models require temperature=1 with thinking; newer
            # ones reject the parameter entirely. Only override when a
            # temperature was explicitly configured.
            if self.temperature is not None:
                kwargs["temperature"] = 1.0

        self.model = ChatBedrockConverse(**kwargs)

    def _initialize_mantle_model(self, callbacks: list = None) -> None:
        """Initialize an AWS Bedrock Mantle model.

        Mantle authenticates with a short-lived bearer token derived from
        standard AWS (SigV4) credentials. Model IDs are bare provider names
        (e.g. ``qwen.qwen3-32b``, ``openai.gpt-5.4``, ``anthropic.claude-haiku-4-5``).

        The protocol is chosen with ``API_PROTOCOL`` (chat_completions |
        responses | anthropic); see ``models.mantle_factory``.
        """
        from personal_ai_assistant.models.mantle_factory import build_mantle_model

        self.model = build_mantle_model(
            self.model_id,
            callbacks=callbacks,
            streaming=self.stream,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
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

        self.model = ChatLiteLLM(model_kwargs=model_kwargs, **kwargs)

    def _initialize_ollama_model(self, callbacks: list = None) -> None:
        """Initialize Ollama model using LangChain."""
        logger.info("Initializing Ollama model...")

        host = self.model_id.get("HOST", "localhost")
        port = self.model_id.get("PORT", 11434)
        base_url = f"http://{host}:{port}"

        # Build kwargs
        passthrough, _ = build_kwargs("MODEL_ID", "ollama", self)
        kwargs = {
            "model": self.model_name,
            "base_url": base_url,
            "callbacks": callbacks,
            **passthrough,
        }

        # Set context window
        kwargs["num_ctx"] = self.max_conversation_tokens

        # Enable reasoning output for thinking models when verbose
        if self.verbose_mode:
            kwargs["reasoning"] = True

        self.model = ChatOllamaWrapper(**kwargs)

    def _initialize_openai_model(self, callbacks: list = None) -> None:
        """Initialize OpenAI model using LangChain."""
        from langchain_openai import ChatOpenAI

        logger.info("Initializing OpenAI model...")

        passthrough, model_kwargs = build_kwargs("MODEL_ID", "openai", self)
        kwargs = {
            "model": self.model_name,
            "callbacks": callbacks,
            "streaming": self.stream,
            **passthrough,
        }
        # reasoning_effort (o1/o3 models) goes in model_kwargs.
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs

        self.model = ChatOpenAI(**kwargs)

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

        self.model = ChatSageMaker(**kwargs)

    def get_model(self) -> BaseChatModel:
        """Get the initialized LLM model instance, initializing if needed.

        Returns:
            Initialized LangChain chat model instance
        """
        if self.model is None:
            self.initialize_model()
        return self.model

    def get_model_type(self) -> str:
        """Get the model type string.

        Returns:
            Model type (bedrock, ollama, openai, sagemaker)
        """
        return self.model_type
