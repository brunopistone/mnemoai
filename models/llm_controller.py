"""LangChain-based LLM controller for multi-provider support."""

from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_litellm import ChatLiteLLM
from models.base_model_controller import BaseModelController
from models.classes.chat_ollama_wrapper import ChatOllamaWrapper
from models.classes.sagemaker_chat import ChatSageMaker
from utils.config import config
from utils.logger import logger


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
        self.reasoning_model = self.model_id.get("REASONING", False)
        self.thinking_tokens = self.model_id.get("THINKING_TOKENS", 1024 * 2)
        self.max_tokens = self.model_id.get("MAX_TOKENS", None)
        self.max_conversation_tokens = config.get("MAX_CONVERSATION_TOKENS", 1024 * 8)
        self.min_p = self.model_id.get("MIN_P", None)
        self.frequency_penalty = self.model_id.get("FREQUENCY_PENALTY", None)
        self.presence_penalty = self.model_id.get("PRESENCE_PENALTY", None)
        self.reasoning_effort = self.model_id.get("REASONING_EFFORT", None)
        self.repetition_penalty = self.model_id.get("REPETITION_PENALTY", None)
        self.stop = self.model_id.get("STOP", None)
        self.stream = self.model_id.get("STREAM", True)
        self.temperature = self.model_id.get("TEMPERATURE", 0.1)
        self.top_p = self.model_id.get("TOP_P", None)
        self.top_k = self.model_id.get("TOP_K", None)

        self.model: Optional[BaseChatModel] = None

    def initialize_model(self, callbacks: list[BaseCallbackHandler] = None) -> None:
        """Initialize the LLM model based on configured type.

        Args:
            callbacks: Optional list of callback handlers for streaming
        """
        if self.model_type == "bedrock":
            self._initialize_bedrock_model(callbacks)
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
        """Initialize AWS Bedrock model using LangChain."""
        from langchain_aws import ChatBedrock

        logger.info("Initializing Bedrock model via LangChain...")

        model_kwargs = {}

        if self.temperature is not None:
            model_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            model_kwargs["top_p"] = self.top_p
        if self.max_tokens is not None:
            model_kwargs["max_tokens"] = self.max_tokens
        if self.stop:
            model_kwargs["stop_sequences"] = self.stop

        # Enable thinking/reasoning for Claude models if verbose
        if self.reasoning_model:
            model_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_tokens,
            }
            # Thinking requires temperature=1
            model_kwargs["temperature"] = 1.0

        self.model = ChatBedrock(
            model_id=self.model_name,
            region_name=self.region,
            model_kwargs=model_kwargs,
            streaming=self.stream,
            callbacks=callbacks,
        )

    def _initialize_litellm_model(self, callbacks: list = None) -> None:
        """Initialize LiteLLM model using langchain-litellm."""
        logger.info("Initializing LiteLLM model...")

        kwargs = {
            "model": self.model_name,
            "callbacks": callbacks,
            "streaming": self.stream,
        }

        model_kwargs = dict()

        if self.model_id.get("API_BASE"):
            kwargs["api_base"] = self.model_id["API_BASE"]
        if self.model_id.get("API_KEY"):
            kwargs["api_key"] = self.model_id["API_KEY"]
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        if self.stop:
            model_kwargs["stop"] = self.stop
        if self.repetition_penalty is not None:
            model_kwargs["repeat_penalty"] = self.repetition_penalty

        self.model = ChatLiteLLM(model_kwargs=model_kwargs, **kwargs)

    def _initialize_ollama_model(self, callbacks: list = None) -> None:
        """Initialize Ollama model using LangChain."""
        logger.info("Initializing Ollama model...")

        host = self.model_id.get("HOST", "localhost")
        port = self.model_id.get("PORT", 11434)
        base_url = f"http://{host}:{port}"

        # Build kwargs
        kwargs = {
            "model": self.model_name,
            "base_url": base_url,
            "callbacks": callbacks,
        }

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.max_tokens is not None:
            kwargs["num_predict"] = self.max_tokens
        if self.stop:
            kwargs["stop"] = self.stop
        if self.repetition_penalty is not None:
            kwargs["repeat_penalty"] = self.repetition_penalty
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.frequency_penalty

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

        kwargs = {
            "model": self.model_name,
            "callbacks": callbacks,
            "streaming": self.stream,
        }

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty

        # For reasoning models like o1
        if self.reasoning_effort is not None:
            kwargs["model_kwargs"] = {"reasoning_effort": self.reasoning_effort}

        self.model = ChatOpenAI(**kwargs)

    def _initialize_sagemaker_model(self, callbacks: list = None) -> None:
        """Initialize SageMaker model using ChatSageMaker wrapper."""

        logger.info("Initializing SageMaker model...")

        endpoint_name = self.model_name
        input_format = self.model_id.get("INPUT_FORMAT", "openai_chat")

        kwargs = {
            "endpoint_name": endpoint_name,
            "region_name": self.region,
            "input_format": input_format,
            "callbacks": callbacks,
        }

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.stop:
            kwargs["stop"] = self.stop

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
