"""LangChain-based vision model controller for multi-provider support."""

import base64
from typing import Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from personal_ai_assistant.models.controllers.base_model_controller import BaseModelController
from personal_ai_assistant.models.provider_params import build_kwargs
from personal_ai_assistant.utils.config import config
from personal_ai_assistant.utils.logger import logger


class VisionModelController(BaseModelController):
    """Vision model controller using LangChain abstractions."""

    def __init__(self, verbose: bool = False) -> None:
        """Initialize vision model controller.

        Args:
            verbose: Enable verbose mode
        """
        self.verbose_mode = verbose
        self.model_id = config.get("VISION_MODEL_ID")
        self.model_name = self.model_id["NAME"]
        self.model_type = self.model_id["TYPE"]
        # Optional custom Bedrock endpoint (e.g. Bedrock Mantle).
        self.endpoint_url = self.model_id.get("ENDPOINT_URL", None)
        self.max_tokens = self.model_id.get("MAX_TOKENS", None)
        self.max_conversation_tokens = config.get("MAX_CONVERSATION_TOKENS", 1024 * 8)
        # No default: newer Bedrock Claude models reject `temperature` as
        # deprecated, so only send it when explicitly configured.
        self.temperature = self.model_id.get("TEMPERATURE", None)
        self.top_p = self.model_id.get("TOP_P", None)
        self.top_k = self.model_id.get("TOP_K", None)
        self.stop = self.model_id.get("STOP", None)
        # SageMaker / LiteLLM connection.
        self.region = self.model_id.get("REGION", "us-east-1")
        self.input_format = self.model_id.get("INPUT_FORMAT", "openai_chat")
        self.api_base = self.model_id.get("API_BASE")
        self.api_key = self.model_id.get("API_KEY")
        # ChatSageMaker requires concrete numeric defaults.
        self.stream = self.model_id.get("STREAM", True)

        self.model: Optional[BaseChatModel] = None

    def initialize_model(self) -> None:
        """Initialize the vision model based on configured type."""
        if self.model_type == "bedrock":
            self._initialize_bedrock_model()
        elif self.model_type == "mantle":
            self._initialize_mantle_model()
        elif self.model_type == "ollama":
            self._initialize_ollama_model()
        elif self.model_type == "openai":
            self._initialize_openai_model()
        elif self.model_type == "sagemaker":
            self._initialize_sagemaker_model()
        elif self.model_type == "litellm":
            self._initialize_litellm_model()
        else:
            raise ValueError(f"Unsupported vision model type: {self.model_type}")

    def _initialize_bedrock_model(self) -> None:
        """Initialize AWS Bedrock vision model using LangChain."""
        from langchain_aws import ChatBedrock

        logger.debug("Initializing Bedrock vision model via LangChain...")

        _, model_kwargs = build_kwargs("VISION_MODEL_ID", "bedrock", self)

        region = self.model_id.get("REGION", "us-east-1")

        bedrock_kwargs = {
            "model_id": self.model_name,
            "region_name": region,
            "model_kwargs": model_kwargs,
        }
        if self.endpoint_url:
            bedrock_kwargs["endpoint_url"] = self.endpoint_url
            logger.debug(f"Using custom Bedrock endpoint: {self.endpoint_url}")

        self.model = ChatBedrock(**bedrock_kwargs)

    def _initialize_ollama_model(self) -> None:
        """Initialize Ollama vision model using LangChain."""
        from langchain_ollama import ChatOllama

        logger.debug("Initializing Ollama vision model via LangChain...")

        host = self.model_id.get("HOST", "localhost")
        port = self.model_id.get("PORT", 11434)
        base_url = f"http://{host}:{port}"

        passthrough, _ = build_kwargs("VISION_MODEL_ID", "ollama", self)
        kwargs = {
            "model": self.model_name,
            "base_url": base_url,
            **passthrough,
        }

        self.model = ChatOllama(**kwargs)

    def _initialize_openai_model(self) -> None:
        """Initialize OpenAI vision model using LangChain."""
        from langchain_openai import ChatOpenAI

        logger.debug("Initializing OpenAI vision model via LangChain...")

        passthrough, _ = build_kwargs("VISION_MODEL_ID", "openai", self)
        kwargs = {
            "model": self.model_name,
            **passthrough,
        }

        self.model = ChatOpenAI(**kwargs)

    def _initialize_mantle_model(self) -> None:
        """Initialize a Bedrock Mantle vision model.

        Delegates to the shared Mantle factory so the vision path supports the
        same protocols as the chat controller (chat_completions | responses |
        anthropic), selected via ``API_PROTOCOL``. The existing
        ``format_request`` emits the OpenAI ``image_url`` content format, which
        the OpenAI-compatible protocols accept directly.
        """
        from personal_ai_assistant.models.mantle_factory import build_mantle_model

        self.model = build_mantle_model(
            self.model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )

    def _initialize_sagemaker_model(self) -> None:
        """Initialize a SageMaker vision model.

        Reuses ``ChatSageMaker`` with the ``openai_chat`` input format, which
        passes the multimodal ``content`` list (text + ``image_url`` blocks
        from ``format_request``) straight through to the endpoint. The endpoint
        must serve a vision-capable model that accepts the OpenAI image format.
        """
        from personal_ai_assistant.models.chat_models.sagemaker_chat import ChatSageMaker

        logger.debug("Initializing SageMaker vision model via ChatSageMaker...")

        kwargs = {
            "endpoint_name": self.model_name,
            "region_name": self.region,
            "input_format": self.input_format,
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

    def _initialize_litellm_model(self) -> None:
        """Initialize a LiteLLM vision model.

        ``ChatLiteLLM`` forwards the OpenAI-style multimodal ``content`` list
        (text + ``image_url`` base64 data URI) that ``format_request`` produces,
        so any LiteLLM vision-capable model works. ``API_BASE``/``API_KEY`` are
        optional — omitted, LiteLLM uses the provider's own env vars.
        """
        from langchain_litellm import ChatLiteLLM

        logger.debug("Initializing LiteLLM vision model via ChatLiteLLM...")

        kwargs = {"model": self.model_name}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key

        self.model = ChatLiteLLM(**kwargs)

    def get_model(self) -> BaseChatModel:
        """Get the initialized vision model instance.

        Returns:
            Initialized LangChain chat model instance
        """
        if self.model is None:
            self.initialize_model()
        return self.model

    def format_request(
        self, question: str, image_data: bytes, image_ext: str = "png"
    ) -> HumanMessage:
        """Format request for vision model using LangChain message format.

        Args:
            question: Question string
            image_data: Raw image bytes
            image_ext: Image file extension (default: "png")

        Returns:
            LangChain HumanMessage with multimodal content
        """
        # Convert image bytes to base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Determine MIME type
        mime_types = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        mime_type = mime_types.get(image_ext.lower(), "image/png")

        # Create multimodal message content
        content = [
            {"type": "text", "text": question},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
            },
        ]

        return HumanMessage(content=content)

    def describe_image(
        self, question: str, image_data: bytes, image_ext: str = "png"
    ) -> str:
        """Describe an image using the vision model.

        Args:
            question: Question about the image
            image_data: Raw image bytes
            image_ext: Image file extension

        Returns:
            Model's description/response
        """
        if self.model is None:
            self.initialize_model()

        message = self.format_request(question, image_data, image_ext)
        response = self.model.invoke([message])

        content = response.content if hasattr(response, "content") else response
        return self._content_to_text(content)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """Normalize model content to a plain string.

        Different protocols return different shapes: Chat Completions yields a
        string, while the OpenAI Responses API (and other multimodal paths)
        yield a list of content blocks like ``[{"type": "text", "text": ...}]``.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    # Only collect text blocks; skip reasoning/tool/other blocks.
                    # Blocks default to "text" when no type is given (Bedrock).
                    if block.get("type", "text") == "text" and "text" in block:
                        parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts).strip()
        return str(content)
