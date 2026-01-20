"""LangChain-based vision model controller for multi-provider support."""

import base64
from typing import Union, Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from models.base_model_controller import BaseModelController
from utils.config import config
from utils.logger import logger


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
        self.max_tokens = self.model_id.get("MAX_TOKENS", None)
        self.max_conversation_tokens = config.get("MAX_CONVERSATION_TOKENS", 1024 * 8)
        self.temperature = self.model_id.get("TEMPERATURE", 0.1)
        self.top_p = self.model_id.get("TOP_P", None)
        self.top_k = self.model_id.get("TOP_K", None)
        self.stop = self.model_id.get("STOP", None)

        self.model: Optional[BaseChatModel] = None

    def initialize_model(self) -> None:
        """Initialize the vision model based on configured type."""
        if self.model_type == "bedrock":
            self._initialize_bedrock_model()
        elif self.model_type == "ollama":
            self._initialize_ollama_model()
        elif self.model_type == "openai":
            self._initialize_openai_model()
        else:
            raise ValueError(f"Unsupported vision model type: {self.model_type}")

    def _initialize_bedrock_model(self) -> None:
        """Initialize AWS Bedrock vision model using LangChain."""
        from langchain_aws import ChatBedrock

        logger.debug("Initializing Bedrock vision model via LangChain...")

        model_kwargs = {}
        if self.temperature is not None:
            model_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            model_kwargs["top_p"] = self.top_p
        if self.max_tokens is not None:
            model_kwargs["max_tokens"] = self.max_tokens

        region = self.model_id.get("REGION", "us-east-1")

        self.model = ChatBedrock(
            model_id=self.model_name,
            region_name=region,
            model_kwargs=model_kwargs,
        )

    def _initialize_ollama_model(self) -> None:
        """Initialize Ollama vision model using LangChain."""
        from langchain_ollama import ChatOllama

        logger.debug("Initializing Ollama vision model via LangChain...")

        host = self.model_id.get("HOST", "localhost")
        port = self.model_id.get("PORT", 11434)
        base_url = f"http://{host}:{port}"

        kwargs = {
            "model": self.model_name,
            "base_url": base_url,
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

        self.model = ChatOllama(**kwargs)

    def _initialize_openai_model(self) -> None:
        """Initialize OpenAI vision model using LangChain."""
        from langchain_openai import ChatOpenAI

        logger.debug("Initializing OpenAI vision model via LangChain...")

        kwargs = {
            "model": self.model_name,
        }

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        self.model = ChatOpenAI(**kwargs)

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

        return response.content if hasattr(response, "content") else str(response)
