"""LangChain-based vision model controller for multi-provider support."""

import base64
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from mnemoai.models.chat_models.bedrock_stream_compat import harden_converse_stream
from mnemoai.models.controllers.base_model_controller import BaseModelController
from mnemoai.models.provider_params import build_kwargs, extra_params
from mnemoai.utils.config import config
from mnemoai.utils.logger import logger


class VisionModelController(BaseModelController):
    """Vision model controller using LangChain abstractions."""

    CONFIG_SECTION = "VISION_MODEL_ID"
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
    PROVIDER_LABEL = "vision model"

    def __init__(self, verbose: bool = False) -> None:
        self.verbose_mode = verbose
        # Shared reads (model_id/name/type, region, endpoint_url, max_tokens,
        # max_conversation_tokens, temperature, top_p, top_k, stop, stream).
        # The config.get calls stay HERE so tests can patch this module's config.
        self._init_model_config(
            config.get("VISION_MODEL_ID"),
            config.get("MAX_CONVERSATION_TOKENS", 1024 * 8),
        )
        # Vision-only: SageMaker / LiteLLM connection details.
        self.input_format = self.model_id.get("INPUT_FORMAT", "openai_chat")
        self.api_base = self.model_id.get("API_BASE")
        self.api_key = self.model_id.get("API_KEY")

        self.model: Optional[BaseChatModel] = None

    def initialize_model(self) -> None:
        """Initialize the vision model based on configured type."""
        self._dispatch_provider(self.model_type)

    def _boto_config(self):
        """botocore Config for the Bedrock client (see ``_boto_request_config``).

        The ``config.get`` stays here, not in the base, so a test can patch this
        module's ``config``.
        """
        return self._boto_request_config(config.get("LLM", {}))

    def _initialize_bedrock_model(self) -> None:
        """Initialize AWS Bedrock vision model using the Converse API.

        Converse, not the legacy ``ChatBedrock``/InvokeModel client this used to
        build — the same migration the chat path made, in the one path that never
        followed. InvokeModel takes inference params as RAW BODY fields
        (``model_kwargs``), and each family defines its own body: an OpenAI GPT
        model on Bedrock rejects ours outright (``ValidationException:
        Unsupported parameter: 'max_tokens' is not supported with this model``),
        so EVERY describe_image call failed for a configuration the app was happy
        to accept — while the same model id worked for chat. Converse takes them
        as top-level client fields that become ``inferenceConfig``, which every
        family understands; hence the registry sends them to ``main`` now, and
        there is no ``model_kwargs`` bucket to fill.

        Two deliberate differences from the chat path: no ``disable_streaming``
        (a description is one ``invoke``, so there is no stream to force on), and
        no Claude thinking fields (a vision reply is a caption, not a reasoning
        turn). ``harden_converse_stream`` is applied anyway — it is inert unless
        something streams, and leaving it off would make the two Bedrock paths
        disagree the day one does.
        """
        from langchain_aws import ChatBedrockConverse

        logger.debug("Initializing Bedrock vision model via LangChain...")

        passthrough, _ = build_kwargs("VISION_MODEL_ID", "bedrock", self)
        kwargs = {
            "model": self.model_name,
            "region_name": self.region,
            "config": self._boto_config(),
            **passthrough,
        }
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
            logger.debug(f"Using custom Bedrock endpoint: {self.endpoint_url}")

        kwargs.update(extra_params(self.model_id))

        self.model = harden_converse_stream(ChatBedrockConverse(**kwargs))

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
        kwargs.update(extra_params(self.model_id))

        self.model = ChatOllama(**kwargs)

    def _initialize_mlx_model(self) -> None:
        """Initialize a vision model served by a local MLX server.

        Same OpenAI-shaped client and HOST/PORT convenience as the chat path; a
        vision model there is a ``model_type: multimodal`` entry and accepts the
        standard ``image_url`` content blocks this controller already builds.
        ``KEEP_ALIVE`` (and ``EXTRA_PARAMS``) go in ``extra_body`` for the same
        reason as the chat path: the openai SDK rejects unknown top-level keys.
        """
        from langchain_openai import ChatOpenAI

        logger.debug("Initializing MLX vision model via LangChain...")

        passthrough, extra_body = build_kwargs("VISION_MODEL_ID", "mlx", self)
        kwargs = {
            "model": self.model_name,
            **passthrough,
        }

        base_url = self.api_base or self.endpoint_url
        if not base_url:
            host = self.model_id.get("HOST", "127.0.0.1")
            port = self.model_id.get("PORT", 8000)
            base_url = f"http://{host}:{port}/v1"
        kwargs["base_url"] = base_url
        kwargs["api_key"] = self.api_key or "sk-local"

        keep_alive = self.model_id.get("KEEP_ALIVE")
        if keep_alive is not None:
            extra_body["keep_alive"] = keep_alive

        extra_body.update(extra_params(self.model_id))
        if extra_body:
            kwargs["extra_body"] = extra_body

        self.model = ChatOpenAI(**kwargs)

    def _initialize_openai_model(self) -> None:
        """Initialize OpenAI vision model using LangChain."""
        from langchain_openai import ChatOpenAI

        logger.debug("Initializing OpenAI vision model via LangChain...")

        passthrough, _ = build_kwargs("VISION_MODEL_ID", "openai", self)
        kwargs = {
            "model": self.model_name,
            **passthrough,
        }
        # Point at an OpenAI-compatible server (local llama-server / LM Studio /
        # vLLM with a vision model, etc.) when configured. API_BASE is canonical;
        # ENDPOINT_URL is an accepted alias.
        # API_BASE (canonical) or ENDPOINT_URL (alias) → OpenAI-compatible server.
        base_url = self.model_id.get("API_BASE") or self.endpoint_url
        if base_url:
            kwargs["base_url"] = base_url
        if self.model_id.get("API_KEY"):
            kwargs["api_key"] = self.model_id["API_KEY"]
        elif base_url:
            kwargs["api_key"] = "sk-local"
        # reasoning_effort is a first-class arg; the rest go via model_kwargs.
        extra = extra_params(self.model_id)
        if "reasoning_effort" in extra:
            kwargs["reasoning_effort"] = extra.pop("reasoning_effort")
        if extra:
            kwargs["model_kwargs"] = extra

        self.model = ChatOpenAI(**kwargs)

    def _initialize_anthropic_model(self) -> None:
        """Initialize a direct Anthropic API vision model (Claude is multimodal;
        accepts the OpenAI ``image_url`` content directly)."""
        from langchain_anthropic import ChatAnthropic

        logger.debug("Initializing Anthropic vision model via LangChain...")

        passthrough, _ = build_kwargs("VISION_MODEL_ID", "anthropic", self)
        # ChatAnthropic requires max_tokens; default when not configured.
        passthrough.setdefault("max_tokens", self.max_tokens or 4096)
        kwargs = {
            "model": self.model_name,
            **passthrough,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.endpoint_url:
            kwargs["base_url"] = self.endpoint_url
        kwargs.update(extra_params(self.model_id))

        self.model = ChatAnthropic(**kwargs)

    def _initialize_mantle_model(self) -> None:
        """Initialize a Bedrock Mantle vision model via the shared factory (same
        protocols as the chat controller)."""
        from mnemoai.models.mantle_factory import build_mantle_model

        self.model = build_mantle_model(
            self.model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            extra_params=extra_params(self.model_id),
        )

    def _initialize_sagemaker_model(self) -> None:
        """Initialize a SageMaker vision model (``ChatSageMaker`` + ``openai_chat``
        format; endpoint must accept the OpenAI image format)."""
        from mnemoai.models.chat_models.sagemaker_chat import ChatSageMaker

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
        """Initialize a LiteLLM vision model (forwards the OpenAI multimodal
        content; API_BASE/API_KEY optional, else the provider's env vars)."""
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
        """The initialized vision model, initializing it if needed."""
        if self.model is None:
            self.initialize_model()
        return self.model

    def format_request(
        self, question: str, image_data: bytes, image_ext: str = "png"
    ) -> HumanMessage:
        """Build a multimodal ``HumanMessage`` (text + base64 image)."""
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        mime_types = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        mime_type = mime_types.get(image_ext.lower(), "image/png")

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
        """Describe an image using the vision model."""
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
                    # Text blocks only (type defaults to "text" for Bedrock).
                    if block.get("type", "text") == "text" and "text" in block:
                        parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts).strip()
        return str(content)
