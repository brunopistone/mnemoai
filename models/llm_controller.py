from typing import Union
from models.base_model_controller import BaseModelController
from models.classes.thinking_ollama import ThinkingOllamaModel
from strands.models import BedrockModel
from strands.models.openai import OpenAIModel
from strands.models.sagemaker import SageMakerAIModel
from utils.config import config
from utils.logger import logger


class LLMController(BaseModelController):
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
        self.thinking_tokens = self.model_id.get("THINKING_TOKENS", 1024 * 2)
        self.max_tokens = self.model_id.get("MAX_TOKENS", None)
        self.max_conversation_tokens = config.get("MAX_CONVERSATION_TOKENS", 1024 * 8)
        self.min_p = self.model_id.get("MIN_P", None)
        self.presence_penalty = self.model_id.get("PRESENCE_PENALTY", None)
        self.reasoning_effort = self.model_id.get("REASONING_EFFORT", None)
        self.repetition_penalty = self.model_id.get("REPETITION_PENALTY", None)
        self.repeat_last_n = self.model_id.get("REPEAT_LAST_N", None)
        self.stop = self.model_id.get("STOP", None)
        self.stream = self.model_id.get("STREAM", True)
        # No default: newer Bedrock Claude models reject `temperature` as
        # deprecated, so only send it when explicitly configured.
        self.temperature = self.model_id.get("TEMPERATURE", None)
        self.top_p = self.model_id.get("TOP_P", None)
        self.top_k = self.model_id.get("TOP_K", None)
        # Bedrock Mantle: OpenAI-compatible (chat/responses) or Anthropic API,
        # selected via API_PROTOCOL. Optional ENDPOINT_URL overrides the default.
        self.api_protocol = self.model_id.get("API_PROTOCOL", "chat_completions")
        self.endpoint_url = self.model_id.get("ENDPOINT_URL", None)

        self.model = None

    def initialize_model(self) -> None:
        """Initialize the LLM model based on configured type (Ollama, Bedrock, or SageMaker)."""
        if self.model_type == "bedrock":
            logger.info("Initializing Bedrock model...")

            args, additional_request_fields = self._set_bedrock_inference_parameters()

            self.model = BedrockModel(
                model_id=self.model_name,
                **args,
                **additional_request_fields,
            )
        elif self.model_type == "ollama":
            logger.info("Initializing Ollama model...")

            args, additional_args, options = self._set_ollama_inference_parameters()

            self.model = ThinkingOllamaModel(
                host=f"http://{self.model_id['HOST']}:{self.model_id['PORT']}",
                model_id=self.model_name,
                options=options,
                additional_args=additional_args,
                **args,
            )
        elif self.model_type == "openai":
            logger.info("Initializing OpenAI model...")

            args = self._set_openai_inference_parameters()

            self.model = OpenAIModel(
                model_id=self.model_name,
                params=args,
            )
        elif self.model_type == "sagemaker":
            logger.info("Initializing Sagemaker model...")

            payload_config = self._set_sagemaker_inference_parameters()

            self.model = SageMakerAIModel(
                endpoint_config={
                    "endpoint_name": self.model_name,
                    "region_name": self.region,
                },
                payload_config=payload_config,
            )
        elif self.model_type == "mantle":
            self._initialize_mantle_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _initialize_mantle_model(self) -> None:
        """Initialize an AWS Bedrock Mantle model.

        Mantle authenticates with a short-lived bearer token minted from
        standard AWS (SigV4) credentials. Model IDs are bare provider names
        (e.g. ``qwen.qwen3-32b``, ``openai.gpt-5.4``, ``anthropic.claude-haiku-4-5``).

        ``API_PROTOCOL`` selects the wire protocol:
          - ``chat_completions`` (default): OpenAI Chat Completions at ``/v1``.
            Uses Strands' native ``bedrock_mantle_config`` (token handled by SDK).
          - ``responses``: OpenAI Responses at ``/openai/v1`` (e.g. GPT-5.4).
            Strands' mantle helper targets ``/v1``, so we supply the
            ``/openai/v1`` base_url + bearer token via ``client_args`` instead.
          - ``anthropic``: Anthropic Messages at ``/anthropic`` (Claude models).
        """
        protocol = self.api_protocol
        logger.info(f"Initializing Bedrock Mantle model (protocol={protocol})...")

        root = f"https://bedrock-mantle.{self.region}.api.aws"
        params = {}
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p

        if protocol == "chat_completions":
            from strands.models.openai import OpenAIModel

            # Native Mantle support: SDK derives base_url (/v1) + bearer token.
            self.model = OpenAIModel(
                model_id=self.model_name,
                bedrock_mantle_config={"region": self.region},
                params=params or None,
            )
        elif protocol == "responses":
            from strands.models.openai_responses import OpenAIResponsesModel
            from aws_bedrock_token_generator import provide_token

            # Mantle serves Responses under /openai/v1 (not /v1), so override
            # base_url + token via client_args rather than bedrock_mantle_config.
            base_url = self.endpoint_url or f"{root}/openai/v1"
            token = provide_token(region=self.region)
            # The Responses API uses `max_output_tokens`, not `max_tokens`.
            responses_params = {
                k: v for k, v in params.items() if k != "max_tokens"
            }
            if self.max_tokens:
                responses_params["max_output_tokens"] = self.max_tokens
            self.model = OpenAIResponsesModel(
                model_id=self.model_name,
                client_args={"api_key": token, "base_url": base_url},
                params=responses_params or None,
            )
        elif protocol == "anthropic":
            from strands.models.anthropic import AnthropicModel
            from aws_bedrock_token_generator import provide_token

            base_url = self.endpoint_url or f"{root}/anthropic"
            token = provide_token(region=self.region)
            # AnthropicModel takes max_tokens as a top-level (required) kwarg;
            # other inference params go in `params`. Anthropic requires
            # max_tokens, so default it when unset.
            inference_params = {
                k: v for k, v in params.items() if k != "max_tokens"
            }
            self.model = AnthropicModel(
                model_id=self.model_name,
                max_tokens=self.max_tokens or 4096,
                client_args={"api_key": token, "base_url": base_url},
                params=inference_params or None,
            )
        else:
            raise ValueError(
                f"Unknown Mantle API_PROTOCOL '{protocol}'. Expected: "
                "chat_completions, responses, anthropic"
            )

    def get_model(self) -> Union[BedrockModel, SageMakerAIModel, ThinkingOllamaModel]:
        """Get the initialized LLM model instance, initializing if needed.

        Returns:
            Initialized model instance (BedrockModel, SageMakerAIModel, or ThinkingOllamaModel)
        """
        if self.model is None:
            self.initialize_model()
        return self.model
