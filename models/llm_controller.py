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
        self.max_tokens = self.model_id.get("MAX_TOKENS", None)
        self.max_conversation_tokens = config.get("MAX_CONVERSATION_TOKENS", 1024 * 8)
        self.min_p = self.model_id.get("MIN_P", None)
        self.presence_penalty = self.model_id.get("PRESENCE_PENALTY", None)
        self.reasoning_effort = self.model_id.get("REASONING_EFFORT", None)
        self.repetition_penalty = self.model_id.get("REPETITION_PENALTY", None)
        self.repeat_last_n = self.model_id.get("REPEAT_LAST_N", None)
        self.stop = self.model_id.get("STOP", None)
        self.stream = self.model_id.get("STREAM", False)
        self.temperature = self.model_id.get("TEMPERATURE", 0.1)
        self.top_p = self.model_id.get("TOP_P", None)
        self.top_k = self.model_id.get("TOP_K", None)

        self.model = None

    def initialize_model(self) -> None:
        """Initialize the LLM model based on configured type (Ollama, Bedrock, or SageMaker)."""
        if self.model_type == "bedrock":
            logger.info("Initializing Bedrock model...")

            args = self._set_bedrock_inference_parameters()

            self.model = BedrockModel(
                model_id=self.model_name,
                **args,
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
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def get_model(self) -> Union[BedrockModel, SageMakerAIModel, ThinkingOllamaModel]:
        """Get the initialized LLM model instance, initializing if needed.

        Returns:
            Initialized model instance (BedrockModel, SageMakerAIModel, or ThinkingOllamaModel)
        """
        if self.model is None:
            self.initialize_model()
        return self.model
