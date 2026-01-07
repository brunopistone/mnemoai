import base64
from models.base_model_controller import BaseModelController
from models.classes.sagemaker_vision_model import SageMakerVisionModel
from strands.models import BedrockModel
from strands.models.ollama import OllamaModel
from strands.models.openai import OpenAIModel
from typing import Union
from utils.config import config
from utils.logger import logger


class VisionModelController(BaseModelController):
    def __init__(self, verbose: bool = False) -> None:
        """Initialize vision model controller.

        Args:
            verbose: Enable verbose mode
        """
        self.verbose_mode = verbose
        self.model_id = config.get("VISION_MODEL_ID")
        self.max_tokens = self.model_id.get("MAX_TOKENS", None)
        self.max_conversation_tokens = config.get("MAX_CONVERSATION_TOKENS", 1024 * 8)
        self.min_p = self.model_id.get("MIN_P", None)
        self.presence_penalty = self.model_id.get("PRESENCE_PENALTY", None)
        self.reasoning_effort = self.model_id.get("REASONING_EFFORT", None)
        self.repetition_penalty = self.model_id.get("REPETITION_PENALTY", None)
        self.repeat_last_n = self.model_id.get("REPEAT_LAST_N", None)
        self.stream = self.model_id.get("STREAM", False)
        self.stop = self.model_id.get("STOP", None)
        self.temperature = self.model_id.get("TEMPERATURE", 0.1)
        self.top_p = self.model_id.get("TOP_P", None)
        self.top_k = self.model_id.get("TOP_K", None)

        self.model = None

    def initialize_model(self) -> None:
        """Initialize the vision model based on configured type (Ollama or SageMaker)."""
        if self.model_id["TYPE"] == "bedrock":
            logger.debug("Initializing Bedrock vision model...")

            args = self._set_bedrock_inference_parameters()

            self.model = BedrockModel(
                model_id=self.model_id["NAME"],
                **args,
            )
        elif self.model_id["TYPE"] == "openai":
            logger.debug("Initializing OpenAI vision model...")

            args = self._set_openai_inference_parameters()

            self.model = OpenAIModel(
                model_id=self.model_id["NAME"],
                params=args,
            )
        elif self.model_id["TYPE"] == "sagemaker":
            logger.debug("Initializing SageMaker vision model...")

            payload_config = self._set_sagemaker_inference_parameters()

            self.model = SageMakerVisionModel(
                endpoint_config={
                    "endpoint_name": self.model_id["NAME"],
                    "region_name": self.model_id.get("REGION", "us-east-1"),
                },
                payload_config=payload_config,
            )
        elif self.model_id["TYPE"] == "ollama":
            logger.debug("Initializing Ollama vision model...")

            args, additional_args, options = self._set_ollama_inference_parameters()

            self.model = OllamaModel(
                host=f"http://{self.model_id['HOST']}:{self.model_id['PORT']}",
                model_id=self.model_id["NAME"],
                options=options,
                additional_args=additional_args,
                **args,
            )
        else:
            raise ValueError(f"Unsupported vision model type: {self.model_id['TYPE']}")

    def get_model(self) -> Union[SageMakerVisionModel, OllamaModel]:
        """Get the initialized vision model instance, initializing if needed.

        Returns:
            Initialized vision model instance (SageMakerVisionModel or OllamaModel)
        """
        if self.model is None:
            self.initialize_model()
        return self.model

    def format_request(
        self, question: str, image_data: bytes, image_ext: str = "png"
    ) -> dict:
        """
        Format request for vision model.
        Args:
            question: Question string
            image_data: Raw image bytes
            image_ext: Image file extension (default: "png")

        Returns:
            JSON with formatted request.

        """

        if self.model_id["TYPE"] == "bedrock":
            # Bedrock expects raw bytes
            return {
                "role": "user",
                "content": [
                    {"text": question},
                    {
                        "image": {
                            "format": image_ext,
                            "source": {"bytes": image_data},
                        }
                    },
                ],
            }
        elif self.model_id["TYPE"] == "ollama":
            # Ollama expects base64 string
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            return {
                "role": "user",
                "content": [
                    {"text": question},
                    {"image": {"source": {"bytes": image_base64}}},
                ],
            }
        elif self.model_id["TYPE"] == "openai":
            # OpenAI expects base64 string with data URI
            return {
                "role": "user",
                "content": [
                    {"text": question},
                    {
                        "image": {
                            "format": image_ext,
                            "source": {"bytes": image_data},
                        }
                    },
                ],
            }
        elif self.model_id["TYPE"] == "sagemaker":
            # SageMaker expects base64 string with data URI
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            return {
                "role": "user",
                "content": [
                    {"text": question},
                    {
                        "image": {
                            "source": {"bytes": f"data:image/png;base64,{image_base64}"}
                        }
                    },
                ],
            }
        else:
            return None
