from typing import Dict, Tuple, Any


class BaseModelController:
    """Base class for model controllers with shared parameter setup logic."""

    def _set_bedrock_inference_parameters(self) -> Dict[str, Any]:
        """Set inference parameters for Bedrock models.

        Returns:
            Dictionary of Bedrock inference parameters
        """
        args = {}

        if self.max_tokens:
            args["max_tokens"] = self.max_tokens

        if self.temperature:
            args["temperature"] = self.temperature

        if self.top_p:
            args["top_p"] = self.top_p

        if self.stop:
            args["stop_sequences"] = self.stop

        # Bedrock does not support streaming for vision requests
        args["streaming"] = False

        return args

    def _set_ollama_inference_parameters(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Set inference parameters for Ollama models.

        Returns:
            Tuple of (args, additional_args, options) dictionaries
        """
        args = {}
        additional_args = {}
        options = {}

        if self.max_tokens:
            args["max_tokens"] = self.max_tokens

        if self.top_p:
            args["top_p"] = self.top_p

        if self.top_k:
            options["top_k"] = self.top_k

        if self.min_p:
            options["min_p"] = self.min_p

        if self.repetition_penalty:
            options["repeat_penalty"] = self.repetition_penalty

        if self.presence_penalty:
            options["presence_penalty"] = self.presence_penalty

        if self.stop:
            args["stop_sequences"] = self.stop

        if not self.verbose_mode:
            additional_args["think"] = self.verbose_mode

        return args, additional_args, options

    def _set_sagemaker_inference_parameters(self) -> Dict[str, Any]:
        """Set inference parameters for SageMaker models.

        Returns:
            Dictionary of SageMaker payload configuration
        """
        payload_config = {
            "temperature": self.temperature,
            "stream": True,
        }

        if self.max_tokens:
            payload_config["max_tokens"] = self.max_tokens

        if self.top_p:
            payload_config["top_p"] = self.top_p

        if self.top_k:
            payload_config["top_k"] = self.top_k

        if self.min_p:
            payload_config["min_p"] = self.min_p

        additional_args = {}

        if self.repetition_penalty:
            additional_args["do_sample"] = True
            additional_args["repetition_penalty"] = self.repetition_penalty

        if self.presence_penalty:
            additional_args["presence_penalty"] = self.presence_penalty

        if additional_args:
            payload_config["additional_args"] = additional_args

        if self.stop:
            payload_config["stop"] = self.stop

        return payload_config
