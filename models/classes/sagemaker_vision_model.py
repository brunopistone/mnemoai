"""Extended SageMaker model with vision support."""

import json
import logging
from typing import Any, Optional
from typing_extensions import override

from strands.models.sagemaker import SageMakerAIModel
from strands.types.content import ContentBlock, Messages
from strands.types.tools import ToolChoice, ToolSpec


class SageMakerVisionModel(SageMakerAIModel):
    """SageMaker model with proper vision content handling."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize SageMaker vision model.

        Args:
            *args: Positional arguments for SageMakerAIModel
            **kwargs: Keyword arguments for SageMakerAIModel
        """
        super().__init__(*args, **kwargs)
        # Suppress logging for this instance
        logging.getLogger("strands.models.sagemaker").setLevel(logging.WARNING)

    @override
    @classmethod
    def format_request_message_content(cls, content: ContentBlock) -> dict[str, Any]:
        """Format content block, converting reasoning to text.

        Args:
            content: Content block to format

        Returns:
            Formatted content dictionary
        """
        if "reasoningContent" in content and content["reasoningContent"]:
            return {
                "signature": content["reasoningContent"]
                .get("reasoningText", {})
                .get("signature", ""),
                "thinking": content["reasoningContent"]
                .get("reasoningText", {})
                .get("text", ""),
                "type": "thinking",
            }
        elif not content.get("reasoningContent", None):
            content.pop("reasoningContent", None)

        if "image" in content:
            image_data = content["image"]["source"]["bytes"]
            if not image_data.startswith("data:"):
                image_data = f"data:image/png;base64,{image_data}"
            return {
                "type": "image_url",
                "image_url": {"url": image_data},
            }

        if "video" in content:
            return {
                "type": "video_url",
                "video_url": {
                    "detail": "auto",
                    "url": content["video"]["source"]["bytes"],
                },
            }

        return super().format_request_message_content(content)

    @override
    def format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        tool_choice: ToolChoice | None = None,
    ) -> dict[str, Any]:
        """Format request with proper handling for image_url and video_url content.

        Args:
            messages: List of messages
            tool_specs: Optional tool specifications
            system_prompt: Optional system prompt
            tool_choice: Optional tool choice configuration

        Returns:
            Formatted request dictionary
        """
        formatted_messages = self.format_request_messages(messages, system_prompt)

        payload = {
            "messages": formatted_messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_spec["name"],
                        "description": tool_spec["description"],
                        "parameters": tool_spec["inputSchema"]["json"],
                    },
                }
                for tool_spec in tool_specs or []
            ],
            **{
                k: v
                for k, v in self.payload_config.items()
                if k not in ["additional_args", "tool_results_as_user_messages"]
            },
        }

        if not payload["tools"]:
            payload.pop("tools")
            payload.pop("tool_choice", None)
        else:
            payload["tool_choice"] = "auto"

        for message in payload["messages"]:
            if message.get("role") == "assistant" and message.get("tool_calls", []):
                message.pop("content", None)
            if message.get("role") == "tool" and self.payload_config.get(
                "tool_results_as_user_messages", False
            ):
                tool_call_id = message.get("tool_call_id", "ABCDEF")
                content = message.get("content", "")
                message["role"] = "user"
                message["content"] = (
                    f"Tool call ID '{tool_call_id}' returned: {content}"
                )

            # Keep text, image_url, and video_url content
            content_list = message.get("content", [])
            if isinstance(content_list, list) and content_list:
                message["content"] = [
                    c
                    for c in content_list
                    if "text" in c or "image_url" in c or "video_url" in c
                ]

        request = {
            "EndpointName": self.endpoint_config["endpoint_name"],
            "Body": json.dumps(payload),
            "ContentType": "application/json",
            "Accept": "application/json",
        }

        if self.endpoint_config.get("inference_component_name"):
            request["InferenceComponentName"] = self.endpoint_config[
                "inference_component_name"
            ]
        if self.endpoint_config.get("target_model"):
            request["TargetModel"] = self.endpoint_config["target_model"]
        if self.endpoint_config.get("target_variant"):
            request["TargetVariant"] = self.endpoint_config["target_variant"]
        if self.endpoint_config.get("additional_args"):
            request.update(self.endpoint_config["additional_args"].__dict__)

        return request
