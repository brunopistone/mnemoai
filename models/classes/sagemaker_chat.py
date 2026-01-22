"""LangChain ChatModel wrapper for SageMaker endpoints."""

import json
import re
from typing import Any, Dict, Iterator, List, Optional

import boto3
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from utils.logger import logger


class ChatSageMaker(BaseChatModel):
    """Chat model for SageMaker endpoints.

    Supports multiple input formats:
    - openai_chat: OpenAI-compatible chat format (messages array)
    - text_generation: HuggingFace text generation format (inputs string)

    Args:
        endpoint_name: SageMaker endpoint name
        region_name: AWS region
        input_format: Input format type (openai_chat or text_generation)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        stop: Stop sequences
    """

    endpoint_name: str
    region_name: str = "us-east-1"
    input_format: str = "openai_chat"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop: Optional[List[str]] = None
    credentials_profile_name: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        session_kwargs = {"region_name": self.region_name}
        if self.credentials_profile_name:
            session_kwargs["profile_name"] = self.credentials_profile_name
        session = boto3.Session(**session_kwargs)
        object.__setattr__(self, "_client", session.client("sagemaker-runtime"))

    @property
    def _llm_type(self) -> str:
        return "sagemaker-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "endpoint_name": self.endpoint_name,
            "region_name": self.region_name,
            "input_format": self.input_format,
        }

    def _convert_messages_to_openai(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        """Convert LangChain messages to OpenAI chat format."""
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                msg_dict = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                result.append(msg_dict)
            elif isinstance(msg, ToolMessage):
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
        return result

    def _convert_messages_to_text(self, messages: List[BaseMessage]) -> str:
        """Convert LangChain messages to a single text prompt."""
        parts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                parts.append(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                parts.append(f"Assistant: {msg.content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _build_payload(
        self,
        messages: List[BaseMessage],
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build request payload based on input format."""
        if self.input_format == "openai_chat":
            payload = {
                "messages": self._convert_messages_to_openai(messages),
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            if stream:
                payload["stream"] = True
            if self.top_p is not None:
                payload["top_p"] = self.top_p
            if self.stop:
                payload["stop"] = kwargs.get("stop", self.stop)
        else:
            # text_generation format
            payload = {
                "inputs": self._convert_messages_to_text(messages),
                "parameters": {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
                },
            }
            if self.top_p is not None:
                payload["parameters"]["top_p"] = self.top_p
            if self.top_k is not None:
                payload["parameters"]["top_k"] = self.top_k
            if self.stop:
                payload["parameters"]["stop"] = kwargs.get("stop", self.stop)

        return payload

    def _parse_response(self, response_body: bytes) -> AIMessage:
        """Parse response based on input format."""
        result = json.loads(response_body)

        if self.input_format == "openai_chat":
            choice = result["choices"][0]
            message = choice["message"]
            content = message.get("content", "")

            # Extract tool calls if present
            tool_calls = []
            if message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    tool_calls.append(
                        {
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "args": json.loads(tc["function"]["arguments"]),
                        }
                    )

            # Check for reasoning content
            additional_kwargs = {}
            if message.get("reasoning_content"):
                additional_kwargs["reasoning_content"] = message["reasoning_content"]

            return AIMessage(
                content=content,
                tool_calls=tool_calls or [],
                additional_kwargs=additional_kwargs,
            )
        else:
            # text_generation format
            if isinstance(result, list):
                content = result[0].get("generated_text", "")
            else:
                content = result.get("generated_text", "")
            return AIMessage(content=content)

    def _extract_thinking(self, content: str) -> tuple[str, Optional[str]]:
        """Extract thinking/reasoning from </think> tags if present.

        Args:
            content: Raw response content

        Returns:
            Tuple of (final_content, reasoning_content)
        """
        match = re.search(r"^(.*?)</think>\s*(.*)$", content, re.DOTALL)
        if match:
            return match.group(2).strip(), match.group(1).strip()
        return content, None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """Generate a response from the SageMaker endpoint."""
        payload = self._build_payload(messages, stream=False, stop=stop, **kwargs)

        try:
            response = self._client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            response_body = response["Body"].read()
            message = self._parse_response(response_body)

            # Check for inline thinking tags
            if message.content and "</think>" in message.content:
                content, reasoning = self._extract_thinking(message.content)
                additional_kwargs = dict(message.additional_kwargs)
                if reasoning:
                    additional_kwargs["reasoning_content"] = reasoning
                message = AIMessage(
                    content=content,
                    tool_calls=message.tool_calls or [],
                    additional_kwargs=additional_kwargs,
                )

            return ChatResult(generations=[ChatGeneration(message=message)])

        except Exception as e:
            logger.error(f"SageMaker endpoint error: {e}")
            raise

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream responses from SageMaker endpoint."""
        payload = self._build_payload(messages, stream=True, stop=stop, **kwargs)

        try:
            response = self._client.invoke_endpoint_with_response_stream(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )

            buffer = ""
            in_thinking = False  # Only true after we detect thinking content

            for event in response["Body"]:
                if "PayloadPart" not in event:
                    continue

                chunk_data = event["PayloadPart"]["Bytes"].decode("utf-8")

                for line in chunk_data.split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        buffer += line
                        try:
                            data = json.loads(buffer)
                            buffer = ""
                        except json.JSONDecodeError:
                            continue

                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    reasoning = delta.get("reasoning_content", "")

                    # Handle reasoning_content from delta (standard format)
                    additional_kwargs = {}
                    if reasoning:
                        additional_kwargs["reasoning_content"] = reasoning

                    if content or reasoning:
                        # Handle inline </think> tag in content
                        if content and "</think>" in content:
                            parts = content.split("</think>", 1)
                            in_thinking = False
                            if parts[0]:
                                yield ChatGenerationChunk(
                                    message=AIMessageChunk(
                                        content="",
                                        additional_kwargs={
                                            "reasoning_content": parts[0]
                                        },
                                    )
                                )
                            if len(parts) > 1 and parts[1]:
                                yield ChatGenerationChunk(
                                    message=AIMessageChunk(content=parts[1])
                                )
                        elif content and in_thinking:
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content="",
                                    additional_kwargs={"reasoning_content": content},
                                )
                            )
                        else:
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content=content,
                                    additional_kwargs=additional_kwargs,
                                )
                            )

        except Exception as e:
            logger.error(f"SageMaker streaming error: {e}")
            raise

    def bind_tools(self, tools: List[Any], **kwargs) -> "ChatSageMaker":
        """Bind tools to the model for function calling.

        Args:
            tools: List of tools to bind

        Returns:
            New ChatSageMaker instance with tools bound
        """
        # Store tools for use in payload building
        # This is a simplified implementation - full implementation would
        # convert tools to OpenAI function format
        return self
