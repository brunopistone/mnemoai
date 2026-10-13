"""SageMaker transport parsing with fake event streams, no AWS requests."""

from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from mnemoai.models.chat_models.sagemaker_chat import ChatSageMaker


@pytest.mark.parametrize("fmt", ["openai_chat", "text_generation"])
@pytest.mark.parametrize("configured,call,expected", [
    (["configured"], None, ["configured"]), (None, ["call"], ["call"]),
    (["configured"], ["call"], ["call"]), (["configured"], [], []),
    (None, None, None),
])
def test_stop_sequences_resolve_call_override_without_serializing_null(fmt, configured, call, expected):
    model = ChatSageMaker.model_construct(endpoint_name="unused", input_format=fmt, stop=configured)
    payload = model._build_payload([HumanMessage(content="hello")], stop=call)
    parameters = payload if fmt == "openai_chat" else payload["parameters"]
    if expected is None:
        assert "stop" not in parameters
    else:
        assert parameters["stop"] == expected


def test_sse_frames_split_across_transport_chunks():
    payload = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'
    with patch("mnemoai.models.chat_models.sagemaker_chat.boto3.Session") as session:
        model = ChatSageMaker(endpoint_name="fake")
        session.return_value.client.return_value.invoke_endpoint_with_response_stream.return_value = {
            "Body": [
                {"PayloadPart": {"Bytes": payload[:20]}},
                {"PayloadPart": {"Bytes": payload[20:]}},
            ]
        }
        chunks = list(model._stream([HumanMessage(content="hello")]))
    assert "".join(chunk.message.content for chunk in chunks) == "hello"
