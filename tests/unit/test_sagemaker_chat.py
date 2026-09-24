"""SageMaker transport parsing with fake event streams, no AWS requests."""

from unittest.mock import patch

from langchain_core.messages import HumanMessage

from mnemoai.models.chat_models.sagemaker_chat import ChatSageMaker


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
