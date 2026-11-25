from langchain_ollama import ChatOllama
from pydantic import Field

from mnemoai.models.provider_params import DEFAULT_OLLAMA_NUM_BATCH


class ChatOllamaWrapper(ChatOllama):
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    # LangChain's ChatOllama does not currently forward this Ollama option.
    num_batch: int = Field(default=DEFAULT_OLLAMA_NUM_BATCH, gt=0, strict=True)

    def _chat_params(self, messages, stop=None, **kwargs):
        params = super()._chat_params(messages, stop, **kwargs)
        # Caller-supplied options may be reused for other requests.
        params["options"] = dict(params["options"])
        params["options"].setdefault("num_batch", self.num_batch)

        # Add penalties to options dict
        if self.presence_penalty is not None:
            params["options"]["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            params["options"]["frequency_penalty"] = self.frequency_penalty

        return params
