from langchain_ollama import ChatOllama


class ChatOllamaWrapper(ChatOllama):
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    def _chat_params(self, messages, stop=None, **kwargs):
        params = super()._chat_params(messages, stop, **kwargs)

        # Add penalties to options dict
        if self.presence_penalty is not None:
            params["options"]["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            params["options"]["frequency_penalty"] = self.frequency_penalty

        return params
