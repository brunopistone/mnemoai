"""Submission identity for pending input, without changing its text contract."""

from uuid import uuid4


class PendingInput(str):
    """A string with an opaque identity, shared by queue/delivery notifications.

    Existing agent callbacks consume strings. Keeping that contract also keeps
    IDs out of model prompts and saved history. Two identical submissions still
    have different IDs; expanding a paste or trimming text preserves the ID.
    ``cancelled`` is used only by the UI's FIFO consumer, on its event-loop thread.
    """

    def __new__(cls, text: str, *, message_id: str = None):
        value = super().__new__(cls, text)
        value.message_id = message_id or uuid4().hex
        value.cancelled = False
        return value

    def with_text(self, text: str):
        return type(self)(text, message_id=self.message_id)
