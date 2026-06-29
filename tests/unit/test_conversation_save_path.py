"""Unit tests for save/load file tracking (current_conversation_path).

After loading a conversation (or saving one), a later bare ``/save`` must write
back to the SAME file, not spawn a new timestamped copy. ``/clear`` resets this
so a fresh conversation saves to a new file. These drive ``save_conversation`` /
``load_conversation`` on a minimally-stubbed client (no LLM/agent graph).
"""

import json

from mnemoai.client.client import LangGraphClient


class _StubAgent:
    def __init__(self, messages=None):
        self.messages = messages or []


def _client(tmp_path):
    """A bare client with just the attributes save/load touch."""
    c = LangGraphClient.__new__(LangGraphClient)
    c.agent = _StubAgent()
    c.system_prompt = "sys"
    c.tools = []
    c.current_conversation_path = None
    return c


def test_bare_save_uses_timestamped_file(tmp_path, monkeypatch):
    import mnemoai.client.client as mod

    monkeypatch.setattr(mod, "conversations_dir", lambda: tmp_path)
    c = _client(tmp_path)
    c.save_conversation(timestamp="20260629_120000")
    expected = tmp_path / "conversation_20260629_120000.json"
    assert expected.is_file()
    assert c.current_conversation_path == str(expected)


def test_save_after_load_overwrites_same_file(tmp_path, monkeypatch):
    import mnemoai.client.client as mod

    monkeypatch.setattr(mod, "conversations_dir", lambda: tmp_path)

    # A pre-existing saved conversation to "load".
    existing = tmp_path / "conversation_20260625_151836.json"
    existing.write_text(json.dumps({"messages": [], "tools": []}))

    c = _client(tmp_path)
    assert c.load_conversation(str(existing)) is True
    assert c.current_conversation_path == str(existing)

    # A bare /save (new timestamp) must overwrite the loaded file, NOT create
    # a new conversation_<ts>.json.
    c.save_conversation(timestamp="20260629_120557")
    files = sorted(p.name for p in tmp_path.glob("*.json"))
    assert files == ["conversation_20260625_151836.json"]  # only the original
    assert c.current_conversation_path == str(existing)


def test_clear_resets_then_save_makes_new_file(tmp_path, monkeypatch):
    import mnemoai.client.client as mod

    monkeypatch.setattr(mod, "conversations_dir", lambda: tmp_path)
    c = _client(tmp_path)
    c.save_conversation(timestamp="20260629_120000")  # sets current path

    # Simulate /clear resetting the open-conversation pointer.
    c.current_conversation_path = None
    c.save_conversation(timestamp="20260629_130000")

    files = sorted(p.name for p in tmp_path.glob("*.json"))
    assert files == [
        "conversation_20260629_120000.json",
        "conversation_20260629_130000.json",
    ]


def test_explicit_path_updates_current(tmp_path, monkeypatch):
    import mnemoai.client.client as mod

    monkeypatch.setattr(mod, "conversations_dir", lambda: tmp_path)
    c = _client(tmp_path)
    target = tmp_path / "mychat.json"
    c.save_conversation(timestamp="20260629_120000", path=str(target))
    assert target.is_file()
    # The explicit path becomes the open conversation for subsequent bare saves.
    assert c.current_conversation_path == str(target)
