"""Unit tests for the ctrl+o "expand last turn's tool calls" feature.

Two pure pieces:
  * LangGraphAgent.record_turn_tool_calls — captures raw, un-elided tool calls;
  * PromptReader._dump_last_tool_calls — reprints them fully (newlines kept),
    in contrast to the elided marker line (_format_tool_call).
No LLM, no prompt_toolkit Application, no TTY.
"""

from prompt_toolkit.formatted_text import HTML

from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.ui.tui import PromptReader

LONG_CMD = "python - <<'PY'\nimport inspect\nprint(inspect.getsource(obj)[:7000])\nPY"
LONG_PATH = "/Users/x/Downloads/" + "segment/" * 12 + "file.pdf"


class TestRecordTurnToolCalls:
    def test_preserves_raw_args_no_elision(self):
        rec = LangGraphAgent.record_turn_tool_calls(
            [{"name": "execute_bash", "args": {"command": LONG_CMD, "timeout": 30}}]
        )
        assert rec[0]["name"] == "execute_bash"
        # Full command, newlines intact (the marker would collapse+elide these).
        assert rec[0]["args"]["command"] == LONG_CMD
        assert "\n" in rec[0]["args"]["command"]

    def test_skips_non_dict_entries(self):
        rec = LangGraphAgent.record_turn_tool_calls(
            ["oops", None, {"name": "fs_read", "args": {"path": "/a"}}]
        )
        assert len(rec) == 1
        assert rec[0]["name"] == "fs_read"

    def test_missing_name_and_args_defaults(self):
        rec = LangGraphAgent.record_turn_tool_calls([{}])
        assert rec[0] == {"name": "tool", "args": {}}

    def test_empty_and_none(self):
        assert LangGraphAgent.record_turn_tool_calls([]) == []
        assert LangGraphAgent.record_turn_tool_calls(None) == []

    def test_contrast_with_marker_elision(self):
        # The marker elides a long value; the capture does not — the whole point.
        tc = {"name": "fs_read", "args": {"path": LONG_PATH}}
        marker = LangGraphAgent._format_tool_call(tc)
        rec = LangGraphAgent.record_turn_tool_calls([tc])
        assert "…" in marker  # marker is elided (LONG_PATH > 72 chars)
        assert "…" not in rec[0]["args"]["path"]  # capture is full
        assert rec[0]["args"]["path"] == LONG_PATH


class TestPanelToggle:
    def _reader(self, calls):
        return PromptReader(
            prompt_text=lambda: HTML("> "),
            commands=[],
            tool_calls_provider=lambda: calls,
        )

    def test_hidden_sets_toolbar_none(self):
        # Panel starts hidden → session.bottom_toolbar is None (bar removed, not
        # a blank/reverse-video bar).
        r = self._reader([{"name": "fs_read", "args": {"path": "/a"}}])
        r._show_tools = False
        r._sync_toolbar()
        assert r._session.bottom_toolbar is None

    def test_toggle_on_sets_toolbar_text(self):
        r = self._reader([{"name": "fs_read", "args": {"path": "/a"}}])
        r._show_tools = True  # what ctrl+o flips
        r._sync_toolbar()
        panel = r._session.bottom_toolbar
        assert panel is not None and "fs_read" in panel


class TestPanelContent:
    def test_empty_message(self):
        text = PromptReader._format_tool_calls([])
        assert "no tool calls in the last turn" in text

    def test_full_command_untruncated(self):
        text = PromptReader._format_tool_calls(
            [{"name": "execute_bash", "args": {"command": LONG_CMD}}]
        )
        assert "1 tool call(s)" in text
        assert "execute_bash" in text
        assert "import inspect" in text  # multi-line body survives
        assert "…" not in text  # not elided

    def test_multiple_calls_all_listed(self):
        text = PromptReader._format_tool_calls(
            [
                {"name": "execute_bash", "args": {"command": "ls"}},
                {"name": "fs_read", "args": {"path": LONG_PATH, "mode": "PDF"}},
            ]
        )
        assert "2 tool call(s)" in text
        assert "[1]" in text and "[2]" in text
        assert LONG_PATH in text  # full path, not elided

    def test_no_args_call(self):
        text = PromptReader._format_tool_calls([{"name": "list_dir", "args": {}}])
        assert "(no arguments)" in text
