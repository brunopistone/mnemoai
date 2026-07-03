"""Unit tests for the styled turn renderer (client/ui/turn_view.py).

Pure string builders for the pinned-input UI: a collapsed reasoning block and a
tool-call block. Tests assert structure/content, not exact ANSI codes (kept
loose so restyling colors doesn't churn tests) — but do check that the reasoning
text and tool args survive verbatim and that empty inputs collapse to nothing.
"""

from mnemoai.client.ui.turn_view import (
    ReasoningStatus,
    format_duration,
    render_conversation,
    render_live_reasoning,
    render_reasoning_block,
    render_tool_call,
)


class TestFormatDuration:
    def test_sub_minute(self):
        assert format_duration(0.4) == "0s"
        assert format_duration(1) == "1s"
        assert format_duration(12.7) == "13s"
        assert format_duration(59) == "59s"

    def test_minutes(self):
        assert format_duration(60) == "1m0s"
        assert format_duration(90) == "1m30s"
        assert format_duration(125) == "2m5s"


class TestReasoningBlock:
    def test_empty_reasoning_collapses_to_nothing(self):
        assert render_reasoning_block("", 1.0) == ""
        assert render_reasoning_block("   \n  ", 1.0) == ""

    def test_header_shows_duration(self):
        out = render_reasoning_block("thinking about it", 2.0)
        assert "Thought for 2s" in out

    def test_reasoning_text_preserved(self):
        out = render_reasoning_block("step one\nstep two", 1.0)
        assert "step one" in out
        assert "step two" in out

    def test_multiline_produces_multiple_lines(self):
        out = render_reasoning_block("a\nb\nc", 1.0)
        # header + 3 content lines (blank lines would add bars, none here)
        assert len(out.split("\n")) >= 4


class TestToolCall:
    def test_name_and_args_present(self):
        out = render_tool_call("web_search", {"query": "AWS news", "num": 10})
        assert "web_search" in out
        assert "query=AWS news" in out
        assert "num=10" in out

    def test_no_args_shows_only_name(self):
        out = render_tool_call("get_status", {})
        assert out.strip().endswith("get_status") or "get_status" in out
        assert "↳" not in out  # no arg connectors when there are no args

    def test_newlines_in_value_flattened(self):
        out = render_tool_call("fs_write", {"file_text": "line1\nline2"})
        # The value is flattened to keep one arg per line.
        assert "line1 line2" in out
        # The arg line itself is single-line (no raw newline inside the value).
        arg_lines = [ln for ln in out.split("\n") if "file_text=" in ln]
        assert len(arg_lines) == 1

    def test_missing_name_falls_back(self):
        out = render_tool_call("", {"a": 1})
        assert "tool" in out


class TestLiveReasoning:
    def test_shows_thinking_header_and_text(self):
        out = render_live_reasoning("weighing options", 5)
        assert "Thinking…" in out
        assert "5s" in out
        assert "weighing options" in out

    def test_empty_collapses(self):
        assert render_live_reasoning("", 3) == ""


class TestReasoningStatus:
    def test_idle_renders_empty(self):
        assert ReasoningStatus().render(now=10.0) == ""

    def test_active_renders_appended_text(self):
        s = ReasoningStatus()
        s.start(now=0.0)
        s.append("first ")
        s.append("second")
        out = s.render(now=4.0)
        assert "first second" in out
        assert "4s" in out  # elapsed = now - started

    def test_stop_clears(self):
        s = ReasoningStatus()
        s.start(now=0.0)
        s.append("thinking")
        assert s.render(now=1.0) != ""
        s.stop()
        assert s.render(now=2.0) == ""

    def test_restart_resets_text_and_clock(self):
        s = ReasoningStatus()
        s.start(now=0.0)
        s.append("old")
        s.start(now=10.0)  # new turn
        s.append("new")
        out = s.render(now=12.0)
        assert "new" in out
        assert "old" not in out
        assert "2s" in out


# Duck-typed message stubs: render_conversation reads by class NAME + attributes
# (no langchain import), so a per-role dynamic type with the right attrs suffices.
def _msg(cls, content="", tool_calls=None, reasoning=None):
    m = type(cls, (), {})()
    m.content = content
    m.tool_calls = tool_calls or []
    m.additional_kwargs = {"reasoning_content": reasoning} if reasoning else {}
    return m


def _human(text):
    return _msg("HumanMessage", content=text)


def _ai(content="", tool_calls=None, reasoning=None):
    return _msg("AIMessage", content=content, tool_calls=tool_calls, reasoning=reasoning)


def _tool(content):
    return _msg("ToolMessage", content=content)


class TestRenderConversation:
    def test_user_prompt_and_answer(self):
        out = render_conversation([_human("hello there"), _ai("hi back")])
        assert "hello there" in out
        assert "hi back" in out
        # Answer carries the ● marker prefix, user prompt the > prefix.
        assert "●" in out and ">" in out

    def test_reasoning_block_rendered(self):
        out = render_conversation([_ai(content="done", reasoning="thinking hard")])
        assert "Thought" in out
        assert "thinking hard" in out

    def test_tool_calls_rendered(self):
        msg = _ai(tool_calls=[{"name": "web_search", "args": {"query": "x"}}])
        out = render_conversation([msg])
        assert "web_search" in out
        assert "query=x" in out

    def test_tool_result_messages_omitted(self):
        # A ToolMessage's content isn't replayed (its call block is enough).
        out = render_conversation([_tool("big tool output blob")])
        assert "big tool output blob" not in out

    def test_block_list_content_extracts_text(self):
        # Bedrock/Responses answer content is a block list.
        msg = _ai(content=[{"type": "text", "text": "block answer"}])
        out = render_conversation([msg])
        assert "block answer" in out

    def test_empty_messages_render_nothing(self):
        assert render_conversation([]) == ""
        assert render_conversation([_ai(content="")]) == ""
