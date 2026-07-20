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
    render_plan,
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
        # A generic tool (not a file-op) flattens a multi-line arg to one line.
        out = render_tool_call("web_crawler", {"content": "line1\nline2"})
        assert "line1 line2" in out
        arg_lines = [ln for ln in out.split("\n") if "content=" in ln]
        assert len(arg_lines) == 1

    def test_missing_name_falls_back(self):
        out = render_tool_call("", {"a": 1})
        assert "tool" in out


class TestFileOpRendering:
    """file_edit / fs_write get a structured block (Update/Create header +
    a red/green diff or numbered content) instead of a flattened ↳ arg line."""

    def test_file_edit_shows_update_header_and_diff(self):
        out = render_tool_call("file_edit", {
            "file_path": "/tmp/x/requirements.txt",
            "old_string": "pkg>=1.0",
            "new_string": "pkg==1.2\nother==2.0",
        })
        assert "Update" in out and "requirements.txt" in out
        assert "- pkg>=1.0" in out           # removed line
        assert "+ pkg==1.2" in out           # added lines
        assert "+ other==2.0" in out
        # NOT flattened onto an ↳ old_string=… line
        assert "old_string=" not in out

    def test_file_edit_delete_and_insert_summaries(self):
        deleted = render_tool_call("file_edit", {
            "file_path": "a.txt", "old_string": "x", "new_string": ""})
        assert "deleted text" in deleted
        inserted = render_tool_call("file_edit", {
            "file_path": "a.txt", "old_string": "", "new_string": "x"})
        assert "inserted text" in inserted

    def test_fs_write_create_numbers_lines(self):
        out = render_tool_call("fs_write", {
            "path": "Desktop/note.md", "command": "create",
            "file_text": "# Title\n\nbody\n",
        })
        assert "Create file" in out and "Desktop/note.md" in out
        # numbered content lines (trailing newline trimmed -> 3 lines). The line
        # number and text are separated by an ANSI reset, so check them loosely.
        import re
        plain = re.sub(r"\x1b\[[0-9;]*m", "", out)
        assert "1 # Title" in plain
        assert "3 body" in plain
        assert "file_text=" not in out

    def test_fs_write_str_replace_is_a_diff(self):
        out = render_tool_call("fs_write", {
            "path": "c.yaml", "command": "str_replace",
            "old_str": "a: 1", "new_str": "a: 2",
        })
        assert "Update" in out and "c.yaml" in out
        assert "- a: 1" in out and "+ a: 2" in out

    def test_home_path_shortened(self):
        import os
        home = os.path.expanduser("~")
        out = render_tool_call("file_edit", {
            "file_path": f"{home}/proj/file.py", "old_string": "a", "new_string": "b"})
        assert "~/proj/file.py" in out
        assert home not in out.replace("~/proj/file.py", "")  # no full home path


class TestRenderPlan:
    def test_empty_plan_shows_header(self):
        out = render_plan("")
        assert "Plan" in out

    def test_line_structure_preserved_not_flattened(self):
        # The bug: a plan rendered as one flattened ↳ plan=… line. render_plan
        # must keep the plan's own line breaks (multiple output lines).
        plan = "## Heading\n\n- item one\n- item two\nfinal line"
        out = render_plan(plan, width=80)
        assert out.count("\n") >= 5
        # Heading marker stripped, list markers kept.
        assert "Heading" in out and "#" not in out
        assert "- item one" in out and "- item two" in out

    def test_long_line_wraps_to_width(self):
        long = "word " * 40  # ~200 chars on one logical line
        out = render_plan(long.strip(), width=60)
        # Wrapped into several bar-prefixed lines, none absurdly long.
        body_lines = [ln for ln in out.split("\n") if "word" in ln]
        assert len(body_lines) >= 3

    def test_inline_markdown_markers_stripped(self):
        out = render_plan("This is **bold** and `code` text", width=80)
        assert "bold" in out and "code" in out
        assert "**" not in out and "`" not in out

    def test_list_continuation_hangs_under_marker(self):
        plan = "- " + ("x " * 40).strip()
        out = render_plan(plan, width=50)
        lines = [ln for ln in out.split("\n") if "x x" in ln]
        assert len(lines) >= 2  # wrapped list body


class TestExitPlanModeInConversation:
    def test_exit_plan_mode_rendered_as_plan_not_flattened(self):
        msg = _ai(
            tool_calls=[
                {"name": "exit_plan_mode", "args": {"plan": "## P\n- step one\n- step two"}}
            ]
        )
        out = render_conversation([msg])
        # Rendered as a plan block (bulleted lines), not a flattened plan= arg.
        assert "- step one" in out and "- step two" in out
        assert "plan=" not in out


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

    def test_loaded_answer_markdown_is_rendered_not_raw(self):
        # Regression: a loaded answer must go through the live formatter, so
        # markdown syntax (bold, code fences) is rendered, not printed literally.
        md = "This is **important**\n\n```python\ndef f():\n    return 1\n```"
        out = render_conversation([_ai(md)])
        assert "**important**" not in out  # bold markers consumed
        assert "```" not in out            # fence consumed by the parser
        assert "important" in out          # the text survives
        assert "\033[" in out              # ANSI styling was applied

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
