"""Unit tests for the styled turn renderer (client/ui/turn_view.py).

Pure string builders for the pinned-input UI: a collapsed reasoning block and a
tool-call block. Tests assert structure/content, not exact ANSI codes (kept
loose so restyling colors doesn't churn tests) — but do check that the reasoning
text and tool args survive verbatim and that empty inputs collapse to nothing.
"""

import re
import time

from mnemoai.client.ui.turn_view import (
    ReasoningStatus,
    StepStatus,
    format_duration,
    render_agent_detail,
    render_conversation,
    render_live_reasoning,
    render_plan,
    render_reasoning_block,
    render_session_notice,
    render_step_done,
    render_step_list,
    render_tool_call,
    render_turn_end,
    user_prompt_text,
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
        assert format_duration(173) == "2m53s"

    def test_hours(self):
        assert format_duration(3600) == "1h0m0s"
        assert format_duration(3725) == "1h2m5s"
        assert format_duration(7384) == "2h3m4s"


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

    def test_name_is_colored_like_the_file_op_headers(self):
        """The header must not be plain bold — bold white read as answer text.

        Asserted against the color the ``Update(...)``/``Create file`` blocks
        already use, so the two kinds of tool block can't drift apart again."""
        generic = render_tool_call("web_search", {"query": "x"})
        file_op = render_tool_call(
            "file_edit", {"file_path": "/tmp/a.py", "old_string": "a", "new_string": "b"}
        )
        accent = file_op.split("Update")[0]
        assert accent and accent in generic.split("web_search")[0]


class TestStepList:
    """Multi-step checklist: done / running (green) / not started."""

    _STEPS = ["Read the config", "Update the loader", "Run the tests"]

    def test_empty_plan_renders_nothing(self):
        assert render_step_list([]) == ""

    def test_lists_every_step_with_a_progress_count(self):
        out = render_step_list(self._STEPS, running={1}, done={0})
        for step in self._STEPS:
            assert step in out
        assert "1/3" in out  # one done of three

    def test_done_steps_are_checked_and_others_are_not(self):
        out = render_step_list(self._STEPS, running={1}, done={0})
        rows = {ln.split("]")[0]: ln for ln in out.split("\n")[1:]}
        assert sum("[✓]" in ln for ln in rows.values()) == 1
        assert sum("[ ]" in ln for ln in rows.values()) == 2

    def test_running_step_is_the_only_green_description(self):
        green = "\033[32m"
        out = render_step_list(self._STEPS, running={1}, done=set())
        running_row = next(ln for ln in out.split("\n") if self._STEPS[1] in ln)
        pending_row = next(ln for ln in out.split("\n") if self._STEPS[2] in ln)
        assert running_row.index(green) < running_row.index(self._STEPS[1])
        assert green not in pending_row

    def test_a_parallel_wave_marks_several_steps_running(self):
        out = render_step_list(self._STEPS, running={0, 1}, done=set())
        green_rows = [
            ln for ln in out.split("\n")[1:] if "\033[32m[ ]" in ln
        ]
        assert len(green_rows) == 2

    def test_long_descriptions_are_truncated(self):
        out = render_step_list(["x" * 300], running={0}, width=40)
        assert "…" in out
        assert max(len(ln) for ln in out.split("\n")) < 200

    def test_long_plan_is_windowed_and_says_what_it_elided(self):
        steps = [f"step {i}" for i in range(20)]
        out = render_step_list(steps, running={12}, done=set(range(12)))
        assert "step 12" in out              # the running step is always shown
        assert "earlier step" in out         # elided head is counted
        assert "more step" in out            # elided tail too
        assert len(out.split("\n")) <= 11    # header + 8 rows + 2 elision markers

    def test_out_of_range_indices_are_ignored(self):
        out = render_step_list(self._STEPS, running={99}, done={-1})
        assert "0/3" in out

    def test_a_multi_line_description_stays_on_one_row(self):
        out = render_step_list(["do a\nthen b"], running={0})
        assert len(out.split("\n")) == 2  # header + one row
        assert "do a then b" in out


class TestStepDone:
    """Per-completion tick line: a wave's block is printed before it starts."""

    def test_marks_the_step_and_carries_the_count(self):
        out = render_step_done("Update the loader", 2, 3)
        assert "[✓]" in out and "2/3" in out and "Update the loader" in out
        assert "\n" not in out

    def test_long_description_is_truncated(self):
        out = render_step_done("x" * 300, 1, 2, width=40)
        assert out.endswith("…\033[0m") and len(out) < 100

    def test_a_multi_line_description_stays_on_one_line(self):
        assert "\n" not in render_step_done("do a\nthen b", 1, 1)


class TestStepStatus:
    """Live checklist sink: the SAME rows tick, no line is appended per step."""

    _STEPS = ["Read the config", "Update the loader", "Run the tests"]

    def test_idle_renders_empty(self):
        assert StepStatus().render() == ""

    def test_an_empty_plan_stays_inactive(self):
        s = StepStatus()
        s.start([])
        assert s.active is False and s.render() == ""

    def test_a_finished_step_is_checked_in_the_existing_row(self):
        # The bug this replaces: rows stayed "[ ]" and each completion was
        # APPENDED as its own "[✓] 1/3 …" line below the block.
        s = StepStatus()
        s.start(self._STEPS)
        s.set_running([0, 1, 2])
        s.mark_done(1)
        out = s.render()
        rows = out.split("\n")[1:]
        assert len(rows) == 3  # still one row per step — nothing appended
        done_row = next(r for r in rows if self._STEPS[1] in r)
        assert "[✓]" in done_row
        assert "1/3" in out  # header counts up as rows tick

    def test_a_done_step_stops_being_reported_as_running(self):
        s = StepStatus()
        s.start(self._STEPS)
        s.set_running([0, 1])
        s.mark_done(0)
        s.set_running([0, 1])  # a later wave update must not un-tick it
        assert "[✓]" in next(
            r for r in s.render().split("\n") if self._STEPS[0] in r
        )

    def test_stop_hands_the_region_back(self):
        s = StepStatus()
        s.start(self._STEPS)
        s.set_running([0])
        assert s.render() != ""
        s.stop()
        assert s.render() == ""

    def test_restart_forgets_the_previous_plan(self):
        s = StepStatus()
        s.start(self._STEPS)
        s.mark_done(0)
        s.start(["Something else"])
        out = s.render()
        assert "Something else" in out
        assert self._STEPS[0] not in out
        assert "0/1" in out  # the old done set is gone too


class TestSessionNotice:
    def test_marks_a_resume_without_brackets(self):
        out = render_session_notice("resumed  20260827_170510_84272_173b8c")
        assert "20260827_170510_84272_173b8c" in out
        assert "⟲" in out
        plain = re.sub(r"\033\[[0-9;]*m", "", out)  # the notice itself, unstyled
        assert "[" not in plain and "]" not in plain

    def test_collapses_whitespace_and_tolerates_empty(self):
        assert "a b" in render_session_notice(" a \n b ")
        assert render_session_notice("").endswith("\033[0m")


class TestTurnEnd:
    """The one-line full stop under a finished turn (duration + clock time)."""

    # 2026-08-28 11:08:03 local — asserted as HH:MM, so the test is TZ-agnostic.
    _AT = time.mktime((2026, 8, 28, 11, 8, 3, 0, 0, -1))

    def test_carries_the_duration_and_the_clock_time(self):
        out = render_turn_end(442.0, self._AT)
        assert "7m22s" in out
        assert time.strftime("%H:%M", time.localtime(self._AT)) in out
        assert "\n" not in out

    def test_a_fast_turn_is_marked_too(self):
        # The terminator has to be unconditional to be worth reading.
        assert "0s" in render_turn_end(0.2, self._AT)

    def test_a_cancelled_turn_says_stopped_with_the_time_spent(self):
        out = render_turn_end(12.0, self._AT, stopped=True)
        assert "⊘" in out and "stopped" in out and "12s" in out
        assert "done" not in out


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


class TestLineLevelDiff:
    """A file-edit diff pairs old/new LINE BY LINE, so an edit buried in a large
    replacement shows only the lines that changed. The previous renderer printed
    every old line then every new line, leaving the reader to diff by eye."""

    @staticmethod
    def _plain(out):
        import re

        return re.sub(r"\x1b\[[0-9;]*m", "", out)

    def _diff(self, old, new):
        return self._plain(
            render_tool_call(
                "file_edit",
                {"file_path": "m.py", "old_string": old, "new_string": new},
            )
        )

    def test_unchanged_lines_appear_once_as_context(self):
        old = "def f():\n    return 1\n\nprint(f())"
        new = "def f():\n    return 2\n\nprint(f())"
        out = self._diff(old, new)
        assert "-     return 1" in out
        assert "+     return 2" in out
        # The surrounding lines are context: neither removed nor added.
        assert "- def f():" not in out and "+ def f():" not in out
        assert "- print(f())" not in out and "+ print(f())" not in out
        assert "def f():" in out and "print(f())" in out

    def test_only_the_changed_line_is_marked(self):
        old = "\n".join(f"line {i}" for i in range(6))
        new = old.replace("line 3", "line THREE")
        out = self._diff(old, new)
        assert "- line 3" in out and "+ line THREE" in out
        for i in (0, 1, 2, 4, 5):
            assert f"- line {i}" not in out and f"+ line {i}" not in out

    def test_a_long_unchanged_run_is_elided_with_a_count(self):
        # 20 identical middle lines between two edits: kept ends + a count.
        middle = "\n".join(f"pad {i}" for i in range(20))
        old = f"first old\n{middle}\nlast old"
        new = f"first new\n{middle}\nlast new"
        out = self._diff(old, new)
        assert "unchanged line" in out
        assert "pad 0" in out and "pad 19" in out   # kept ends
        assert "pad 10" not in out                  # elided middle

    def test_leading_and_trailing_context_is_trimmed_to_the_change(self):
        # 20 unchanged lines BEFORE the only change: only the ones next to it
        # are worth showing, so the far end is elided.
        pad = "\n".join(f"pad {i}" for i in range(20))
        out = self._diff(f"{pad}\nold tail", f"{pad}\nnew tail")
        assert "- old tail" in out and "+ new tail" in out
        assert "pad 19" in out      # adjacent to the change
        assert "pad 0" not in out   # far from it

    def test_whole_block_insert_and_delete_are_all_one_sign(self):
        inserted = self._diff("", "a\nb")
        assert "+ a" in inserted and "+ b" in inserted and "- " not in inserted
        deleted = self._diff("a\nb", "")
        assert "- a" in deleted and "- b" in deleted and "+ " not in deleted

    def test_identical_old_and_new_shows_the_text_not_an_elision(self):
        out = self._diff("same line", "same line")
        assert "same line" in out
        assert "unchanged line" not in out

    def test_a_pure_insertion_keeps_the_original_lines_as_context(self):
        old = "a\nc"
        new = "a\nb\nc"
        out = self._diff(old, new)
        assert "+ b" in out
        assert "- a" not in out and "- c" not in out


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


class TestReplayShowsOnlyWhatTheUserTyped:
    """A stored user message still carries the context the client PREPENDED. The
    replay used to print it raw, so every `--resume` opened with a ~30-line wall
    of episodic-memory tool names, and the first real prompt lost its `>` marker
    inside that block. Stripping is shared with the picker label and `/export` via
    ``turn_view.user_prompt_text`` so the three can't drift apart.
    """

    EPISODIC = (
        '[Episodic Memory - Similar Past Tasks]\n'
        '1. "please continue" \u2192 execute_bash, fs_read (similarity: 0.72)\n\n'
        "Remember the word RED."
    )

    def test_the_episodic_block_is_not_replayed(self):
        out = render_conversation([_human(self.EPISODIC), _ai("RED - got it.")])
        assert "Episodic Memory" not in out
        assert "similarity" not in out
        assert "execute_bash" not in out

    def test_the_real_prompt_survives_with_its_marker(self):
        out = render_conversation([_human(self.EPISODIC), _ai("RED - got it.")])
        assert "Remember the word RED." in out
        line = next(ln for ln in out.splitlines() if "Remember the word RED." in ln)
        assert ">" in line  # the prompt marker, previously swallowed by the block

    def test_the_steering_block_is_not_replayed(self):
        out = render_conversation(
            [_human("<steering>always use tabs</steering>\nfix the parser")]
        )
        assert "always use tabs" not in out
        assert "fix the parser" in out

    def test_the_plan_mode_reminder_is_not_replayed(self):
        out = render_conversation(
            [_human("<plan-mode-active>read only</plan-mode-active>\ndo research")]
        )
        assert "read only" not in out
        assert "do research" in out

    def test_an_auto_delivered_subagent_report_is_not_a_prompt(self):
        # It carries role=user but it's the agent talking to itself.
        out = render_conversation(
            [_human("Your background sub-agent finished: here is the report")]
        )
        assert "background sub-agent" not in out

    def test_an_injection_only_message_leaves_no_empty_marker(self):
        assert render_conversation([_human("<steering>x</steering>")]) == ""

    def test_an_ordinary_prompt_is_untouched(self):
        out = render_conversation([_human("what does this function do?")])
        assert "what does this function do?" in out


class TestUserPromptText:
    """The shared stripper — the single definition of "what the user typed"."""

    def test_a_plain_prompt_passes_through(self):
        assert user_prompt_text("hello") == "hello"

    def test_episodic_context_is_dropped(self):
        assert user_prompt_text('[Episodic Memory - x]\n1. "y"\n\nreal') == "real"

    def test_an_episodic_block_with_no_prompt_yields_nothing(self):
        assert user_prompt_text("[Episodic Memory - x] nothing follows") == ""

    def test_both_ephemeral_block_kinds_are_dropped(self):
        assert user_prompt_text("<steering>a</steering>b") == "b"
        assert user_prompt_text("<plan-mode-active>a</plan-mode-active>b") == "b"

    def test_a_background_report_yields_nothing(self):
        assert user_prompt_text("Your background sub-agent finished") == ""

    def test_empty_and_none_are_safe(self):
        assert user_prompt_text("") == ""
        assert user_prompt_text(None) == ""

    def test_the_word_episodic_inside_a_real_prompt_is_kept(self):
        # Only a LEADING injected block is stripped, not a mention of it.
        text = "why does [Episodic Memory ...] show up in my transcript?"
        assert user_prompt_text(text) == text


class TestRenderAgentDetail:
    """render_agent_detail replays a captured ActivityRun as a main-thread-style
    transcript (tool blocks + result/error lines + markdown final answer)."""

    def _run(self):
        from mnemoai.client.agent.agent_activity import AgentActivityStore

        store = AgentActivityStore()
        sink = store.open_run("explore", "Find Ray script", "spawn")
        sink.tool_call("grep_search", {"pattern": "fully_shard"})
        sink.tool_result("grep_search", "3 matches")
        sink.tool_error("glob_search", "error executing tool")
        sink.final("# Found it\n\nThe script is at **train.py**.")
        sink.finish("done")
        return store.get(store.snapshot()[0].run_id)

    def test_renders_header_tools_and_answer(self):
        out = render_agent_detail(self._run())
        assert "explore" in out and "spawn" in out
        assert "Find Ray script" in out          # description
        assert "grep_search" in out              # tool call block
        assert "pattern=fully_shard" in out      # tool arg
        assert "3 matches" in out                # result line
        assert "glob_search" in out and "error executing tool" in out  # error line
        assert "Found it" in out                 # final answer text
        assert "**train.py**" not in out         # markdown consumed, not raw
        assert "\033[" in out                    # ANSI styling applied

    def test_running_run_shows_running_marker(self):
        from mnemoai.client.agent.agent_activity import AgentActivityStore

        store = AgentActivityStore()
        store.open_run("plan", "still working", "orchestrator")
        out = render_agent_detail(store.snapshot()[0])
        assert "running" in out.lower()

    def test_failed_and_no_events_does_not_crash(self):
        from mnemoai.client.agent.agent_activity import AgentActivityStore

        store = AgentActivityStore()
        sink = store.open_run("explore", "d", "background")
        sink.finish("failed")
        out = render_agent_detail(store.snapshot()[0])
        assert "failed" in out.lower()
