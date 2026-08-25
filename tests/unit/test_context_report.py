"""Unit tests for the ``/context`` window breakdown (client/context_report.py).

Pure-logic: no model, no MCP, no terminal. The collector is exercised against a
stub client so the report's shape is pinned without a live session.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from mnemoai.client import context_injection, context_report
from mnemoai.client.context_report import Part


class TestSplitSystemPrompt:
    """Segmenting the live system prompt into the blocks it is built from."""

    def test_base_prompt_only(self):
        parts = context_report.split_system_prompt("You are an assistant.")
        assert parts == [("System prompt", "You are an assistant.")]

    def test_empty_prompt_has_no_parts(self):
        assert context_report.split_system_prompt("") == []
        assert context_report.split_system_prompt("   \n ") == []

    def test_every_block_is_labelled_in_order(self):
        prompt = "\n\n".join(
            [
                "BASE PROMPT",
                "<profile>\nStyle: concise\n</profile>",
                "[Persistent Memory]\nUser prefers uv.",
                "<available_skills>\nskills here\n</available_skills>",
                "<available_subagents>\nagents here\n</available_subagents>",
                "[Playbook - Learned Strategies]\nstrategy",
                "<conversation_summary>\nearlier turns\n</conversation_summary>",
            ]
        )
        labels = [label for label, _ in context_report.split_system_prompt(prompt)]
        assert labels == [
            "System prompt",
            "Learned profile",
            "Persistent memory (MEMORY.md)",
            "Skills listing",
            "Sub-agent types",
            "Learned strategies",
            "Compaction summary",
        ]

    def test_block_text_is_attributed_to_its_own_label(self):
        prompt = (
            "BASE\n\n[Persistent Memory]\nremember this\n\n"
            "<available_skills>\none skill\n</available_skills>"
        )
        blocks = dict(context_report.split_system_prompt(prompt))
        assert blocks["System prompt"] == "BASE"
        assert "remember this" in blocks["Persistent memory (MEMORY.md)"]
        assert "one skill" not in blocks["Persistent memory (MEMORY.md)"]
        assert "one skill" in blocks["Skills listing"]

    def test_absent_blocks_are_omitted(self):
        prompt = "BASE\n\n<available_subagents>\nagents\n</available_subagents>"
        labels = [label for label, _ in context_report.split_system_prompt(prompt)]
        assert labels == ["System prompt", "Sub-agent types"]

    def test_marker_inside_a_block_body_does_not_split(self):
        # A user's MEMORY.md may quote these words; only a marker that OPENS a
        # block (start of prompt or after a blank line) is a boundary.
        prompt = (
            "BASE\n\n[Persistent Memory]\nI told you about <available_skills> once\n"
            "and <profile> too."
        )
        parts = context_report.split_system_prompt(prompt)
        assert [label for label, _ in parts] == [
            "System prompt",
            "Persistent memory (MEMORY.md)",
        ]
        assert "<available_skills>" in parts[1][1]

    def test_real_session_prompt_splits_into_known_blocks(self):
        # Guards the markers against a rename in the injection code.
        client = SimpleNamespace(
            profile_manager=SimpleNamespace(get_profile_summary=lambda: ""),
            playbook=None,
        )
        prompt = context_injection.build_system_prompt(client)
        labels = [label for label, _ in context_report.split_system_prompt(prompt)]
        assert labels[0] == "System prompt"
        # Sub-agent types are always injected (the built-in types always exist).
        assert "Sub-agent types" in labels


class TestScale:
    """Estimated parts re-proportioned onto the provider's exact total."""

    def _parts(self):
        return [
            Part("a", 100, detail=(("a1", 60), ("a2", 40))),
            Part("b", 300),
        ]

    def test_parts_sum_to_the_target(self):
        scaled = context_report.scale(self._parts(), 200)
        assert sum(p.tokens for p in scaled) == 200

    def test_shares_are_preserved(self):
        scaled = context_report.scale(self._parts(), 800)
        assert [p.tokens for p in scaled] == [200, 600]

    def test_detail_rows_scale_too(self):
        scaled = context_report.scale(self._parts(), 800)
        assert scaled[0].detail == (("a1", 120), ("a2", 80))

    def test_no_target_leaves_parts_untouched(self):
        parts = self._parts()
        assert context_report.scale(parts, 0) == parts
        assert context_report.scale(parts, -5) == parts

    def test_empty_estimate_is_not_divided_by_zero(self):
        parts = [Part("a", 0), Part("b", 0)]
        assert context_report.scale(parts, 1000) == parts

    def test_labels_and_groups_survive(self):
        scaled = context_report.scale([Part("x", 10, group="tools")], 100)
        assert scaled[0].label == "x"
        assert scaled[0].group == "tools"


class TestRender:
    """The plain-text report."""

    def _parts(self):
        return [
            Part("System prompt", 4000, group="prompt"),
            Part("Steering: 1 file", 12000, detail=(("~/CLAUDE.md", 11900),), group="steering"),
            Part("Tool schemas: 30 tools", 4000, group="tools"),
        ]

    def test_header_shows_total_limit_and_percentage(self):
        text = context_report.render(self._parts(), 20000, limit=100000)
        assert "20,000 tokens of 100,000 (20%)" in text

    def test_bar_reflects_usage(self):
        text = context_report.render(self._parts(), 50000, limit=100000)
        bar = [line for line in text.split("\n") if "█" in line][0]
        assert bar.count("█") == bar.count("░")

    def test_a_tiny_context_still_shows_one_block(self):
        text = context_report.render([Part("System prompt", 10)], 10, limit=1000000)
        assert "█" in text

    def test_rows_carry_label_tokens_and_share(self):
        text = context_report.render(self._parts(), 20000, limit=100000)
        assert "Steering: 1 file" in text
        assert "12,000" in text
        assert "60%" in text  # 12000 of the 20000 in play

    def test_detail_rows_are_indented_under_their_part(self):
        lines = context_report.render(self._parts(), 20000).split("\n")
        parent = next(i for i, ln in enumerate(lines) if "Steering: 1 file" in ln)
        assert "~/CLAUDE.md" in lines[parent + 1]
        assert lines[parent + 1].startswith("    ")

    def test_groups_are_separated_by_blank_lines(self):
        body = context_report.render(self._parts(), 20000)
        # prompt / steering / tools → two blank lines between the three groups.
        rows = body.split("\n\n")
        assert len(rows) > 3

    def test_free_and_compaction_threshold(self):
        text = context_report.render(self._parts(), 20000, limit=100000, high_water=80000)
        assert "Free: 80,000 tokens" in text
        assert "compaction starts at 80,000" in text

    def test_no_limit_hides_the_free_line_and_bar(self):
        text = context_report.render(self._parts(), 20000)
        assert "Free:" not in text
        assert "█" not in text

    def test_measured_and_estimated_notes_differ(self):
        measured = context_report.render(self._parts(), 20000, measured=True)
        estimated = context_report.render(self._parts(), 20000, measured=False)
        assert "provider's exact count for the last turn" in measured
        assert "every number here is an estimate" in estimated

    def test_empty_report_explains_itself(self):
        text = context_report.render([], 0)
        assert "Nothing is loaded" in text

    def test_nothing_is_wider_than_a_standard_terminal(self):
        # Includes the footer notes, which are hand-wrapped.
        deep = "~/a/very/deeply/nested/project/tree/that/keeps/going/CLAUDE.md"
        parts = self._parts() + [Part("Steering: 1 file", 100, detail=((deep, 90),))]
        for measured in (True, False):
            lines = context_report.render(
                parts, 20100, limit=100000, high_water=80000, measured=measured
            ).split("\n")
            assert max(len(ln) for ln in lines) <= 80
        # A long path is clipped from the front: the filename still identifies it.
        assert "CLAUDE.md" in "\n".join(lines)

    def test_no_ansi_escapes(self):
        assert "\033" not in context_report.render(self._parts(), 20000, limit=100000)


class TestToolSchemaText:
    """What a bound tool contributes to every request."""

    def test_dict_schema_is_measured(self):
        tool = SimpleNamespace(
            name="grep_search",
            description="Search files",
            args_schema={"properties": {"pattern": {"type": "string"}}},
        )
        text = context_report.tool_schema_text(tool)
        assert "grep_search" in text and "Search files" in text and "pattern" in text

    def test_pydantic_schema_is_measured(self):
        pydantic = pytest.importorskip("pydantic")

        class Args(pydantic.BaseModel):
            pattern: str

        tool = SimpleNamespace(name="t", description="d", args_schema=Args)
        assert "pattern" in context_report.tool_schema_text(tool)

    def test_missing_schema_is_not_fatal(self):
        tool = SimpleNamespace(name="t", description="d", args_schema=None)
        assert context_report.tool_schema_text(tool) == "t\nd"

    def test_broken_schema_costs_only_itself(self):
        class Exploding:
            def model_json_schema(self):
                raise RuntimeError("nope")

            def schema(self):
                raise RuntimeError("also nope")

        tool = SimpleNamespace(name="t", description="d", args_schema=Exploding())
        assert context_report.tool_schema_text(tool).startswith("t\nd")


class TestCollect:
    """Gathering the parts from a (stubbed) live client."""

    @pytest.fixture(autouse=True)
    def _no_real_steering(self, monkeypatch):
        # collect() would otherwise read the developer's own STEERING.md/CLAUDE.md.
        monkeypatch.setattr(context_injection, "steering_reminder", lambda client: "")

    def _client(self, **kwargs):
        base = dict(
            system_prompt="BASE\n\n[Persistent Memory]\nnote",
            agent=SimpleNamespace(
                system_prompt="BASE\n\n[Persistent Memory]\nnote",
                messages=[],
                _last_input_tokens=None,
            ),
            tools=[],
            conversation_manager=SimpleNamespace(max_tokens=100000),
        )
        base.update(kwargs)
        return SimpleNamespace(**base)

    def test_system_prompt_blocks_become_parts(self):
        parts = context_report.collect(self._client())
        assert [p.label for p in parts] == [
            "System prompt",
            "Persistent memory (MEMORY.md)",
        ]
        assert all(p.tokens > 0 for p in parts)

    def test_the_agents_prompt_wins_over_the_clients(self):
        # After a compaction the agent's prompt carries the summary; the client's
        # copy can lag, and the agent's is what gets sent.
        client = self._client()
        client.agent.system_prompt = (
            "BASE\n\n<conversation_summary>\nearlier\n</conversation_summary>"
        )
        labels = [p.label for p in context_report.collect(client)]
        assert "Compaction summary" in labels
        assert "Persistent memory (MEMORY.md)" not in labels

    def test_tools_are_one_row_counting_every_schema(self):
        tools = [
            SimpleNamespace(name=f"t{i}", description="d" * 200, args_schema=None)
            for i in range(3)
        ]
        parts = context_report.collect(self._client(tools=tools))
        row = next(p for p in parts if p.label.startswith("Tool schemas"))
        assert row.label == "Tool schemas: 3 tools"
        assert row.tokens > 0

    def test_no_tools_no_row(self):
        parts = context_report.collect(self._client())
        assert not [p for p in parts if p.label.startswith("Tool schemas")]

    def test_history_is_split_by_who_produced_it(self):
        client = self._client()
        client.agent.messages = [
            HumanMessage(content="a question"),
            AIMessage(content="an answer"),
            ToolMessage(content="x" * 8000, tool_call_id="1"),
        ]
        row = next(
            p for p in context_report.collect(client) if p.label.startswith("Conversation")
        )
        assert row.label == "Conversation: 3 messages"
        detail = dict(row.detail)
        assert detail["tool results"] > detail["assistant replies"]
        assert set(detail) <= {"tool results", "assistant replies", "your prompts", "other"}

    def test_a_streamed_reply_counts_as_the_assistant(self):
        # AIMessageChunk is an AIMessage subclass; a class-name check would file
        # a live conversation's replies under "other".
        client = self._client()
        client.agent.messages = [AIMessageChunk(content="streamed " * 500)]
        row = next(
            p for p in context_report.collect(client) if p.label.startswith("Conversation")
        )
        assert dict(row.detail).get("assistant replies", 0) > 0

    def test_reasoning_and_tool_calls_are_counted(self):
        client = self._client()
        plain = AIMessage(content="hi")
        thinking = AIMessage(
            content="hi", additional_kwargs={"reasoning_content": "because " * 300}
        )
        client.agent.messages = [plain]
        small = next(
            p for p in context_report.collect(client) if p.label.startswith("Conversation")
        ).tokens
        client.agent.messages = [thinking]
        big = next(
            p for p in context_report.collect(client) if p.label.startswith("Conversation")
        ).tokens
        assert big > small

    def test_steering_row_lists_each_file(self, monkeypatch, tmp_path):
        global_file = tmp_path / "STEERING.md"
        project_file = tmp_path / "project" / "CLAUDE.md"
        project_file.parent.mkdir()
        global_file.write_text("global rules " * 50)
        project_file.write_text("project rules " * 200)
        monkeypatch.setattr(
            context_injection, "steering_reminder", lambda client: "<steering>\nx\n</steering>"
        )
        monkeypatch.setattr(
            context_report,
            "SteeringStore",
            lambda: SimpleNamespace(
                sizes=lambda: [
                    (global_file, global_file.read_text()),
                    (project_file, project_file.read_text()),
                ]
            ),
        )
        row = next(
            p for p in context_report.collect(self._client()) if p.label.startswith("Steering")
        )
        assert row.label == "Steering: 2 files"
        assert [label for label, _ in row.detail] == [str(global_file), str(project_file)]
        # The project file is the bigger of the two, and both are counted.
        assert row.detail[1][1] > row.detail[0][1]

    def test_no_steering_no_row(self):
        parts = context_report.collect(self._client())
        assert not [p for p in parts if p.label.startswith("Steering")]

    def test_a_failing_steering_read_does_not_break_the_report(self, monkeypatch):
        def boom(client):
            raise OSError("disk gone")

        monkeypatch.setattr(context_injection, "steering_reminder", boom)
        parts = context_report.collect(self._client())
        assert [p.label for p in parts][0] == "System prompt"


class TestReport:
    """End-to-end assembly on a stub client."""

    @pytest.fixture(autouse=True)
    def _no_real_steering(self, monkeypatch):
        monkeypatch.setattr(context_injection, "steering_reminder", lambda client: "")

    def _client(self, last_input=None, max_tokens=100000):
        prompt = "BASE PROMPT " * 200
        return SimpleNamespace(
            system_prompt=prompt,
            agent=SimpleNamespace(
                system_prompt=prompt,
                messages=[HumanMessage(content="hello")],
                _last_input_tokens=last_input,
            ),
            tools=[],
            conversation_manager=SimpleNamespace(max_tokens=max_tokens),
        )

    def test_uses_the_providers_exact_total_when_there_is_one(self):
        text = context_report.report(self._client(last_input=12345))
        assert "12,345 tokens of 100,000" in text
        assert "provider's exact count" in text

    def test_falls_back_to_the_estimate_before_the_first_turn(self):
        text = context_report.report(self._client())
        assert "every number here is an estimate" in text

    def test_compaction_threshold_defaults_to_80_percent(self):
        text = context_report.report(self._client(last_input=1000, max_tokens=200000))
        assert "compaction starts at 160,000" in text

    def test_no_agent_yet_reports_nothing_loaded(self):
        client = SimpleNamespace(
            system_prompt="", agent=None, tools=[], conversation_manager=None
        )
        assert "Nothing is loaded" in context_report.report(client)
