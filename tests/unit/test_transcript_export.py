"""Unit tests for ``/export`` — the shareable one-way transcript renderer.

Distinct from ``/save`` (re-importable JSON): this renders a human-readable
artifact for a bug report or PR, so the tests care about what a READER sees —
no ANSI, no injected context, no thousand-line tool results.
"""

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)

from mnemoai.client import transcript_export as tx


def _convo():
    return [
        HumanMessage(content="How do I run the tests?"),
        AIMessage(
            content="Run `python -m pytest`.",
            additional_kwargs={"reasoning_content": "They want the test command."},
        ),
    ]


class TestItIsReadableAndAnsiFree:
    """An escape-laden file is useless in a bug report — which is exactly why
    this doesn't reuse ``turn_view.render_conversation``."""

    def test_no_ansi_escapes_survive(self):
        out = tx.render(_convo())
        assert "\033" not in out

    def test_both_sides_of_the_conversation_appear(self):
        out = tx.render(_convo())
        assert "How do I run the tests?" in out
        assert "python -m pytest" in out

    def test_markdown_uses_headings(self):
        out = tx.render(_convo(), "md")
        assert "### User" in out and "### Assistant" in out

    def test_plain_text_has_no_markdown_markup(self):
        out = tx.render(_convo(), "txt")
        assert "USER" in out and "###" not in out

    def test_an_unknown_format_falls_back_to_markdown(self):
        assert "### User" in tx.render(_convo(), "pdf")

    def test_a_leading_dot_in_the_format_is_tolerated(self):
        assert "USER" in tx.render(_convo(), ".txt")

    def test_the_header_carries_the_title_and_model(self):
        out = tx.render(_convo(), title="My chat", model="claude-opus-5")
        assert "My chat" in out and "claude-opus-5" in out

    def test_an_empty_conversation_renders_nothing(self):
        # The caller refuses to write a file rather than leaving an empty one.
        assert tx.render([]) == ""

    def test_a_conversation_of_only_injected_text_renders_nothing(self):
        msgs = [HumanMessage(content="<steering>be nice</steering>")]
        assert tx.render(msgs) == ""


class TestStreamedRepliesAreNotDropped:
    """A streamed answer lands in history as an ``AIMessageChunk`` (an
    ``AIMessage`` SUBCLASS) with block-list content. Matching on the exact class
    NAME silently dropped every assistant reply, exporting a file of nothing but
    user prompts — which is what a live run actually produced."""

    def test_a_streamed_chunk_answer_is_exported(self):
        msgs = [
            HumanMessage(content="Reply with exactly: RED"),
            AIMessageChunk(content=[{"text": "RED", "type": "text", "index": 0}]),
        ]
        out = tx.render(msgs)
        assert "RED" in out
        assert "### Assistant" in out

    def test_a_streamed_chunks_tool_calls_are_exported(self):
        msgs = [
            AIMessageChunk(
                content=[{"text": "", "type": "text"}],
                tool_calls=[{"name": "fs_read", "args": {"path": "a.py"}, "id": "1"}],
            )
        ]
        assert "fs_read" in tx.render(msgs)

    def test_the_transcript_is_not_all_user_prompts(self):
        # The exact shape of the bug: every turn present, no answers.
        msgs = []
        for word in ("RED", "GREEN"):
            msgs.append(HumanMessage(content=f"Reply with exactly: {word}"))
            msgs.append(AIMessageChunk(content=[{"text": word, "type": "text"}]))
        out = tx.render(msgs)
        assert out.count("### Assistant") == 2


class TestInjectedContextIsStripped:
    """None of this was typed by the user; unstripped it dominates the export."""

    def test_the_steering_block_is_removed(self):
        msgs = [HumanMessage(content="<steering>always use tabs</steering>\nhi there")]
        out = tx.render(msgs)
        assert "always use tabs" not in out
        assert "hi there" in out

    def test_the_episodic_memory_block_is_removed(self):
        msgs = [
            HumanMessage(
                content='[Episodic Memory - Similar Past Tasks]\n1. "x" → y\n\nreal question'
            )
        ]
        out = tx.render(msgs)
        assert "Episodic Memory" not in out
        assert "real question" in out

    def test_a_tool_result_message_is_not_rendered_as_a_turn(self):
        msgs = [
            HumanMessage(content="go"),
            AIMessage(content="", tool_calls=[
                {"name": "fs_read", "args": {"path": "a.py"}, "id": "1"}
            ]),
            ToolMessage(content="x" * 5000, tool_call_id="1", name="fs_read"),
            AIMessage(content="Done."),
        ]
        out = tx.render(msgs)
        assert "x" * 100 not in out  # the 5000-char result body is absent
        assert "fs_read" in out       # but the CALL is recorded
        assert "Done." in out


class TestToolCallsAreSummarizedNotDumped:
    def test_a_tool_call_shows_its_name_and_args(self):
        msgs = [AIMessage(content="", tool_calls=[
            {"name": "grep_search", "args": {"pattern": "def foo"}, "id": "1"}
        ])]
        out = tx.render(msgs)
        assert "grep_search" in out and "def foo" in out

    def test_a_bulky_arg_is_replaced_by_its_size(self):
        # A whole file body pasted into an argument defeats the purpose.
        msgs = [AIMessage(content="", tool_calls=[
            {"name": "fs_write", "args": {"path": "a.py", "content": "y" * 9000}, "id": "1"}
        ])]
        out = tx.render(msgs)
        assert "y" * 100 not in out
        assert "9000 chars" in out
        assert "a.py" in out

    def test_a_long_ordinary_arg_is_truncated(self):
        msgs = [AIMessage(content="", tool_calls=[
            {"name": "t", "args": {"q": "z" * 900}, "id": "1"}
        ])]
        out = tx.render(msgs)
        assert "z" * 400 not in out
        assert "…" in out

    def test_newlines_in_args_are_flattened_to_one_line(self):
        msgs = [AIMessage(content="", tool_calls=[
            {"name": "t", "args": {"cmd": "a\nb\nc"}, "id": "1"}
        ])]
        line = [ln for ln in tx.render(msgs).splitlines() if "cmd=" in ln][0]
        assert "a b c" in line

    def test_a_call_with_no_args_still_renders(self):
        msgs = [AIMessage(content="", tool_calls=[
            {"name": "git_status_safe", "args": {}, "id": "1"}
        ])]
        assert "git_status_safe()" in tx.render(msgs)


class TestReasoningIsOptOut:
    """Thinking blocks usually dwarf the conversation, so they're off by default."""

    def test_reasoning_is_omitted_by_default(self):
        assert "They want the test command." not in tx.render(_convo())

    def test_reasoning_is_included_on_request(self):
        out = tx.render(_convo(), include_reasoning=True)
        assert "They want the test command." in out

    def test_markdown_reasoning_is_collapsible(self):
        out = tx.render(_convo(), "md", include_reasoning=True)
        assert "<details>" in out

    def test_plain_text_reasoning_is_labelled(self):
        out = tx.render(_convo(), "txt", include_reasoning=True)
        assert "[reasoning]" in out


class TestSuggestedFilename:
    def test_it_slugs_the_first_prompt(self):
        name = tx.suggest_filename([HumanMessage(content="Fix the login bug!")])
        assert name.endswith(".md")
        assert "fix-the-login-bug" in name

    def test_the_extension_follows_the_format(self):
        assert tx.suggest_filename(_convo(), "txt").endswith(".txt")

    def test_it_falls_back_when_there_is_no_prompt(self):
        name = tx.suggest_filename([])
        assert name.startswith("conversation_") and name.endswith(".md")

    def test_injected_text_does_not_become_the_filename(self):
        name = tx.suggest_filename(
            [HumanMessage(content="<steering>x</steering>\nreal prompt")]
        )
        assert "steering" not in name and "real-prompt" in name

    def test_the_slug_is_bounded(self):
        name = tx.suggest_filename([HumanMessage(content="word " * 200)])
        assert len(name) < 80


class TestTheClientAndCommandWiring:
    """The pure renderer passing proves nothing about reachability — a live run
    exported a file of only user prompts while every unit test was green."""

    def _client(self, tmp_path, messages):
        from mnemoai.client.client import LangGraphClient

        c = LangGraphClient.__new__(LangGraphClient)
        c.agent = type("_A", (), {"messages": messages})()
        c.model_name_for_log = lambda: "test-model"
        return c

    def test_export_writes_a_file_and_returns_its_path(self, tmp_path):
        c = self._client(tmp_path, _convo())
        out = c.export_transcript(path=str(tmp_path / "t.md"))
        assert out and open(out).read().count("### ") >= 2

    def test_the_extension_selects_the_format(self, tmp_path):
        c = self._client(tmp_path, _convo())
        body = open(c.export_transcript(path=str(tmp_path / "t.txt"))).read()
        assert "USER" in body and "###" not in body

    def test_a_directory_target_gets_a_generated_name(self, tmp_path):
        c = self._client(tmp_path, _convo())
        out = c.export_transcript(path=str(tmp_path) + "/")
        assert out.endswith(".md") and "conversation_" in out

    def test_an_extensionless_path_gains_the_format_suffix(self, tmp_path):
        c = self._client(tmp_path, _convo())
        assert c.export_transcript(path=str(tmp_path / "notes")).endswith(".md")

    def test_an_empty_conversation_writes_nothing(self, tmp_path):
        c = self._client(tmp_path, [])
        assert c.export_transcript(path=str(tmp_path / "e.md")) is None
        assert not (tmp_path / "e.md").exists()

    def test_the_title_comes_from_the_live_messages(self, tmp_path):
        # conversation_title() reads a saved FILE and is a staticmethod requiring
        # a path — calling it here raised TypeError in a live run.
        c = self._client(tmp_path, _convo())
        assert "How do I run the tests?" in open(c.export_transcript(
            path=str(tmp_path / "t.md")
        )).read()

    def test_a_streamed_conversation_exports_its_answers(self, tmp_path):
        msgs = [
            HumanMessage(content="Reply with exactly: RED"),
            AIMessageChunk(content=[{"text": "RED", "type": "text"}]),
        ]
        body = open(self._client(tmp_path, msgs).export_transcript(
            path=str(tmp_path / "s.md")
        )).read()
        assert "RED" in body and "### Assistant" in body
