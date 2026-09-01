"""Unit tests for subtask parsing (client/orchestrator.py)."""

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.agent.orchestrator import parse_subtasks

VALID = {"simple_qa", "code", "research", "knowledge", "full"}
FALLBACK = "original query"


class TestParseSubtasks:
    def test_parses_clean_json_array(self):
        content = (
            '[{"description": "read the file", "category": "code"}, '
            '{"description": "summarize it", "category": "simple_qa"}]'
        )
        result = parse_subtasks(content, FALLBACK, VALID)
        assert len(result) == 2
        assert result[0] == {
            "description": "read the file", "category": "code", "depends_on": []
        }
        assert result[1]["category"] == "simple_qa"

    def test_strips_markdown_code_fences(self):
        content = '```json\n[{"description": "do x", "category": "code"}]\n```'
        result = parse_subtasks(content, FALLBACK, VALID)
        assert result == [{"description": "do x", "category": "code", "depends_on": []}]

    def test_strips_thinking_tags_before_json(self):
        content = (
            "<think>let me decompose this</think>"
            '[{"description": "task a", "category": "research"}]'
        )
        result = parse_subtasks(content, FALLBACK, VALID)
        assert result == [
            {"description": "task a", "category": "research", "depends_on": []}
        ]

    def test_invalid_category_normalized_to_full(self):
        content = '[{"description": "do y", "category": "nonsense"}]'
        result = parse_subtasks(content, FALLBACK, VALID)
        assert result[0]["category"] == "full"

    def test_missing_category_defaults_to_full(self):
        content = '[{"description": "no category here"}]'
        result = parse_subtasks(content, FALLBACK, VALID)
        assert result[0]["category"] == "full"

    def test_malformed_json_falls_back_to_single_subtask(self):
        result = parse_subtasks("this is not json at all", FALLBACK, VALID)
        assert result == [{"description": FALLBACK, "category": "full", "depends_on": []}]

    def test_empty_string_falls_back(self):
        result = parse_subtasks("", FALLBACK, VALID)
        assert result == [{"description": FALLBACK, "category": "full", "depends_on": []}]

    def test_non_list_json_falls_back(self):
        result = parse_subtasks('{"description": "x"}', FALLBACK, VALID)
        assert result == [{"description": FALLBACK, "category": "full", "depends_on": []}]

    def test_entries_without_description_are_skipped(self):
        content = '[{"category": "code"}, {"description": "keep me", "category": "code"}]'
        result = parse_subtasks(content, FALLBACK, VALID)
        assert result == [{"description": "keep me", "category": "code", "depends_on": []}]

    def test_bedrock_list_content_blocks(self):
        content = [
            {"type": "thinking", "thinking": "decomposing"},
            {"type": "text", "text": '[{"description": "t", "category": "full"}]'},
        ]
        result = parse_subtasks(content, FALLBACK, VALID)
        assert result == [{"description": "t", "category": "full", "depends_on": []}]


class TestParseSubtasksDependsOn:
    def test_valid_backward_dependency_kept(self):
        content = (
            '[{"description": "a", "category": "code"}, '
            '{"description": "b", "category": "code", "depends_on": [0]}]'
        )
        result = parse_subtasks(content, FALLBACK, VALID)
        assert result[0]["depends_on"] == []
        assert result[1]["depends_on"] == [0]

    def test_forward_and_self_references_dropped(self):
        # index 0 depends on 1 (forward) and 0 (self) — both invalid → dropped.
        content = (
            '[{"description": "a", "category": "code", "depends_on": [1, 0]}, '
            '{"description": "b", "category": "code"}]'
        )
        result = parse_subtasks(content, FALLBACK, VALID)
        assert result[0]["depends_on"] == []

    def test_out_of_range_and_non_int_dropped(self):
        content = (
            '[{"description": "a", "category": "code"}, '
            '{"description": "b", "category": "code", "depends_on": [0, 9, "x", true]}]'
        )
        result = parse_subtasks(content, FALLBACK, VALID)
        assert result[1]["depends_on"] == [0]  # 9 out of range, "x"/true rejected

    def test_dedup_and_sorted(self):
        content = (
            '[{"description": "a", "category": "code"}, '
            '{"description": "b", "category": "code"}, '
            '{"description": "c", "category": "code", "depends_on": [1, 0, 1]}]'
        )
        result = parse_subtasks(content, FALLBACK, VALID)
        assert result[2]["depends_on"] == [0, 1]

    def test_non_list_depends_on_ignored(self):
        content = '[{"description": "a", "category": "code", "depends_on": "0"}]'
        result = parse_subtasks(content, FALLBACK, VALID)
        assert result[0]["depends_on"] == []


class _DecomposerModel(BaseModel):
    """A copyable stand-in for the shared chat model the decomposer borrows."""

    reasoning: bool = True
    callbacks: object = None
    seen: list = []

    def invoke(self, messages, config=None):
        self.seen.append(self.reasoning)
        return AIMessage(content='[{"description": "a", "category": "code"}]')


class TestDecomposeDoesNotMutateTheSharedModel:
    """Decomposition borrows self.model, which a concurrent worker may be using.

    The old save/disable/restore ran on that shared object: the disable was
    visible mid-call elsewhere, and on the scalar-attribute providers interleaved
    restores could leave reasoning off for the whole session.
    """

    def test_decomposes_on_a_twin_and_leaves_the_parent_alone(self):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a.orchestrator_model = None
        sentinel = object()
        a.model = _DecomposerModel(reasoning=True, callbacks=sentinel, seen=[])
        a._disable_reasoning = lambda model=None: (_ for _ in ()).throw(
            AssertionError("must not mutate the shared model when a twin is available")
        )
        out = a._decompose_task("do a thing", "decompose", VALID)
        assert out == [{"description": "a", "category": "code", "depends_on": []}]
        assert a.model.seen == [False]  # the call ran with reasoning off...
        assert a.model.reasoning is True  # ...but on a twin
        assert a.model.callbacks is sentinel

    def test_falls_back_to_in_place_disable_when_untwinnable(self):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a.orchestrator_model = None

        class _M:
            callbacks = None

            def invoke(self, messages, config=None):
                return AIMessage(content='[{"description": "a", "category": "code"}]')

        a.model = _M()
        disabled = []
        a._disable_reasoning = lambda model=None: (
            disabled.append(True) or {"reasoning": True}
        )
        a._restore_reasoning = lambda saved, model=None: None
        assert a._decompose_task("do a thing", "decompose", VALID)
        assert disabled == [True]
