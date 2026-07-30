"""Unit tests for the model-initiated ``ask_user_question`` tool.

Covers the pure validation/formatting in ``client/agent/ask_user.py`` and the
``ask`` driver's refusal paths — the important ones, since a question that reaches
a UI nobody can see would block the turn forever (a background sub-agent parked on
a picker that never paints has no way out).
"""

import pytest

from mnemoai.client.agent import ask_user
from mnemoai.client.agent.agent import LangGraphAgent


class _Stub:
    """A bare agent-like object (the ``__new__`` stub pattern the gate tests use)."""

    def __init__(self, ui=None, headless=False, depth=0):
        self._question_ui = ui
        self._headless = headless
        self._spawn_depth = depth
        self.spinner_calls = []

    def _is_headless(self):
        return self._headless

    def _spinner_snapshot(self):
        return (True, "Thinking")

    def _stop_spinner(self):
        self.spinner_calls.append("stop")

    def _start_spinner(self, label="Thinking"):
        self.spinner_calls.append(f"start:{label}")


class TestOptionsAreNormalized:
    def test_plain_list_is_kept_in_order(self):
        assert ask_user.normalize_options(["b", "a"]) == ["b", "a"]

    def test_a_bare_string_becomes_one_option(self):
        assert ask_user.normalize_options("only") == ["only"]

    def test_dict_entries_are_unwrapped_by_label(self):
        got = ask_user.normalize_options(
            [{"label": "Postgres"}, {"text": "SQLite"}, {"value": "MySQL"}]
        )
        assert got == ["Postgres", "SQLite", "MySQL"]

    def test_newlines_collapse_so_each_option_is_one_row(self):
        assert ask_user.normalize_options(["a\n  b\tc"]) == ["a b c"]

    def test_blank_and_none_entries_are_dropped(self):
        assert ask_user.normalize_options(["a", "", None, "   "]) == ["a"]

    def test_duplicates_are_dropped_case_insensitively(self):
        assert ask_user.normalize_options(["Yes", "yes", "No"]) == ["Yes", "No"]

    def test_option_count_is_capped(self):
        got = ask_user.normalize_options([f"o{i}" for i in range(50)])
        assert len(got) == ask_user.MAX_OPTIONS

    def test_a_long_label_is_truncated_to_the_total_width(self):
        got = ask_user.normalize_options(["x" * 500])
        assert len(got[0]) == ask_user.MAX_LABEL_CHARS
        assert got[0].endswith("…")

    def test_unusable_shapes_yield_nothing(self):
        assert ask_user.normalize_options(None) == []
        assert ask_user.normalize_options(42) == []


class TestAMalformedCallNeverReachesTheUser:
    """An empty or single-option picker is worse than no picker — the model
    should have just asked in its reply."""

    def test_no_question_is_refused(self):
        _, _, err = ask_user.validate("", ["a", "b"])
        assert err and "non-empty question" in err

    def test_fewer_than_two_options_is_refused(self):
        _, _, err = ask_user.validate("Which?", ["only"])
        assert err and "at least 2" in err

    def test_options_collapsing_to_one_is_refused(self):
        # Two entries that normalize to the same label are one real choice.
        _, opts, err = ask_user.validate("Which?", ["Yes", "yes"])
        assert opts == ["Yes"]
        assert err is not None

    def test_a_well_formed_call_passes(self):
        q, opts, err = ask_user.validate("  Which   db? ", ["a", "b"])
        assert (q, opts, err) == ("Which db?", ["a", "b"], None)

    def test_a_long_question_is_truncated(self):
        q, _, err = ask_user.validate("y" * 900, ["a", "b"])
        assert err is None
        assert len(q) == ask_user.MAX_QUESTION_CHARS

    def test_the_ui_is_not_invoked_for_a_malformed_call(self):
        called = []
        stub = _Stub(ui=lambda q, o: called.append(q) or "a")
        ask_user.ask(stub, "Which?", ["only"])
        assert called == []


class TestAQuestionNobodyCanSeeIsRefused:
    """Each of these would otherwise park a thread on an unanswerable prompt."""

    def test_a_subagent_cannot_ask(self):
        stub = _Stub(ui=lambda q, o: "a", depth=1)
        out = ask_user.ask(stub, "Which?", ["a", "b"])
        assert "cannot ask" in out and "sub-agent" in out

    def test_a_headless_agent_cannot_ask(self):
        stub = _Stub(ui=lambda q, o: "a", headless=True)
        out = ask_user.ask(stub, "Which?", ["a", "b"])
        assert "cannot ask" in out

    def test_a_subagent_is_refused_even_with_a_ui_hook_present(self):
        # The hook is inherited from the parent agent object, so presence alone
        # must not be read as "there is someone to ask".
        called = []
        stub = _Stub(ui=lambda q, o: called.append(q) or "a", depth=1)
        ask_user.ask(stub, "Which?", ["a", "b"])
        assert called == []

    def test_no_ui_hook_means_non_interactive(self):
        out = ask_user.ask(_Stub(ui=None), "Which?", ["a", "b"])
        assert "not interactive" in out

    def test_every_refusal_tells_the_model_to_decide_itself(self):
        for stub in (_Stub(ui=None), _Stub(ui=lambda q, o: "a", depth=1)):
            assert "best judgment" in ask_user.ask(stub, "Which?", ["a", "b"])


class TestTheAnswerReachesTheModel:
    def test_the_chosen_option_is_reported(self):
        stub = _Stub(ui=lambda q, o: "Postgres")
        out = ask_user.ask(stub, "Which db?", ["Postgres", "SQLite"])
        assert "Postgres" in out
        assert "re-ask" in out or "second-guess" in out

    def test_the_ui_receives_the_normalized_question_and_options(self):
        seen = {}

        def _ui(q, opts):
            seen["q"], seen["opts"] = q, opts
            return opts[0]

        ask_user.ask(_Stub(ui=_ui), "  Which\n db? ", ["a", "a", "b"])
        assert seen == {"q": "Which db?", "opts": ["a", "b"]}

    def test_a_dismissal_tells_the_model_to_proceed_not_re_ask(self):
        stub = _Stub(ui=lambda q, o: None)
        out = ask_user.ask(stub, "Which?", ["a", "b"])
        assert "Do NOT ask again" in out

    def test_a_failing_dialog_does_not_kill_the_turn(self):
        def _boom(q, o):
            raise RuntimeError("dialog exploded")

        out = ask_user.ask(_Stub(ui=_boom), "Which?", ["a", "b"])
        assert "Do NOT ask again" in out  # degrades to the dismissed path

    def test_a_non_string_choice_is_still_reported(self):
        out = ask_user.ask(_Stub(ui=lambda q, o: 3), "Which?", ["a", "b"])
        assert "3" in out


class TestTheSpinnerIsHandedBack:
    """Nothing else restarts the spinner on this client-side path, so a lost
    restore leaves the terminal at a dead `>` for the rest of the turn."""

    def test_the_spinner_is_stopped_then_restored(self):
        stub = _Stub(ui=lambda q, o: "a")
        ask_user.ask(stub, "Which?", ["a", "b"])
        assert stub.spinner_calls == ["stop", "start:Thinking"]

    def test_the_spinner_is_restored_even_when_the_dialog_raises(self):
        def _boom(q, o):
            raise RuntimeError("boom")

        stub = _Stub(ui=_boom)
        ask_user.ask(stub, "Which?", ["a", "b"])
        assert stub.spinner_calls == ["stop", "start:Thinking"]

    def test_an_idle_spinner_is_not_started(self):
        stub = _Stub(ui=lambda q, o: "a")
        stub._spinner_snapshot = lambda: (False, "Thinking")
        ask_user.ask(stub, "Which?", ["a", "b"])
        assert stub.spinner_calls == ["stop"]


class TestTheToolIsWiredIn:
    def test_it_is_available_on_every_route(self):
        # Routing must never hide it: any query can hit a genuine fork.
        assert "ask_user_question" in LangGraphAgent._ALWAYS_AVAILABLE_TOOLS

    def test_the_intercept_returns_a_tool_message(self):
        agent = LangGraphAgent.__new__(LangGraphAgent)
        agent._question_ui = lambda q, o: "SQLite"
        agent._headless_tl = None
        agent._spawn_depth_plain = 0
        agent._spawn_depth_tl = None
        msg = agent._client_side_tool_message(
            "ask_user_question",
            {"question": "Which db?", "options": ["Postgres", "SQLite"]},
            "call-1",
        )
        assert msg is not None
        assert msg.tool_call_id == "call-1"
        assert msg.name == "ask_user_question"
        assert "SQLite" in msg.content

    def test_it_is_stripped_from_a_subagents_toolset(self):
        from mnemoai.client.agent import subagent_runner

        class _Tool:
            def __init__(self, name):
                self.name = name

        class _Def:
            tools = None  # "all tools"
            disallowed_tools = None

        agent = LangGraphAgent.__new__(LangGraphAgent)
        agent.tools = [_Tool("fs_read"), _Tool("ask_user_question"), _Tool("spawn_agent")]
        names = [t.name for t in subagent_runner.subagent_tools(agent, _Def())]
        assert names == ["fs_read"]

    def test_its_marker_is_suppressed_so_the_picker_is_not_duplicated(self):
        agent = LangGraphAgent.__new__(LangGraphAgent)
        agent.styled_turn_view = True
        # No exception and no output path taken: the method returns before
        # touching the renderer (which would need more wiring on a bare stub).
        assert agent._print_tool_marker(
            {"name": "ask_user_question", "args": {"question": "q"}}
        ) is None


class TestTheServerStubIsSafeIfDrivenDirectly:
    def test_the_tool_is_registered_and_documents_its_args(self):
        pytest.importorskip("mcp.server.fastmcp")
        import asyncio

        from mcp.server.fastmcp import FastMCP

        from mnemoai.server.tools.ask_user_question import register_ask_user_tools

        mcp = FastMCP("test")
        register_ask_user_tools(mcp)
        tools = asyncio.run(mcp.list_tools())
        tool = next(t for t in tools if t.name == "ask_user_question")
        assert set(tool.inputSchema["properties"]) == {"question", "options"}

    def test_driven_without_the_client_it_tells_the_model_to_decide(self):
        # The stub body is what a directly-driven server returns; it must not
        # imply an answer is coming.
        pytest.importorskip("mcp.server.fastmcp")
        import asyncio

        from mcp.server.fastmcp import FastMCP

        from mnemoai.server.tools.ask_user_question import register_ask_user_tools

        mcp = FastMCP("test")
        register_ask_user_tools(mcp)
        out = asyncio.run(
            mcp.call_tool("ask_user_question", {"question": "q", "options": ["a", "b"]})
        )
        assert "best judgment" in str(out)
