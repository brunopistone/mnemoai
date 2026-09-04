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


class TestTheNoteRidesAlongWithTheChoice:
    """The options were guessed by the model, so "this one, but…" has to be
    expressible — otherwise the user's only outlet is the closest wrong row."""

    def test_a_note_is_collapsed_to_one_line(self):
        assert ask_user.normalize_note(" only\n for  local\truns ") == (
            "only for local runs"
        )

    def test_no_note_is_empty_not_none(self):
        assert ask_user.normalize_note(None) == ""
        assert ask_user.normalize_note("   ") == ""

    def test_a_long_note_is_capped(self):
        got = ask_user.normalize_note("z" * (ask_user.MAX_NOTE_CHARS + 500))
        assert len(got) == ask_user.MAX_NOTE_CHARS
        assert got.endswith("…")

    def test_the_note_is_far_more_generous_than_an_option_label(self):
        # It's prose the model reads, not a row to render.
        assert ask_user.MAX_NOTE_CHARS > ask_user.MAX_LABEL_CHARS

    def test_the_choice_and_the_note_both_reach_the_model(self):
        out = ask_user.format_answer("SQLite", "only for local runs")
        assert "SQLite" in out and "only for local runs" in out

    def test_the_wording_is_unchanged_when_there_is_no_note(self):
        assert ask_user.format_answer("SQLite") == ask_user.format_answer("SQLite", "")

    def test_ask_reports_a_choice_with_its_note(self):
        stub = _Stub(ui=lambda q, o: ("Postgres", "as long as it's managed"))
        out = ask_user.ask(stub, "Which db?", ["Postgres", "SQLite"])
        assert "Postgres" in out and "as long as it's managed" in out
        assert "re-ask" in out or "second-guess" in out


class TestDecliningEveryOptionIsItsOwnAnswer:
    """Three outcomes, not two. Dismissing the picker means "decide for me";
    taking the escape row means the opposite — settle it in conversation. Before
    the row existed, a user who agreed with none of the options could only Esc,
    which told the model to press on with its own guess."""

    def test_the_escape_row_is_appended_to_every_question(self):
        rows = ask_user.picker_rows(["a", "b"])
        assert [r[0] for r in rows] == ["a", "b", ask_user.DISCUSS]

    def test_its_row_is_last_so_the_options_keep_their_order(self):
        rows = ask_user.picker_rows(["first", "second"])
        assert rows[-1] == (ask_user.DISCUSS, ask_user.DISCUSS_LABEL)

    def test_even_a_two_option_question_gets_it(self):
        assert len(ask_user.picker_rows(["a", "b"])) == 3

    def test_a_model_option_cannot_impersonate_it(self):
        # The row's identity must not depend on its wording.
        rows = ask_user.picker_rows(["a", ask_user.DISCUSS])
        assert [r[0] for r in rows].count(ask_user.DISCUSS) == 1

    def test_picking_it_yields_no_choice_but_keeps_the_note(self):
        assert ask_user.picker_reply(ask_user.DISCUSS, " neither fits ") == (
            None,
            "neither fits",
        )

    def test_picking_an_option_yields_the_pair(self):
        assert ask_user.picker_reply("a", "sure") == ("a", "sure")

    def test_cancelling_is_a_dismissal_even_with_a_note_typed(self):
        # A note accompanies an answer; it is not one on its own.
        assert ask_user.picker_reply(None, "typed then escaped") is None

    def test_the_model_is_told_to_reply_not_to_proceed(self):
        out = ask_user.format_discussion("why not both?")
        assert "why not both?" in out
        assert "don't proceed" in out.lower()
        assert "again" in out  # and not to re-open the picker

    def test_it_works_with_no_note_at_all(self):
        out = ask_user.format_discussion()
        assert "talk it through" in out and "don't proceed" in out.lower()

    def test_it_does_not_read_like_a_dismissal(self):
        # The dismissed wording hands the decision BACK to the model; this must
        # not, or the escape row would be indistinguishable from pressing Esc.
        assert "best judgment" not in ask_user.format_discussion()
        assert "best judgment" in ask_user.format_dismissed()

    def test_ask_routes_the_escape_row_to_the_discussion_wording(self):
        stub = _Stub(ui=lambda q, o: (None, "neither, they'd both leak"))
        out = ask_user.ask(stub, "Which db?", ["Postgres", "SQLite"])
        assert "neither, they'd both leak" in out
        assert "Do NOT ask again" not in out  # not the dismissed path

    def test_ask_still_distinguishes_a_real_dismissal(self):
        assert "Do NOT ask again" in ask_user.ask(
            _Stub(ui=lambda q, o: None), "Q", ["a", "b"]
        )


class TestTheUiReplyShapesAreTolerated:
    """``question_ui`` grew a note without breaking its old contract: a bare
    string still means "this option was chosen"."""

    def test_a_bare_string_is_still_a_choice(self):
        assert ask_user.normalize_reply("SQLite") == ("SQLite", "", True)

    def test_none_is_a_dismissal(self):
        assert ask_user.normalize_reply(None) == (None, "", False)

    def test_a_pair_carries_the_note(self):
        assert ask_user.normalize_reply(("a", " x  y ")) == ("a", "x y", True)

    def test_a_pair_with_no_choice_is_answered_but_unchosen(self):
        assert ask_user.normalize_reply((None, "nope")) == (None, "nope", True)

    def test_an_answered_reply_with_neither_is_still_not_a_dismissal(self):
        # The escape row taken with an empty note: nothing chosen, nothing said,
        # but the user DID answer — so the model must not decide for itself.
        assert ask_user.normalize_reply((None, "")) == (None, "", True)

    def test_a_lone_value_in_a_tuple_is_a_choice(self):
        assert ask_user.normalize_reply(("a",)) == ("a", "", True)

    def test_an_empty_string_is_read_as_a_dismissal(self):
        # Nothing chosen and nothing said, with no pair to prove it was submitted.
        assert ask_user.normalize_reply("") == (None, "", False)

    def test_an_empty_pair_is_read_as_the_escape_row(self):
        # A pair at all means the dialog was submitted, so it can't be a dismissal.
        assert ask_user.normalize_reply(()) == (None, "", True)

    def test_a_non_string_choice_survives(self):
        assert ask_user.normalize_reply((3, "")) == ("3", "", True)


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
