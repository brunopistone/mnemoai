"""Unit tests for the TUI dialog helpers' non-TTY fallbacks.

select_from_list and confirm_inline present prompt_toolkit dialogs on a TTY but
degrade to plain input() when stdin isn't a TTY (pipes / CI / tests). These tests
drive the non-TTY branch with input() mocked — no modal, no terminal.
"""

import builtins
import sys

import pytest

from mnemoai.client.agent import ask_user
from mnemoai.client.ui import tui


@pytest.fixture
def not_a_tty(monkeypatch):
    # The helpers do `import sys` locally, so patch the real sys.std* streams.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)


def _drive_keys(reader, keys: str):
    """Run the reader's real merged key bindings against a pipe input on a dummy
    output, feed `keys`, then exit — no real terminal."""
    import asyncio

    from prompt_toolkit.application import Application
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.key_binding import merge_key_bindings
    from prompt_toolkit.key_binding.defaults import load_key_bindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import BufferControl
    from prompt_toolkit.output import DummyOutput

    inp = Window(BufferControl(buffer=reader._buffer))
    with create_pipe_input() as pipe:
        app = Application(
            layout=Layout(HSplit([inp]), focused_element=inp),
            key_bindings=merge_key_bindings(
                [load_key_bindings(), reader._make_bindings()]
            ),
            full_screen=False,
            input=pipe,
            output=DummyOutput(),
        )
        reader._app = app

        async def run():
            async def feed():
                await asyncio.sleep(0.1)
                pipe.send_text(keys)
                await asyncio.sleep(0.2)
                app.exit()

            asyncio.ensure_future(feed())
            await app.run_async()

        asyncio.run(run())


class TestSelectFromList:
    OPTIONS = [("/a/one.json", "one"), ("/a/two.json", "two")]

    def test_empty_returns_none(self):
        assert tui.select_from_list("Pick", []) is None

    def test_pick_by_number(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda *_: "2")
        assert tui.select_from_list("Pick", self.OPTIONS) == "/a/two.json"

    def test_blank_cancels(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda *_: "")
        assert tui.select_from_list("Pick", self.OPTIONS) is None

    def test_out_of_range_cancels(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda *_: "9")
        assert tui.select_from_list("Pick", self.OPTIONS) is None

    def test_non_number_cancels(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda *_: "abc")
        assert tui.select_from_list("Pick", self.OPTIONS) is None

    def test_eof_cancels(self, not_a_tty, monkeypatch):
        def _raise(*_):
            raise EOFError

        monkeypatch.setattr(builtins, "input", _raise)
        assert tui.select_from_list("Pick", self.OPTIONS) is None


class TestConfirmInline:
    @pytest.mark.parametrize("answer,expected", [
        ("y", True), ("yes", True), ("Y", True),
        ("n", False), ("no", False), ("", False), ("garbage", False),
    ])
    def test_answers(self, not_a_tty, monkeypatch, answer, expected):
        monkeypatch.setattr(builtins, "input", lambda *_: answer)
        assert tui.confirm_inline("Clear?") is expected

    def test_eof_is_no(self, not_a_tty, monkeypatch):
        def _raise(*_):
            raise EOFError

        monkeypatch.setattr(builtins, "input", _raise)
        assert tui.confirm_inline("Clear?") is False


class TestConfirmDialog:
    """confirm_dialog is a full-screen Yes/No on a TTY; off-TTY it degrades to a
    y/N input() prompt (used by the /load Delete flow)."""

    @pytest.mark.parametrize("answer,expected", [
        ("y", True), ("yes", True), ("Y", True),
        ("n", False), ("", False), ("nope", False),
    ])
    def test_non_tty_answers(self, not_a_tty, monkeypatch, answer, expected):
        monkeypatch.setattr(builtins, "input", lambda *_: answer)
        assert tui.confirm_dialog("Delete this conversation?") is expected

    def test_non_tty_eof_is_no(self, not_a_tty, monkeypatch):
        def _raise(*_):
            raise EOFError

        monkeypatch.setattr(builtins, "input", _raise)
        assert tui.confirm_dialog("Delete this?") is False


class TestArrowKeysCommitTheHighlightedRow:
    """A ``RadioList`` keeps the highlighted row (``_selected_index``) separate
    from its committed ``current_value``, and only its OWN enter/space binding
    reconciles them. ``_radio_pick`` overrides enter to confirm the dialog
    directly, so without ``select_on_focus=True`` arrows moved the highlight while
    ``current_value`` stayed on row 1 — ↓↓Enter returned the FIRST entry.

    That silently opened the wrong conversation in ``--resume`` / ``/load`` and
    made the Delete button delete the wrong one. Verified against a real
    ``RadioList`` (a stub wouldn't model the two-value split).
    """

    def _radio(self, select_on_focus=True):
        from prompt_toolkit.widgets import RadioList

        return RadioList(
            values=[("id1", "one"), ("id2", "two"), ("id3", "three")],
            select_on_focus=select_on_focus,
        )

    @staticmethod
    def _press_down(radio, times=1):
        """Invoke the widget's own ``down`` handler (not via key lookup, which also
        matches its catch-all type-to-search binding)."""
        from prompt_toolkit.keys import Keys

        handler = next(
            b.handler
            for b in radio.control.key_bindings.bindings
            if b.keys == (Keys.Down,)
        )
        for _ in range(times):
            handler(None)

    def test_moving_the_highlight_updates_the_value_read_on_enter(self):
        radio = self._radio()
        assert radio.current_value == "id1"
        self._press_down(radio)
        assert radio._selected_index == 1
        # The assertion that matters: _radio_pick exits with current_value.
        assert radio.current_value == "id2"

    def test_it_still_tracks_after_several_moves(self):
        radio = self._radio()
        self._press_down(radio, times=2)
        assert radio.current_value == "id3"

    def test_without_select_on_focus_the_value_would_lag(self):
        # Pins WHY the flag is required: this is the old, broken behavior.
        radio = self._radio(select_on_focus=False)
        self._press_down(radio)
        assert radio._selected_index == 1
        assert radio.current_value == "id1"  # highlight moved, value did not

    def test_the_picker_asks_for_select_on_focus(self, monkeypatch):
        # Guards the call site itself: the flag lives in _radio_pick, and a future
        # refactor dropping it would silently restore the wrong-row bug.
        seen = {}
        import prompt_toolkit.widgets as widgets

        real = widgets.RadioList

        def _spy(values, **kwargs):
            seen.update(kwargs)
            return real(values, **kwargs)

        monkeypatch.setattr(tui, "RadioList", _spy)
        monkeypatch.setattr(tui, "Application", lambda **k: type(
            "_A", (), {"run": lambda self: tui._CANCEL}
        )())
        tui._radio_pick("t", [("a", "a"), ("b", "b")])
        assert seen.get("select_on_focus") is True

    def test_every_single_pick_list_in_the_app_asks_for_it(self):
        # One key must not behave differently depending on which dialog is open.
        # The /model + /config pickers live in utils.configurator, not here, and
        # shipped without the flag: arrows moved the marker in --resume but not
        # there, where the pick had to be committed with Space first. Scanning the
        # SOURCE covers a picker added in a third module later on.
        import pathlib
        import re

        package = pathlib.Path(tui.__file__).parents[2]
        sites = [
            (path.name, args)
            for path in sorted(package.rglob("*.py"))
            for args in re.findall(r"RadioList\(([^)]*)\)", path.read_text())
        ]
        assert sites, "no RadioList construction found — did the pickers move?"
        for name, args in sites:
            assert "select_on_focus=True" in args, (
                f"{name}: RadioList({args}) — arrows would move the highlight "
                f"while the committed value stayed behind"
            )


class TestSelectAllowDelete:
    """allow_delete adds a Delete button on a TTY (returns (_DELETE, value)); the
    non-TTY fallback just picks and ignores the delete affordance."""

    OPTIONS = [("/a/one.json", "one"), ("/a/two.json", "two")]

    def test_non_tty_allow_delete_still_picks(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda *_: "1")
        assert tui.select_from_list("Pick", self.OPTIONS, allow_delete=True) == (
            "/a/one.json"
        )


class TestQuestionUi:
    """``question_ui`` backs the ``ask_user_question`` tool: a picker raised from
    the WORKER thread mid-turn, so it goes through ``run_dialog`` (a full-screen
    dialog can't nest inside the running pinned app)."""

    def _reader(self, dialog=None):
        r = tui.PinnedPromptReader.__new__(tui.PinnedPromptReader)
        r._app = object()
        r._loop = object()
        if dialog is not None:
            r.run_dialog = dialog
        return r

    @staticmethod
    def _answers(*lines):
        """An ``input()`` stub that returns each line in turn (then EOF)."""
        it = iter(lines)

        def _input(*_):
            try:
                return next(it)
            except StopIteration:
                raise EOFError
        return _input

    def test_it_goes_through_run_dialog(self, not_a_tty, monkeypatch):
        # Calling the dialog directly would try to nest a full-screen app inside
        # the running one.
        seen = {}
        monkeypatch.setattr(builtins, "input", self._answers("2", ""))

        def _run_dialog(func):
            seen["called"] = True
            return func()

        got = self._reader(_run_dialog).question_ui("Which?", ["a", "b"])
        assert seen.get("called") is True
        assert got == ("b", "")

    def test_no_running_app_yields_none(self):
        # Off-TTY / plain loop: there is no app to suspend, so there is no one to
        # ask — None means "dismissed", which the tool turns into decide-yourself.
        r = tui.PinnedPromptReader.__new__(tui.PinnedPromptReader)
        r._app = None
        r._loop = None
        assert r.question_ui("Which?", ["a", "b"]) is None

    def test_a_dismissed_picker_returns_none(self):
        assert self._reader(lambda func: None).question_ui("Q", ["a", "b"]) is None

    def test_the_answer_is_echoed_to_scrollback(self, capsys, monkeypatch):
        # The picker is full-screen, so without this echo the turn would leave no
        # trace of what was asked or chosen.
        monkeypatch.setattr(tui, "question_dialog", lambda *a, **k: ("SQLite", ""))
        self._reader(lambda func: func()).question_ui("Which db?", ["x", "y"])
        out = capsys.readouterr().out
        assert "Which db?" in out and "SQLite" in out

    def test_a_dismissal_is_echoed_too(self, capsys, monkeypatch):
        monkeypatch.setattr(tui, "question_dialog", lambda *a, **k: None)
        self._reader(lambda func: func()).question_ui("Which db?", ["x", "y"])
        assert "dismissed" in capsys.readouterr().out

    def test_the_note_is_echoed_beside_the_choice(self, capsys, monkeypatch):
        monkeypatch.setattr(
            tui, "question_dialog", lambda *a, **k: ("SQLite", "only for local runs")
        )
        self._reader(lambda func: func()).question_ui("Which db?", ["x", "y"])
        out = capsys.readouterr().out
        assert "SQLite" in out and "only for local runs" in out

    def test_declining_every_option_is_echoed_as_such_not_as_a_dismissal(
        self, capsys, monkeypatch
    ):
        # The two are different answers, so they must not read identically:
        # dismissed = "decide for me", none-of-these = "talk to me".
        monkeypatch.setattr(tui, "question_dialog", lambda *a, **k: (None, "why both?"))
        self._reader(lambda func: func()).question_ui("Which db?", ["x", "y"])
        out = capsys.readouterr().out
        assert "dismissed" not in out
        assert "none of these" in out and "why both?" in out

    def test_the_question_and_options_reach_the_dialog(self, monkeypatch):
        seen = {}

        def _dialog(question, options, multiple):
            seen["question"], seen["options"] = question, options
            seen["multiple"] = multiple
            return None

        monkeypatch.setattr(tui, "question_dialog", _dialog)
        self._reader(lambda func: func()).question_ui("Q", ["a", "b"])
        assert seen == {"question": "Q", "options": ["a", "b"], "multiple": False}

    def test_the_pick_any_flag_reaches_the_dialog(self, monkeypatch):
        seen = {}

        def _dialog(question, options, multiple):
            seen["multiple"] = multiple
            return None

        monkeypatch.setattr(tui, "question_dialog", _dialog)
        self._reader(lambda func: func()).question_ui("Q", ["a", "b"], True)
        assert seen["multiple"] is True

    def test_every_pick_is_echoed_on_its_own_line(self, capsys, monkeypatch):
        # A list literal in the scrollback record would be the one thing the user
        # is most likely to read back.
        monkeypatch.setattr(
            tui, "question_dialog", lambda *a, **k: (["SQLite", "Postgres"], "")
        )
        self._reader(lambda func: func()).question_ui("Which?", ["x", "y"], True)
        out = capsys.readouterr().out
        assert "['" not in out and "', '" not in out
        assert "SQLite" in out and "Postgres" in out

    def test_an_empty_multi_selection_is_echoed_as_declining_them_all(
        self, capsys, monkeypatch
    ):
        monkeypatch.setattr(tui, "question_dialog", lambda *a, **k: ([], "hmm"))
        self._reader(lambda func: func()).question_ui("Which?", ["x", "y"], True)
        out = capsys.readouterr().out
        assert "none of these" in out and "dismissed" not in out


class TestQuestionDialog:
    """The ``ask_user_question`` picker is not just a list: the options were
    GUESSED by the model, so it also offers a free-text note and a row for
    declining all of them. Driven here through the non-TTY fallback, which must
    offer the same three affordances as the full-screen dialog."""

    def _input(self, *lines):
        it = iter(lines)

        def _stub(*_):
            try:
                return next(it)
            except StopIteration:
                raise EOFError
        return _stub

    def test_an_option_is_returned_with_an_empty_note(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", self._input("1", ""))
        assert tui.question_dialog("Q", ["a", "b"]) == ("a", "")

    def test_a_note_rides_along_with_the_choice(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", self._input("2", "  but only in CI "))
        assert tui.question_dialog("Q", ["a", "b"]) == ("b", "but only in CI")

    def test_the_escape_row_is_always_offered_last(self, not_a_tty, monkeypatch, capsys):
        monkeypatch.setattr(builtins, "input", self._input("3", "neither fits"))
        # Row 3 of a 2-option question is the escape row: no choice, just a note.
        assert tui.question_dialog("Q", ["a", "b"]) == (None, "neither fits")
        assert ask_user.DISCUSS_LABEL in capsys.readouterr().out

    def test_the_escape_row_needs_no_note(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", self._input("3", ""))
        assert tui.question_dialog("Q", ["a", "b"]) == (None, "")

    def test_blank_dismisses(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", self._input(""))
        assert tui.question_dialog("Q", ["a", "b"]) is None

    def test_out_of_range_dismisses(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", self._input("9"))
        assert tui.question_dialog("Q", ["a", "b"]) is None

    def test_eof_on_the_note_is_no_note_not_a_dismissal(self, not_a_tty, monkeypatch):
        # A pipe with one line of input must still deliver the choice.
        monkeypatch.setattr(builtins, "input", self._input("1"))
        assert tui.question_dialog("Q", ["a", "b"]) == ("a", "")

    def test_eof_on_the_pick_dismisses(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", self._input())
        assert tui.question_dialog("Q", ["a", "b"]) is None

    def test_it_does_not_reuse_the_plain_picker(self, not_a_tty, monkeypatch):
        # select_from_list backs /load, --resume and the configurator, where the
        # options ARE the whole answer — growing a note there would be wrong.
        monkeypatch.setattr(
            tui, "select_from_list", lambda *a, **k: pytest.fail("wrong picker")
        )
        monkeypatch.setattr(builtins, "input", self._input("1", ""))
        tui.question_dialog("Q", ["a", "b"])


class TestQuestionDialogPickAny:
    """``multiple`` off-TTY: a comma-separated list, since a pipe has no
    checkboxes. Same three outcomes as the pick-one fallback beside it."""

    def _input(self, *lines):
        it = iter(lines)

        def _stub(*_):
            try:
                return next(it)
            except StopIteration:
                raise EOFError
        return _stub

    def test_several_indices_are_all_returned(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", self._input("1,3", ""))
        assert tui.question_dialog("Q", ["a", "b", "c"], True) == (["a", "c"], "")

    def test_spaces_and_repeats_are_tolerated(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", self._input(" 2 , 1 , 2 ", ""))
        assert tui.question_dialog("Q", ["a", "b"], True) == (["b", "a"], "")

    def test_one_index_still_works(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", self._input("2", "later"))
        assert tui.question_dialog("Q", ["a", "b"], True) == (["b"], "later")

    def test_junk_and_out_of_range_entries_are_dropped_not_fatal(
        self, not_a_tty, monkeypatch
    ):
        monkeypatch.setattr(builtins, "input", self._input("1,zz,99", ""))
        assert tui.question_dialog("Q", ["a", "b"], True) == (["a"], "")

    def test_nothing_parseable_is_a_dismissal(self, not_a_tty, monkeypatch):
        monkeypatch.setattr(builtins, "input", self._input("zz,99"))
        assert tui.question_dialog("Q", ["a", "b"], True) is None

    def test_the_escape_row_wins_over_an_option_beside_it(self, not_a_tty, monkeypatch):
        # Row 3 of a 2-option question is the escape row.
        monkeypatch.setattr(builtins, "input", self._input("1,3", "neither, really"))
        assert tui.question_dialog("Q", ["a", "b"], True) == (None, "neither, really")

    def test_the_prompt_says_more_than_one_is_accepted(
        self, not_a_tty, monkeypatch, capsys
    ):
        seen = []
        monkeypatch.setattr(builtins, "input", lambda p="": seen.append(p) or "")
        tui.question_dialog("Q", ["a", "b"], True)
        assert "1,2" in seen[0]


class TestQuestionPickKeys:
    """Drives the REAL full-screen question dialog with REAL key presses over a
    pipe input — the non-TTY fallback above can't show that Tab reaches the note,
    that Enter submits from either side, or that the highlighted row is the one
    committed (the ``select_on_focus`` trap, in the one dialog that also has a
    second focusable widget to Tab into)."""

    OPTIONS = ["Postgres", "SQLite"]

    def _drive(self, keys: str, options=None):
        import asyncio
        import unittest.mock as m

        from prompt_toolkit.application import Application
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        holder = {}
        real_init = Application.__init__

        def cap_init(self, *a, **k):
            k = dict(k)
            k["input"] = holder["pipe"]
            k["output"] = DummyOutput()
            real_init(self, *a, **k)

        def fake_run(self):
            # wait_for so a dialog that never exits FAILS instead of hanging.
            return asyncio.run(asyncio.wait_for(self.run_async(), timeout=10))

        with create_pipe_input() as pipe:
            holder["pipe"] = pipe
            # Fed BEFORE the app starts: the bytes sit in the pipe, so nothing can
            # be dropped by racing the first render.
            pipe.send_text(keys)
            with m.patch.object(tui, "_dialog_is_tty", lambda: True), m.patch.object(
                Application, "__init__", cap_init
            ), m.patch.object(Application, "run", fake_run):
                return tui.question_dialog("Which db?", options or self.OPTIONS)

    def test_enter_confirms_the_highlighted_row(self):
        assert self._drive("\r") == ("Postgres", "")

    def test_arrows_move_the_committed_row_not_just_the_highlight(self):
        assert self._drive("\x1b[B\r") == ("SQLite", "")

    def test_tab_reaches_the_note_and_enter_submits_from_there(self):
        # If focus were still on the rows, these letters would drive the widget's
        # type-ahead search instead of landing in the note.
        assert self._drive("\tonly for local runs\r") == (
            "Postgres",
            "only for local runs",
        )

    def test_the_escape_row_is_reachable_by_arrows(self):
        assert self._drive("\x1b[B\x1b[B\r") == (None, "")

    def test_a_note_on_the_escape_row_is_kept(self):
        assert self._drive("\x1b[B\x1b[B\tneither, both leak\r") == (
            None,
            "neither, both leak",
        )

    def test_escape_dismisses(self):
        assert self._drive("\x1b") is None


class TestMultiPickKeys:
    """The pick-any dialog driven with REAL keys. Its widget keeps the answer in
    ``current_values``, so the highlight/commit reconciliation the radio list
    needs is not just unnecessary here but actively wrong: ``select_on_focus``
    would tick every row the highlight passes over."""

    OPTIONS = ["Postgres", "SQLite", "DuckDB"]
    DOWN = "\x1b[B"

    def _drive(self, keys: str, options=None):
        import asyncio
        import unittest.mock as m

        from prompt_toolkit.application import Application
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        holder = {}
        real_init = Application.__init__

        def cap_init(self, *a, **k):
            k = dict(k)
            k["input"] = holder["pipe"]
            k["output"] = DummyOutput()
            real_init(self, *a, **k)

        def fake_run(self):
            return asyncio.run(asyncio.wait_for(self.run_async(), timeout=10))

        with create_pipe_input() as pipe:
            holder["pipe"] = pipe
            pipe.send_text(keys)
            with m.patch.object(tui, "_dialog_is_tty", lambda: True), m.patch.object(
                Application, "__init__", cap_init
            ), m.patch.object(Application, "run", fake_run):
                return tui.question_dialog(
                    "Which?", options or self.OPTIONS, True
                )

    def test_space_ticks_the_highlighted_row(self):
        assert self._drive(" \r") == (["Postgres"], "")

    def test_several_rows_can_be_ticked(self):
        assert self._drive(f" {self.DOWN}{self.DOWN} \r") == (
            ["Postgres", "DuckDB"],
            "",
        )

    def test_moving_over_a_row_does_not_tick_it(self):
        # select_on_focus on a checkbox list would tick everything the highlight
        # crossed, so the answer would be every row above the last one.
        assert self._drive(f"{self.DOWN}{self.DOWN} \r") == (["DuckDB"], "")

    def test_space_again_unticks(self):
        # Tick the first row, change your mind, tick the second.
        assert self._drive(f"  {self.DOWN} \r") == (["SQLite"], "")

    def test_enter_with_nothing_ticked_hands_the_decision_back(self):
        assert self._drive("\r") == (None, "")

    def test_tab_reaches_the_note_and_enter_submits_from_there(self):
        assert self._drive(" \tall of them\r") == (["Postgres"], "all of them")

    def test_the_escape_row_is_a_row_here_too(self):
        assert self._drive(f"{self.DOWN}{self.DOWN}{self.DOWN} \r") == (None, "")

    def test_escape_dismisses(self):
        assert self._drive("\x1b") is None

    def test_the_hint_names_the_key_that_ticks(self):
        # Nothing else on screen says a row has to be ticked before Enter.
        import inspect

        assert "Space to tick" in inspect.getsource(tui._multi_pick)


class TestAgentsPanelVisibility:
    """The bottom agents panel shows while ANY sub-agent is running (finished
    ones stay listed), and hides once ALL have finished.
    Nav-mode forces it visible so an in-progress navigation isn't yanked away."""

    def _reader(self, store):
        return tui.PinnedPromptReader(
            prompt_text=lambda: ">",
            commands=[],
            dispatch=lambda q: None,
            agents_provider=store.snapshot,
            agents_get=store.get,
        )

    def _store(self):
        from mnemoai.client.agent.agent_activity import AgentActivityStore

        return AgentActivityStore()

    def test_hidden_when_no_agents(self):
        r = self._reader(self._store())
        assert r._panel_showable() is False

    def test_shown_while_running_hidden_when_all_done(self):
        store = self._store()
        a = store.open_run("explore", "a", "spawn")
        b = store.open_run("explore", "b", "spawn")
        r = self._reader(store)
        assert r._panel_showable() is True  # both running
        a.finish("done")
        assert r._panel_showable() is True  # one still running → still shown
        # the finished one is still listed (so "2 done / 1 going" reads right)
        assert any("a" in txt for _cls, txt in r._agents_text())
        b.finish("done")
        assert r._panel_showable() is False  # all done → hidden

    def test_lone_agent_hides_when_done(self):
        store = self._store()
        x = store.open_run("explore", "solo", "spawn")
        r = self._reader(store)
        assert r._panel_showable() is True
        x.finish("done")
        assert r._panel_showable() is False

    def test_nav_mode_forces_visible(self):
        store = self._store()
        store.open_run("explore", "d", "spawn").finish("done")  # all done
        r = self._reader(store)
        assert r._panel_showable() is False
        r._nav_mode = True
        assert r._panel_showable() is True


class TestAgentsPanelScrolling:
    """The panel is a VIEWPORT over every retained run, not the last 6.

    It used to slice ``rows[-6:]`` when rendering, so in a run with 11 sub-agents
    the older 5 were not merely off-screen: they could not be selected, viewed or
    stopped, and a still-running early agent was invisible with no way to reach it.
    """

    def _reader(self, store):
        return tui.PinnedPromptReader(
            prompt_text=lambda: ">",
            commands=[],
            dispatch=lambda q: None,
            agents_provider=store.snapshot,
            agents_get=store.get,
            agents_stop=store.request_stop,
            agents_stop_all=store.request_stop_all,
        )

    def _store(self, n=11, running=()):
        """A store with `n` runs named a0..a{n-1}; only indexes in `running` are
        left running. Returns (store, sinks)."""
        from mnemoai.client.agent.agent_activity import AgentActivityStore

        store = AgentActivityStore()
        sinks = [store.open_run("explore", f"a{i}", "spawn") for i in range(n)]
        for i, s in enumerate(sinks):
            if i not in running:
                s.finish("done")
        return store, sinks

    def _rows_text(self, reader):
        return "".join(txt for _cls, txt in reader._agents_text())

    def test_every_run_stays_in_the_list(self):
        store, _ = self._store(11, running={0})
        r = self._reader(store)
        assert len(r._agent_rows()) == 11

    def test_the_viewport_draws_only_max_rows(self):
        store, _ = self._store(11, running={0})
        r = self._reader(store)
        # hint line + at most _PANEL_MAX_ROWS agent rows.
        assert len(r._agents_text()) == r._PANEL_MAX_ROWS + 1

    def test_idle_panel_reports_the_hidden_running_agent(self):
        store, _ = self._store(11, running={0})  # the OLDEST is still going
        r = self._reader(store)
        hint = r._agents_text()[0][1]
        assert "+5 more" in hint and "(1 running)" in hint

    def test_no_running_suffix_when_the_hidden_ones_are_done(self):
        store, _ = self._store(11, running={10})
        r = self._reader(store)
        hint = r._agents_text()[0][1]
        assert "+5 more" in hint and "running" not in hint

    def test_hint_is_bare_when_everything_fits(self):
        store, _ = self._store(3, running={0})
        r = self._reader(store)
        assert r._agents_text()[0][1].strip() == "Ctrl+A: agents"

    def test_nav_scrolls_the_viewport_up_to_the_cursor(self):
        store, _ = self._store(11, running={0})
        r = self._reader(store)
        r._nav_mode = True
        r._nav_index = 0
        body = self._rows_text(r)
        assert " a0 " in body.replace("›", " ")  # oldest now on screen…
        assert "a10" not in body  # …and the newest scrolled off
        assert "1/11" in r._agents_text()[0][1]

    def test_the_cursor_marks_the_selected_row_after_scrolling(self):
        store, _ = self._store(11, running={0})
        r = self._reader(store)
        r._nav_mode = True
        r._nav_index = 7
        selected = [
            txt for cls, txt in r._agents_text() if cls == "class:pinned-panel-sel"
        ]
        assert len(selected) == 1 and " a7 " in selected[0]

    def test_x_stops_an_agent_the_old_panel_could_not_reach(self):
        store, sinks = self._store(11, running={0})
        r = self._reader(store)
        r._nav_mode = True
        r._nav_index = 0  # oldest run — outside the 6-row viewport
        r._stop_selected_agent()
        assert sinks[0].is_cancelled() is True

    def test_ctrl_a_lands_on_the_oldest_running_agent(self):
        store, _ = self._store(11, running={2, 9})
        r = self._reader(store)
        _drive_keys(r, "\x01")  # Ctrl+A
        assert r._nav_mode is True
        assert r._nav_index == 2

    def test_ctrl_a_lands_on_the_newest_when_none_run(self):
        store, _ = self._store(11, running=())
        r = self._reader(store)
        _drive_keys(r, "\x01")
        assert r._nav_index == 10

    def test_finished_runs_stay_navigable_after_the_panel_hides(self):
        store, _ = self._store(4, running=())
        r = self._reader(store)
        assert r._panel_showable() is False  # nothing running → panel gone
        assert r._agents_navigable() is True  # but the reports are still there
        _drive_keys(r, "\x01")
        assert r._nav_mode is True and r._panel_showable() is True

    def test_nothing_to_navigate_with_no_runs(self):
        store, _ = self._store(0)
        r = self._reader(store)
        assert r._agents_navigable() is False
        _drive_keys(r, "\x01")
        assert r._nav_mode is False


class TestAgentsPanelTurnScoping:
    """The LIVE panel glances at THIS turn's agents; earlier turns are counted,
    not listed.

    Every run of the session stays in the store so a finished agent's report can
    still be opened, which is what made the panel grow: after three turns its six
    rows were previous turns' ✓ entries and the agents actually running had
    scrolled off. Scoping the glance must not cost any of that reach — nav mode
    still walks, views and stops every retained run.
    """

    def _reader(self, store):
        return tui.PinnedPromptReader(
            prompt_text=lambda: ">",
            commands=[],
            dispatch=lambda q: None,
            agents_provider=store.snapshot,
            agents_get=store.get,
            agents_stop=store.request_stop,
            agents_stop_all=store.request_stop_all,
            agents_turn=lambda: store.current_turn,
        )

    def _store(self, old=3, new=2, old_running=()):
        """`old` runs from a previous turn (finished unless their index is in
        `old_running`), then `new` running ones from this turn.
        Returns (store, old_sinks, new_sinks)."""
        from mnemoai.client.agent.agent_activity import AgentActivityStore

        store = AgentActivityStore()
        store.begin_turn()
        olds = [store.open_run("research", f"old{i}", "spawn") for i in range(old)]
        for i, s in enumerate(olds):
            if i not in old_running:
                s.finish("done")
        store.begin_turn()
        news = [store.open_run("code", f"new{i}", "spawn") for i in range(new)]
        return store, olds, news

    def _body(self, reader):
        return "".join(txt for _cls, txt in reader._agents_text()[1:])

    def test_only_this_turns_agents_are_listed(self):
        store, _, _ = self._store()
        r = self._reader(store)
        body = self._body(r)
        assert "new0" in body and "new1" in body
        assert "old0" not in body and "old2" not in body

    def test_the_hint_counts_what_it_scoped_out(self):
        # Silently dropping rows would read as the panel losing the reports; the
        # count is how the user knows Ctrl+A still reaches them.
        store, _, _ = self._store()
        assert "+3 earlier" in self._reader(store)._agents_text()[0][1]

    def test_no_earlier_count_when_there_is_nothing_earlier(self):
        store, _, _ = self._store(old=0, new=2)
        assert self._reader(store)._agents_text()[0][1].strip() == "Ctrl+A: agents"

    def test_an_earlier_agent_still_running_is_listed(self):
        # A background sub-agent outlives its turn, and one still working is
        # exactly what the panel is for.
        store, _, _ = self._store(old=2, new=1, old_running={0})
        body = self._body(self._reader(store))
        assert "old0" in body and "new0" in body  # the runner and this turn's
        assert "old1" not in body  # the finished one is counted, not listed
        assert "+1 earlier" in self._reader(store)._agents_text()[0][1]

    def test_nav_mode_shows_every_retained_run(self):
        store, _, _ = self._store()
        r = self._reader(store)
        r._nav_mode = True
        body = self._body(r)
        assert "old0" in body and "new1" in body
        assert "earlier" not in r._agents_text()[0][1]  # nav lists them, not counts

    def test_a_scoped_out_agents_report_is_still_reachable(self):
        store, olds, _ = self._store(old=1, new=1, old_running={0})
        olds[0].finish_ok("what the research found")  # the real success path
        r = self._reader(store)
        assert "old0" not in self._body(r)  # not in the glance…
        assert len(r._agent_rows()) == 2  # …but the nav list still has it
        r._nav_mode = True
        r._nav_index = 0
        row = r._agent_rows()[r._nav_index]
        assert row.description == "old0"
        # And the detail view's source still resolves it (the point of retaining).
        run = r._agents_get(row.run_id)
        assert any(e.kind == "final" for e in run.events)

    def test_a_scoped_out_agent_can_still_be_stopped(self):
        # `x` must reach a run the glance isn't showing — a stale stamp must not
        # become a way to lose control of a worker.
        store, olds, _ = self._store(old=2, new=1, old_running={1})
        r = self._reader(store)
        r._nav_mode = True
        r._nav_index = 1  # the earlier turn's runner
        r._stop_selected_agent()
        assert olds[1].is_cancelled() is True

    def test_a_turn_that_spawned_nothing_lists_nothing(self):
        # The accumulation case at its plainest: a new prompt with no agents of
        # its own must not inherit the last turn's ✓ rows.
        store, _, news = self._store()
        for s in news:
            s.finish("done")
        store.begin_turn()
        r = self._reader(store)
        assert r._agents_text() == []
        assert r._panel_showable() is False
        assert r._agents_navigable() is True  # Ctrl+A brings the reports back
        _drive_keys(r, "\x01")
        assert r._nav_mode is True and "old0" in self._body(r)

    def test_a_reader_with_no_turn_provider_scopes_nothing(self):
        # The default (no agents_turn wired: off-TTY callers, the existing tests)
        # must keep showing everything.
        store, _, _ = self._store()
        r = tui.PinnedPromptReader(
            prompt_text=lambda: ">",
            commands=[],
            dispatch=lambda q: None,
            agents_provider=store.snapshot,
            agents_get=store.get,
        )
        assert "old0" in self._body(r)

    def test_a_raising_turn_provider_does_not_blank_the_panel(self):
        store, _, _ = self._store()

        def _boom():
            raise RuntimeError("no turn for you")

        r = tui.PinnedPromptReader(
            prompt_text=lambda: ">",
            commands=[],
            dispatch=lambda q: None,
            agents_provider=store.snapshot,
            agents_get=store.get,
            agents_turn=_boom,
        )
        assert "new0" in self._body(r)


class TestStopAgentBindings:
    """x stops the selected agent (nav-mode); Ctrl+X Ctrl+K stops ALL — the
    latter is GLOBAL (armed whenever any agent runs, no nav-mode needed) to match
    the documented chord. Drives the real merged bindings via a pipe input."""

    def _store(self):
        from mnemoai.client.agent.agent_activity import AgentActivityStore

        return AgentActivityStore()

    def _drive(self, reader, keys: str):
        _drive_keys(reader, keys)

    def _reader(self, store):
        return tui.PinnedPromptReader(
            prompt_text=lambda: ">",
            commands=[],
            dispatch=lambda q: None,
            agents_provider=store.snapshot,
            agents_get=store.get,
            agents_stop=store.request_stop,
            agents_stop_all=store.request_stop_all,
        )

    def test_ctrl_x_ctrl_k_stops_all_without_nav_mode(self):
        store = self._store()
        a = store.open_run("explore", "a", "spawn")
        b = store.open_run("explore", "b", "background")
        r = self._reader(store)
        r._nav_mode = False  # GLOBAL: chord must fire from the normal prompt
        self._drive(r, "\x18\x0b")  # Ctrl+X, Ctrl+K
        assert a.is_cancelled() and b.is_cancelled()

    def test_x_stops_selected_in_nav_mode(self):
        store = self._store()
        a = store.open_run("explore", "a", "spawn")
        b = store.open_run("explore", "b", "background")
        r = self._reader(store)
        r._nav_mode = True
        r._nav_index = 0  # 'a'
        self._drive(r, "x")
        assert a.is_cancelled() is True
        assert b.is_cancelled() is False  # only the selected one

    def test_panel_row_shows_cancelling_while_stopping(self):
        store = self._store()
        a = store.open_run("explore", "core", "spawn")
        a.tool_call("fs_read", {})
        r = self._reader(store)
        r._nav_mode = True
        # Before stop: normal tool/elapsed counters.
        row = next(t for _c, t in r._agents_text() if "core" in t)
        assert "cancelling" not in row and "tool" in row
        # After stop request (worker not yet finished): live cancelling label.
        store.request_stop(a._run_id)
        row = next(t for _c, t in r._agents_text() if "core" in t)
        assert "cancelling" in row and "tool" not in row
        # Once the worker finishes, the ✗ row shows counters again (no label).
        a.finish("stopped")
        row = next(t for _c, t in r._agents_text() if "core" in t)
        assert "cancelling" not in row and "✗" in row


class TestDetailScroll:
    """The detail viewer owns its scroll position directly (get_vertical_scroll)
    with a visible scrollbar, so a single ↓ scrolls with no dead zone."""

    def test_scroll_position_advances_on_down(self):
        import asyncio

        from prompt_toolkit.application import Application
        from prompt_toolkit.data_structures import Size
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.margins import ScrollbarMargin
        from prompt_toolkit.output import DummyOutput

        class SmallOut(DummyOutput):
            def get_size(self):
                return Size(rows=10, columns=80)

        text = "\n".join(f"line {i}" for i in range(60))
        holder = {}
        real_init = Application.__init__

        def cap_init(self, *a, **k):
            k = dict(k)
            k["input"] = holder["pipe"]
            k["output"] = SmallOut()
            real_init(self, *a, **k)
            holder["app"] = self
            for w in k["layout"].walk():
                if isinstance(w, Window) and w.right_margins:
                    holder["body"] = w

        import unittest.mock as m

        with create_pipe_input() as pipe:
            holder["pipe"] = pipe
            with m.patch.object(tui, "_dialog_is_tty", lambda: True), m.patch.object(
                Application, "__init__", cap_init
            ), m.patch.object(Application, "run", lambda self: None):
                tui._run_detail_app(text)
            body = holder["body"]
            # Scrollbar present (visible position indicator).
            assert any(isinstance(mg, ScrollbarMargin) for mg in body.right_margins)
            # Direct scroll control: it owns vertical scroll.
            assert body.get_vertical_scroll is not None
            # Drive the app briefly and press Down; vertical_scroll must advance.
            app = holder["app"]

            # POLL for each stage instead of racing fixed sleeps. Two fixed 0.1s
            # sleeps (send-then-read) made this fail deterministically on some CI
            # runners: the app hadn't finished its FIRST render when the key was
            # sent, so the key was dropped and vertical_scroll stayed 0. Waiting
            # on the observable state (render_info present, then scroll advanced)
            # keeps the assertion strict while removing the timing dependence.
            async def _until(pred, timeout=10.0, step=0.02):
                waited = 0.0
                while waited < timeout:
                    if pred():
                        return True
                    await asyncio.sleep(step)
                    waited += step
                return False

            async def run():
                async def feed():
                    # 1. Wait for the first render, so the key can't be dropped.
                    await _until(lambda: body.render_info is not None)
                    pipe.send_text("\x1b[B")  # Down
                    # 2. Wait for the scroll to actually advance.
                    await _until(
                        lambda: body.render_info is not None
                        and body.render_info.vertical_scroll >= 1
                    )
                    holder["vscroll"] = (
                        body.render_info.vertical_scroll if body.render_info else -1
                    )
                    app.exit()

                asyncio.ensure_future(feed())
                await app.run_async()

            asyncio.run(run())
            assert holder["vscroll"] >= 1  # scrolled down (no dead zone)
