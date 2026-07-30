"""Unit tests for the TUI dialog helpers' non-TTY fallbacks.

select_from_list and confirm_inline present prompt_toolkit dialogs on a TTY but
degrade to plain input() when stdin isn't a TTY (pipes / CI / tests). These tests
drive the non-TTY branch with input() mocked — no modal, no terminal.
"""

import builtins
import sys

import pytest

from mnemoai.client.ui import tui


@pytest.fixture
def not_a_tty(monkeypatch):
    # The helpers do `import sys` locally, so patch the real sys.std* streams.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)


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

    def test_it_goes_through_run_dialog(self, not_a_tty, monkeypatch):
        # Calling select_from_list directly would try to nest a full-screen app
        # inside the running one.
        seen = {}
        monkeypatch.setattr(builtins, "input", lambda *_: "2")

        def _run_dialog(func):
            seen["called"] = True
            return func()

        got = self._reader(_run_dialog).question_ui("Which?", ["a", "b"])
        assert seen.get("called") is True
        assert got == "b"

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
        monkeypatch.setattr(tui, "select_from_list", lambda *a, **k: "SQLite")
        self._reader(lambda func: func()).question_ui("Which db?", ["x", "y"])
        out = capsys.readouterr().out
        assert "Which db?" in out and "SQLite" in out

    def test_a_dismissal_is_echoed_too(self, capsys, monkeypatch):
        monkeypatch.setattr(tui, "select_from_list", lambda *a, **k: None)
        self._reader(lambda func: func()).question_ui("Which db?", ["x", "y"])
        assert "dismissed" in capsys.readouterr().out

    def test_options_are_passed_as_value_label_pairs(self, monkeypatch):
        seen = {}

        def _pick(title, opts, **k):
            seen["title"], seen["opts"] = title, opts
            return None

        monkeypatch.setattr(tui, "select_from_list", _pick)
        self._reader(lambda func: func()).question_ui("Q", ["a", "b"])
        assert seen["opts"] == [("a", "a"), ("b", "b")]
        assert "Q" in seen["title"]


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


class TestStopAgentBindings:
    """x stops the selected agent (nav-mode); Ctrl+X Ctrl+K stops ALL — the
    latter is GLOBAL (armed whenever any agent runs, no nav-mode needed) to match
    the documented chord. Drives the real merged bindings via a pipe input."""

    def _store(self):
        from mnemoai.client.agent.agent_activity import AgentActivityStore

        return AgentActivityStore()

    def _drive(self, reader, keys: str):
        # Run the reader's real merged key bindings against a pipe input on a
        # dummy output, feed `keys`, then exit — no real terminal.
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
