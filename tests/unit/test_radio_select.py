"""Unit tests for the picker's "which row is highlighted" helper.

A ``RadioList`` moves its highlight (``_selected_index``) in more bindings than it
commits ``current_value`` in, and every dialog here confirms on Enter without a
Tab-to-OK step — so a confirm that read ``current_value`` answered with the row
the user had already moved off. Pinned in three layers: the helper's own contract
(shape-independent, so it holds on any prompt_toolkit version), the paging binding
it installs, and the three real dialogs driven by real keys through a pipe.
"""

import inspect

import pytest
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.widgets import RadioList

from mnemoai.client.ui import tui
from mnemoai.utils import configurator
from mnemoai.utils.radio_select import commit_paging, highlighted_value

PAGEDOWN = "\x1b[6~"
PAGEUP = "\x1b[5~"
DOWN = "\x1b[B"
ENTER = "\r"


def _rows(n: int = 40):
    return [(f"v{i}", f"row {i}") for i in range(n)]


def _drive(dialog, keys: str, *args, **kwargs):
    """Run one of the real dialog helpers against piped keys on a dummy output."""
    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            pipe.send_text(keys)
            return dialog(*args, **kwargs)


# --- the helper's own contract -------------------------------------------------


def test_highlighted_value_reads_the_highlight_not_the_stale_commit():
    radio = RadioList(values=_rows(5), select_on_focus=True)
    radio._selected_index = 3  # moved by a binding that doesn't commit
    assert radio.current_value == "v0"
    assert highlighted_value(radio) == "v3"


def test_highlighted_value_falls_back_when_the_shape_is_unreadable():
    class Odd:
        values = None
        _selected_index = 2
        current_value = "committed"

    # A picker must always return something; the commit is the best answer left.
    assert highlighted_value(Odd()) == "committed"


# --- the paging binding --------------------------------------------------------


class _Event:
    """The one thing ``commit_paging`` reads off an event: the visible height.

    ``invalidate`` is here because prompt_toolkit's ``Binding.call`` repaints after
    any handler that doesn't return ``NotImplemented`` — which is how the moved
    ``(*)`` marker reaches the screen.
    """

    def __init__(self, height):
        info = type("Info", (), {"displayed_lines": list(range(height))})()
        window = type("Win", (), {"render_info": info})()
        layout = type("Layout", (), {"current_window": window})()
        self.app = type("App", (), {"layout": layout, "invalidate": lambda self: None})()


def _paging_handlers(radio):
    """The handlers a key press would actually reach — last binding wins."""
    from prompt_toolkit.keys import Keys

    kb = radio.control.key_bindings
    return (
        kb.get_bindings_for_keys((Keys.PageUp,))[-1],
        kb.get_bindings_for_keys((Keys.PageDown,))[-1],
    )


def test_commit_paging_moves_the_highlight_and_the_commit_together():
    radio = RadioList(values=_rows(40), select_on_focus=True)
    commit_paging(radio)
    up, down = _paging_handlers(radio)

    down.call(_Event(10))
    assert radio._selected_index == 10
    assert radio.current_value == "v10"

    up.call(_Event(10))
    assert radio._selected_index == 0
    assert radio.current_value == "v0"


def test_commit_paging_clamps_at_both_ends():
    radio = RadioList(values=_rows(5), select_on_focus=True)
    commit_paging(radio)
    up, down = _paging_handlers(radio)

    down.call(_Event(100))
    assert (radio._selected_index, radio.current_value) == (4, "v4")
    up.call(_Event(100))
    assert (radio._selected_index, radio.current_value) == (0, "v0")


def test_commit_paging_never_raises_on_an_event_it_cannot_read():
    radio = RadioList(values=_rows(5), select_on_focus=True)
    commit_paging(radio)
    _up, down = _paging_handlers(radio)
    # The handler itself, not Binding.call (whose own repaint needs a real event):
    # paging is a convenience, and a dialog must not die because it failed.
    down.handler(None)
    assert radio.current_value == "v0"


# --- the real dialogs, driven by real keys ------------------------------------


def test_paging_then_enter_returns_a_row_further_down():
    # The bug: this returned the FIRST row, because PgDn moved only the highlight.
    got = _drive(tui._radio_pick, PAGEDOWN + ENTER, "pick", _rows())
    assert got != "v0"
    assert got in {v for v, _ in _rows()}


def test_paging_down_and_back_up_returns_where_it_started():
    got = _drive(tui._radio_pick, PAGEDOWN + PAGEUP + ENTER, "pick", _rows())
    assert got == "v0"


def test_arrow_keys_still_confirm_the_row_they_land_on():
    got = _drive(tui._radio_pick, DOWN + DOWN + ENTER, "pick", _rows())
    assert got == "v2"


def test_type_to_find_then_enter_returns_the_row_it_found():
    # Same defect through a different key: type-to-find moves the highlight only.
    rows = [("a", "alpha"), ("b", "beta"), ("g", "gamma")]
    assert _drive(tui._radio_pick, "g" + ENTER, "pick", rows) == "g"


def test_delete_button_targets_the_row_that_is_highlighted():
    # Tab past OK to Delete, then Enter — deleting the wrong row is unrecoverable.
    got = _drive(
        tui._radio_pick, PAGEDOWN + "\t\t" + ENTER, "pick", _rows(), allow_delete=True
    )
    assert isinstance(got, tuple) and got[0] is tui._DELETE
    assert got[1] != "v0"


def test_question_picker_answers_with_the_row_on_screen():
    got = _drive(tui._question_pick, PAGEDOWN + ENTER, "q", _rows())
    assert isinstance(got, tuple)
    assert got[0] != "v0"


def test_configurator_radio_confirms_the_row_on_screen():
    got = _drive(configurator._dialog_radio, PAGEDOWN + ENTER, "cfg", _rows())
    assert got != "v0"
    assert got is not configurator._DIALOG_CANCEL


# --- drift guard ---------------------------------------------------------------


@pytest.mark.parametrize(
    "func",
    [tui._radio_pick, tui._question_pick, configurator._dialog_radio],
)
def test_every_picker_confirms_the_highlight(func):
    """A new dialog copied from one of these must not go back to the commit."""
    src = inspect.getsource(func)
    assert "highlighted_value(radio)" in src
    assert "commit_paging(radio)" in src
    assert "radio.current_value" not in src
