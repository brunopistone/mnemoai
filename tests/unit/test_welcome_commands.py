"""Unit tests for the launch banner and the ``/help`` command box.

Two lists describe the same commands — ``_COMMANDS`` (the autocomplete tokens) and
``_COMMAND_GROUPS`` (the ``/help`` box) — so they can drift: a command added to one
and not the other is either undocumented or untypeable via the menu. These tests
pin that they agree, and that the box stays inside an 80-column terminal.

The banner does NOT print that box, and the first class here holds it to that: the
box was 31 of the 41 lines a launch printed, and everything it listed is one
keystroke away via ``/help`` or the ``/`` menu. So the banner's job is only to say
which app, which version, and where the rest is.
"""

import re

import pytest

from mnemoai.client.ui import chat_interface as chat_interface_mod
from mnemoai.client.ui.chat_interface import ChatInterface
from mnemoai.client.user_commands import UserCommand

_ANSI = re.compile(r"\033\[[0-9;]*m")


@pytest.fixture(autouse=True)
def _no_user_commands(tmp_path, monkeypatch):
    """Render the box against an EMPTY commands dir.

    The box now grows a "Yours" group from ``~/.mnemoai/commands/``, so without
    this the width assertions would depend on whatever the developer running the
    suite happens to have authored.
    """
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))


class _FakeStore:
    """Stand-in for UserCommandStore with a fixed command list."""

    def __init__(self, commands):
        self._commands = commands

    def list_commands(self):
        return list(self._commands)

    def completions(self):
        return [(f"/{c.name}", c.description) for c in self._commands]


def _box_commands():
    """Bare command tokens listed in the ``/help`` box (``/exit, /quit`` → both)."""
    out = []
    for _, items in ChatInterface._COMMAND_GROUPS:
        for label, _desc in items:
            for part in label.split(","):
                token = part.strip().split()[0]  # drop the "[path]" arg hint
                if token:
                    out.append(token)
    return out


class TestTheTwoCommandListsAgree:
    def test_every_autocompletable_command_is_documented_in_the_box(self):
        box = set(_box_commands())
        missing = [c for c, _ in ChatInterface._COMMANDS if c not in box]
        assert not missing, f"missing from the /help reference: {missing}"

    def test_every_documented_command_can_be_typed_via_the_menu(self):
        known = {c for c, _ in ChatInterface._COMMANDS}
        missing = [c for c in _box_commands() if c not in known]
        assert not missing, f"documented but not autocompletable: {missing}"

    def test_no_command_is_listed_twice_in_the_box(self):
        shown = _box_commands()
        assert len(shown) == len(set(shown))


class TestTheLaunchBanner:
    """What a launch prints. Short, by design — the reference lives in ``/help``."""

    MAX_COLS = 80
    MAX_LINES = 12

    def _rendered(self, capsys):
        ci = ChatInterface.__new__(ChatInterface)
        ci._ChatInterface__welcome_message()
        return capsys.readouterr().out

    def test_it_does_not_print_the_command_box(self, capsys):
        # The whole point: the box is a screenful, and /help is one keystroke.
        rendered = _ANSI.sub("", self._rendered(capsys))
        assert "│" not in rendered and "╭" not in rendered
        # A description only the box carries — proof no row leaked through.
        assert "Summarize & shrink context" not in rendered

    def test_it_stays_short_enough_to_read_at_a_glance(self, capsys):
        lines = [ln for ln in self._rendered(capsys).split("\n") if ln.strip()]
        assert len(lines) <= self.MAX_LINES, f"banner is {len(lines)} lines"

    def test_it_fits_an_80_column_terminal(self, capsys):
        widths = [
            len(_ANSI.sub("", line)) for line in self._rendered(capsys).split("\n")
        ]
        assert max(widths) <= self.MAX_COLS, f"widest row is {max(widths)} cols"

    def test_it_names_the_three_ways_in(self, capsys):
        # With no box printed, this line is the ONLY thing at launch that says the
        # reference exists — and @ is invisible anywhere else.
        rendered = _ANSI.sub("", self._rendered(capsys))
        for way in ("/help", "to search", "@"):
            assert way in rendered, f"{way} is not offered at launch"

    def test_the_wordmark_is_flush_left(self, capsys):
        # It used to be indented to center over the box; with the box gone there
        # is nothing to center over, and the prompt + footer below are flush.
        lines = [_ANSI.sub("", ln) for ln in self._rendered(capsys).split("\n")]
        wordmark = [ln for ln in lines if "█" in ln]
        assert wordmark, "no wordmark printed"
        assert all(not ln.startswith(" ") for ln in wordmark)

    def test_the_tagline_sits_under_the_wordmark(self, capsys):
        # Centered on the WORDMARK's own width — under the letters, not the screen.
        lines = [_ANSI.sub("", ln) for ln in self._rendered(capsys).split("\n")]
        wordmark = next(ln for ln in lines if "█" in ln)
        tagline = next(ln for ln in lines if "learns & remembers" in ln)

        def span(s):
            return len(s) - len(s.lstrip()), len(s.rstrip())

        wl, wr = span(wordmark)
        tl, tr = span(tagline)
        assert tl >= wl and tr <= wr, "tagline runs wider than the wordmark"
        assert abs((tl - wl) - (wr - tr)) <= 1, "tagline is not centered"

    def test_the_version_is_shown(self, capsys, monkeypatch):
        # The one fact neither /help nor the pinned footer carries.
        monkeypatch.setattr(chat_interface_mod, "app_version", lambda: "9.9.9")
        assert "v9.9.9" in _ANSI.sub("", self._rendered(capsys))

    def test_a_checkout_prints_no_version_line(self, capsys, monkeypatch):
        # Running from a checkout has no distribution metadata; a placeholder
        # ("v", "unknown") would be worse than the line being absent.
        monkeypatch.setattr(chat_interface_mod, "app_version", lambda: "")
        lines = [
            _ANSI.sub("", ln).strip()
            for ln in self._rendered(capsys).split("\n")
            if ln.strip()
        ]
        assert not any(ln.startswith("v") for ln in lines), lines


class TestHelpScreen:
    """``/help`` is the command reference — the only place the box is printed.

    The banner points here instead of carrying the list itself, so these are the
    tests that the box is complete, square, and readable: the fixed-width frame
    does not wrap or adapt, so one over-long description silently breaks it on a
    standard terminal.
    """

    MAX_COLS = 80

    def _rendered(self):
        return ChatInterface._help_text(ChatInterface.__new__(ChatInterface))

    def test_each_group_stays_small_enough_to_scan(self):
        # The reason for grouping in the first place: one group of nine reads as a
        # wall. Split a group before it grows past this.
        for heading, items in ChatInterface._COMMAND_GROUPS:
            assert len(items) <= 5, f"group '{heading}' has {len(items)} entries"

    def test_a_group_heading_labels_only_its_first_row(self):
        # The heading is inlined into the gutter of its first command's row, so it
        # must not repeat down the group (that's what the blank-line spacers used
        # to cost). Checked in the GUTTER only — "Exit" also appears in a
        # description, so counting the bare substring would be misleading.
        rendered = _ANSI.sub("", self._rendered())
        headings = {h for h, _ in ChatInterface._COMMAND_GROUPS}
        gutters = [
            line.split("│")[1][:12].strip()
            for line in rendered.split("\n")
            if line.count("│") >= 2
        ]
        labelled = [g for g in gutters if g in headings]
        assert sorted(labelled) == sorted(headings)

    def test_it_fits_an_80_column_terminal(self):
        widths = [len(_ANSI.sub("", line)) for line in self._rendered().split("\n")]
        assert max(widths) <= self.MAX_COLS, f"widest row is {max(widths)} cols"

    def test_every_frame_row_is_the_same_width(self):
        rows = [
            _ANSI.sub("", line)
            for line in self._rendered().split("\n")
            if "│" in line
        ]
        widths = {len(r) for r in rows if r.strip()}
        assert len(widths) == 1, f"ragged frame widths: {sorted(widths)}"

    def test_it_lists_every_command_the_banner_does(self):
        rendered = _ANSI.sub("", self._rendered())
        missing = [c for c in _box_commands() if c not in rendered]
        assert not missing, f"missing from /help: {missing}"

    def test_it_documents_the_keys_that_are_not_commands(self):
        rendered = _ANSI.sub("", self._rendered())
        # Esc (interrupt) and Ctrl+A (agents panel) have no slash command at all,
        # so /help is the only place they are ever shown.
        for key in ("Enter", "Ctrl+J", "Esc", "Ctrl+A", "Ctrl+C"):
            assert key in rendered, f"{key} is undocumented"

    def test_help_is_itself_listed(self):
        # Otherwise the one command you needed to find this screen is invisible.
        assert "/help" in _ANSI.sub("", self._rendered())


class TestUserCommandsInTheBox:
    """The user's own commands are listed too — without resizing the reference.

    Both columns are padded to their widest member and the frame widens to its
    longest row, so ONE verbose description in a file the user wrote would push the
    whole built-in reference past an 80-column terminal.
    """

    MAX_COLS = 80

    def _render(self, commands):
        ci = ChatInterface.__new__(ChatInterface)
        ci._user_commands = _FakeStore(commands)
        return ChatInterface._help_text(ci)

    def _width(self, rendered):
        return max(len(_ANSI.sub("", line)) for line in rendered.split("\n"))

    def test_a_user_command_is_listed_in_its_own_group(self):
        rendered = _ANSI.sub("", self._render([UserCommand("deploy", "Ship a release")]))
        assert "Yours" in rendered
        assert "/deploy" in rendered
        assert "Ship a release" in rendered

    def test_a_verbose_user_command_cannot_widen_the_box(self):
        baseline = self._width(self._render([]))
        wide = self._render(
            [
                UserCommand(
                    "deploy",
                    "d" * 200,
                    argument_hint="<" + "h" * 80 + ">",
                )
            ]
        )
        assert self._width(wide) == baseline
        assert self._width(wide) <= self.MAX_COLS

    def test_a_long_label_keeps_the_name_and_drops_the_hint(self):
        # The name is what you type; the hint still shows in the / menu.
        rendered = _ANSI.sub(
            "",
            self._render(
                [UserCommand("deploy", "Ship it", argument_hint="<a very long hint here>")]
            ),
        )
        assert "/deploy" in rendered
        assert "very long hint" not in rendered

    def test_the_frame_stays_square_with_user_rows(self):
        rendered = self._render([UserCommand("deploy", "d" * 200)])
        rows = [_ANSI.sub("", ln) for ln in rendered.split("\n") if "│" in ln]
        widths = {len(r) for r in rows if r.strip()}
        assert len(widths) == 1, f"ragged frame widths: {sorted(widths)}"

    def test_many_commands_collapse_to_a_count(self):
        # A user with 30 commands must not push the built-in reference off screen.
        many = [UserCommand(f"cmd{i}", f"desc {i}") for i in range(12)]
        rendered = _ANSI.sub("", self._render(many))
        listed = [f"/cmd{i}" for i in range(12) if f"/cmd{i}" in rendered]
        assert len(listed) == ChatInterface._MAX_USER_COMMAND_ROWS
        assert f"+{12 - len(listed)} more" in rendered

    def test_no_group_when_the_user_has_none(self):
        assert "Yours" not in _ANSI.sub("", self._render([]))

    def test_a_broken_commands_dir_does_not_break_the_box(self):
        # A diagnostic (or a banner) must never become the failure it reports.
        class _Boom:
            def list_commands(self):
                raise OSError("commands dir is on fire")

        ci = ChatInterface.__new__(ChatInterface)
        ci._user_commands = _Boom()
        rendered = ChatInterface._help_text(ci)
        assert "/help" in _ANSI.sub("", rendered)
        assert "Yours" not in _ANSI.sub("", rendered)
