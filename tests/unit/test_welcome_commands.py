"""Unit tests for the launch banner's command box.

Two lists describe the same commands — ``_COMMANDS`` (the autocomplete tokens) and
``_COMMAND_GROUPS`` (the banner box) — so they can drift: a command added to one and
not the other is either invisible at launch or untypeable via the menu. These tests
pin that they agree, and that the box stays inside an 80-column terminal.
"""

import re

from mnemoai.client.ui.chat_interface import ChatInterface

_ANSI = re.compile(r"\033\[[0-9;]*m")


def _box_commands():
    """Bare command tokens shown in the banner (``/exit, /quit`` → both)."""
    out = []
    for _, items in ChatInterface._COMMAND_GROUPS:
        for label, _desc in items:
            for part in label.split(","):
                token = part.strip().split()[0]  # drop the "[path]" arg hint
                if token:
                    out.append(token)
    return out


class TestTheTwoCommandListsAgree:
    def test_every_autocompletable_command_is_documented_in_the_banner(self):
        box = set(_box_commands())
        missing = [c for c, _ in ChatInterface._COMMANDS if c not in box]
        assert not missing, f"not shown at launch: {missing}"

    def test_every_banner_command_can_be_typed_via_the_menu(self):
        known = {c for c, _ in ChatInterface._COMMANDS}
        missing = [c for c in _box_commands() if c not in known]
        assert not missing, f"in the banner but not autocompletable: {missing}"

    def test_no_command_is_listed_twice_in_the_box(self):
        shown = _box_commands()
        assert len(shown) == len(set(shown))


class TestTheBoxStaysReadable:
    """The box is a fixed width — it does not wrap or adapt — so an over-long
    description silently breaks the frame on a standard terminal."""

    MAX_COLS = 80

    def _rendered(self, capsys):
        ci = ChatInterface.__new__(ChatInterface)
        ci._ChatInterface__welcome_message()
        return capsys.readouterr().out

    def test_it_fits_an_80_column_terminal(self, capsys):
        widths = [
            len(_ANSI.sub("", line)) for line in self._rendered(capsys).split("\n")
        ]
        assert max(widths) <= self.MAX_COLS, f"widest row is {max(widths)} cols"

    def test_every_frame_row_is_the_same_width(self, capsys):
        # A mis-padded row shows as a ragged right border.
        rows = [
            _ANSI.sub("", line)
            for line in self._rendered(capsys).split("\n")
            if line.strip().startswith("\033[90m│") or "│" in line
        ]
        widths = {len(r) for r in rows if r.strip()}
        assert len(widths) == 1, f"ragged frame widths: {sorted(widths)}"

    def test_each_group_stays_small_enough_to_scan(self):
        # The reason for grouping in the first place: one group of nine reads as a
        # wall. Split a group before it grows past this.
        for heading, items in ChatInterface._COMMAND_GROUPS:
            assert len(items) <= 5, f"group '{heading}' has {len(items)} entries"

    def test_the_wordmark_is_centered_over_the_box(self, capsys):
        # The box widens to its longest row (well past the wordmark's fixed 64
        # columns), so without an indent the logo sits visibly left of the frame.
        lines = [_ANSI.sub("", ln) for ln in self._rendered(capsys).split("\n")]
        wordmark = next(ln for ln in lines if "█" in ln)
        frame = next(ln for ln in lines if ln.startswith("╭"))

        def span(s):
            return len(s) - len(s.lstrip()), len(s.rstrip())

        wl, wr = span(wordmark)
        fl, fr = span(frame)
        assert abs((wl - fl) - (fr - wr)) <= 1, "wordmark is not centered on the box"

    def test_the_tagline_sits_under_the_wordmark(self, capsys):
        # Centered on the WORDMARK, not the box — under the letters, not the frame.
        lines = [_ANSI.sub("", ln) for ln in self._rendered(capsys).split("\n")]
        wordmark = next(ln for ln in lines if "█" in ln)
        tagline = next(ln for ln in lines if "learns & remembers" in ln)

        def span(s):
            return len(s) - len(s.lstrip()), len(s.rstrip())

        wl, wr = span(wordmark)
        tl, tr = span(tagline)
        assert tl >= wl and tr <= wr, "tagline runs wider than the wordmark"
        assert abs((tl - wl) - (wr - tr)) <= 1, "tagline is not centered"

    def test_the_hint_mentions_the_completion_menu(self, capsys):
        # It's how a command is found WITHOUT this box, which is what allows the
        # box to stay short.
        assert "to search commands" in self._rendered(capsys)

    def test_a_group_heading_labels_only_its_first_row(self, capsys):
        # The heading is inlined into the gutter of its first command's row, so it
        # must not repeat down the group (that's what the blank-line spacers used
        # to cost). Checked in the GUTTER only — "Exit" also appears in a
        # description, so counting the bare substring would be misleading.
        rendered = _ANSI.sub("", self._rendered(capsys))
        headings = {h for h, _ in ChatInterface._COMMAND_GROUPS}
        gutters = [
            line.split("│")[1][:12].strip()
            for line in rendered.split("\n")
            if line.count("│") >= 2
        ]
        labelled = [g for g in gutters if g in headings]
        assert sorted(labelled) == sorted(headings)
