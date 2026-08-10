"""Unit tests for URL and code formatters (utils/formatting/)."""


import pytest

from mnemoai.utils.formatting.code_formatter import CodeFormatter
from mnemoai.utils.formatting.url_formatter import (
    format_url,
    highlight_urls,
    make_urls_clickable,
)

# OSC 8 hyperlink introducer used by clickable terminal links.
OSC8 = "\033]8;;"


@pytest.fixture
def no_hyperlink_term(monkeypatch):
    """Force the 'terminal without hyperlink support' branch."""
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.delenv("ITERM_SESSION_ID", raising=False)
    for k in list(__import__("os").environ):
        if "ITERM" in k:
            monkeypatch.delenv(k, raising=False)


@pytest.fixture
def hyperlink_term(monkeypatch):
    """Force the 'terminal with hyperlink support' branch."""
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")


class TestFormatUrl:
    def test_wraps_url_in_osc8_sequence(self):
        out = format_url("https://example.com")
        assert OSC8 in out
        assert "https://example.com" in out

    def test_custom_display_text(self):
        out = format_url("https://example.com", "click here")
        assert "click here" in out


class TestHighlightUrls:
    def test_plain_url_gets_ansi_color(self):
        out = highlight_urls("see https://example.com now")
        assert "\033[36;4m" in out
        assert "https://example.com" in out

    def test_markdown_link_rendered(self):
        out = highlight_urls("[docs](https://example.com)")
        assert "docs" in out
        assert "https://example.com" in out

    def test_text_without_urls_unchanged_content(self):
        text = "no links here"
        out = highlight_urls(text)
        assert "no links here" in out


class TestMakeUrlsClickable:
    def test_falls_back_to_highlight_without_hyperlink_support(
        self, no_hyperlink_term
    ):
        out = make_urls_clickable("visit https://example.com")
        # Fallback path uses color codes, not OSC8 hyperlinks.
        assert "\033[36;4m" in out

    def test_uses_osc8_with_hyperlink_support(self, hyperlink_term):
        out = make_urls_clickable("visit https://example.com")
        assert OSC8 in out

    def test_markdown_link_with_hyperlink_support(self, hyperlink_term):
        out = make_urls_clickable("[docs](https://example.com)")
        assert OSC8 in out
        assert "docs" in out

    def test_reformatting_is_a_noop(self, hyperlink_term):
        """A URL already inside an OSC 8 link is not wrapped a second time."""
        once = make_urls_clickable("see https://example.com/x now")
        assert make_urls_clickable(once) == once


class TestNoAnsiLeaksIntoVisibleText:
    """Link formatting runs over text emphasis has ALREADY turned into ANSI, so
    its patterns must treat escapes as structure. Both leaks below were live: a
    literal `1m` printed before a bold name (the markdown-link `[` matched the
    `[` of `\\x1b[1m`, stranding the ESC), and a literal `[0m` after a URL (the
    plain-URL pass re-matched an already-wrapped URL and its trailing character
    class ate the ESC of the following reset)."""

    # Text a terminal would show: every valid SGR sequence removed. Anything
    # escape-shaped still left is something the user sees as garbage.
    @staticmethod
    def _visible(rendered: str) -> str:
        import re

        return re.sub(r"\x1b\[[0-9;]*m", "", rendered)

    @pytest.mark.parametrize(
        "source",
        [
            "**Bold text** before [a link](https://example.com), then more",
            "text with **[a bold link](https://example.com)** in it",
            "- **Bold item** — see [the docs](https://example.org/guide)",
            "See [a link](https://example.com) and https://example.org/x too",
            "*[italic link](https://example.com)* with `code` and https://a.bc/d.",
        ],
    )
    def test_emphasis_next_to_links_leaves_no_escape_fragments(
        self, source, no_hyperlink_term
    ):
        visible = self._visible(CodeFormatter.render_to_string(source))
        assert "\x1b" not in visible
        # The tell-tales: an SGR body that lost its ESC.
        for fragment in ("[0m", "[1m", "[3m", "[36;4m", "1mBold", "1ma link"):
            assert fragment not in visible

    def test_markdown_link_url_is_highlighted_once(self, no_hyperlink_term):
        """The URL inside a markdown link must not be re-wrapped by the plain
        pass — doubling the opening code is what stranded the reset's ESC."""
        out = highlight_urls("[a link](https://example.com) tail")
        assert out.count("\033[36;4m") == 2  # display text + URL, not 3
        assert "\033[36;4m\033[36;4m" not in out

    def test_highlighting_twice_is_a_noop(self, no_hyperlink_term):
        once = highlight_urls("see https://example.com/x now")
        assert highlight_urls(once) == once

    def test_bold_link_text_keeps_its_emphasis(self, no_hyperlink_term):
        """Guarding against escapes must not stop emphasis INSIDE link text from
        surviving — the display text may legitimately contain SGR runs."""
        out = CodeFormatter.render_to_string("[**bold link**](https://example.com)")
        assert "\033[1m" in out
        assert "bold link" in self._visible(out)


class TestCodeFormatter:
    def test_plain_text_passthrough(self, capsys):
        cf = CodeFormatter()
        cf.process_chunk("just some text")
        cf.flush()
        captured = capsys.readouterr()
        assert "just some text" in captured.out

    def test_complete_code_block_highlighted(self, capsys):
        cf = CodeFormatter()
        # Stream the fences/content across chunks as the model would emit them.
        cf.process_chunk("```python\n")
        cf.process_chunk("print('hi')\n")
        cf.process_chunk("```\n")
        cf.flush()
        captured = capsys.readouterr()
        # Content should appear (possibly with ANSI highlight codes around it).
        assert "print" in captured.out

    def test_state_resets_after_closed_block(self, capsys):
        cf = CodeFormatter()
        # Closing fence must arrive in a chunk that balances the opening one.
        cf.process_chunk("```python\nx = 1\n")
        cf.process_chunk("```\n")
        cf.flush()
        assert cf._in_code_block is False

    def test_text_after_closed_block_is_preserved(self, capsys):
        cf = CodeFormatter()
        for chunk in ["```python\n", "y = 2\n", "```\n", "after the block"]:
            cf.process_chunk(chunk)
        cf.flush()
        captured = capsys.readouterr()
        assert "after the block" in captured.out
        assert "y" in captured.out

    def test_inline_code_is_bold_cyan(self, capsys):
        # inline code / identifiers in bold cyan.
        cf = CodeFormatter()
        cf.process_chunk("use `foo.py` now")
        cf.flush()
        out = capsys.readouterr().out
        assert "\033[1;36m" in out  # bold cyan
        assert "foo.py" in out

    def test_unclosed_code_block_is_flushed(self, capsys):
        # Regression: a response that ends INSIDE an unclosed ``` fence must
        # still emit the code, not silently drop it in the buffer.
        cf = CodeFormatter()
        cf.process_chunk("here:\n```python\nprint('hi')\n")
        cf.flush()
        out = capsys.readouterr().out
        assert "print" in out and "hi" in out
        assert cf._in_code_block is False

    def test_dangling_backtick_not_dropped(self, capsys):
        # A trailing solo backtick (held back as a possible ``` fence) that
        # never completes must print as a literal backtick, not vanish.
        cf = CodeFormatter()
        cf.process_chunk("see ")
        cf.process_chunk("`")
        cf.flush()
        out = capsys.readouterr().out
        assert "`" in out

    def test_unbalanced_inline_backtick_resets_color(self, capsys):
        # An unterminated inline backtick must reset the terminal color on flush
        # so the prompt isn't left stuck in any styling.
        cf = CodeFormatter()
        cf.process_chunk("start `unterminated")
        cf.flush()
        out = capsys.readouterr().out
        assert out.rstrip().endswith("\033[0m")


class TestMarkdownRendering:
    """Lightweight Markdown rendering of streamed non-code text (no rich)."""

    def _render(self, *chunks):
        import contextlib
        import io

        cf = CodeFormatter()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for c in chunks:
                cf.process_chunk(c)
            cf.flush()
        return buf.getvalue()

    def test_header_is_bold_without_hashes(self):
        out = self._render("## What the script does\n")
        assert "\033[1m" in out  # bold
        assert "What the script does" in out
        assert "##" not in out  # the marker itself is stripped

    def test_bullet_becomes_glyph(self):
        out = self._render("- first item\n")
        assert "•" in out
        assert "first item" in out

    def test_numbered_list_preserved(self):
        out = self._render("1. do this\n")
        assert "1." in out and "do this" in out

    def test_bold_span_rendered(self):
        out = self._render("this is **important** text\n")
        assert "\033[1m" in out
        assert "important" in out
        assert "**" not in out

    def test_spaced_asterisks_not_italicized(self):
        # Regression: a math expression with spaced asterisks must NOT be
        # treated as italic (would swallow the asterisks and restyle the text).
        out = self._render("cost = instance_count * price_per_hour * hours\n")
        assert "instance_count * price_per_hour * hours" in out
        assert "\033[3m" not in out  # no italic styling applied

    def test_inline_code_in_markdown_line(self):
        out = self._render("use the `foo.py` file\n")
        assert "\033[1;36m" in out  # inline code bold cyan
        assert "foo.py" in out

    def test_bold_inside_inline_code_stays_literal(self):
        # `**x**` inside backticks must NOT be bolded — it's literal code.
        out = self._render("run `a ** b` now\n")
        assert "a ** b" in out

    def test_plain_paragraph_unchanged_text(self):
        out = self._render("just a normal sentence.\n")
        assert "just a normal sentence." in out

    def test_code_block_still_highlighted_with_markdown_around(self):
        out = self._render("# Title\n", "```python\n", "x = 1\n", "```\n", "- done\n")
        assert "Title" in out and "x" in out and "done" in out

    def _plain(self, s):
        import re

        return re.sub(r"\033\[[0-9;]*m|\033\]8;;[^\033]*\033\\", "", s)

    def _render_by_char(self, text):
        import contextlib
        import io

        cf = CodeFormatter()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for ch in text:
                cf.process_chunk(ch)
            cf.flush()
        return buf.getvalue()

    def test_language_label_never_leaks_as_text(self):
        # Regression: the old split('```') state machine could emit a bare
        # "python" line before a code block. The parser consumes the fence's
        # language into the token, so it can never render as visible text.
        for render in (lambda t: self._render(t), self._render_by_char):
            p = self._plain(render("```python\nx = 1\n```\n"))
            assert not any(line.strip() == "python" for line in p.splitlines())
            assert "x = 1" in p

    def test_adjacent_code_fences_render_as_two_blocks(self):
        # Two back-to-back fenced blocks — the classic case that desyncs a naive
        # split('```') scanner. Both bodies must appear, no fence markers leak.
        text = "```\nAAA\n```\n```python\nBBB\n```\n"
        for render in (lambda t: self._render(t), self._render_by_char):
            p = self._plain(render(text))
            assert "AAA" in p and "BBB" in p
            assert "```" not in p

    def test_no_literal_backslash_n_in_output(self):
        # Regression: a chunk boundary once caused an escaped "\n" to render
        # literally. Streaming char-by-char must never emit a literal \n.
        text = "line one\n\n```python\ny = 2\n```\nline `two` after.\n"
        p = self._plain(self._render_by_char(text))
        assert "\\n" not in p

    def test_inline_code_after_code_block_still_styled_char_stream(self):
        # Regression: a paragraph with inline code following adjacent fences was
        # rendered raw (backticks shown) when streamed char-by-char.
        text = "```\nA\n```\n```python\nB\n```\nBoth `explore` and `plan` here.\n"
        out = self._render_by_char(text)
        assert "\033[1;36m" in out  # inline code styled
        assert "`explore`" not in self._plain(out)  # no raw backticks


class TestParserFailureFallback:
    """When markdown-it itself raises, ``_render_pending`` falls back to emitting
    the unrendered tail. That fallback called ``self._render_text_block()``, a
    method that exists nowhere — so a recoverable render failure became an
    ``AttributeError`` that lost the answer entirely. Only reachable via a parser
    exception, which is why it was never hit in normal use.
    """

    def _formatter(self, buffer):
        f = CodeFormatter()
        f._buffer = buffer
        f._rendered_lines = 0

        class _Boom:
            def parse(self, text):
                raise RuntimeError("parser blew up")

        f._md_inst = _Boom()
        return f

    def test_the_missing_method_is_never_called(self):
        # Assert on CODE, not raw source: the name appears in the comment that
        # explains the bug.
        import inspect

        from mnemoai.utils.formatting import code_formatter

        code = [
            line
            for line in inspect.getsource(code_formatter).splitlines()
            if not line.lstrip().startswith("#")
        ]
        assert not any("_render_text_block" in line for line in code)

    def test_a_parser_failure_still_renders_the_tail(self, capsys):
        self._formatter("line one\nline two")._render_pending(final=True)
        out = capsys.readouterr().out
        assert "line one" in out and "line two" in out

    def test_markdown_in_the_tail_is_still_styled(self, capsys):
        self._formatter("## a heading\n- a bullet")._render_pending(final=True)
        out = capsys.readouterr().out
        assert "a heading" in out and "a bullet" in out
        assert "##" not in out  # prefix consumed, not printed
        assert "•" in out

    def test_nothing_is_double_rendered(self, capsys):
        f = self._formatter("only line")
        f._render_pending(final=True)
        f._render_pending(final=True)
        assert capsys.readouterr().out.count("only line") == 1

    def test_a_blank_tail_emits_nothing(self, capsys):
        self._formatter("   \n  ")._render_pending(final=True)
        assert capsys.readouterr().out.strip() == ""
