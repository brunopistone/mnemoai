"""Unit tests for URL and code formatters (utils/formatting/)."""

import re

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
