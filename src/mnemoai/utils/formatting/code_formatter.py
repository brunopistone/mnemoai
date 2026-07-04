"""Utility for formatting code blocks, inline code, and lightweight Markdown
in streamed terminal output.

Fenced code blocks (```lang) are syntax-highlighted with Pygments. Everything
else is rendered with a small, dependency-free Markdown pass (no ``rich``):
headers, bullet/numbered lists, **bold**, *italic*, inline ``code``, and
clickable URLs. Non-code text is buffered by LINE so block- and span-level
Markdown can be applied to whole lines as they stream in.
"""

import re

from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import TextLexer, get_lexer_by_name, guess_lexer

from mnemoai.utils.formatting.url_formatter import make_urls_clickable


class CodeFormatter:
    """Highlight code and render lightweight Markdown during streaming."""

    # Inline code / identifiers: bold cyan (and the Rich library default ``bold cyan``). 
    # Plain cyan was washed out next to the surrounding text; the bold weight gives 
    # the same crisp distinction.
    _INLINE_CODE = "\033[1;36m"
    _BOLD = "\033[1m"
    _ITALIC = "\033[3m"
    _DIM = "\033[90m"
    _RESET = "\033[0m"

    # --- Markdown line patterns (block level) --------------------------------
    _HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
    _BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
    _NUMBER_RE = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
    _QUOTE_RE = re.compile(r"^>\s?(.*)$")
    _HR_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
    # --- Markdown span patterns (inline; both ends must hug non-space so a
    # spaced ``a * b`` math expression is never italicized) ------------------
    _BOLD_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
    _ITALIC_RE = re.compile(r"\*(?=\S)(.+?)(?<=\S)\*")
    _CODE_SPAN_RE = re.compile(r"`([^`]+)`")

    def __init__(self) -> None:
        """Initialize code formatter."""
        self._in_code_block = False
        self._code_buffer = ""
        self._code_lang = ""
        # Non-code text accumulates here until a newline completes a line.
        self._text_buffer = ""
        self._backtick_buffer = ""

    # --- code-fence handling (unchanged behavior) ----------------------------

    def _process_code_blocks(self, data: str) -> None:
        """Process data containing ``` delimiters.

        Args:
            data: Text containing code block delimiters
        """
        parts = data.split("```")
        # Track if we're in a code block at the start of this chunk
        in_code = self._in_code_block

        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Even index - depends on initial state
                if in_code:
                    # We're in a code block, accumulate this part
                    self._code_buffer += part
                else:
                    # We're outside, render as Markdown text
                    if part:
                        self._emit_markdown(part)
            else:
                # Odd index - toggle state
                if in_code:
                    # End of code block
                    self._print_highlighted_code()
                    self._in_code_block = False
                    self._code_buffer = ""
                    self._code_lang = ""
                    in_code = False
                else:
                    # Start of code block — flush any partial text line first so
                    # it isn't stranded behind the (about-to-print) code block.
                    self._flush_text_line()
                    self._in_code_block = True
                    # Extract language from the part after ```
                    # The part could be: "python\ncode", "\npython\ncode", "python", or ""
                    if part:
                        lines = part.split("\n", 1)
                        self._code_lang = lines[0].strip()
                        self._code_buffer = lines[1] if len(lines) > 1 else ""
                    else:
                        # Empty part, language will come in next chunk
                        self._code_lang = ""
                        self._code_buffer = ""
                    in_code = True

    def _process_in_code_blocks(self, data: str) -> None:
        """Process data while inside a code block (accumulate, don't print yet).

        Args:
            data: Text chunk arriving while inside a fenced block
        """
        # If language not set yet, try to extract from first line
        if not self._code_lang and data:
            lines = data.split("\n", 1)
            first_line = lines[0].strip()
            # Check if first line looks like a language identifier (short, alphanumeric)
            if first_line and len(first_line) < 20 and first_line.isalnum():
                self._code_lang = first_line
                self._code_buffer += lines[1] if len(lines) > 1 else ""
            else:
                self._code_buffer += data
        else:
            self._code_buffer += data

    def _print_highlighted_code(self) -> None:
        """Print code with syntax highlighting."""
        if not self._code_buffer:
            return

        try:
            lexer = None
            if self._code_lang:
                try:
                    lexer = get_lexer_by_name(self._code_lang, stripall=True)
                except Exception:
                    lexer = None

            if not lexer:
                try:
                    lexer = guess_lexer(self._code_buffer)
                except Exception:
                    lexer = TextLexer()

            # Use Terminal256Formatter for better colors
            highlighted = highlight(
                self._code_buffer, lexer, Terminal256Formatter(style="monokai")
            )
            print(highlighted, end="", flush=True)
        except Exception:
            # Fallback: plain cyan, so a highlighter failure still shows the code.
            print(f"\033[36m{self._code_buffer}\033[0m", end="", flush=True)

    # --- Markdown text rendering (line-buffered) -----------------------------

    def _emit_markdown(self, text: str) -> None:
        """Buffer non-code text and render each completed line as Markdown."""
        self._text_buffer += text
        while "\n" in self._text_buffer:
            line, self._text_buffer = self._text_buffer.split("\n", 1)
            print(self._render_line(line), flush=True)

    def _flush_text_line(self) -> None:
        """Render whatever partial (newline-less) text line is buffered."""
        if self._text_buffer:
            print(self._render_line(self._text_buffer), end="", flush=True)
            self._text_buffer = ""

    def _render_line(self, line: str) -> str:
        """Render one Markdown line (block prefix + inline spans) to ANSI text."""
        line = line.rstrip("\r")

        m = self._HEADER_RE.match(line)
        if m:
            return f"{self._BOLD}{self._render_spans(m.group(2))}{self._RESET}"

        if self._HR_RE.match(line):
            return f"{self._DIM}────────────────────{self._RESET}"

        m = self._BULLET_RE.match(line)
        if m:
            indent, content = m.group(1), m.group(2)
            return f"{indent}  • {self._render_spans(content)}"

        m = self._NUMBER_RE.match(line)
        if m:
            indent, num, content = m.group(1), m.group(2), m.group(3)
            return f"{indent}  {num}. {self._render_spans(content)}"

        m = self._QUOTE_RE.match(line)
        if m:
            return f"{self._DIM}│ {self._render_spans(m.group(1))}{self._RESET}"

        return self._render_spans(line)

    def _render_spans(self, text: str) -> str:
        """Render inline Markdown spans: code, bold, italic, and URLs.

        Inline code is rendered first and its content is left untouched by the
        emphasis/URL passes (so ``**x**`` inside backticks stays literal).
        """
        rendered = []
        pos = 0
        for m in self._CODE_SPAN_RE.finditer(text):
            if m.start() > pos:
                rendered.append(self._render_text_span(text[pos:m.start()]))
            rendered.append(f"{self._INLINE_CODE}{m.group(1)}{self._RESET}")
            pos = m.end()
        if pos < len(text):
            rendered.append(self._render_text_span(text[pos:]))
        return "".join(rendered)

    def _render_text_span(self, text: str) -> str:
        """Apply bold/italic emphasis and clickable URLs to a plain text span."""
        text = self._BOLD_RE.sub(lambda m: f"{self._BOLD}{m.group(1)}{self._RESET}", text)
        text = self._ITALIC_RE.sub(
            lambda m: f"{self._ITALIC}{m.group(1)}{self._RESET}", text
        )
        return make_urls_clickable(text)

    # --- streaming entry points ---------------------------------------------

    def flush(self) -> None:
        """Emit anything still buffered at end-of-stream.

        Must be called once the stream ends. Without it:
        * a trailing backtick held back for a possible ``\\`\\`\\``` is dropped;
        * a response that ends INSIDE an unclosed code fence loses its entire
          code body (it sits unprinted in ``_code_buffer``);
        * a partial last line sits unrendered in ``_text_buffer``.

        Always resets terminal styling at the very end so the prompt is never
        left mid-color.
        """
        # Any backticks we were holding back never became a fence — literal text.
        pending_backticks = self._backtick_buffer
        self._backtick_buffer = ""

        if self._in_code_block:
            # Stream ended mid-code-block (no closing fence). Emit the code we
            # accumulated so it isn't lost.
            if self._code_buffer or pending_backticks:
                self._code_buffer += pending_backticks
                self._print_highlighted_code()
            self._in_code_block = False
            self._code_buffer = ""
            self._code_lang = ""
        else:
            # Pending backticks were a possible ``` fence that never arrived —
            # they're literal text; render the final partial line including them.
            if pending_backticks:
                self._text_buffer += pending_backticks
            self._flush_text_line()

        # Never leave the terminal styled at end of a response.
        print(self._RESET, end="", flush=True)

    def process_chunk(self, data: str) -> None:
        """Process a streaming chunk with code highlighting + Markdown.

        Args:
            data: Text chunk to process
        """
        # Handle buffered backticks from previous chunk
        data = self._backtick_buffer + data
        self._backtick_buffer = ""

        if not data:
            return

        # Buffer trailing backticks only if they might form ``` in next chunk
        if "```" not in data:
            # Count trailing backticks
            trailing_backticks = 0
            for i in range(len(data) - 1, -1, -1):
                if data[i] == "`":
                    trailing_backticks += 1
                else:
                    break

            # Buffer 1 or 2 trailing backticks only if preceded by whitespace/newline or at start
            # This prevents buffering inline code backticks like "text``"
            if trailing_backticks in [1, 2]:
                char_before = (
                    data[-(trailing_backticks + 1)]
                    if len(data) > trailing_backticks
                    else None
                )
                if char_before is None or char_before in [" ", "\n", "\t", "\r"]:
                    self._backtick_buffer = data[-trailing_backticks:]
                    data = data[:-trailing_backticks]

        if not data:
            return

        if "```" in data:
            self._process_code_blocks(data)
        elif self._in_code_block:
            self._process_in_code_blocks(data)
        else:
            # Normal text - render as Markdown (line-buffered)
            self._emit_markdown(data)
