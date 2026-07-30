"""Streamed terminal rendering of Markdown + code, using a real parser.

Robustness is architectural, not hand-rolled: instead of a ``split("```")``
state machine fed streaming deltas (which desyncs on adjacent fences, a bare
language label, half-finished spans, etc.), this **buffers the full response and
re-tokenizes it with a real Markdown parser** (``markdown-it-py``) as it streams.

Streaming model: keep a "stable prefix" of the text whose top-level blocks are
complete. On each chunk, re-parse the whole buffer, render any newly-completed
blocks exactly once, and hold back the trailing (still-growing) block — it gets
re-parsed cleanly on the next chunk. Because the parser always re-reads the whole
tail, an unterminated code fence or an open ``**bold`` simply resolves on the
next delta; there is no fence bookkeeping to get out of sync. Fenced code becomes
a single token whose language is consumed by the parser, so a language label can
never leak as visible text.

Fenced code blocks are syntax-highlighted with Pygments; everything else is
rendered with a small token→ANSI pass (headers, lists, blockquotes, rules,
**bold**, *italic*, inline ``code``, links). No ``rich`` dependency.
"""

from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import TextLexer, get_lexer_by_name, guess_lexer

from mnemoai.utils.formatting.url_formatter import make_urls_clickable


class CodeFormatter:
    """Render Markdown + highlighted code from a streaming text buffer."""

    # Inline code / identifiers: bold cyan (crisp against surrounding text).
    _INLINE_CODE = "\033[1;36m"
    _BOLD = "\033[1m"
    _ITALIC = "\033[3m"
    _DIM = "\033[90m"
    _RESET = "\033[0m"

    # A shared parser: CommonMark + tables/strikethrough off (the model uses ~
    # for "approximately", so leave it literal). block+inline tokens, no HTML.
    _md = MarkdownIt("commonmark").enable("table")

    def __init__(self, out=None, own_parser: bool = False) -> None:
        """Initialize the formatter.

        ``out`` is the sink (defaults to the builtin ``print``, i.e. live stdout
        streaming). ``own_parser`` gives this instance its OWN ``MarkdownIt``
        rather than the class-shared one. Both default to the streaming behavior;
        :meth:`render_to_string` sets them so an off-stream caller (e.g. the
        agent-detail view) can render to a string on the UI thread WITHOUT the
        process-global ``redirect_stdout`` + shared-parser race.
        """
        self._out = out if out is not None else print
        self._md_inst = (
            MarkdownIt("commonmark").enable("table") if own_parser else self._md
        )
        # Full accumulated response text.
        self._buffer = ""
        # Number of source LINES already finalized+rendered (their blocks are
        # complete). We re-parse the whole buffer each chunk but only render
        # blocks that start at/after this line, and only finalize a block once a
        # later block exists after it (so a growing tail is never emitted early).
        self._rendered_lines = 0
        # Back-compat flag some callers/tests read: is the tail an open fence?
        self._in_code_block = False

    # --- streaming entry points ---------------------------------------------

    def process_chunk(self, data: str) -> None:
        """Append a streamed chunk and render any newly-completed blocks."""
        if not data:
            return
        self._buffer += data
        self._render_pending(final=False)

    def flush(self) -> None:
        """Render everything still buffered at end-of-stream, then reset color.

        The trailing (possibly unterminated) block is rendered now — an unclosed
        code fence is emitted as a code block rather than dropped. Always resets
        terminal styling so the prompt is never left mid-color.
        """
        self._render_pending(final=True)
        self._out(self._RESET, end="", flush=True)

    @classmethod
    def render_to_string(cls, text: str) -> str:
        """Render a complete markdown string to ANSI and RETURN it.

        Thread-safe alternative to the module-level ``render_markdown`` helper:
        uses a per-call sink (no process-global ``redirect_stdout``) AND this
        instance's OWN ``MarkdownIt`` (not the class-shared ``_md``), so it can
        run on the UI thread while a live turn streams to stdout on the worker
        thread without racing either. Used by the agent-detail view.
        """
        parts: list = []

        def _sink(*args, **kwargs) -> None:
            # Mimic print(): join args, honor end= (default "\n"), ignore flush.
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            parts.append(sep.join(str(a) for a in args) + end)

        fmt = cls(out=_sink, own_parser=True)
        fmt.process_chunk(text)
        fmt.flush()
        return "".join(parts)

    # --- core: re-parse the buffer, render newly-finalized blocks ------------

    def _render_pending(self, final: bool) -> None:
        """Render top-level blocks that are complete (or, on ``final``, all).

        Re-tokenizes the WHOLE buffer every call (the parser handles unterminated
        fences / half-finished spans by re-reading them each time — no state
        machine to desync). Renders each top-level token whose block starts at a
        line ``>= self._rendered_lines``. While streaming, the LAST top-level
        block is held back (it may still be growing) and re-rendered next chunk;
        on ``final`` it's rendered too.
        """
        text = self._buffer
        lines = text.split("\n")

        try:
            tokens = [t for t in self._md_inst.parse(text) if t.level == 0 and t.map]
        except Exception:
            tokens = []

        if not tokens:
            # Parser produced nothing mappable (e.g. only whitespace so far, or a
            # partial line). Nothing to finalize yet; render the tail on final.
            if final and self._rendered_lines < len(lines):
                # Emit the tail the same way _render_token does (line renderer +
                # _out), rather than calling a `_render_text_block` that never
                # existed — this branch is only reached when the parser itself
                # raised, so the AttributeError would have replaced a recoverable
                # render failure with a crash, losing the answer entirely.
                for line in lines[self._rendered_lines:]:
                    self._out(self._render_line(line), flush=True)
                self._rendered_lines = len(lines)
            return

        # The tail token is "unstable" while streaming — don't finalize it.
        stop = len(tokens) if final else len(tokens) - 1
        self._in_code_block = (
            not final and tokens[-1].type == "fence"
            and not self._token_fence_closed(tokens[-1], lines)
        )

        for idx in range(stop):
            tok = tokens[idx]
            l0, l1 = tok.map
            if l0 < self._rendered_lines:
                continue  # already rendered in a prior chunk
            # Emit blank lines between this block and the previously rendered one
            # (preserves paragraph spacing) then render the block.
            if l0 > self._rendered_lines:
                for _ in range(l0 - self._rendered_lines):
                    self._out(flush=True)
            self._render_token(tok, lines)
            self._rendered_lines = l1

    def _token_fence_closed(self, tok, lines) -> bool:
        """True if a fence token's source includes its closing ``` line."""
        l0, l1 = tok.map
        src = "\n".join(lines[l0:l1])
        return src.rstrip().endswith("```") and src.count("```") >= 2

    def _render_token(self, tok, lines) -> None:
        """Render one top-level markdown-it block token to the terminal.

        Code fences use the parser's separated ``.content``/``.info`` (so a
        language label can never render as text). Every other top-level block
        (heading, paragraph, list, blockquote, rule) is rendered from its SOURCE
        lines through the lightweight line renderer, which handles ``#``/``-``/
        ``>`` prefixes and inline spans."""
        if tok.type == "fence":
            self._print_code(tok.info.strip(), tok.content)
            return
        if tok.type == "code_block":
            self._print_code("", tok.content)
            return
        if tok.type == "hr":
            self._out(f"{self._DIM}────────────────────{self._RESET}", flush=True)
            return
        # Heading / paragraph / list / blockquote / table → render source lines.
        l0, l1 = tok.map
        for line in lines[l0:l1]:
            self._out(self._render_line(line), flush=True)

    def _print_code(self, lang: str, body: str) -> None:
        """Print a code body syntax-highlighted (language already separated out by
        the parser, so it's never rendered as text)."""
        body = body.rstrip("\n")
        if not body:
            return
        try:
            lexer = None
            if lang:
                try:
                    lexer = get_lexer_by_name(lang, stripall=True)
                except Exception:
                    lexer = None
            if lexer is None:
                try:
                    lexer = guess_lexer(body)
                except Exception:
                    lexer = TextLexer()
            highlighted = highlight(
                body, lexer, Terminal256Formatter(style="monokai")
            )
            self._out(highlighted, end="", flush=True)
        except Exception:
            # Highlighter failure: plain cyan so the code still shows.
            self._out(f"\033[36m{body}\033[0m", end="", flush=True)

    # --- Markdown line rendering (block prefix + inline spans) ---------------

    def _render_line(self, line: str) -> str:
        """Render one Markdown line (block prefix + inline spans) to ANSI text."""
        import re

        line = line.rstrip("\r")

        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            return f"{self._BOLD}{self._render_spans(m.group(2))}{self._RESET}"

        if re.match(r"^\s*([-*_])(?:\s*\1){2,}\s*$", line):
            return f"{self._DIM}────────────────────{self._RESET}"

        m = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
        if m:
            indent, content = m.group(1), m.group(2)
            return f"{indent}  • {self._render_spans(content)}"

        m = re.match(r"^(\s*)(\d+)[.)]\s+(.*)$", line)
        if m:
            indent, num, content = m.group(1), m.group(2), m.group(3)
            return f"{indent}  {num}. {self._render_spans(content)}"

        m = re.match(r"^>\s?(.*)$", line)
        if m:
            return f"{self._DIM}│ {self._render_spans(m.group(1))}{self._RESET}"

        return self._render_spans(line)

    def _render_spans(self, text: str) -> str:
        """Render inline spans: code (untouched by other passes), bold, italic,
        links. Inline code is rendered first so ``**x**`` inside backticks stays
        literal."""
        import re

        rendered = []
        pos = 0
        for m in re.finditer(r"`([^`]+)`", text):
            if m.start() > pos:
                rendered.append(self._render_text_span(text[pos:m.start()]))
            rendered.append(f"{self._INLINE_CODE}{m.group(1)}{self._RESET}")
            pos = m.end()
        if pos < len(text):
            rendered.append(self._render_text_span(text[pos:]))
        return "".join(rendered)

    def _render_text_span(self, text: str) -> str:
        """Apply bold/italic emphasis and clickable URLs to a plain span.

        Emphasis markers must hug non-space on both ends, so a spaced ``a * b``
        math expression is never italicized.
        """
        import re

        text = re.sub(
            r"\*\*(?=\S)(.+?)(?<=\S)\*\*",
            lambda m: f"{self._BOLD}{m.group(1)}{self._RESET}",
            text,
        )
        text = re.sub(
            r"\*(?=\S)(.+?)(?<=\S)\*",
            lambda m: f"{self._ITALIC}{m.group(1)}{self._RESET}",
            text,
        )
        return make_urls_clickable(text)
