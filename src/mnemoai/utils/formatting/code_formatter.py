"""Utility for formatting code blocks and inline code in terminal output."""

from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import TextLexer, get_lexer_by_name, guess_lexer

from mnemoai.utils.formatting.url_formatter import make_urls_clickable


class CodeFormatter:
    """Handles syntax highlighting for code blocks and inline code during streaming."""

    # Inline code / identifiers: bold cyan, matching Claude Code's look (and the
    # Rich library default ``bold cyan``). Plain cyan was washed out next to the
    # surrounding text; the bold weight gives the same crisp distinction.
    _INLINE_CODE = "\033[1;36m"
    _RESET = "\033[0m"

    def __init__(self) -> None:
        """Initialize code formatter."""
        self._in_code_block = False
        self._code_buffer = ""
        self._code_lang = ""
        self._in_inline_code = False
        self._backtick_buffer = ""

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
                    # We're outside, print normally
                    if part:
                        self._print_with_inline_code(part)
            else:
                # Odd index - toggle state
                if in_code:
                    # End of code block
                    self._print_highlighted_code()
                    self._in_code_block = False
                    self._code_buffer = ""
                    self._code_lang = ""
                    if self._in_inline_code:
                        print(self._RESET, end="", flush=True)
                        self._in_inline_code = False
                    in_code = False
                else:
                    # Start of code block
                    if self._in_inline_code:
                        print(self._RESET, end="", flush=True)
                        self._in_inline_code = False
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
        """Process data containing ``` delimiters.

        Args:
            data: Text containing code block delimiters
        """
        # Inside code block - accumulate, don't print yet
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

    def _print_with_inline_code(self, text: str) -> None:
        """Print text with inline code highlighting for single backticks.

        Args:
            text: Text to print with inline code formatting
        """

        i = 0
        while i < len(text):
            if text[i] == "`":
                # Toggle inline code state
                if self._in_inline_code:
                    print(self._RESET, end="", flush=True)
                    self._in_inline_code = False
                else:
                    print(self._INLINE_CODE, end="", flush=True)
                    self._in_inline_code = True
                i += 1
            else:
                # Find next backtick or end of string
                next_backtick = text.find("`", i)
                if next_backtick == -1:
                    chunk = text[i:]
                    i = len(text)
                else:
                    chunk = text[i:next_backtick]
                    i = next_backtick

                # Print chunk
                if self._in_inline_code:
                    print(chunk, end="", flush=True)
                else:
                    print(make_urls_clickable(chunk), end="", flush=True)

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

    def flush(self) -> None:
        """Emit anything still buffered at end-of-stream.

        Must be called once the stream ends. Without it:
        * a trailing backtick held back for a possible ``\\`\\`\\``` is dropped;
        * a response that ends INSIDE an unclosed code fence loses its entire
          code body (it sits unprinted in ``_code_buffer``);
        * an unbalanced inline backtick leaves the terminal stuck in cyan.

        Handles all three: flush pending backticks as text, highlight/emit any
        unclosed code block, and reset inline-code color.
        """
        # Any backticks we were holding back never became a fence — they're
        # literal text. Append them to whatever we emit below.
        pending_backticks = self._backtick_buffer
        self._backtick_buffer = ""

        if self._in_code_block:
            # Stream ended mid-code-block (no closing fence). Emit the code we
            # accumulated so it isn't lost; highlight it with whatever language
            # we detected (guess_lexer falls back when unknown).
            if self._code_buffer or pending_backticks:
                self._code_buffer += pending_backticks
                self._print_highlighted_code()
            self._in_code_block = False
            self._code_buffer = ""
            self._code_lang = ""
        elif pending_backticks:
            # These backticks were held back as a possible ``` fence that never
            # arrived. They're literal text now — print them verbatim, NOT via
            # _print_with_inline_code (which would treat ` as a mode toggle and
            # swallow the character).
            if self._in_inline_code:
                print(self._RESET, end="", flush=True)
                self._in_inline_code = False
            print(pending_backticks, end="", flush=True)
            return

        # Never leave the terminal mid-inline-code (stuck cyan).
        if self._in_inline_code:
            print(self._RESET, end="", flush=True)
            self._in_inline_code = False

    def process_chunk(self, data: str) -> None:
        """Process a streaming chunk with code highlighting.

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
            # Normal text - print immediately with inline code processing
            self._print_with_inline_code(data)
