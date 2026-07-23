"""Styled ANSI rendering of a turn's reasoning and tool calls (pinned-input UI).

Pure string builders (no I/O, no prompt_toolkit) written into native scrollback
above the pinned input; colors degrade harmlessly where unsupported.
"""

import contextlib
import io
import re
import threading

from mnemoai.utils.formatting.code_formatter import CodeFormatter

_GREEN = "\033[32m"
_GRAY = "\033[90m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
# Diff colors for file edits (added / removed lines), like a git diff.
_ADD = "\033[32m"   # green
_DEL = "\033[31m"   # red
_HEADER = "\033[38;5;63m"  # indigo, matches the launch banner

_BAR = f"{_GREEN}▌{_RESET}"
_CONNECTOR = "↳"


def format_duration(seconds: float) -> str:
    """Compact duration for the header: 0.4→"0s", 90→"1m30s"."""
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    return f"{minutes}m{secs}s"


def _bordered(header: str, text: str) -> str:
    """Green-border block: bold header, then reasoning indented under the bar."""
    lines = [f"{_BAR} {_BOLD}{header}{_RESET}"]
    # Blank lines kept as a bare bar so paragraphs read.
    for i, raw in enumerate(text.split("\n")):
        connector = _CONNECTOR if i == 0 else " "
        if raw.strip():
            lines.append(f"{_BAR}   {_GRAY}{connector} {raw}{_RESET}")
        else:
            lines.append(_BAR)
    return "\n".join(lines)


def render_reasoning_block(reasoning: str, seconds: float) -> str:
    """Final "Thought for Ns…" block committed to scrollback; "" when empty."""
    text = (reasoning or "").strip()
    if not text:
        return ""
    return _bordered(f"Thought for {format_duration(seconds)}…", text)


def render_live_reasoning(reasoning: str, seconds: float) -> str:
    """In-progress "Thinking… (Ns)" block for the transient region; "" when empty."""
    text = (reasoning or "").strip()
    if not text:
        return ""
    return _bordered(f"Thinking… ({format_duration(seconds)})", text)


class ReasoningStatus:
    """Thread-safe live-reasoning buffer shared between the agent (worker thread,
    appends chunks) and the pinned UI (renders the transient block)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._parts: list = []
        self._started: float = 0.0
        self.active = False

    def start(self, now: float) -> None:
        with self._lock:
            self._parts = []
            self._started = now
            self.active = True

    def append(self, text: str) -> None:
        with self._lock:
            self._parts.append(text)

    def stop(self) -> None:
        with self._lock:
            self.active = False

    def render(self, now: float) -> str:
        """The live block to show now, or "" when idle/empty."""
        with self._lock:
            if not self.active:
                return ""
            text = "".join(self._parts)
            elapsed = now - self._started
        return render_live_reasoning(text, elapsed)


def _wrap(text: str, width: int) -> list:
    """Word-wrap a single logical line to ``width`` cols; keeps at least one line."""
    words = text.split()
    if not words:
        return [""]
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    lines.append(cur)
    return lines


def _style_markdown_inline(text: str) -> str:
    """Light inline markdown → ANSI: **bold** bold, `code` cyan. Strips markers."""
    text = re.sub(r"\*\*(.+?)\*\*", rf"{_BOLD}\1{_RESET}", text)
    text = re.sub(r"`([^`]+?)`", rf"{_CYAN}\1{_RESET}", text)
    return text


def render_plan(plan: str, width: int = 80) -> str:
    """Render a plan as a bordered, word-wrapped, markdown-aware block.

    Unlike a flattened ``↳ plan=…`` tool line, this preserves the plan's line
    structure: headings are bold, list items keep their bullet + hanging indent,
    long lines wrap to ``width``. Every line sits under the green bar.
    """
    text = (plan or "").strip()
    if not text:
        return f"{_BAR} {_BOLD}Plan{_RESET}"
    out = [f"{_BAR} {_BOLD}Plan{_RESET}", _BAR]
    for raw in text.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            out.append(_BAR)
            continue
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        # List item: keep the marker and hang-indent wrapped continuation lines.
        m = re.match(r"([-*+]|\d+\.)\s+(.*)", stripped)
        if m:
            marker, body = m.group(1), m.group(2)
            hang = " " * (len(marker) + 1)
            wrapped = _wrap(body, max(20, width - len(indent) - len(hang)))
            first = _style_markdown_inline(wrapped[0])
            out.append(f"{_BAR}   {indent}{marker} {first}")
            for cont in wrapped[1:]:
                out.append(f"{_BAR}   {indent}{hang}{_style_markdown_inline(cont)}")
            continue
        # Heading (## ...) → bold, markers dropped.
        h = re.match(r"#{1,6}\s+(.*)", stripped)
        if h:
            for i, cont in enumerate(_wrap(h.group(1), max(20, width - len(indent)))):
                body = f"{_BOLD}{cont}{_RESET}"
                out.append(f"{_BAR}   {indent}{body}")
            continue
        for cont in _wrap(stripped, max(20, width - len(indent))):
            out.append(f"{_BAR}   {indent}{_style_markdown_inline(cont)}")
    return "\n".join(out)


def _short_path(path: str) -> str:
    """Home-relative, compact path for a file-op header (``~/…`` when under home)."""
    import os

    p = str(path or "")
    home = os.path.expanduser("~")
    if p.startswith(home):
        p = "~" + p[len(home):]
    return p


def _diff_lines(old: str, new: str) -> list:
    """Render an old→new change as colored ``-``/``+`` diff lines (whole-block).

    Not a line-level LCS diff (we don't have the file); we show the removed text
    in red and the added text in green, which is what a str-replace edit is."""
    out = []
    for line in (old or "").split("\n"):
        out.append(f"  {_DEL}- {line}{_RESET}")
    for line in (new or "").split("\n"):
        out.append(f"  {_ADD}+ {line}{_RESET}")
    return out


def _update_block(path: str, summary: str, old: str, new: str) -> str:
    """``Update(path)`` header + gray summary line + red/green old→new diff."""
    header = f"{_HEADER}{_BOLD}Update{_RESET}{_HEADER}({path}){_RESET}"
    lines = [header, f"  {_GRAY}{summary}{_RESET}"]
    lines.extend(_diff_lines(old, new))
    return "\n".join(lines)


def render_file_edit(args: dict) -> str:
    """``Update(path)`` block with a red/green old→new diff."""
    path = _short_path(args.get("file_path", ""))
    old = str(args.get("old_string", ""))
    new = str(args.get("new_string", ""))
    if not old and new:
        summary = "inserted text"
    elif old and not new:
        summary = "deleted text"
    else:
        summary = "replaced text"
    return _update_block(path, summary, old, new)


def render_fs_write(args: dict) -> str:
    """Style block for fs_write: ``Create file`` with numbered content, or an
    ``Update(path)`` diff for str_replace/insert/append."""
    path = _short_path(args.get("path", ""))
    command = str(args.get("command", "create")).lower()

    if command == "create":
        header = f"{_HEADER}{_BOLD}Create file{_RESET}"
        text = str(args.get("file_text", ""))
        body = text.split("\n")
        # Trim a trailing empty line from a final newline so the count is clean.
        if body and body[-1] == "":
            body = body[:-1]
        width = len(str(len(body))) if body else 1
        lines = [header, f"  {_GRAY}{path}{_RESET}"]
        for i, line in enumerate(body, 1):
            lines.append(f"  {_GRAY}{str(i).rjust(width)}{_RESET} {line}")
        return "\n".join(lines)

    if command == "str_replace":
        return _update_block(
            path,
            "replaced text",
            str(args.get("old_str", "")),
            str(args.get("new_str", "")),
        )

    # insert / append: show the added text as green lines.
    verb = "Insert" if command == "insert" else "Append"
    header = f"{_HEADER}{_BOLD}{verb}{_RESET}{_HEADER}({path}){_RESET}"
    added = str(args.get("new_str", "") or args.get("file_text", ""))
    lines = [header]
    for line in added.split("\n"):
        lines.append(f"  {_ADD}+ {line}{_RESET}")
    return "\n".join(lines)


def render_tool_call(name: str, args: dict) -> str:
    """Build one tool block: bold ``ToolName`` then dimmed ``↳ key=value`` lines.

    Values are shown in full (unlike the elided ``[⚙ …]`` marker); newlines are
    flattened to one arg per line. File-op tools use their own richer renderers.
    """
    if name == "file_edit":
        return render_file_edit(args or {})
    if name == "fs_write":
        return render_fs_write(args or {})
    lines = [f"{_BOLD}{name or 'tool'}{_RESET}"]
    for key, value in (args or {}).items():
        flat = str(value).replace("\n", " ")
        lines.append(f"  {_GRAY}{_CONNECTOR} {key}={flat}{_RESET}")
    return "\n".join(lines)


_ANSWER_MARKER = "\033[36m●\033[0m "
_USER_PROMPT = "\033[34m>\033[0m "


def _visible_text(content) -> str:
    """Answer text from a message's content (string or Bedrock/Responses blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and (b.get("type") == "text" or "text" in b)
        ).strip()
    return ""


def render_markdown(text: str) -> str:
    """Render Markdown text to an ANSI string using the SAME formatter a live
    turn streams through, so a replayed answer looks identical to a fresh one.

    ``CodeFormatter`` prints to stdout as it renders; we run a fresh instance
    with stdout captured so the replay path can embed the result in the
    transcript string instead of streaming it.
    """
    if not text:
        return ""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fmt = CodeFormatter()
            fmt.process_chunk(text)
            fmt.flush()
    except Exception:
        return text  # never let rendering break a load
    return buf.getvalue().rstrip("\n")


def render_conversation(messages: list) -> str:
    """Replay loaded messages to a scrollback transcript: user prompts, 
    ``Thought for…`` blocks, tool calls, and answers.

    Duck-types LangChain messages by attributes (no import) so this stays pure:
    HumanMessage → ``> text``; AIMessage → reasoning block + tool blocks +
    ``● answer``; ToolMessage results are omitted (their call is already shown).
    """
    out = []
    for msg in messages:
        cls = type(msg).__name__
        content = getattr(msg, "content", "")
        if cls == "HumanMessage":
            text = _visible_text(content)
            if text:
                out.append(f"{_USER_PROMPT}{text}")
        elif cls == "AIMessage":
            reasoning = (getattr(msg, "additional_kwargs", {}) or {}).get(
                "reasoning_content"
            )
            if reasoning and reasoning.strip():
                # Duration isn't persisted; show the block without a time.
                out.append(_bordered("Thought…", reasoning.strip()))
            for tc in getattr(msg, "tool_calls", None) or []:
                if tc.get("name") == "exit_plan_mode":
                    out.append(render_plan((tc.get("args") or {}).get("plan", "")))
                else:
                    out.append(
                        render_tool_call(tc.get("name", "tool"), tc.get("args") or {})
                    )
            answer = _visible_text(content)
            if answer:
                # Render through the live formatter so a loaded answer matches a
                # freshly-streamed one (markdown, code highlighting, no raw
                # ``**bold``/fences/tables leaking through).
                out.append(f"{_ANSWER_MARKER}{render_markdown(answer)}")
    return "\n\n".join(out)
