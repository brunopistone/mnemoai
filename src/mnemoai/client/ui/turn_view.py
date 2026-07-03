"""Styled ANSI rendering of a turn's reasoning and tool calls (pinned-input UI).

Pure string builders (no I/O, no prompt_toolkit) written into native scrollback
above the pinned input; colors degrade harmlessly where unsupported.
"""

import threading

_GREEN = "\033[32m"
_GRAY = "\033[90m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

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


def render_tool_call(name: str, args: dict) -> str:
    """Build one tool block: bold ``ToolName`` then dimmed ``↳ key=value`` lines.

    Values are shown in full (unlike the elided ``[⚙ …]`` marker); newlines are
    flattened to one arg per line.
    """
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


def render_conversation(messages: list) -> str:
    """Replay loaded messages to a scrollback transcript (like Claude Code's
    --resume): user prompts, ``Thought for…`` blocks, tool calls, and answers.

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
                out.append(render_tool_call(tc.get("name", "tool"), tc.get("args") or {}))
            answer = _visible_text(content)
            if answer:
                out.append(f"{_ANSWER_MARKER}{answer}")
    return "\n\n".join(out)
