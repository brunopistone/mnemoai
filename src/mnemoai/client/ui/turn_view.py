"""Styled ANSI rendering of a turn's reasoning and tool calls (pinned-input UI).

Pure string builders (no I/O, no prompt_toolkit) written into native scrollback
above the pinned input; colors degrade harmlessly where unsupported.
"""

import difflib
import re
import threading
import time

from mnemoai.utils.formatting.code_formatter import CodeFormatter

_GREEN = "\033[32m"
_GRAY = "\033[90m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
# Diff colors for file edits (added / removed lines), like a git diff.
_ADD = "\033[32m"   # green
_DEL = "\033[31m"   # red
# Unchanged lines kept on each side of a change in a file-edit diff.
_DIFF_CONTEXT = 3
# A LIGHTER blue than the launch banner's indigo (63): these headers sit inline
# among gray args and white prose all turn long, so they need to read as a label
# at a glance without going as dark as a one-off banner can afford to be.
_HEADER = "\033[38;5;111m"

_BAR = f"{_GREEN}▌{_RESET}"
_CONNECTOR = "↳"

# Context the client PREPENDS to a prompt before it is stored: the episodic-memory
# block, and the ephemeral `<steering>`/`<plan-mode-active>` reminders. None of it
# was typed by the user, so none of it belongs in anything a person reads back.
_EPISODIC_PREFIX = "[Episodic Memory"
_EPHEMERAL_RE = re.compile(r"<(plan-mode-active|steering)>.*?</\1>\s*", re.DOTALL)
# A background sub-agent's report is auto-delivered AS a user message.
_BG_REPORT_PREFIX = "Your background sub-agent"


def user_prompt_text(text: str) -> str:
    """The part of a stored user message the user actually typed ("" if none).

    Strips the injected context listed above. Shared by the replay renderer, the
    ``--resume`` picker label, and ``/export`` so all three agree on what "what the
    user said" means — the replay used to print the raw episodic block, which
    dumped a ~30-line wall of tool names above the first prompt on every resume.
    """
    text = _EPHEMERAL_RE.sub("", text or "")
    if text.lstrip().startswith(_EPISODIC_PREFIX):
        # Shape: "[Episodic Memory …]\n<entries>\n\n<the real prompt>".
        text = text.split("\n\n", 1)[1] if "\n\n" in text else ""
    if text.lstrip().startswith(_BG_REPORT_PREFIX):
        return ""
    return text.strip()


def format_duration(seconds: float) -> str:
    """Compact duration: 0.4→"0s", 90→"1m30s", 3725→"1h2m5s"."""
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes}m{secs}s"
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


def _context_lines(lines: list, leading: bool, trailing: bool) -> list:
    """Unchanged lines as gray context, collapsing a long run to its useful ends.

    Only the side of the run that FACES a change carries information, so the
    unchanged head of an edit (``leading``) keeps its LAST few lines and the
    unchanged tail (``trailing``) keeps its FIRST few. An interior run keeps both
    ends and elides the middle.
    """
    head = 0 if leading else _DIFF_CONTEXT
    tail = 0 if trailing else _DIFF_CONTEXT
    gray = [f"  {_GRAY}  {line}{_RESET}" for line in lines]
    if len(gray) <= head + tail:
        return gray
    dropped = len(gray) - head - tail
    return (
        gray[:head]
        + [f"  {_GRAY}  … {dropped} unchanged line{'s' if dropped != 1 else ''}{_RESET}"]
        + (gray[len(gray) - tail:] if tail else [])
    )


def _diff_lines(old: str, new: str) -> list:
    """Render an old→new change as a line-level diff (red ``-`` / green ``+``).

    Pairs the two blobs line-by-line (``difflib`` LCS) so an edit buried in a
    large block shows only the lines that actually changed, with the surrounding
    unchanged lines kept as gray context and long runs of them elided. The naive
    version printed every old line then every new line, which for a one-word fix
    in a 40-line replacement made the reader diff it by eye.
    """
    old_lines = (old or "").split("\n")
    new_lines = (new or "").split("\n")
    # Whole-block insert/delete: nothing to pair, and no context to keep.
    if not old:
        return [f"  {_ADD}+ {line}{_RESET}" for line in new_lines]
    if not new:
        return [f"  {_DEL}- {line}{_RESET}" for line in old_lines]

    opcodes = difflib.SequenceMatcher(None, old_lines, new_lines).get_opcodes()
    # Identical blobs: there is no change to point at, so show the text plainly
    # rather than eliding it down to a bare "N unchanged lines".
    if all(tag == "equal" for tag, *_ in opcodes):
        return [f"  {_GRAY}  {line}{_RESET}" for line in old_lines]

    out = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            out.extend(
                _context_lines(
                    old_lines[i1:i2],
                    leading=(i1 == 0),
                    trailing=(i2 == len(old_lines)),
                )
            )
            continue
        if tag in ("replace", "delete"):
            out.extend(f"  {_DEL}- {line}{_RESET}" for line in old_lines[i1:i2])
        if tag in ("replace", "insert"):
            out.extend(f"  {_ADD}+ {line}{_RESET}" for line in new_lines[j1:j2])
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
    """Build one tool block: the ``ToolName`` header then dimmed ``↳ key=value``
    lines.

    The name carries the accent color the file-op blocks already use for theirs —
    bold white read as ordinary answer text, so a tool call didn't separate from
    the prose around it. Values are shown in full (unlike the elided ``[⚙ …]``
    marker); newlines are flattened to one arg per line. File-op tools use their
    own richer renderers.
    """
    if name == "file_edit":
        return render_file_edit(args or {})
    if name == "fs_write":
        return render_fs_write(args or {})
    lines = [f"{_HEADER}{_BOLD}{name or 'tool'}{_RESET}"]
    for key, value in (args or {}).items():
        flat = str(value).replace("\n", " ")
        lines.append(f"  {_GRAY}{_CONNECTOR} {key}={flat}{_RESET}")
    return "\n".join(lines)


# Steps shown at once in a multi-step checklist; a longer plan elides the parts
# that are neither running nor about to (see :func:`render_step_list`).
_STEP_MAX_ROWS = 8
_STEP_LEAD = 2  # rows of already-done context kept above the running step


def _step_window(total: int, running: set, max_rows: int) -> tuple:
    """``(start, end)`` slice of a step list to display, centered on the work.

    A 20-step plan re-printed in full for every wave would bury the answer, so
    only a window around the running step is shown; the elided ends are counted
    in the rendered output rather than dropped silently.
    """
    if total <= max_rows:
        return 0, total
    first = min(running) if running else 0
    start = max(0, min(first - _STEP_LEAD, total - max_rows))
    return start, start + max_rows


def render_step_list(
    descriptions: list,
    running=(),
    done=(),
    width: int = 76,
    max_rows: int = _STEP_MAX_ROWS,
) -> str:
    """Checklist of a multi-step task, the step(s) executing now in green.

    ``[✓]`` for a finished step (dim), green for whatever is running, dim for
    what hasn't started. A parallel wave runs several steps at once, so
    ``running`` is a collection, not a single index. "" for an empty plan.
    """
    steps = [str(d or "") for d in descriptions or []]
    if not steps:
        return ""
    running = {i for i in running if 0 <= i < len(steps)}
    done = {i for i in done if 0 <= i < len(steps)}
    total = len(steps)
    body = max(20, width - 6)

    header = (
        f"{_HEADER}{_BOLD}Steps{_RESET} {_GRAY}{len(done)}/{total}{_RESET}"
    )
    out = [header]
    start, end = _step_window(total, running, max_rows)
    if start:
        out.append(f"  {_GRAY}… {start} earlier step{'s' if start != 1 else ''}{_RESET}")
    for i in range(start, end):
        text = " ".join(steps[i].split())  # a newline would shred the block
        if len(text) > body:
            text = text[: body - 1] + "…"
        if i in done:
            out.append(f"  {_GREEN}[✓]{_RESET} {_GRAY}{text}{_RESET}")
        elif i in running:
            out.append(f"  {_GREEN}[ ] {text}{_RESET}")
        else:
            out.append(f"  {_GRAY}[ ] {text}{_RESET}")
    left = total - end
    if left:
        out.append(f"  {_GRAY}… {left} more step{'s' if left != 1 else ''}{_RESET}")
    return "\n".join(out)


def render_step_done(text: str, done: int, total: int, width: int = 76) -> str:
    """One line marking a step that just finished, with the running count.

    For the PRINTED checklist only: scrollback can't be rewritten, so a wave's
    block — printed before any of its steps start — would otherwise sit at
    ``0/N`` for as long as it runs, and each completion gets its own line
    instead. With a live :class:`StepStatus` the rows themselves tick, and no
    line like this is emitted. Emitted from the scheduling thread only.
    """
    label = " ".join(str(text or "").split())
    body = max(20, width - 6)
    if len(label) > body:
        label = label[: body - 1] + "…"
    return f"  {_GREEN}[✓]{_RESET} {_GRAY}{done}/{total} {label}{_RESET}"


class StepStatus:
    """Thread-safe live checklist shared between the wave scheduler (worker
    thread, marks steps running/done) and the pinned UI (re-renders it).

    A printed block is frozen the moment it lands — so ticking the EXISTING rows
    of a plan means rendering it in the transient pinned region, where every
    repaint replaces the whole block. The scheduler updates this sink instead of
    printing per wave, and one final all-checked block is committed to
    scrollback when the plan ends (see :class:`ReasoningStatus`, same shape).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps: list = []
        self._running: set = set()
        self._done: set = set()
        self.active = False

    def start(self, descriptions) -> None:
        with self._lock:
            self._steps = [str(d or "") for d in descriptions or []]
            self._running = set()
            self._done = set()
            self.active = bool(self._steps)

    def set_running(self, running) -> None:
        """Replace the set of steps executing now (a wave runs several at once)."""
        with self._lock:
            self._running = {i for i in running if i not in self._done}

    def mark_done(self, index: int) -> None:
        with self._lock:
            self._done.add(index)
            self._running.discard(index)

    def stop(self) -> None:
        with self._lock:
            self.active = False

    def render(self, width: int = 76) -> str:
        """The live checklist to show now, or "" when idle/empty."""
        with self._lock:
            if not self.active:
                return ""
            steps = list(self._steps)
            running, done = set(self._running), set(self._done)
        return render_step_list(steps, running=running, done=done, width=width)


def render_session_notice(text: str) -> str:
    """One accent line for a session-level event (a resume) — not a turn.

    Distinct from a turn's own output: the glyph carries the accent color and the
    text stays dim, so it reads as a marker rather than something the model said.
    """
    return f"{_HEADER}⟲{_RESET} {_GRAY}{' '.join(str(text or '').split())}{_RESET}"


def render_turn_end(seconds: float, finished: float, stopped: bool = False) -> str:
    """One dim line closing a turn: how long it took and when it ended.

    A streamed answer simply STOPS — the last chunk looks like any other, so
    nothing says whether the model is finished or the next paragraph is still
    coming, and the pinned prompt sits there looking identical either way (the
    spinner that meant "working" is gone from the toolbar the moment the turn
    ends, and a toolbar is not scrollback: it can't answer the question later).
    This is the missing full stop, and the two facts you want once a long turn
    has ended without you watching — the wait, and the clock time it landed.

    Printed for EVERY turn including a fast one: a terminator you can only
    sometimes rely on doesn't terminate anything. ``stopped`` words a cancelled
    turn instead, resolving the transient "(cancelling…)" the same way.
    """
    clock = time.strftime("%H:%M", time.localtime(finished))
    duration = format_duration(seconds)
    if stopped:
        return f"{_GRAY}⊘ stopped after {duration} · {clock}{_RESET}"
    return f"{_GRAY}· done in {duration} · {clock}{_RESET}"


def render_command_expansion(name: str, path=None) -> str:
    """One dim line naming the user-defined command that produced this prompt.

    The turn that follows asks the FILE's question, not the line the user typed,
    so without this the transcript (and the ``/export`` of it) shows an answer to
    a question that appears nowhere. The file is named because that's what you
    edit when the expansion wasn't what you meant.
    """
    where = f" · {path.name}" if path is not None else ""
    return f"{_GRAY}⌘ /{name}{where}{_RESET}"


def render_mention_notice(label: str) -> str:
    """One dim line per ``@path`` a prompt pulled in — or failed to.

    A mention attaches a file's contents silently, so without this the only
    evidence is the answer itself, and a typo'd path (nothing attached, the model
    guessing) looks exactly like a correct one. It is also where a truncated file
    says so. The label already leads with the ``@``, which is marker enough.
    """
    return f"{_GRAY}{label}{_RESET}"


def render_hook_notice(text: str) -> str:
    """One dim line for something a user hook did (fired, blocked, errored).

    Hooks are invisible by nature — a command in a config file, firing on a tool
    call the user didn't watch — so every consequence of one gets a line here.
    """
    return f"  {_GRAY}⇢ {' '.join(text.split())}{_RESET}"


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

    Uses ``CodeFormatter.render_to_string`` (a per-call sink + its own parser),
    so it is safe to call on the UI thread even while a live turn is streaming to
    stdout on the worker thread — no process-global ``redirect_stdout`` and no
    shared-parser race.
    """
    if not text:
        return ""
    try:
        return CodeFormatter.render_to_string(text).rstrip("\n")
    except Exception:
        return text  # never let rendering break a load


def render_agent_detail(run) -> str:
    """Full transcript for the agent-detail view, from a captured ActivityRun.

    Reproduces the main-thread look: each tool call as a ``ToolName``/``↳ arg``
    block, each result/error as a dimmed line, and the final answer through the
    same markdown renderer. ``run`` is an ``agent_activity.ActivityRun`` (or any
    object exposing ``agent_type``/``description``/``origin``/``status``/
    ``events`` where each event has ``kind``/``name``/``args``/``text``). Pure:
    returns a string, no I/O — safe to build on the UI thread.
    """
    status = getattr(run, "status", "?")
    dot = {
        "running": "\033[33m●",
        "done": "\033[32m✓",
        "failed": "\033[31m✗",
        "stopped": "\033[31m✗",
    }.get(status, "○")
    head = (
        f"{_HEADER}{_BOLD}{getattr(run, 'agent_type', 'agent')}{_RESET} "
        f"{_GRAY}[{getattr(run, 'origin', '')}]{_RESET}  {dot} {status}{_RESET}"
    )
    desc = getattr(run, "description", "")
    out = [head]
    if desc:
        out.append(f"{_GRAY}{desc}{_RESET}")
    out.append("")
    final_text = ""
    for ev in getattr(run, "events", []) or []:
        kind = getattr(ev, "kind", "")
        if kind == "tool_call":
            out.append(render_tool_call(getattr(ev, "name", "") or "tool", getattr(ev, "args", None) or {}))
        elif kind == "tool_result":
            out.append(f"  {_GRAY}{_CONNECTOR} {getattr(ev, 'text', '')}{_RESET}")
        elif kind == "tool_error":
            out.append(f"  {_DEL}✗ {getattr(ev, 'name', '')}: {getattr(ev, 'text', '')}{_RESET}")
        elif kind == "final":
            final_text = getattr(ev, "text", "") or ""
    if final_text:
        out.append("")
        out.append(f"{_ANSWER_MARKER}{render_markdown(final_text)}")
    if status == "running":
        cancelling = hasattr(run, "is_cancelling") and run.is_cancelling()
        out.append("")
        out.append(f"{_GRAY}({'cancelling…' if cancelling else 'running…'}){_RESET}")
    return "\n".join(out)


def render_conversation(messages: list) -> str:
    """Replay loaded messages to a scrollback transcript: user prompts, 
    ``Thought for…`` blocks, tool calls, and answers.

    Duck-types LangChain messages by attributes (no import) so this stays pure:
    HumanMessage → ``> text``; AIMessage → reasoning block + tool blocks +
    ``● answer``; ToolMessage results are omitted (their call is already shown).

    Matches on the class name AND its bases, because a *streamed* reply is an
    ``AIMessageChunk`` (an ``AIMessage`` subclass): an exact-name check drops it.
    Today's callers only replay decoded history (plain ``AIMessage``), so this is
    a guard against a caller that hands over live messages, not an active fix.
    """
    out = []
    for msg in messages:
        names = {c.__name__ for c in type(msg).__mro__}
        cls = (
            "HumanMessage" if "HumanMessage" in names
            else "AIMessage" if "AIMessage" in names
            else type(msg).__name__
        )
        content = getattr(msg, "content", "")
        if cls == "HumanMessage":
            # Show only what the user typed: a stored prompt still carries the
            # prepended episodic-memory block, which otherwise replayed as a wall
            # of tool names above the first message on every resume.
            text = user_prompt_text(_visible_text(content))
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
