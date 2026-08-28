"""Getting the last answer out of the terminal (`/copy`).

An answer that has to leave the terminal — into a commit message, an editor, a
ticket — is currently extracted with the mouse, and a mouse selection over a
streamed reply takes the wrapping with it: hard line breaks where the terminal
wrapped, the pinned footer, half a tool block. The text is right here in the
conversation; `/copy` puts the exact string on the clipboard, and `/copy code`
narrows it to the last fenced block, which is the part most often wanted alone
(`/copy 2` reaches an earlier answer).

**Two transports, and the order matters more than it looks.** A local helper
(`pbcopy`, `wl-copy`, `xclip`, `xsel`, `clip.exe`) is exact and unlimited; **OSC
52** asks the terminal emulator itself to set the clipboard, which is the only
mechanism that works **over SSH** — there, a helper would either not exist or
copy to the clipboard of the machine nobody is sitting at, so an SSH session
tries OSC 52 FIRST. Inside tmux/screen the sequence needs the same DCS
passthrough as a notification, so both share `notify.wrap_for_multiplexer`.

OSC 52 is also the one with limits worth naming: terminals bound the sequence
length (and tmux ignores it entirely unless `set-clipboard on`), so a large
payload reports what happened instead of silently truncating.

Extraction is pure (`answer_text`, `last_code_block`) and separate from the
transport, so what gets copied is unit-testable without a terminal or a
clipboard.
"""

import base64
import os
import shutil
import subprocess
import sys
from typing import Any, List, Optional, Tuple

from langchain_core.messages import AIMessage

from mnemoai.client.ui import notify

# The local clipboard helpers, in preference order per platform. `clip.exe` is
# last because it is reachable from WSL, where a Linux helper is more likely right.
_HELPERS = (
    ("pbcopy", []),
    ("wl-copy", []),
    ("xclip", ["-selection", "clipboard"]),
    ("xsel", ["--clipboard", "--input"]),
    ("clip.exe", []),
)

# A helper that hangs (an X server that isn't answering) must not hang the REPL.
_TIMEOUT = 5

# Past this, OSC 52 is refused rather than sent: terminals cap the sequence and a
# truncated clipboard is worse than a clear refusal. Local helpers have no limit.
_MAX_OSC52_CHARS = 100_000

_FENCES = ("```", "~~~")


def _text_of(content: Any) -> str:
    """Visible text from a message's content (string or provider block list)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and (b.get("type") == "text" or "text" in b)
        ).strip()
    return ""


def answer_text(messages: List[Any], back: int = 1) -> str:
    """The text of the ``back``-th most recent assistant answer ("" if none).

    ``isinstance``, never a class-name check: a streamed reply is an
    ``AIMessageChunk`` (an ``AIMessage`` subclass), and matching on the name is
    the bug that once exported a transcript containing no answers at all.
    Tool-call-only messages carry no text and are skipped, so "the last answer"
    means the last thing the user was actually shown.
    """
    if back < 1:
        back = 1
    seen = 0
    for msg in reversed(messages or []):
        if not isinstance(msg, AIMessage):
            continue
        text = _text_of(getattr(msg, "content", ""))
        if not text:
            continue
        seen += 1
        if seen == back:
            return text
    return ""


def last_code_block(text: str) -> Tuple[str, str]:
    """``(code, language)`` of the LAST fenced block in ``text`` ("" if none).

    Pure. Handles both fence styles and an UNTERMINATED fence — a turn cut short
    mid-block still has code worth copying, and the alternative is telling the
    user there is none while it's on the screen in front of them.
    """
    lines = (text or "").split("\n")
    blocks: List[Tuple[List[str], str]] = []
    fence: Optional[str] = None
    language = ""
    body: List[str] = []
    for line in lines:
        stripped = line.strip()
        if fence is None:
            for mark in _FENCES:
                if stripped.startswith(mark):
                    fence = mark
                    language = stripped[len(mark):].strip().split()[0:1]
                    language = language[0] if language else ""
                    body = []
                    break
            continue
        if stripped.startswith(fence):
            blocks.append((body, language))
            fence = None
            continue
        body.append(line)
    if fence is not None and body:
        blocks.append((body, language))  # unterminated: keep what's there
    if not blocks:
        return "", ""
    code, lang = blocks[-1]
    return "\n".join(code).strip("\n"), lang


def _over_ssh() -> bool:
    """Are we on the far end of an SSH session? (Then OSC 52 goes first.)"""
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))


def _helper_copy(text: str) -> Optional[str]:
    """Copy via the first available local helper; its name, or None."""
    for name, args in _HELPERS:
        path = shutil.which(name)
        if not path:
            continue
        try:
            proc = subprocess.run(
                [path, *args],
                input=text,
                text=True,
                capture_output=True,
                timeout=_TIMEOUT,
            )
        except (OSError, subprocess.SubprocessError):
            continue  # try the next one: an unusable helper is not an answer
        if proc.returncode == 0:
            return name
    return None


def osc52_sequence(text: str) -> str:
    """The OSC 52 sequence that sets the terminal's clipboard to ``text``."""
    payload = base64.b64encode(text.encode("utf-8", "replace")).decode("ascii")
    return notify.wrap_for_multiplexer(
        f"\033]52;c;{payload}\007",
        tmux=bool(os.environ.get("TMUX")),
        screen=os.environ.get("TERM", "").startswith("screen")
        and not os.environ.get("TMUX"),
    )


def _osc52_copy(text: str) -> Optional[str]:
    """Ask the terminal to set the clipboard; ``"the terminal"``, or None.

    There is no reply to an OSC 52 write, so "sent" is the most this can honestly
    claim — hence it is the FALLBACK everywhere except SSH, where it is the only
    thing that reaches the machine the user is sitting at.
    """
    if len(text) > _MAX_OSC52_CHARS:
        return None
    try:
        if not sys.stdout.isatty():
            return None
        sys.stdout.write(osc52_sequence(text))
        sys.stdout.flush()
    except Exception:  # noqa: BLE001 — a failed copy must not break the REPL
        return None
    return "the terminal"


def copy(text: str) -> Tuple[bool, str]:
    """Put ``text`` on the clipboard. Returns ``(ok, how)``.

    Over SSH the terminal is tried first (a local helper there would target the
    wrong machine); otherwise a helper first, because it is exact and unbounded.
    """
    if not text:
        return False, ""
    order = (_osc52_copy, _helper_copy) if _over_ssh() else (_helper_copy, _osc52_copy)
    for attempt in order:
        how = attempt(text)
        if how:
            return True, how
    return False, ""


def _describe(text: str, what: str, how: str) -> str:
    """The one-line confirmation: what was copied, how much of it, and where to."""
    lines = text.count("\n") + 1
    size = f"{lines} line{'s' if lines != 1 else ''}, {len(text)} char{'s' if len(text) != 1 else ''}"
    return f"Copied {what} to the clipboard ({size}) via {how}."


def report(client: Any, arg: str = "") -> str:
    """``/copy [code|N]``: copy the last answer (or its last code block).

    Returns the line to print. Never raises — a clipboard is a convenience, and a
    failed copy has to say so rather than look like it worked.
    """
    try:
        arg = (arg or "").strip().lower()
        back, want_code = 1, False
        if arg in ("code", "block"):
            want_code = True
        elif arg.isdigit():
            back = max(1, int(arg))
        elif arg:
            return f"Unknown option: {arg}  ·  /copy [code|N]"

        agent = getattr(client, "agent", None)
        answer = answer_text(getattr(agent, "messages", None) or [], back=back)
        if not answer:
            which = "an answer" if back == 1 else f"{back} answers back"
            return f"Nothing to copy — this conversation has no {which} yet."

        if want_code:
            code, language = last_code_block(answer)
            if not code:
                return "That answer has no code block — /copy copies the whole answer."
            ok, how = copy(code)
            what = f"the last {language} block" if language else "the last code block"
        else:
            ok, how = copy(answer)
            what = "the answer" if back == 1 else f"answer -{back}"

        if not ok:
            return (
                "Could not reach a clipboard (no pbcopy/wl-copy/xclip/xsel, and the "
                "terminal refused OSC 52).\n  /export writes the conversation to a "
                "file instead."
            )
        return _describe(code if want_code else answer, what, how)
    except Exception as e:  # noqa: BLE001
        return f"Copy failed: {e}"
