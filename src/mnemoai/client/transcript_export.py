"""Render a conversation to a shareable Markdown/plain-text transcript (``/export``).

Distinct from ``/save``: that writes re-importable JSON for ``/load``. This is a
**one-way human-readable artifact** — the thing you paste into a bug report, a PR
description, or a message to a colleague. Nothing here is designed to be read
back, so it favors legibility over fidelity.

Pure string building (no I/O), so it is unit-testable and safe to call from any
thread. Deliberately emits **no ANSI** — an escape-laden file is useless in a bug
report, which is exactly why this doesn't reuse ``turn_view.render_conversation``.

Message kinds are matched with ``isinstance``, NOT by class name: a *streamed*
reply lands in history as an ``AIMessageChunk`` (an ``AIMessage`` subclass), so an
exact name check silently dropped every assistant answer and exported a file of
nothing but user prompts.

What's included, and why: user prompts and assistant answers in full (the
conversation), tool calls as a one-line summary with their arguments (what the
assistant did). What's left out: tool RESULTS (often thousands of lines of file
content, and the single biggest source of noise), reasoning blocks unless asked
for (``include_reasoning``), and injected context — the steering/plan blocks and
the prepended episodic-memory block are stripped, because the user never typed
them and they'd dominate the transcript.
"""

import re
from datetime import datetime
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mnemoai.client.ui import turn_view

# Tool args worth showing inline. A tool call's value in a transcript is "what did
# it do", and a whole file body pasted into an argument defeats that.
_MAX_ARG_CHARS = 200

# Args that are pure payload — their content is the file being written, which the
# following diff/answer already conveys and which would swamp the transcript.
_BULKY_ARGS = {"content", "file_text", "new_str", "old_str", "text", "body"}

_FORMATS = ("md", "txt")


def _clean_user_text(text: str) -> str:
    """Strip injected context from a stored user prompt (see module docstring).

    Shares :func:`turn_view.user_prompt_text` with the replay renderer and the
    ``--resume`` picker so all three agree on what the user actually typed.
    """
    return turn_view.user_prompt_text(text)


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


def _format_args(args: Any) -> str:
    """One-line argument summary for a tool call, bulky values elided."""
    if not isinstance(args, dict) or not args:
        return ""
    parts = []
    for key, value in args.items():
        if key in _BULKY_ARGS:
            body = str(value)
            parts.append(f"{key}=<{len(body)} chars>")
            continue
        flat = " ".join(str(value).split())
        if len(flat) > _MAX_ARG_CHARS:
            flat = flat[: _MAX_ARG_CHARS - 1] + "…"
        parts.append(f"{key}={flat}")
    return ", ".join(parts)


def render(
    messages: List[Any],
    fmt: str = "md",
    *,
    title: str = "",
    model: str = "",
    include_reasoning: bool = False,
    started: Optional[float] = None,
) -> str:
    """Render ``messages`` to a shareable transcript.

    ``fmt`` is ``"md"`` (headings + fenced answers) or ``"txt"`` (plain, no
    markup). Both are ANSI-free. Returns "" for a conversation with nothing
    visible in it, so a caller can refuse to write an empty file.
    """
    fmt = (fmt or "md").lower().lstrip(".")
    if fmt not in _FORMATS:
        fmt = "md"
    md = fmt == "md"

    body: List[str] = []
    for msg in messages or []:
        if isinstance(msg, ToolMessage):
            continue  # results are noise in a transcript; the CALL is recorded
        if isinstance(msg, HumanMessage):
            text = _clean_user_text(_text_of(getattr(msg, "content", "")))
            if not text:
                continue  # tool-result-only / pure-injection message
            body.append(f"### User\n\n{text}" if md else f"USER\n\n{text}")
        elif isinstance(msg, AIMessage):
            chunk: List[str] = []
            if include_reasoning:
                reasoning = (getattr(msg, "additional_kwargs", {}) or {}).get(
                    "reasoning_content"
                )
                if reasoning and str(reasoning).strip():
                    r = str(reasoning).strip()
                    chunk.append(
                        f"<details><summary>Reasoning</summary>\n\n{r}\n\n</details>"
                        if md else f"[reasoning]\n{r}"
                    )
            for tc in getattr(msg, "tool_calls", None) or []:
                name = tc.get("name", "tool")
                args = _format_args(tc.get("args"))
                line = f"{name}({args})" if args else f"{name}()"
                chunk.append(f"- `{line}`" if md else f"  [tool] {line}")
            answer = _text_of(getattr(msg, "content", ""))
            if answer:
                chunk.append(answer)
            if chunk:
                head = "### Assistant\n\n" if md else "ASSISTANT\n\n"
                body.append(head + "\n\n".join(chunk))

    if not body:
        return ""

    when = datetime.fromtimestamp(started) if started else datetime.now()
    head: List[str] = []
    if md:
        head.append(f"# {title}" if title else "# Conversation")
        meta = [when.strftime("%Y-%m-%d %H:%M")]
        if model:
            meta.append(f"model: {model}")
        head.append("_" + " · ".join(meta) + "_")
    else:
        head.append(title or "Conversation")
        meta = [when.strftime("%Y-%m-%d %H:%M")]
        if model:
            meta.append(f"model: {model}")
        head.append(" · ".join(meta))
        head.append("=" * 60)

    sep = "\n\n---\n\n" if md else "\n\n" + "-" * 60 + "\n\n"
    return "\n\n".join(head) + "\n\n" + sep.join(body) + "\n"


def suggest_filename(messages: List[Any], fmt: str = "md", when=None) -> str:
    """A timestamped, slugged default filename derived from the first prompt.

    Mirrors the ``--resume`` picker's labelling: the file is identifiable at a
    glance in a directory listing instead of being one of many
    ``conversation_<ts>`` files.
    """
    fmt = (fmt or "md").lower().lstrip(".")
    if fmt not in _FORMATS:
        fmt = "md"
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    slug = ""
    for msg in messages or []:
        if not isinstance(msg, HumanMessage):
            continue
        text = _clean_user_text(_text_of(getattr(msg, "content", "")))
        if not text:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40].strip("-")
        break
    return f"conversation_{stamp}{'_' + slug if slug else ''}.{fmt}"
