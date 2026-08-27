"""The pinned footer line: what the NEXT turn costs, always in view.

Replaces the per-turn ``[Context: N tokens]`` notice — a number that mattered
every turn but scrolled away with the turn that printed it, and read as debug
output. Here it sits under the input, with a meter that colors amber past 70% of
the window and red past 90%, beside the model and directory the session is
running in.

Pure formatting + layout (no prompt_toolkit, no terminal I/O): the caller passes
the numbers and gets styled segments back, so the whole line is unit-testable.
"""

import math
import os
import re

# Style classes the pinned app registers (see ``tui._pinned_style``).
OK = "class:pinned-footer"
MODEL = "class:pinned-footer-model"
WARN = "class:pinned-footer-warn"
CRIT = "class:pinned-footer-crit"

_WARN_AT = 0.70
_CRIT_AT = 0.90
_CELLS = 8
_FILLED = "▓"
_EMPTY = "░"
_GAP = 2  # minimum spaces between two groups
_MAX_MODEL = 28

_VERSION_SUFFIX = re.compile(r"-v\d+(?::\d+)?$")
_DATE_SUFFIX = re.compile(r"-\d{6,8}$")


def short_model_name(name: str, limit: int = _MAX_MODEL) -> str:
    """``us.anthropic.claude-opus-5-20260514-v1:0`` → ``claude-opus-5``.

    Drops the provider prefix and the date/version tails a footer has no room
    for. A dotted tail that STARTS WITH A DIGIT is part of the model's own
    version (``llama3.1:8b``), not a provider prefix, so it is left alone.
    """
    label = (name or "").strip().rsplit("/", 1)[-1]
    tail = label.rsplit(".", 1)[-1]
    if tail and not tail[0].isdigit():
        label = tail
    label = _DATE_SUFFIX.sub("", _VERSION_SUFFIX.sub("", label))
    if len(label) > limit:
        label = label[: max(1, limit - 1)] + "…"
    return label


def format_tokens(n: int) -> str:
    """Compact count: 512 → ``512``, 90096 → ``90.1k``, 1166221 → ``1.17M``."""
    n = max(0, int(n or 0))
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.2f}M"


def meter(fraction: float, cells: int = _CELLS) -> str:
    """``▓░░░░░░░`` — at least one cell for any nonzero usage, never over-full."""
    f = max(0.0, min(1.0, float(fraction or 0.0)))
    filled = 0 if f <= 0 else max(1, math.ceil(f * cells))
    return _FILLED * filled + _EMPTY * (cells - filled)


def level(fraction: float) -> str:
    """Style class for a fill fraction: dim, amber past 70%, red past 90%."""
    f = max(0.0, float(fraction or 0.0))
    if f >= _CRIT_AT:
        return CRIT
    if f >= _WARN_AT:
        return WARN
    return OK


def home_path(path: str) -> str:
    """``/Users/me/dev/x`` → ``~/dev/x`` (a footer has no room for $HOME)."""
    path = str(path or "")
    home = os.path.expanduser("~")
    if home and home != os.sep and path.startswith(home):
        return "~" + path[len(home) :]
    return path


def context_group(tokens: int, window: int, estimated: bool = False) -> tuple:
    """``(text, style)`` for the context readout: meter, count, % of the window.

    ``estimated`` marks the count with ``~``: before a turn has run there is no
    provider-reported ``input_tokens``, and the local estimate over-counts — a
    persistent number must say so rather than quietly showing the wrong one.
    """
    tokens = max(0, int(tokens or 0))
    if not tokens:
        return "—", OK
    count = ("~" if estimated else "") + format_tokens(tokens)
    if not window or window <= 0:
        return count, OK
    fraction = tokens / float(window)
    pct = f"{fraction * 100:.0f}%"
    if pct == "0%":
        pct = "<1%"
    return f"{meter(fraction)} {count} · {pct}", level(fraction)


def _variants(model: str, provider: str, cwd: str):
    """Layouts from widest to narrowest — the first that fits is used."""
    path = home_path(cwd)
    base = os.path.basename(path.rstrip(os.sep))
    short = f"…{os.sep}{base}" if base and base != path else path
    yield model, provider, path
    yield model, provider, short
    yield model, provider, ""
    yield model, "", ""
    yield "", "", ""


def segments(
    *,
    model: str = "",
    provider: str = "",
    cwd: str = "",
    tokens: int = 0,
    window: int = 0,
    estimated: bool = False,
    width: int = 80,
) -> list:
    """Styled ``(class, text)`` segments for the footer, right group flush right.

    Degrades by dropping groups (full path → basename → no path → no provider)
    until the line fits ``width``, so a narrow terminal loses the context least
    likely to be missed rather than wrapping the bar onto a second line.
    """
    right_text, right_cls = context_group(tokens, window, estimated)
    width = max(1, int(width or 1))
    name = short_model_name(model)

    for label, prov, mid in _variants(name, (provider or "").strip().lower(), cwd):
        groups = []
        if label:
            group = [(MODEL, label)]
            if prov:
                group.append((OK, f" · {prov}"))
            groups.append(group)
        if mid:
            groups.append([(OK, mid)])
        groups.append([(right_cls, right_text)])

        used = sum(len(text) for group in groups for _, text in group)
        gaps = len(groups) - 1
        slack = width - used - _GAP * gaps
        if slack < 0:
            continue
        pads = []
        if gaps:
            share, extra = divmod(slack, gaps)
            pads = [_GAP + share] * gaps
            pads[-1] += extra

        out = []
        for i, group in enumerate(groups):
            out.extend(group)
            if i < gaps:
                out.append((OK, " " * pads[i]))
        return out

    # Narrower than the readout itself: show what fits, never wrap.
    return [(right_cls, right_text[:width])]
