"""Session token accounting (``/usage``).

Adds up what the providers *report* — input, output and (where offered) cache
tokens — per model, for the life of the process. Every model call in the app
funnels its response through :meth:`UsageTracker.record`, including the ones the
user never sees: sub-agents, orchestrator workers, the query router, and the
compaction summarizer. Those are exactly the ones worth surfacing, since a
delegated task can spend more than the visible turn that triggered it.

**No cost estimate, deliberately.** A price table would be wrong more often than
right here: Ollama and a local OpenAI-compatible server are marginal-free,
SageMaker bills by endpoint-hour rather than by token, and LiteLLM proxies an open
set of models at prices this process cannot know. A confidently wrong dollar figure
is worse than none, so this reports tokens and says whose numbers they are.

**Reported, not measured.** ``usage_metadata`` is normalized by LangChain but not
universally populated — several providers omit it, or fill only part of it. A call
whose response carries no usage is counted separately (``calls_without_usage``) so
a partial total can never masquerade as a complete one.

Thread-safe: sub-agents and orchestrator waves record concurrently from pool
threads.
"""

import threading
from typing import Any, Dict, List, Optional


class ModelUsage:
    """Running totals for one model."""

    __slots__ = (
        "model", "calls", "input_tokens", "output_tokens",
        "cache_read", "cache_write", "calls_without_usage",
    )

    def __init__(self, model: str):
        self.model = model
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read = 0
        self.cache_write = 0
        self.calls_without_usage = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "calls_without_usage": self.calls_without_usage,
        }


def _usage_of(response: Any) -> Optional[Dict[str, Any]]:
    """The ``usage_metadata`` mapping from a response, or None when absent."""
    um = getattr(response, "usage_metadata", None)
    return um if isinstance(um, dict) and um else None


def _int(value: Any) -> int:
    """Coerce a reported count to a non-negative int (providers vary)."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


# Per-TTL cache-write keys. LangChain's normalized `cache_creation` is NOT filled
# by every provider: both Bedrock and Mantle reported a write of several thousand
# tokens under these keys with `cache_creation: 0`, which showed a prompt-cache
# session as "0 written" — the one number that proves caching engaged.
_CACHE_WRITE_KEYS = ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")


def _cache_write(details: Dict[str, Any]) -> int:
    """Cache-creation tokens from a report, per-TTL keys as the fallback."""
    raw = details.get("cache_creation")
    if isinstance(raw, dict):  # some providers nest the per-TTL breakdown here
        return sum(_int(v) for v in raw.values())
    normalized = _int(raw)
    if normalized:
        return normalized
    return sum(_int(details.get(k)) for k in _CACHE_WRITE_KEYS)


class UsageTracker:
    """Accumulates reported token usage per model for this session."""

    def __init__(self):
        self._lock = threading.Lock()
        self._by_model: Dict[str, ModelUsage] = {}

    def record(self, response: Any, model: str = "") -> None:
        """Add one model call's reported usage. Never raises.

        Called from the streaming/invoke paths, so a bad response shape or an
        unexpected provider payload must not break the turn it belongs to.
        """
        try:
            key = str(model or "unknown")
            um = _usage_of(response)
            with self._lock:
                entry = self._by_model.get(key)
                if entry is None:
                    entry = self._by_model[key] = ModelUsage(key)
                entry.calls += 1
                if um is None:
                    # Counted, but not silently folded into the totals as zeros.
                    entry.calls_without_usage += 1
                    return
                entry.input_tokens += _int(um.get("input_tokens"))
                entry.output_tokens += _int(um.get("output_tokens"))
                details = um.get("input_token_details")
                if isinstance(details, dict):
                    entry.cache_read += _int(details.get("cache_read"))
                    entry.cache_write += _cache_write(details)
        except Exception:  # noqa: BLE001 — accounting must never break a turn
            pass

    def snapshot(self) -> List[Dict[str, Any]]:
        """Per-model totals, busiest first (a stable order for display)."""
        with self._lock:
            rows = [u.as_dict() for u in self._by_model.values()]
        rows.sort(key=lambda r: (-r["total_tokens"], r["model"]))
        return rows

    def totals(self) -> Dict[str, Any]:
        """Aggregate across every model."""
        rows = self.snapshot()
        return {
            "calls": sum(r["calls"] for r in rows),
            "input_tokens": sum(r["input_tokens"] for r in rows),
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "total_tokens": sum(r["total_tokens"] for r in rows),
            "cache_read": sum(r["cache_read"] for r in rows),
            "cache_write": sum(r["cache_write"] for r in rows),
            "calls_without_usage": sum(r["calls_without_usage"] for r in rows),
            "models": len(rows),
        }

    def reset(self) -> None:
        """Clear all counters (``/clear`` starts a fresh conversation)."""
        with self._lock:
            self._by_model.clear()


def _fmt(n: int) -> str:
    """Thousands-separated count."""
    return f"{n:,}"


def render(tracker: UsageTracker, context_tokens: int = 0) -> str:
    """Render the ``/usage`` report as plain text.

    Kept out of the tracker so it stays a pure data structure, and so the report
    is unit-testable without a terminal.
    """
    rows = tracker.snapshot()
    if not rows:
        # No calls made YET — but a resumed or /load-ed conversation already has a
        # context, and reporting "ask something first" while several thousand tokens
        # sit loaded reads as a bug. Say why the spend is zero, and still show the
        # context, which is the number that actually applies to the next turn.
        lines = ["No tokens spent yet in this session."]
        if context_tokens:
            lines.append(
                f"  Current context: {_fmt(context_tokens)} tokens "
                "(what the next turn re-sends)"
            )
            lines.append(
                "  A restored conversation carries its context but no spend — "
                "totals count only\n  the calls THIS session makes."
            )
        else:
            lines.append("  Ask something and the totals appear here.")
        return "\n".join(lines)
    agg = tracker.totals()

    out = ["Token usage this session (as reported by the provider)", ""]
    for r in rows:
        out.append(f"  {r['model']}")
        out.append(
            f"    {_fmt(r['calls'])} call{'s' if r['calls'] != 1 else ''}  ·  "
            f"in {_fmt(r['input_tokens'])}  ·  out {_fmt(r['output_tokens'])}  ·  "
            f"total {_fmt(r['total_tokens'])}"
        )
        if r["cache_read"] or r["cache_write"]:
            out.append(
                f"    cache: {_fmt(r['cache_read'])} read  ·  "
                f"{_fmt(r['cache_write'])} written"
            )
        if r["calls_without_usage"]:
            out.append(
                f"    {_fmt(r['calls_without_usage'])} call"
                f"{'s' if r['calls_without_usage'] != 1 else ''} reported no usage"
            )
    if agg["models"] > 1:
        out.append("")
        out.append(
            f"  All models: {_fmt(agg['calls'])} calls  ·  "
            f"in {_fmt(agg['input_tokens'])}  ·  out {_fmt(agg['output_tokens'])}"
            f"  ·  total {_fmt(agg['total_tokens'])}"
        )
    if context_tokens:
        out.append("")
        out.append(
            f"  Current context: {_fmt(context_tokens)} tokens "
            "(what the next turn re-sends)"
        )
    out.append("")
    out.append(
        "  Totals are cumulative for this session and include work you don't see "
        "(sub-agents,\n  routing, compaction) — not the size of your conversation. "
        "No cost is shown: token\n  pricing doesn't apply uniformly across the "
        "supported providers."
    )
    if agg["calls_without_usage"]:
        out.append(
            f"  {_fmt(agg['calls_without_usage'])} call(s) returned no usage data, "
            "so the totals above are a lower bound."
        )
    return "\n".join(out)
