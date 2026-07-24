"""Live activity store for hidden sub-agent runs (the pinned-TUI "agents" panel).

Foreground spawns, background spawns, and orchestrator subtask workers all run
``quiet`` — their model output and tool calls are deliberately suppressed so
they don't corrupt the main display. That leaves the user blind to what those
agents are doing. This module is the thread-safe capture layer: each hidden run
opens an :class:`ActivitySink`, appends a bounded stream of tool-call / result /
error / final events as it works, and the UI reads immutable snapshots to paint
a live panel + a per-agent detail view.

Pure state + threading, no LLM/agent logic (mirrors ``background_agents.py`` and
``skill_store``: data separated from the agent runner). Distinct from
``BackgroundAgentRegistry`` — that owns result delivery/resume for *background*
spawns only and persists to disk; this owns the *live feed* for ALL three hidden
classes and is in-memory only. The writers are the sub-agent worker/daemon/pool
threads; the reader is the UI event-loop thread.

Locking discipline: ONE ``threading.Lock`` guards every field, held only for the
O(1) body of each method (a deque append or a shallow copy) — never across a
model call, tool call, or I/O. It is a leaf lock (nested under no other), so it
cannot deadlock with the UI read and never serializes real agent work. The
optional ``on_change`` callback is invoked OUTSIDE the lock so it can't re-enter.
"""

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional

# Bounds so a long session / chatty agent can't grow memory without limit.
MAX_RUNS = 32  # retained runs (running always kept; oldest FINISHED evicted first)
EVENTS_PER_RUN = 200  # ring-buffered activity events per run
_TEXT_CAP = 300  # tool result/error text truncated to this in the live feed


@dataclass(frozen=True)
class ActivityEvent:
    """One captured event in a hidden agent's run."""

    kind: str  # "tool_call" | "tool_result" | "tool_error" | "final"
    at: float  # time.monotonic()
    name: Optional[str] = None  # tool name (tool_call/result/error)
    args: Optional[dict] = None  # normalized tool args (tool_call only)
    text: Optional[str] = None  # short result/error, or the final answer


@dataclass
class ActivityRun:
    """One hidden sub-agent run and its live event stream."""

    run_id: str
    agent_type: str
    description: str
    origin: str  # "spawn" | "background" | "orchestrator"
    start: float
    status: str = "running"  # running | done | failed | stopped
    end: Optional[float] = None
    events: Deque[ActivityEvent] = field(default_factory=lambda: deque(maxlen=EVENTS_PER_RUN))
    # Per-run cooperative stop: set by the UI (x / stop-all), polled by the
    # worker loop so ANY agent — foreground or background — can be stopped
    # individually. Shared by the live run AND its snapshot copies (same object),
    # so a stop request against a snapshot's run_id reaches the live worker.
    cancel: "threading.Event" = field(default_factory=lambda: threading.Event())

    def elapsed(self, now: float) -> float:
        """Seconds since start (frozen at ``end`` once finished)."""
        return (self.end if self.end is not None else now) - self.start

    def tool_call_count(self) -> int:
        return sum(1 for e in self.events if e.kind == "tool_call")

    def is_cancelling(self) -> bool:
        """True in the window between a stop request (cancel set) and the worker
        actually returning (status flips to 'stopped'). Drives a live
        'cancelling…' label so the user sees the stop is in progress."""
        return self.status == "running" and self.cancel.is_set()


class ActivitySink:
    """Per-run handle the worker loop writes through. All methods no-op when the
    store is None (plain loop / tests / the main foreground turn), so the quiet
    loop is untouched when there's no UI listening."""

    def __init__(self, store: Optional["AgentActivityStore"], run_id: str) -> None:
        self._store = store
        self._run_id = run_id

    def tool_call(self, name: str, args: Optional[dict]) -> None:
        if self._store is None:
            return
        # Defensive shallow copy so a later mutation of the caller's dict can't
        # change an already-recorded event.
        self._store._record(
            self._run_id,
            ActivityEvent("tool_call", time.monotonic(), name=name, args=dict(args or {})),
        )

    def tool_result(self, name: str, text: str) -> None:
        if self._store is None:
            return
        self._store._record(
            self._run_id,
            ActivityEvent("tool_result", time.monotonic(), name=name, text=(text or "")[:_TEXT_CAP]),
        )

    def tool_error(self, name: str, text: str) -> None:
        if self._store is None:
            return
        self._store._record(
            self._run_id,
            ActivityEvent("tool_error", time.monotonic(), name=name, text=(text or "")[:_TEXT_CAP]),
        )

    def final(self, text: str) -> None:
        if self._store is None:
            return
        # Full answer kept (the detail view renders it as markdown); only the
        # per-tool result/error lines are capped.
        self._store._record(
            self._run_id, ActivityEvent("final", time.monotonic(), text=text or "")
        )

    def finish(self, status: str) -> None:
        if self._store is None:
            return
        self._store._finish(self._run_id, status)

    def finish_ok(self, final_text: str) -> None:
        """Finish a run that returned normally: mark 'stopped' if it was cancelled
        mid-run (x / stop-all), else record ``final_text`` and mark 'done'.

        Encapsulates the identical success-path transition every hidden run does
        (foreground/background spawns, orchestrator subtasks). The except/finally
        failure paths stay explicit at each call site (they differ per caller)."""
        if self.is_cancelled():
            self.finish("stopped")
        else:
            self.final(final_text)
            self.finish("done")

    def finish_if_running(self, status: str) -> None:
        """Finish the run ONLY if it's still 'running' — an idempotent backstop
        for abnormal exits (e.g. a KeyboardInterrupt cancel) that never leaves a
        run counting forever, without clobbering a real done/failed status."""
        if self._store is None:
            return
        self._store._finish_if_running(self._run_id, status)

    def is_cancelled(self) -> bool:
        """True once the UI asked THIS run to stop (x / stop-all). Polled by the
        worker loop so any single agent can be stopped on demand."""
        if self._store is None:
            return False
        return self._store.is_cancelled(self._run_id)


class AgentActivityStore:
    """Thread-safe, in-memory, bounded store of hidden-agent runs for one session."""

    def __init__(self, max_runs: int = MAX_RUNS) -> None:
        self._lock = threading.Lock()
        self._runs: "OrderedDict[str, ActivityRun]" = OrderedDict()
        self._counter = 0
        self._max_runs = max_runs
        # Wired by the UI (TTY only) to force an immediate repaint on change;
        # invoked OUTSIDE the lock. None off-TTY / in tests.
        self.on_change: Optional[Callable[[], None]] = None

    def open_run(self, agent_type: str, description: str, origin: str) -> ActivitySink:
        """Register a new running hidden agent and return its write sink."""
        with self._lock:
            self._counter += 1
            run_id = f"{origin}-{self._counter}"
            self._runs[run_id] = ActivityRun(
                run_id=run_id,
                agent_type=agent_type,
                description=description,
                origin=origin,
                start=time.monotonic(),
            )
            self._evict_locked()
        self._fire_change()
        return ActivitySink(self, run_id)

    def _record(self, run_id: str, event: ActivityEvent) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:  # evicted mid-run — drop silently
                return
            run.events.append(event)
        self._fire_change()

    def _finish(self, run_id: str, status: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.status = status
            run.end = time.monotonic()
        self._fire_change()

    def _finish_if_running(self, run_id: str, status: str) -> None:
        """Set status/end only if the run is still 'running' (idempotent)."""
        changed = False
        with self._lock:
            run = self._runs.get(run_id)
            if run is not None and run.status == "running":
                run.status = status
                run.end = time.monotonic()
                changed = True
        if changed:
            self._fire_change()

    def _evict_locked(self) -> None:
        """Drop oldest FINISHED runs while over the cap (never a running one).
        Caller must hold the lock."""
        while len(self._runs) > self._max_runs:
            victim = next(
                (rid for rid, r in self._runs.items() if r.status != "running"), None
            )
            if victim is None:  # all still running — leave them
                return
            del self._runs[victim]

    def snapshot(self) -> List[ActivityRun]:
        """An ordered list of shallow COPIES (with a copied events list) for the
        UI to render — the caller never touches a live mutable run/deque."""
        with self._lock:
            return [self._copy_locked(r) for r in self._runs.values()]

    def get(self, run_id: str) -> Optional[ActivityRun]:
        """A frozen copy of one run for the detail view (writers keep appending
        to the original while the view iterates this copy)."""
        with self._lock:
            r = self._runs.get(run_id)
            return self._copy_locked(r) if r is not None else None

    @staticmethod
    def _copy_locked(r: ActivityRun) -> ActivityRun:
        """Shallow copy with a copied events deque, SHARING the live cancel Event
        (so stopping via a snapshot's run_id reaches the live worker). Caller
        holds the lock."""
        return ActivityRun(
            run_id=r.run_id,
            agent_type=r.agent_type,
            description=r.description,
            origin=r.origin,
            start=r.start,
            status=r.status,
            end=r.end,
            events=deque(r.events, maxlen=r.events.maxlen),
            cancel=r.cancel,  # SAME object — a snapshot can request the stop
        )

    def any_active(self) -> bool:
        with self._lock:
            return any(r.status == "running" for r in self._runs.values())

    def is_cancelled(self, run_id: str) -> bool:
        """True if a stop was requested for this run (polled by the worker loop)."""
        with self._lock:
            run = self._runs.get(run_id)
            return bool(run and run.cancel.is_set())

    def request_stop(self, run_id: str) -> bool:
        """Ask ONE running agent to stop (the x key). Sets its cancel Event; the
        worker loop polls it and returns cleanly. Returns True if it was running."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.status != "running":
                return False
            run.cancel.set()
        self._fire_change()
        return True

    def request_stop_all(self) -> int:
        """Ask EVERY running agent to stop (ctrl+x ctrl+k). Returns how many."""
        n = 0
        with self._lock:
            for run in self._runs.values():
                if run.status == "running" and not run.cancel.is_set():
                    run.cancel.set()
                    n += 1
        if n:
            self._fire_change()
        return n

    def clear(self) -> None:
        """Drop all runs (used on /clear for a fresh context)."""
        with self._lock:
            self._runs.clear()
        self._fire_change()

    def _fire_change(self) -> None:
        cb = self.on_change
        if cb is not None:
            try:
                cb()
            except Exception:
                pass  # a repaint hook must never break a worker thread
