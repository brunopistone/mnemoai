"""Registry for background (detached) sub-agents.

A background sub-agent runs on a daemon thread while the parent turn continues,
so the user keeps interacting. This module is the thread-safe bookkeeping: it
tracks each background run's status/result and persists a small record under
``tasks_dir()`` so ``list``/``get`` survive within the session. The actual agent
loop still runs via the agent's ``_run_one_subagent`` (quiet) — this only owns
the state, not the execution.

Pure state + persistence, no LLM/agent logic (mirrors ``SkillStore``/subagents'
separation of data from the agent runner). The agent launches runs and drains
completions; the UI surfaces notifications.
"""

import json
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mnemoai.utils.logger import logger
from mnemoai.utils.paths import tasks_dir


@dataclass
class BackgroundAgent:
    """One background sub-agent run."""

    agent_id: str
    agent_type: str
    description: str
    prompt: str
    status: str = "running"  # running | done | failed
    result: Optional[str] = None
    # Persisted transcript (saveable worker messages) for resume, as role dicts.
    transcript: List[dict] = field(default_factory=list)
    # True once a completion notification has been surfaced to the parent, so it
    # isn't injected twice.
    notified: bool = False


class BackgroundAgentRegistry:
    """Thread-safe registry of background sub-agents for one session.

    Session-scoped and in-memory (the source of truth); a compact JSON record is
    also written under ``tasks_dir()`` per run for durability/debugging. All
    mutation goes through the lock so the parent turn and the daemon threads never
    race."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._agents: Dict[str, BackgroundAgent] = {}
        self._counter = 0

    def _next_id(self, agent_type: str) -> str:
        self._counter += 1
        return f"{agent_type}-{self._counter}"

    def register(self, agent_type: str, description: str, prompt: str) -> BackgroundAgent:
        """Create and store a new running background agent; returns it."""
        with self._lock:
            agent_id = self._next_id(agent_type)
            rec = BackgroundAgent(
                agent_id=agent_id,
                agent_type=agent_type,
                description=description,
                prompt=prompt,
            )
            self._agents[agent_id] = rec
        self._persist(rec)
        return rec

    def complete(
        self, agent_id: str, result: str, transcript: Optional[List[dict]] = None,
        failed: bool = False,
    ) -> None:
        """Mark a background agent done (or failed) with its final result."""
        with self._lock:
            rec = self._agents.get(agent_id)
            if rec is None:
                return
            rec.status = "failed" if failed else "done"
            rec.result = result
            if transcript is not None:
                rec.transcript = transcript
        self._persist(rec)

    def get(self, agent_id: str) -> Optional[BackgroundAgent]:
        with self._lock:
            return self._agents.get(agent_id)

    def list_all(self) -> List[BackgroundAgent]:
        with self._lock:
            return list(self._agents.values())

    def drain_completed_unnotified(self) -> List[BackgroundAgent]:
        """Return finished agents whose completion hasn't been surfaced yet,
        marking them notified. Used by the chat loop to inject a notification for
        each newly-finished background agent exactly once."""
        with self._lock:
            ready = [
                r for r in self._agents.values()
                if r.status in ("done", "failed") and not r.notified
            ]
            for r in ready:
                r.notified = True
            return ready

    def any_running(self) -> bool:
        with self._lock:
            return any(r.status == "running" for r in self._agents.values())

    def any_undelivered(self) -> bool:
        """True if a finished background agent hasn't been surfaced yet."""
        with self._lock:
            return any(
                r.status in ("done", "failed") and not r.notified
                for r in self._agents.values()
            )

    def _persist(self, rec: BackgroundAgent) -> None:
        """Best-effort JSON snapshot under tasks_dir (never fatal). Includes the
        original ``prompt`` so a finished agent can be resumed after a restart
        (the in-memory registry is gone, but the record on disk survives)."""
        try:
            path = tasks_dir() / f"subagent_{rec.agent_id}.json"
            path.write_text(
                json.dumps(
                    {
                        "agent_id": rec.agent_id,
                        "agent_type": rec.agent_type,
                        "description": rec.description,
                        "prompt": rec.prompt,
                        "status": rec.status,
                        "result": rec.result,
                    },
                    indent=2,
                )
            )
        except OSError as e:
            logger.debug(f"Could not persist background agent {rec.agent_id}: {e}")

    def load_from_disk(self, agent_id: str) -> Optional[BackgroundAgent]:
        """Load a finished background agent's persisted record (for cross-restart
        resume, when the in-memory registry has no such id). Returns None if the
        record is absent, unreadable, or still marked running (a stale record from
        a process that died mid-run can't be trusted to have a result)."""
        agent_id = (agent_id or "").strip()
        if not agent_id:
            return None
        try:
            path = tasks_dir() / f"subagent_{agent_id}.json"
            if not path.is_file():
                return None
            data = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            logger.debug(f"Could not load background agent {agent_id}: {e}")
            return None
        if data.get("status") not in ("done", "failed"):
            return None
        return BackgroundAgent(
            agent_id=data.get("agent_id", agent_id),
            agent_type=data.get("agent_type", ""),
            description=data.get("description", ""),
            prompt=data.get("prompt", ""),
            status=data.get("status", "done"),
            result=data.get("result"),
        )
