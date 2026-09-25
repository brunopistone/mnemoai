"""Versioned playbook records with evidence, user controls, and archival."""

import copy
import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, List

from mnemoai.client.memory import playbook_records as records
from mnemoai.client.memory.reflector import PlaybookEntry
from mnemoai.utils.atomic_write import atomic_write_json
from mnemoai.utils.file_lock import file_lock
from mnemoai.utils.logger import logger

# Opens the injected block. Public because context_report segments the LIVE system
# prompt by marker, so the two must never drift (a stale marker silently stops
# attributing this block in /context).
#
# Historical heuristic notes remain readable alongside model-extracted lessons.
# Neither a repeated observation nor an LLM's own score proves effectiveness.
PLAYBOOK_BLOCK_MARKER = "[Tool-use notes from past sessions]"
PLAYBOOK_END_MARKER = "[End tool-use notes]"

# Injected on EVERY turn and never reclaimable by compaction, so an unverified note
# is kept cheap. Capped in code rather than by lowering PLAYBOOK.MAX_INJECT, whose
# default reaches only fresh installs — every config.yaml written so far sets it to
# 10 explicitly, so a default change would not reach a single existing user.
_MAX_INJECT_PER_GROUP = 2


class PlaybookStore:
    """Stores and retrieves playbook entries with semantic deduplication."""

    def __init__(
        self,
        persist_path: str,
        embeddings_controller=None,
        max_entries: int = 500,
        similarity_threshold: float = 0.85,
    ):
        """Initialize playbook store.

        Args:
            persist_path: Directory to persist playbook data
            embeddings_controller: For semantic similarity (optional)
            max_entries: Maximum entries before triggering refinement
            similarity_threshold: Threshold for merging similar entries
        """
        self.persist_path = persist_path
        self.embeddings = embeddings_controller
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold

        self.playbook_file = os.path.join(persist_path, "playbook.json")
        self.lock_file = os.path.join(persist_path, ".playbook.lock")
        self.entries: List[Dict[str, Any]] = []
        self.error = None
        self.enabled = True
        try:
            with file_lock(self.lock_file):
                self._load()
        except OSError as e:
            self._unavailable(e)

    def _unavailable(self, error):
        message = str(error)
        if self.error != message:
            logger.warning("Playbook unavailable; file preserved: %s", error)
        self.error = message
        self.entries = []

    def _load(self) -> None:
        """Load playbook from disk."""
        try:
            raw = []
            if os.path.exists(self.playbook_file):
                with open(self.playbook_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            if not isinstance(raw, list):
                raise ValueError("Playbook must contain a list of entries")
            normalized = [records.normalize(entry) for entry in raw]
            if len({entry["id"] for entry in normalized}) != len(normalized):
                raise ValueError("Duplicate playbook IDs")
            if normalized != raw:
                self._backup()
                atomic_write_json(self.playbook_file, normalized)
            self.entries = normalized
            self.error = None
        except Exception as e:
            self._unavailable(e)

    def _backup(self):
        if os.path.isfile(self.playbook_file):
            backup = self.playbook_file + ".backup-" + uuid.uuid4().hex
            shutil.copy2(self.playbook_file, backup)
            return backup
        return None

    def snapshot(self):
        """Fresh copies prevent a second session's edits being silently overwritten."""
        try:
            with file_lock(self.lock_file):
                self._load()
                if self.error:
                    raise ValueError(self.error)
                return copy.deepcopy(self.entries)
        except OSError as e:
            self._unavailable(e)
            raise

    def _transaction(self, change):
        with file_lock(self.lock_file):
            self._load()
            if self.error:
                raise ValueError(self.error)
            before = copy.deepcopy(self.entries)
            try:
                result = change()
                self._save()
                return result
            except BaseException:
                self.entries = before
                raise

    def _save(self) -> None:
        """Persist playbook to disk (atomically -- see utils.atomic_write)."""
        os.makedirs(self.persist_path, exist_ok=True)
        atomic_write_json(self.playbook_file, self.entries)

    def append(self, entry: PlaybookEntry) -> None:
        """Append a new entry (delta update).

        Args:
            entry: PlaybookEntry to add
        """
        self.append_batch([entry])

    def append_batch(self, entries: List[PlaybookEntry]) -> None:
        """Append multiple entries efficiently.

        Args:
            entries: List of PlaybookEntry objects
        """
        incoming = [records.normalize(entry.to_dict()) for entry in entries]

        def add():
            for entry in incoming:
                existing = next(
                    (old for old in self.entries if records.key(old) == records.key(entry)),
                    None,
                )
                if existing is not None:
                    known = {records.ref_key(ref) for ref in existing["source_refs"]}
                    new_refs = [
                        ref for ref in entry["source_refs"]
                        if records.ref_key(ref) not in known
                    ]
                    if new_refs:
                        existing["source_refs"].extend(new_refs)
                        existing["revision"] += 1
                        existing["last_seen"] = records.now()
                    # Repetition never increases confidence or re-enables a note.
                    continue
                if any(
                    records.key(old) == records.key(entry)
                    for current in self.entries for old in current["history"]
                ):
                    continue  # a user edited/superseded this wording
                self.entries.append(entry)
            self._refine()

        if incoming:
            self._transaction(add)

    def _refine(self) -> None:
        """Lazy refinement - merge similar entries when over limit."""
        active = [e for e in self.entries if e["status"] == "active"]
        if len(active) <= self.max_entries:
            return
        logger.info(f"Refining playbook ({len(active)} active entries)...")

        if not self.embeddings:
            keep = sorted(
                active, key=lambda x: x.get("timestamp", ""), reverse=True
            )[:max(0, self.max_entries)]
        else:
            groups = {}
            for entry in active:
                groups.setdefault((entry["scope"], entry["context"]), []).append(entry)
            keep = [
                entry for group in groups.values() for entry in self._merge_similar(group)
            ][:max(0, self.max_entries)]
        ids = {entry["id"] for entry in keep}
        for entry in active:
            if entry["id"] not in ids:
                records.checkpoint(entry, "archive")
                entry["status"] = "archived"

    def _merge_similar(self, entries: List[Dict]) -> List[Dict]:
        """Merge semantically similar entries.

        Args:
            entries: Entries to potentially merge

        Returns:
            Deduplicated entries
        """
        if len(entries) <= 1:
            return entries

        # Use embeddings if available for semantic deduplication.
        if self.embeddings:
            return self._merge_with_embeddings(entries)

        return self._merge_by_strategy_key(entries)

    def _merge_by_strategy_key(self, entries: List[Dict]) -> List[Dict]:
        """Dedup without embeddings: keep the highest-confidence entry per
        exact scope/context/strategy. Used without embeddings and as the
        fallback when the embedding merge fails."""
        sorted_entries = sorted(
            entries,
            key=lambda x: (x.get("confidence", 0), x.get("timestamp", "")),
            reverse=True,
        )

        keep = []
        seen_strategies = set()

        for entry in sorted_entries:
            strategy_key = records.key(entry)
            if strategy_key not in seen_strategies:
                keep.append(entry)
                seen_strategies.add(strategy_key)

        return keep

    def _merge_with_embeddings(self, entries: List[Dict]) -> List[Dict]:
        """Merge entries using semantic similarity.

        Args:
            entries: Entries to deduplicate

        Returns:
            Deduplicated entries
        """
        if len(entries) <= 1:
            return entries

        try:
            # Get embeddings for all strategies
            strategies = [e.get("strategy", "") for e in entries]
            embeddings = self.embeddings.embed(strategies)

            # Find clusters of similar entries
            keep_indices = []
            used = set()

            for i, emb_i in enumerate(embeddings):
                if i in used:
                    continue

                # Find all similar entries
                similar = [i]
                for j, emb_j in enumerate(embeddings):
                    if j <= i or j in used:
                        continue

                    # Cosine similarity
                    similarity = sum(a * b for a, b in zip(emb_i, emb_j))
                    if similarity >= self.similarity_threshold:
                        similar.append(j)
                        used.add(j)

                # Keep the one with highest confidence
                best_idx = max(similar, key=lambda x: entries[x].get("confidence", 0))
                keep_indices.append(best_idx)
                used.add(i)

            return [entries[i] for i in sorted(keep_indices)]

        except Exception as e:
            logger.error(f"Embedding merge failed: {e}")
            # Fallback to the non-embedding keyword dedup (avoids re-dispatching
            # back into this method, which the old __wrapped__ call did wrongly).
            return self._merge_by_strategy_key(entries)

    def get_relevant_entries(
        self, task: str, top_k: int = 10, include_failures: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve entries relevant to a task.

        Args:
            task: Current task context
            top_k: Maximum entries to return
            include_failures: Whether to include failure strategies

        Returns:
            List of relevant playbook entries
        """
        if not getattr(self, "enabled", True):
            return []
        if hasattr(self, "lock_file"):
            try:
                self.snapshot()
            except (ValueError, OSError):
                return []
        if not self.entries:
            return []

        task_lower = task.lower()

        # Score entries by relevance
        scored = []
        for entry in self.entries:
            if not records.eligible(entry):
                continue
            if entry.get("scope") and entry["scope"] != os.path.realpath(os.getcwd()):
                continue
            if not include_failures and entry.get("outcome") == "failure":
                continue

            score = self._relevance_score(entry, task_lower)
            if score > 0:
                scored.append((score, entry))

        # Sort by score and return top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def _relevance_score(self, entry: Dict, task_lower: str) -> float:
        """Calculate relevance score for an entry.

        Args:
            entry: Playbook entry
            task_lower: Lowercase task string

        Returns:
            Relevance score (0-1)
        """
        score = 0.0

        # Context match
        context = entry.get("context", "").lower()
        if context in task_lower or any(word in task_lower for word in context.split()):
            score += 0.4

        # Tool match
        tools = entry.get("tools", [])
        tool_keywords = {
            "fs_read": ["read", "file", "content"],
            "fs_write": ["write", "create", "save"],
            "file_edit": ["edit", "modify", "change", "update"],
            "execute_bash": ["run", "command", "bash", "shell"],
            "grep_search": ["search", "find", "grep"],
            "glob_search": ["find", "files", "list"],
        }
        for tool in tools:
            keywords = tool_keywords.get(tool, [])
            if any(kw in task_lower for kw in keywords):
                score += 0.3
                break

        # Confidence boost
        score += entry.get("confidence", 0.5) * 0.2

        # Recency boost (entries from last 7 days)
        timestamp = entry.get("timestamp", "")
        if timestamp:
            try:
                entry_date = datetime.fromisoformat(timestamp)
                days_old = (datetime.now(entry_date.tzinfo) - entry_date).days
                if days_old < 7:
                    score += 0.1
            except (ValueError, TypeError, OverflowError):
                pass

        return score

    def format_for_prompt(self, entries: List[Dict[str, Any]]) -> str:
        """Format entries for injection into system prompt.

        Args:
            entries: Playbook entries to format

        Returns:
            Formatted string for prompt injection
        """
        entries = self.prompt_entries(entries)
        if not entries or not getattr(self, "enabled", True):
            return ""

        lines = [PLAYBOOK_BLOCK_MARKER]

        # Group by outcome
        successes = [e for e in entries if e.get("outcome") == "success"]
        failures = [e for e in entries if e.get("outcome") == "failure"]

        if failures:
            lines.append("Noted after past errors:")
            for entry in failures[:_MAX_INJECT_PER_GROUP]:
                lines.append(self._prompt_line(entry))

        if successes:
            lines.append("Noted after past successes:")
            for entry in successes[:_MAX_INJECT_PER_GROUP]:
                lines.append(self._prompt_line(entry))

        lines.append(PLAYBOOK_END_MARKER)
        return "\n".join(lines)

    @staticmethod
    def _prompt_line(entry):
        def text(value, limit):
            return " ".join(
                "".join(c for c in str(value) if c.isprintable() or c.isspace()).split()
            )[:limit]
        return f"  · [{text(entry.get('context', ''), 200)}]: {text(entry.get('strategy', ''), 800)}"

    def prompt_entries(self, entries):
        """The same filtering/caps drive both rendering and exposure accounting."""
        selected = [entry for entry in entries if records.eligible(entry)]
        return [
            entry for outcome in ("failure", "success")
            for entry in [e for e in selected if e.get("outcome") == outcome][
                :_MAX_INJECT_PER_GROUP
            ]
        ]

    def update(self, entry_id, revision, *, action, context=None, strategy=None):
        """Apply a user-confirmed edit to exactly the revision that was previewed."""
        def change():
            entry = next((e for e in self.entries if e["id"] == entry_id), None)
            if entry is None or entry["revision"] != revision:
                raise ValueError("Entry changed; inspect it again before editing")
            if action not in {"edit", "disable", "restore", "helpful", "unhelpful"}:
                raise ValueError("Unknown playbook action")
            if action == "edit":
                for value, limit in ((context, 200), (strategy, 800)):
                    if not isinstance(value, str) or not value.strip() or len(value) > limit:
                        raise ValueError("Context (1–200) and strategy (1–800) are required")
                    if any(not c.isprintable() and not c.isspace() for c in value):
                        raise ValueError("Control characters are not allowed")
            records.checkpoint(entry, action)
            if action == "edit":
                entry.update(context=" ".join(context.split()), strategy=" ".join(strategy.split()),
                             provenance="user")
            elif action in {"disable", "restore"}:
                entry["status"] = "disabled" if action == "disable" else "active"
            else:
                entry[action + "_count"] += 1
                entry["confidence"] = (entry["helpful_count"] + 1) / (
                    entry["helpful_count"] + entry["unhelpful_count"] + 2
                )
            return copy.deepcopy(entry)
        return self._transaction(change)

    def record_exposure(self, entry_ids, success=None):
        """Count exposure/outcome associations, never inferred helpfulness."""
        ids = set(entry_ids)
        if not ids:
            return
        field = "injection_count" if success is None else (
            "observed_success_count" if success else "observed_failure_count"
        )
        def change():
            for entry in self.entries:
                if entry["id"] in ids:
                    entry[field] += 1
        self._transaction(change)

    def get_stats(self) -> Dict[str, Any]:
        """Get playbook statistics.

        Returns:
            Dictionary with stats
        """
        successes = sum(1 for e in self.entries if e.get("outcome") == "success")
        failures = sum(1 for e in self.entries if e.get("outcome") == "failure")

        return {
            "total_entries": len(self.entries),
            "successes": successes,
            "failures": failures,
            "contexts": len(set(e.get("context", "") for e in self.entries)),
        }

    def clear(self, expected=None) -> None:
        """Clear all entries."""
        def change():
            if expected is not None and expected != [
                (e["id"], e["revision"]) for e in self.entries
            ]:
                raise ValueError("Playbook changed; review it again before clearing")
            self._backup()
            self.entries = []
        self._transaction(change)
