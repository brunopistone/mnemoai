"""Playbook Store - Manages strategy entries with append-only updates and lazy deduplication."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from client.memory.reflector import PlaybookEntry
from utils.logger import logger


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
        self.entries: List[Dict[str, Any]] = []

        self._load()

    def _load(self) -> None:
        """Load playbook from disk."""
        if os.path.exists(self.playbook_file):
            try:
                with open(self.playbook_file, "r") as f:
                    self.entries = json.load(f)
                logger.debug(f"Loaded {len(self.entries)} playbook entries")
            except Exception as e:
                logger.error(f"Failed to load playbook: {e}")
                self.entries = []

    def _save(self) -> None:
        """Persist playbook to disk."""
        os.makedirs(self.persist_path, exist_ok=True)
        try:
            with open(self.playbook_file, "w") as f:
                json.dump(self.entries, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save playbook: {e}")

    def append(self, entry: PlaybookEntry) -> None:
        """Append a new entry (delta update).

        Args:
            entry: PlaybookEntry to add
        """
        entry_dict = entry.to_dict()

        # Quick duplicate check by exact strategy match
        for existing in self.entries:
            if existing.get("strategy") == entry_dict["strategy"]:
                # Update confidence if same strategy seen again
                existing["confidence"] = min(1.0, existing["confidence"] + 0.1)
                existing["last_seen"] = datetime.now().isoformat()
                logger.debug(
                    f"Updated confidence for existing strategy: {entry.context}"
                )
                self._save()
                return

        self.entries.append(entry_dict)
        logger.debug(f"Appended new playbook entry: {entry.context} ({entry.outcome})")

        # Trigger lazy refinement if over limit
        if len(self.entries) > self.max_entries:
            self._refine()

        self._save()

    def append_batch(self, entries: List[PlaybookEntry]) -> None:
        """Append multiple entries efficiently.

        Args:
            entries: List of PlaybookEntry objects
        """
        for entry in entries:
            self.append(entry)

    def _refine(self) -> None:
        """Lazy refinement - merge similar entries when over limit."""
        logger.info(f"Refining playbook ({len(self.entries)} entries)...")

        if not self.embeddings:
            # Without embeddings, just keep most recent
            self.entries = sorted(
                self.entries, key=lambda x: x.get("timestamp", ""), reverse=True
            )[: self.max_entries]
            return

        # Group by context for semantic comparison
        context_groups: Dict[str, List[Dict]] = {}
        for entry in self.entries:
            ctx = entry.get("context", "general")
            if ctx not in context_groups:
                context_groups[ctx] = []
            context_groups[ctx].append(entry)

        refined = []
        for context, group in context_groups.items():
            if len(group) <= 2:
                refined.extend(group)
                continue

            # Merge similar strategies within context
            merged = self._merge_similar(group)
            refined.extend(merged)

        self.entries = refined[: self.max_entries]
        logger.info(f"Refined to {len(self.entries)} entries")

    def _merge_similar(self, entries: List[Dict]) -> List[Dict]:
        """Merge semantically similar entries.

        Args:
            entries: Entries to potentially merge

        Returns:
            Deduplicated entries
        """
        if len(entries) <= 1:
            return entries

        # Use embeddings if available for semantic deduplication
        if self.embeddings:
            return self._merge_with_embeddings(entries)

        # Fallback: keep highest confidence entries
        sorted_entries = sorted(
            entries,
            key=lambda x: (x.get("confidence", 0), x.get("timestamp", "")),
            reverse=True,
        )

        keep = []
        seen_strategies = set()

        for entry in sorted_entries:
            strategy_key = entry.get("strategy", "")[:50]
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
            # Fallback to simple deduplication
            return self._merge_similar.__wrapped__(self, entries)

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
        if not self.entries:
            return []

        task_lower = task.lower()

        # Score entries by relevance
        scored = []
        for entry in self.entries:
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
                days_old = (datetime.now() - entry_date).days
                if days_old < 7:
                    score += 0.1
            except:
                pass

        return score

    def format_for_prompt(self, entries: List[Dict[str, Any]]) -> str:
        """Format entries for injection into system prompt.

        Args:
            entries: Playbook entries to format

        Returns:
            Formatted string for prompt injection
        """
        if not entries:
            return ""

        lines = ["[Playbook - Learned Strategies]"]

        # Group by outcome
        successes = [e for e in entries if e.get("outcome") == "success"]
        failures = [e for e in entries if e.get("outcome") == "failure"]

        if failures:
            lines.append("Avoid these patterns:")
            for entry in failures[:5]:
                lines.append(f"  ✗ [{entry.get('context')}]: {entry.get('strategy')}")

        if successes:
            lines.append("Effective strategies:")
            for entry in successes[:5]:
                lines.append(f"  ✓ [{entry.get('context')}]: {entry.get('strategy')}")

        return "\n".join(lines)

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

    def clear(self) -> None:
        """Clear all entries."""
        self.entries = []
        self._save()
