"""Curated, bounded markdown memory the agent maintains itself.

A small ``MEMORY.md`` (Hermes-style) holding durable facts the agent chose to
keep — environment details, conventions, lessons, project context. Unlike
episodic memory (vector retrieval) it is injected *whole* into the system prompt
at session start, so it must stay small: a hard character cap forces the agent
to consolidate (via ``replace``/``remove``) rather than grow without bound.

This module is pure file logic — no MCP, no LLM — so it is shared by the
server-side ``memory`` tool and the client-side ``/memory`` command, and is
unit-testable on its own.

Entries are separated by a ``§`` delimiter on its own line, which lets a single
entry span multiple lines while keeping add/replace/remove reliable.
"""

from pathlib import Path
from typing import List, Optional

from mnemoai.utils.logger import logger

DELIMITER = "\n§\n"
DEFAULT_MAX_CHARS = 2200


class MemoryError(Exception):
    """Raised for a memory operation the caller should surface to the agent.

    The message is written for the model to read and act on (e.g. "consolidate
    then retry", "no unique match").
    """


class MemoryStore:
    """Read/edit a profile's ``MEMORY.md`` with a bounded, curated discipline."""

    def __init__(self, path: Optional[Path] = None, max_chars: Optional[int] = None) -> None:
        """Initialize the store.

        Args:
            path: MEMORY.md location; defaults to ``paths.memory_file_path()``.
            max_chars: Hard cap on file size; defaults to config ``MEMORY.MAX_CHARS``
                (or 2200). A write that would exceed it is rejected.
        """
        if path is None:
            from mnemoai.utils.paths import memory_file_path

            path = memory_file_path()
        self.path = Path(path)
        if max_chars is None:
            from mnemoai.utils.config import config

            max_chars = config.get("MEMORY", {}).get("MAX_CHARS", DEFAULT_MAX_CHARS)
        self.max_chars = int(max_chars)

    # --- low-level entry IO --------------------------------------------------

    def read(self) -> str:
        """Return the raw file contents, or "" if absent/unreadable."""
        try:
            return self.path.read_text() if self.path.is_file() else ""
        except OSError as e:
            logger.warning(f"Could not read memory file {self.path}: {e}")
            return ""

    def _entries(self) -> List[str]:
        """Current entries (delimiter-split, blanks dropped)."""
        text = self.read().strip()
        if not text:
            return []
        return [e.strip() for e in text.split(DELIMITER.strip()) if e.strip()]

    def _write_entries(self, entries: List[str]) -> None:
        """Persist entries joined by the delimiter (creates parent dir)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = DELIMITER.join(entries)
        self.path.write_text(body + "\n" if body else "")

    @staticmethod
    def _projected_len(entries: List[str]) -> int:
        """Char length of the file that ``entries`` would produce."""
        if not entries:
            return 0
        return len(DELIMITER.join(entries)) + 1  # trailing newline

    # --- public operations (used by the MCP tool) ---------------------------

    def add(self, text: str) -> str:
        """Append a new entry. Rejects blanks, exact duplicates, and overflow."""
        text = (text or "").strip()
        if not text:
            raise MemoryError("Refusing to add empty memory text.")
        entries = self._entries()
        if text in entries:
            raise MemoryError("That entry already exists; not adding a duplicate.")
        projected = self._projected_len(entries + [text])
        if projected > self.max_chars:
            raise MemoryError(
                f"Adding this would exceed the memory limit "
                f"({projected} > {self.max_chars} chars). Consolidate or remove "
                f"existing entries first (use replace/remove), then add again."
            )
        entries.append(text)
        self._write_entries(entries)
        return f"Added memory entry ({self._projected_len(entries)}/{self.max_chars} chars used)."

    def replace(self, old_text: str, new_text: str) -> str:
        """Replace the entry uniquely matching ``old_text`` with ``new_text``."""
        old_text = (old_text or "").strip()
        new_text = (new_text or "").strip()
        if not old_text or not new_text:
            raise MemoryError("replace requires both old_text and new_text.")
        entries = self._entries()
        matches = [i for i, e in enumerate(entries) if old_text in e]
        if not matches:
            raise MemoryError(f"No memory entry contains: {old_text!r}")
        if len(matches) > 1:
            raise MemoryError(
                f"{len(matches)} entries match {old_text!r}; be more specific."
            )
        candidate = list(entries)
        candidate[matches[0]] = new_text
        projected = self._projected_len(candidate)
        if projected > self.max_chars:
            raise MemoryError(
                f"That replacement would exceed the memory limit "
                f"({projected} > {self.max_chars} chars). Make it shorter or "
                f"remove another entry first."
            )
        self._write_entries(candidate)
        return f"Replaced memory entry ({projected}/{self.max_chars} chars used)."

    def remove(self, old_text: str) -> str:
        """Remove the entry uniquely matching ``old_text``."""
        old_text = (old_text or "").strip()
        if not old_text:
            raise MemoryError("remove requires old_text.")
        entries = self._entries()
        matches = [i for i, e in enumerate(entries) if old_text in e]
        if not matches:
            raise MemoryError(f"No memory entry contains: {old_text!r}")
        if len(matches) > 1:
            raise MemoryError(
                f"{len(matches)} entries match {old_text!r}; be more specific."
            )
        del entries[matches[0]]
        self._write_entries(entries)
        return f"Removed memory entry ({self._projected_len(entries)}/{self.max_chars} chars used)."

    def clear(self) -> None:
        """Delete all entries (used by ``/memory clear``)."""
        self._write_entries([])
