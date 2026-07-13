"""User-authored, always-on steering instructions (mnemoai's ``STEERING.md``).

Instructions the USER writes to steer the agent every turn — conventions, commands, 
"always do X" rules. It is read-only from the app's perspective (the agent never writes it), 
which is the key distinction from ``MEMORY.md`` (agent-curated, bounded).

Discovery is hierarchical (see :func:`utils.paths.steering_files`): a global
``~/.mnemoai/STEERING.md`` plus a project ``./STEERING.md`` found by walking up
to the repo root, concatenated broadest→most-specific.

The content is prepended to the prompt as a leading ``<steering>`` block each turn 
(re-read from disk, so edits apply immediately) and stripped before storage, so it is 
**never summarized by compaction** and always reaches the model verbatim. 
Pure file logic — no MCP, no LLM — so it is unit-testable on its own.
"""

from pathlib import Path
from typing import List, Optional

from mnemoai.utils.logger import logger


class SteeringStore:
    """Read + concatenate the discovered ``STEERING.md`` files (never writes)."""

    def __init__(self, files: Optional[List[Path]] = None, cwd: Optional[Path] = None) -> None:
        """Initialize the store.

        Args:
            files: Explicit list of STEERING.md paths (apply order). Defaults to
                :func:`utils.paths.steering_files` discovery.
            cwd: Working directory for project discovery (defaults to the process
                cwd); ignored when ``files`` is given.
        """
        if files is None:
            from mnemoai.utils.paths import steering_files

            files = steering_files(cwd=cwd)
        self.files = [Path(f) for f in files]

    def read(self) -> str:
        """Concatenated content of all discovered files, each under a ``Contents
        of <path>:`` header, in apply order. "" if none.

        Tolerant: a file that vanished or can't be read is skipped, not fatal.
        """
        blocks: List[str] = []
        for f in self.files:
            try:
                text = f.read_text().strip() if f.is_file() else ""
            except OSError as e:
                logger.warning(f"Could not read steering file {f}: {e}")
                continue
            if text:
                blocks.append(f"Contents of {f} (user steering instructions):\n\n{text}")
        return "\n\n".join(blocks)
