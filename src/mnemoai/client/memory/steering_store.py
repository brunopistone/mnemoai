"""User-authored, always-on steering instructions (``STEERING.md``/``CLAUDE.md``).

Instructions the USER writes to steer the agent every turn — conventions, commands,
"always do X" rules. It is read-only from the app's perspective (the agent never writes it),
which is the key distinction from ``MEMORY.md`` (agent-curated, bounded).

Discovery is hierarchical (see :func:`utils.paths.steering_files`): the app home
plus every directory from the CWD up to the repo root, each contributing at most
one file — ``STEERING.md``, else ``CLAUDE.md`` (:data:`utils.paths.
STEERING_FILENAMES`) — concatenated broadest→most-specific.

The content is prepended to the prompt as a leading ``<steering>`` block each turn 
(re-read from disk, so edits apply immediately) and stripped before storage, so it is 
**never summarized by compaction** and always reaches the model verbatim. 
Pure file logic — no MCP, no LLM — so it is unit-testable on its own.
"""

from pathlib import Path
from typing import List, Optional

from mnemoai.utils.config import config
from mnemoai.utils.logger import logger

# Per-file ceiling on injected instructions. Deliberately generous — this is a
# runaway guard, not a style rule: a normal hand-written file is far under it,
# while a machine-generated one can be tens of thousands of chars and is billed
# on every single turn. Overridable via ``STEERING.MAX_CHARS`` (``0`` disables
# the cap); a code default so no config edit is needed to reach existing installs.
_DEFAULT_MAX_CHARS = 45_000


class SteeringStore:
    """Read + concatenate the discovered instruction files (never writes)."""

    def __init__(self, files: Optional[List[Path]] = None, cwd: Optional[Path] = None) -> None:
        """Initialize the store.

        Args:
            files: Explicit list of instruction-file paths (apply order).
                Defaults to :func:`utils.paths.steering_files` discovery.
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

        Tolerant by necessity: this runs on EVERY turn, so anything that raises
        here breaks the whole conversation, not one feature. A file that
        vanished, can't be read, or isn't valid UTF-8 is skipped or degraded, and
        one bad file never costs the others.
        """
        blocks: List[str] = []
        for f in self.files:
            try:
                # errors="replace", not strict: these files are often authored by
                # other tooling in another encoding, and a stray byte must not
                # raise UnicodeDecodeError — which is a ValueError, so it would
                # escape an OSError-only guard and fail every turn.
                text = (
                    f.read_text(encoding="utf-8", errors="replace").strip()
                    if f.is_file()
                    else ""
                )
            except (OSError, ValueError) as e:
                logger.warning(f"Could not read steering file {f}: {e}")
                continue
            if not text:
                continue
            text = self._bounded(f, text)
            blocks.append(f"Contents of {f} (user steering instructions):\n\n{text}")
        return "\n\n".join(blocks)

    @staticmethod
    def _bounded(path: Path, text: str) -> str:
        """Cap one file's contribution, telling the model what was cut.

        This content is re-injected verbatim every turn and deliberately kept out
        of history, so compaction can never reclaim it — an oversized file is a
        permanent per-turn tax (a 55k-char file is ~14k tokens EVERY turn). The
        cap is generous and never silent: the model is told the file was
        truncated, so it can read the rest with its file tools instead of
        assuming it saw everything.
        """
        try:
            max_chars = int(
                (config.get("STEERING", {}) or {}).get("MAX_CHARS", _DEFAULT_MAX_CHARS)
            )
        except (AttributeError, TypeError, ValueError):
            max_chars = _DEFAULT_MAX_CHARS
        # A non-positive cap means "no cap" — an explicit opt-out for a user who
        # wants a big file honored whole.
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        logger.warning(
            "Steering file %s is %d chars; injecting the first %d "
            "(STEERING.MAX_CHARS). It is re-sent every turn, so consider "
            "trimming it.",
            path,
            len(text),
            max_chars,
        )
        return (
            text[:max_chars]
            + f"\n\n[... truncated: {path} is {len(text)} chars and only the "
            f"first {max_chars} are shown here. Read the file directly if you "
            "need the rest — do not assume the omitted part says nothing.]"
        )
