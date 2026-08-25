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

A file may pull in another with an ``@path`` reference, so a long ruleset can be
split into focused files instead of one wall of text (see :func:`references`).
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from mnemoai.utils.config import config
from mnemoai.utils.logger import logger

# Per-file ceiling on injected instructions. Deliberately generous — this is a
# runaway guard, not a style rule: a normal hand-written file is far under it,
# while a machine-generated one can be tens of thousands of chars and is billed
# on every single turn. Overridable via ``STEERING.MAX_CHARS`` (``0`` disables
# the cap); a code default so no config edit is needed to reach existing installs.
_DEFAULT_MAX_CHARS = 45_000

# An ``@path`` reference: at the start of a line or after whitespace/an opening
# bracket or quote, never mid-word — so an email address (``a@b.com``) can't match.
# The path runs to the first whitespace and may not END on punctuation, so a
# reference that closes a sentence ("see @rules/style.md.") resolves cleanly.
_REF_RE = re.compile(r"(?:(?<=\s)|(?<=[(\[<'\"`])|^)@([\w.~/-]*[\w/])", re.MULTILINE)
# How deep a chain of references is followed (a referenced file may reference
# more), and how many files one turn's steering may pull in. Both are runaway
# guards: this content is re-sent on EVERY turn.
_MAX_INCLUDE_DEPTH = 3
_MAX_INCLUDES = 20


def references(text: str) -> List[str]:
    """The ``@path`` references in ``text``, in order, de-duplicated.

    Pure text scan — resolution and existence are the caller's problem, which is
    what keeps false positives harmless: a Python decorator (``@staticmethod``) or
    a handle (``@someone``) is matched here and then simply doesn't resolve to a
    file, so it stays in the prose untouched.
    """
    seen = set()
    out = []
    for m in _REF_RE.finditer(text or ""):
        ref = m.group(1)
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def resolve_reference(ref: str, base_dir: Path) -> Optional[Path]:
    """Resolve one ``@path`` against the referencing file's directory.

    ``~`` expands, an absolute path is taken as-is, and anything else is relative
    to the file that mentioned it (not the process cwd — a project's steering file
    must mean its own neighbours regardless of where the app was launched).
    Returns None unless it names a readable existing file: a reference to a
    directory or to something absent is left in the prose for the model to chase
    with its file tools.
    """
    try:
        path = Path(ref).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return path if path.is_file() else None
    except (OSError, ValueError, RuntimeError):
        # RuntimeError: expanduser() with no resolvable home.
        return None


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

        Files referenced with ``@path`` are appended after the file that mentioned
        them, under their own header. The reference itself stays in the prose so
        the surrounding sentence still reads, and a file already injected is not
        repeated.

        Tolerant by necessity: this runs on EVERY turn, so anything that raises
        here breaks the whole conversation, not one feature. A file that
        vanished, can't be read, or isn't valid UTF-8 is skipped or degraded, and
        one bad file never costs the others.
        """
        blocks: List[str] = []
        # Seeded with every discovered file so a reference to one of them (a
        # project file pointing at the global ruleset) doesn't inject it twice.
        seen = {self._key(f) for f in self.files}
        cap = self._max_chars()
        for f in self.files:
            text = self._read_text(f)
            if not text:
                continue
            body = self._bounded(f, text)
            blocks.append(f"Contents of {f} (user steering instructions):\n\n{body}")
            # What this file may still pull in shares its own ceiling, so
            # splitting a ruleset across @references can't sidestep the cap.
            # -1 = uncapped (STEERING.MAX_CHARS <= 0).
            budget = -1 if cap <= 0 else max(0, cap - len(body))
            blocks.extend(self._included_blocks(f, text, seen, budget))
        return "\n\n".join(blocks)

    def sizes(self) -> List[Tuple[Path, str]]:
        """``(file, injected text)`` per discovered file with content, apply order.

        Reads and caps exactly as :meth:`read` does, so a per-file size shown to
        the user (``/context``) is the size actually sent rather than the size on
        disk. Files pulled in by ``@path`` are not listed — they belong to the
        block as a whole, not to one row.
        """
        out: List[Tuple[Path, str]] = []
        for f in self.files:
            text = self._read_text(f)
            if text:
                out.append((f, self._bounded(f, text)))
        return out

    def _included_blocks(
        self, source: Path, text: str, seen: set, budget: int
    ) -> List[str]:
        """Blocks for the files ``text`` pulls in with ``@path``, transitively.

        Breadth-first so a file's own references land before their sub-references,
        and bounded three ways — depth, count, and the remaining char budget of
        the steering file that started the chain — because every byte here is paid
        on every turn and compaction can never reclaim it.
        """
        blocks: List[str] = []
        # (referencing file, its text, depth)
        frontier: List[Tuple[Path, str, int]] = [(source, text, 1)]
        omitted = 0
        while frontier:
            origin, body, depth = frontier.pop(0)
            for ref in references(body):
                target = resolve_reference(ref, origin.parent)
                if target is None:
                    continue  # not a file — leave the mention in the prose
                key = self._key(target)
                if key in seen:
                    continue
                seen.add(key)
                if len(blocks) >= _MAX_INCLUDES:
                    omitted += 1
                    continue
                included = self._read_text(target)
                if not included:
                    continue
                if budget >= 0 and len(included) > budget:
                    omitted += 1
                    continue
                blocks.append(
                    f"Contents of {target} (referenced by @{ref}):\n\n{included}"
                )
                if budget >= 0:
                    budget -= len(included)
                if depth < _MAX_INCLUDE_DEPTH:
                    frontier.append((target, included, depth + 1))
        if omitted:
            # Never silent: a dropped include would otherwise look like an
            # instruction the user wrote and the model ignored.
            logger.warning(
                "Steering: %d file(s) referenced from %s were not injected "
                "(reference limit or STEERING.MAX_CHARS budget).",
                omitted,
                source,
            )
            blocks.append(
                f"[... {omitted} file(s) referenced from {source} were not "
                "included here (reference limit or size budget). Read them "
                "directly if they matter.]"
            )
        return blocks

    @staticmethod
    def _key(path: Path) -> str:
        """Identity for de-duplication: resolved, case-normalized path."""
        try:
            return os.path.normcase(os.path.realpath(str(path)))
        except (OSError, ValueError):
            return os.path.normcase(str(path))

    @staticmethod
    def _read_text(path: Path) -> str:
        """One file's stripped text, "" if it is missing or unreadable."""
        try:
            # errors="replace", not strict: these files are often authored by
            # other tooling in another encoding, and a stray byte must not
            # raise UnicodeDecodeError — which is a ValueError, so it would
            # escape an OSError-only guard and fail every turn.
            if not path.is_file():
                return ""
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except (OSError, ValueError) as e:
            logger.warning(f"Could not read steering file {path}: {e}")
            return ""

    @staticmethod
    def _max_chars() -> int:
        """Per-file ceiling from ``STEERING.MAX_CHARS`` (``<= 0`` means no cap)."""
        try:
            return int(
                (config.get("STEERING", {}) or {}).get("MAX_CHARS", _DEFAULT_MAX_CHARS)
            )
        except (AttributeError, TypeError, ValueError):
            return _DEFAULT_MAX_CHARS

    @classmethod
    def _bounded(cls, path: Path, text: str) -> str:
        """Cap one file's contribution, telling the model what was cut.

        This content is re-injected verbatim every turn and deliberately kept out
        of history, so compaction can never reclaim it — an oversized file is a
        permanent per-turn tax (a 55k-char file is ~14k tokens EVERY turn). The
        cap is generous and never silent: the model is told the file was
        truncated, so it can read the rest with its file tools instead of
        assuming it saw everything.
        """
        max_chars = cls._max_chars()
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
