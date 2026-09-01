"""User-defined slash commands: prompt macros the USER invokes by name.

A *command* is one markdown file under the commands root
(``~/.mnemoai/commands/<name>.md``): optional YAML frontmatter (``description``,
``argument_hint``) followed by a markdown body. Typing ``/<name> rest of line``
submits that body as the turn's prompt, with ``$ARGUMENTS`` (and ``$1``…``$9``)
substituted from what followed the name.

The distinction from the two model-invoked markdown directories: a *skill* is
loaded when the MODEL decides it applies (``use_skill``), a sub-agent type when it
delegates. A command is **typed**, and the model never learns one was involved —
the expansion IS the prompt, so nothing about the turn is special afterwards.
That is also why a command needs no toggle: the file's presence is the switch.

Pure file logic (no MCP, no LLM, no prompt_toolkit), so the dispatcher and the
completion menu read the same store, and it unit-tests on its own. Tolerant like
``mcp_config``/``skill_store``: a file that can't be a command is skipped with a
reason rather than being fatal — and a reason the user can see, since an
invisible command is indistinguishable from an unimplemented feature.
"""

import re
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Tuple

import yaml

from mnemoai.client.memory.skill_store import _parse_frontmatter
from mnemoai.utils.console import print_error
from mnemoai.utils.logger import logger

# A command name must be typeable as one token after the slash and unambiguous in
# the completion menu, so the same shape as the built-ins: letters, digits, dash,
# underscore. Case-insensitive at lookup (dispatch lowercases), stored as authored.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

# The built-in slash commands a file must not shadow: the dispatcher matches those
# first, so a file named after one would never fire — it is rejected with a reason
# instead of appearing to work. Listed HERE, not read from the terminal UI's
# command table, so this module stays pure file logic that anything can consult
# (``/doctor`` counts commands too); a test pins the two lists together, so adding
# a built-in without reserving it fails the suite.
BUILTIN_COMMANDS = (
    "auto", "branch", "clear", "compact", "config", "context", "copy", "diff",
    "doctor", "exit", "export", "features", "files", "help", "hooks", "load", "mcp",
    "memory", "model", "params", "plan", "quit", "rename", "rewind", "save", "skills",
    "usage",
)

# A leading `_` marks a file in the commands dir that is NOT a command (notes, a
# draft, the bundled `_README.md`), so authoring docs can live beside the commands
# they document without becoming one.
_IGNORED_PREFIXES = ("_", ".")

# Sanity cap on a command body. It is only paid when the command is invoked (not
# every turn like steering), but the directory is a plain `*.md` glob — a large
# notes file dropped in here would silently become a command that floods the
# prompt, so anything over this is reported and skipped instead.
_MAX_BODY_CHARS = 40_000

# Description shown when the file has no frontmatter: the first non-empty body
# line, trimmed to this. A bare markdown file must work with no ceremony.
_MAX_DESC_CHARS = 80

# Both are rendered in fixed-width UI (the completion menu's meta column, the
# framed command reference), so an unbounded one is a layout bug waiting to
# happen — clipped here, at the one place they're read from disk.
_MAX_HINT_CHARS = 40

# `$ARGUMENTS` and `$1`…`$9`, matched in ONE pass so a substituted value is never
# re-scanned (an argument containing "$1" stays intact). Word-bounded so `$1` does
# not match inside `$12` — a two-digit positional is not supported, and silently
# rewriting half of it would be worse than leaving it alone.
_PLACEHOLDER_RE = re.compile(r"\$ARGUMENTS\b|\$([1-9])(?![0-9])")


class UserCommand(NamedTuple):
    """A parsed user-defined command.

    Attributes:
        name: The file stem — the token typed after the slash.
        description: One-line summary for the ``/`` menu and ``/help``.
        argument_hint: Optional short hint for what to type after the name ("").
        body: The markdown body, i.e. the prompt before substitution.
        path: The file it came from.
    """

    name: str
    description: str
    argument_hint: str = ""
    body: str = ""
    path: Optional[Path] = None

    @property
    def label(self) -> str:
        """``/name`` plus the argument hint, as shown in the command reference."""
        return f"/{self.name} {self.argument_hint}".strip()


class CommandIssue(NamedTuple):
    """A file in the commands dir that was rejected, and why."""

    name: str
    reason: str


def substitute(body: str, arguments: str = "") -> str:
    """Expand ``$ARGUMENTS`` / ``$1``…``$9`` in a command body.

    ``$ARGUMENTS`` becomes the whole argument string; ``$N`` the Nth
    whitespace-separated word (empty when not supplied — a command called with
    fewer args than it references must still run). A body that references NO
    placeholder gets the arguments **appended** instead of dropping them: the
    common case is a one-line instruction plus a target, and silently discarding
    what the user typed is the one outcome that reads as a bug.
    """
    args = (arguments or "").strip()
    words = args.split()

    def _sub(m: "re.Match") -> str:
        if m.group(1) is None:
            return args
        idx = int(m.group(1)) - 1
        return words[idx] if idx < len(words) else ""

    expanded = _PLACEHOLDER_RE.sub(_sub, body)
    if args and expanded == body:
        return f"{body.rstrip()}\n\n{args}"
    return expanded


def _clip(text: str, width: int) -> str:
    """Whitespace-collapsed ``text`` bounded to ``width`` chars, … when cut."""
    text = " ".join(str(text or "").split())
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def _describe(front: dict, body: str) -> str:
    """The command's one-line description: frontmatter, else the first body line."""
    desc = str(front.get("description", "")).strip()
    if not desc:
        for line in body.splitlines():
            line = line.strip().lstrip("#").strip()
            if line:
                desc = line
                break
    return _clip(desc, _MAX_DESC_CHARS) or "custom command"


def _dir_signature(root: Path) -> tuple:
    """Cheap fingerprint of the commands root: (name, mtime) per ``*.md``.

    Changes on any add/remove/edit, so the memoized scan invalidates exactly when
    the on-disk commands do — an edit applies to the very next line typed, without
    re-parsing every file per keystroke of the completion menu.
    """
    sig = []
    for entry in sorted(root.glob("*.md")):
        try:
            sig.append((entry.name, entry.stat().st_mtime_ns))
        except OSError:
            continue
    return tuple(sig)


# Process-global memoization of the scan, keyed by root path and invalidated by the
# directory fingerprint (the store is re-instantiated per dispatch/completion).
_SCAN_CACHE: dict = {}


class UserCommandStore:
    """Scan and expand user-defined slash commands from a commands root."""

    def __init__(
        self, root: Optional[Path] = None, reserved: Iterable[str] = BUILTIN_COMMANDS
    ) -> None:
        """Initialize the store.

        Args:
            root: Commands root directory; defaults to ``paths.commands_dir()``.
            reserved: Built-in command names (slash optional) that a file must not
                shadow; defaults to :data:`BUILTIN_COMMANDS`.
        """
        if root is None:
            from mnemoai.utils.paths import commands_dir

            root = commands_dir()
        self.root = Path(root)
        self.reserved = {str(r).lstrip("/").lower() for r in reserved}

    def _scan(self) -> Tuple[List[UserCommand], List[CommandIssue]]:
        """Scan the root once, returning (valid commands, rejected issues)."""
        if not self.root.is_dir():
            return [], []
        key = (str(self.root), tuple(sorted(self.reserved)))
        try:
            sig = _dir_signature(self.root)
        except OSError:
            sig = None
        if sig is not None:
            cached = _SCAN_CACHE.get(key)
            if cached is not None and cached[0] == sig:
                return cached[1], cached[2]
        commands: List[UserCommand] = []
        issues: List[CommandIssue] = []
        for entry in sorted(self.root.glob("*.md")):
            if not entry.is_file() or entry.name.startswith(_IGNORED_PREFIXES):
                continue
            name = entry.stem
            if not _NAME_RE.match(name):
                issues.append(
                    CommandIssue(entry.name, "name isn't a single /command token")
                )
                continue
            if name.lower() in self.reserved:
                issues.append(
                    CommandIssue(entry.name, f"/{name} is a built-in command")
                )
                continue
            try:
                text = entry.read_text()
            except (OSError, UnicodeDecodeError) as e:
                issues.append(CommandIssue(entry.name, f"could not read it ({e})"))
                continue
            if len(text) > _MAX_BODY_CHARS:
                issues.append(
                    CommandIssue(
                        entry.name,
                        f"too large to be a prompt ({len(text)} > {_MAX_BODY_CHARS} chars)",
                    )
                )
                continue
            # Frontmatter is OPTIONAL here (unlike a skill, whose description is
            # what the model routes on): a plain markdown file is a valid command.
            try:
                front, body = _parse_frontmatter(text)
            except yaml.YAMLError as e:
                issues.append(CommandIssue(entry.name, f"invalid YAML frontmatter ({e})"))
                continue
            body = body.strip()
            if not body:
                issues.append(CommandIssue(entry.name, "no prompt text in the file"))
                continue
            commands.append(
                UserCommand(
                    name=name,
                    description=_describe(front, body),
                    argument_hint=_clip(front.get("argument_hint", ""), _MAX_HINT_CHARS),
                    body=body,
                    path=entry,
                )
            )
        if commands:
            logger.debug(
                "Loaded %d user command(s): %s",
                len(commands),
                ", ".join(f"/{c.name}" for c in commands),
            )
        for issue in issues:
            print_error(f"Command '{issue.name}': {issue.reason}; skipping.")
        if sig is not None:
            _SCAN_CACHE[key] = (sig, commands, issues)
        return commands, issues

    def list_commands(self) -> List[UserCommand]:
        """Return all valid commands under the root (sorted by name)."""
        return self._scan()[0]

    def list_issues(self) -> List[CommandIssue]:
        """Return files that were found but rejected, with reasons."""
        return self._scan()[1]

    def get(self, name: str) -> Optional[UserCommand]:
        """Return the command called ``name`` (slash optional), or None."""
        name = (name or "").strip().lstrip("/").lower()
        if not name:
            return None
        for cmd in self.list_commands():
            if cmd.name.lower() == name:
                return cmd
        return None

    def expand(self, line: str) -> Optional[Tuple[UserCommand, str]]:
        """Resolve a submitted ``/name args`` line to (command, expanded prompt).

        Returns None when the line isn't a user command, so the caller can fall
        through to treating it as an ordinary query — an unknown ``/thing`` keeps
        its current meaning (prose) rather than becoming an error.
        """
        text = (line or "").strip()
        if not text.startswith("/"):
            return None
        head, _, rest = text.partition(" ")
        cmd = self.get(head)
        if cmd is None:
            return None
        return cmd, substitute(cmd.body, rest.strip())

    def completions(self) -> List[Tuple[str, str]]:
        """``[("/name", description)]`` pairs for the ``/`` completion menu.

        The ``argument_hint`` rides in the description here rather than in the
        label: the menu is open at exactly the moment you're about to type the
        arguments, and it's the one place with room for the hint (the framed
        command reference clips to the built-ins' widths).
        """
        return [
            (
                f"/{c.name}",
                f"{c.description} · {c.argument_hint}" if c.argument_hint else c.description,
            )
            for c in self.list_commands()
        ]
