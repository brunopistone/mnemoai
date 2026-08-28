"""What changed on disk, and which part of it was us (`/diff`).

The session edits files; before committing, the question is always the same — what
is different now, and did *this* conversation do it. Both halves are already
knowable, just not together: `git diff` sees the working tree but not who dirtied
it, and the file ledger (`/files`) knows what the session wrote but not what the
change looks like. So this report is `git diff --numstat` with the ledger's
written paths **marked**, which is the one column no other tool can give:
everything unmarked was already dirty when the conversation started.

It exists as a command rather than as a question to the model because a diff is
worth nothing if reading it costs a turn — the model would have to shell out, the
output would land in the context window at full size, and the answer would arrive
in a paraphrase. Here it is instant, exact, and free.

**Read-only, by construction.** Every git invocation in this module inspects
(`rev-parse`, `status`, `diff`, `ls-files`); nothing stages, stashes, checks out or
commits. A report that can change the thing it reports is not a report — the same
rule `/doctor` follows.

**Bounded**, because scrollback is the budget: the summary caps its file list, one
file's diff caps its lines, and each cap prints the exact git command that shows
the rest rather than pretending nothing was cut. An untracked file is counted by
reading it (git has nothing to diff against), which is why it is size-capped too.

Parsing and rendering are pure (`parse_numstat`, `render`, `colorize`), so the
report is unit-testable without a repository; `collect`/`report`/`file_report`
take the client — the `context_report`/`doctor` collaborator pattern.
"""

import os
import subprocess
from typing import Any, List, NamedTuple, Optional, Tuple

from mnemoai.utils.logger import logger, one_line

# Git is local, but a pathological repo (a huge tree, a cold NFS mount) must not
# hang the REPL — the report is a convenience, so it gives up instead.
_TIMEOUT = 15

# Rows in the summary before the rest collapse into a count.
_MAX_ROWS = 40

# Lines of one file's diff before the rest collapse into a count. Roughly two
# screens: enough to read a change, little enough to leave the conversation above
# it reachable.
_MAX_DIFF_LINES = 300

# An untracked file is counted by reading it (there is no blob to diff against),
# so it is capped: past this it is reported as large rather than opened.
_MAX_UNTRACKED_BYTES = 2_000_000

_GRAY = "\033[90m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_ADD = "\033[32m"
_DEL = "\033[31m"
_MARK = "\033[92m"
_HEADER = "\033[38;5;111m"


class Change(NamedTuple):
    """One changed file: what happened to it, and whether this session did it."""

    path: str  # repo-relative, as git reports it
    added: int
    deleted: int
    untracked: bool = False
    binary: bool = False
    mine: bool = False  # this session wrote it (from the file ledger)


def _git(args: List[str], cwd: str) -> Optional[str]:
    """Run a read-only git command; ``None`` when it fails or isn't there."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            stdin=subprocess.DEVNULL,  # never let git open a pager or prompt
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug(f"/diff: git {' '.join(args)} failed: {one_line(e)}")
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _numstat_path(raw: str) -> str:
    """The post-change path from a numstat entry, unwrapping a rename.

    Git writes a rename as ``old => new`` or ``dir/{old => new}/x`` — the name that
    exists on disk now is the one worth showing, and the one the ledger recorded.
    """
    raw = raw.strip()
    if "=>" not in raw:
        return raw
    if "{" in raw and "}" in raw:
        head, rest = raw.split("{", 1)
        middle, tail = rest.split("}", 1)
        return f"{head}{middle.split('=>')[-1].strip()}{tail}".replace("//", "/")
    return raw.split("=>")[-1].strip()


def parse_numstat(text: str) -> List[Change]:
    """Parse ``git diff --numstat`` output into :class:`Change` rows.

    Pure. A binary file's counts are ``-``/``-`` in this format, which is why the
    numbers are parsed defensively rather than cast.
    """
    changes = []
    for line in (text or "").splitlines():
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        added, deleted, raw = fields[0], fields[1], "\t".join(fields[2:])
        path = _numstat_path(raw)
        if not path:
            continue
        binary = added == "-" or deleted == "-"
        changes.append(
            Change(
                path=path,
                added=0 if binary else int(added or 0),
                deleted=0 if binary else int(deleted or 0),
                binary=binary,
            )
        )
    return changes


def _count_lines(absolute: str) -> Tuple[int, bool]:
    """``(lines, binary)`` for an untracked file — read, because git can't diff it."""
    try:
        if os.path.getsize(absolute) > _MAX_UNTRACKED_BYTES:
            return 0, True  # reported as "large" rather than opened
        with open(absolute, "rb") as fh:
            data = fh.read()
    except OSError:
        return 0, True
    if b"\0" in data:
        return 0, True
    return len(data.splitlines()), False


def _mine(root: str, path: str, changed: set) -> bool:
    """Did this session write ``path``? Keyed like the ledger, so spellings agree."""
    if not changed:
        return False
    absolute = os.path.join(root, path)
    try:
        key = os.path.normcase(os.path.realpath(absolute))
    except OSError:
        key = os.path.normcase(os.path.abspath(absolute))
    return key in changed


def repo_root(cwd: str) -> Optional[str]:
    """The working tree containing ``cwd``, or None when it isn't a repo."""
    out = _git(["rev-parse", "--show-toplevel"], cwd)
    return out.strip() if out and out.strip() else None


def branch_name(root: str) -> str:
    """The current branch, ``HEAD detached`` when there isn't one."""
    out = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    name = (out or "").strip()
    return "HEAD detached" if name == "HEAD" else name


def _changed_paths(client: Any) -> set:
    """The resolved paths the session wrote, from the agent's file ledger."""
    ledger = getattr(getattr(client, "agent", None), "files", None)
    if ledger is None:
        return set()
    try:
        return ledger.changed_paths()
    except Exception:  # noqa: BLE001 — a marker column must not break the report
        return set()


def collect(client: Any, cwd: Optional[str] = None):
    """``(root, branch, changes)`` for the repo around ``cwd``, or None with no repo.

    ``changes`` covers everything uncommitted — staged and unstaged against HEAD,
    plus untracked files — because "what have I not committed yet" is one question,
    not three, and the index is invisible from the chat anyway.
    """
    cwd = cwd or os.getcwd()
    root = repo_root(cwd)
    if root is None:
        return None
    changed = _changed_paths(client)

    # HEAD covers staged + unstaged at once. On a repo with no commits yet there is
    # no HEAD, so everything is untracked and this correctly yields nothing.
    tracked = parse_numstat(_git(["diff", "--numstat", "HEAD"], root) or "")

    untracked = []
    for rel in (_git(["ls-files", "--others", "--exclude-standard"], root) or "").splitlines():
        rel = rel.strip()
        if not rel:
            continue
        lines, binary = _count_lines(os.path.join(root, rel))
        untracked.append(Change(path=rel, added=lines, deleted=0, untracked=True, binary=binary))

    changes = [c._replace(mine=_mine(root, c.path, changed)) for c in tracked + untracked]
    # This session's own edits first, then the largest changes: the rows most
    # likely to be read are the ones the conversation just produced.
    changes.sort(key=lambda c: (not c.mine, -(c.added + c.deleted), c.path))
    return root, branch_name(root), changes


def _counts(change: Change) -> Tuple[str, str]:
    """``(plain, colored)`` counts for a file — ``+14 -2``, or what stands in.

    Both forms, because the column is right-padded to its widest member and ANSI
    escapes occupy no columns: measuring the colored string would ragged the tag
    beside it.
    """
    if change.binary:
        return "binary", f"{_GRAY}binary{_RESET}"
    plain, colored = [], []
    if change.added:
        plain.append(f"+{change.added}")
        colored.append(f"{_ADD}+{change.added}{_RESET}")
    if change.deleted:
        plain.append(f"-{change.deleted}")
        colored.append(f"{_DEL}-{change.deleted}{_RESET}")
    if not plain:
        return "no change", f"{_GRAY}no change{_RESET}"
    return " ".join(plain), " ".join(colored)


def render(root: str, branch: str, changes: List[Change]) -> str:
    """The ``/diff`` summary. Pure over already-collected rows."""
    where = _short(root)
    head = f"{_BOLD}Changes in {where}{_RESET}"
    if branch:
        head += f" {_GRAY}({branch}){_RESET}"
    if not changes:
        return (
            f"{head}\n\n  {_GRAY}Nothing uncommitted — the working tree is "
            f"clean.{_RESET}"
        )

    rows = changes[:_MAX_ROWS]
    width = min(max(len(c.path) for c in rows), 60)
    counts = [_counts(c) for c in rows]
    count_w = max(len(plain) for plain, _ in counts)
    out = [head, ""]
    for change, (plain, colored) in zip(rows, counts):
        mark = f"{_MARK}✎{_RESET}" if change.mine else " "
        tag = f"  {_GRAY}new{_RESET}" if change.untracked else ""
        padded = colored + " " * (count_w - len(plain))
        out.append(f"  {mark} {_clip(change.path, width).ljust(width)}  {padded}{tag}".rstrip())
    if len(changes) > _MAX_ROWS:
        out.append(f"    {_GRAY}… +{len(changes) - _MAX_ROWS} more (git status){_RESET}")

    added = sum(c.added for c in changes)
    deleted = sum(c.deleted for c in changes)
    mine = sum(1 for c in changes if c.mine)
    total = f"{len(changes)} file{'s' if len(changes) != 1 else ''}"
    out.append("")
    out.append(
        f"  {_GRAY}{total} · {_RESET}{_ADD}+{added}{_RESET} {_DEL}-{deleted}{_RESET}"
        + (f"{_GRAY} · {_MARK}✎{_GRAY} {mine} written this session{_RESET}" if mine else "")
    )
    out.append(f"  {_GRAY}/diff <path> shows one file's diff.{_RESET}")
    return "\n".join(out)


def colorize(diff: str, limit: Optional[int] = None) -> Tuple[str, int]:
    """``(colored, dropped)`` for a unified diff, in the app's own palette.

    Pure, and deliberately line-level rather than a re-parse: git already decided
    what the hunks are, so the only job here is to make them readable — additions
    green, deletions red, hunk headers highlighted, everything else gray so the
    changed lines are the only thing with color in them.

    The cap is read at CALL time (not as a default argument, which would bind it at
    import and make the module constant unoverridable).
    """
    limit = _MAX_DIFF_LINES if limit is None else limit
    lines = (diff or "").splitlines()
    dropped = max(0, len(lines) - limit)
    out = []
    for line in lines[:limit]:
        if line.startswith("@@"):
            out.append(f"{_HEADER}{line}{_RESET}")
        elif line.startswith(("+++", "---")):
            out.append(f"{_GRAY}{_BOLD}{line}{_RESET}")
        elif line.startswith("+"):
            out.append(f"{_ADD}{line}{_RESET}")
        elif line.startswith("-"):
            out.append(f"{_DEL}{line}{_RESET}")
        elif line.startswith(("diff --git", "index ", "new file", "deleted file", "similarity")):
            out.append(f"{_GRAY}{line}{_RESET}")
        else:
            out.append(f"{_GRAY}{line}{_RESET}")
    return "\n".join(out), dropped


def _is_tracked(root: str, rel: str) -> bool:
    """Does git already know this file? (An empty diff means nothing without it.)"""
    return bool((_git(["ls-files", "--", rel], root) or "").strip())


def _untracked_as_diff(absolute: str, rel: str) -> Optional[str]:
    """A new file rendered as all-additions, since git has no blob to diff."""
    lines, binary = _count_lines(absolute)
    if binary:
        return f"+++ {rel}\n(binary or too large to show — {lines or 'unknown'} lines)"
    try:
        with open(absolute, "r", errors="replace") as fh:
            body = fh.read()
    except OSError:
        return None
    head = f"+++ {rel}\n@@ new file · {lines} line{'s' if lines != 1 else ''} @@"
    return head + "\n" + "\n".join(f"+{line}" for line in body.splitlines())


def file_report(client: Any, path: str, cwd: Optional[str] = None) -> str:
    """``/diff <path>``: one file's uncommitted diff, colored and capped."""
    cwd = cwd or os.getcwd()
    root = repo_root(cwd)
    if root is None:
        return _no_repo(cwd)

    absolute = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(absolute):
        # A repo-relative spelling is the natural one to type after the summary,
        # which lists paths that way.
        candidate = os.path.join(root, path)
        if os.path.exists(candidate):
            absolute = candidate
        else:
            return f"No such file: {path}"
    if os.path.isdir(absolute):
        return f"{path} is a directory — /diff takes one file (or no argument)."

    try:
        rel = os.path.relpath(absolute, root)
    except ValueError:
        rel = absolute
    if rel.startswith(".."):
        return f"{path} is outside {_short(root)}."

    diff = _git(["diff", "HEAD", "--", rel], root)
    if not (diff or "").strip() and not _is_tracked(root, rel):
        # Empty diff, two very different reasons: a committed file that hasn't
        # changed, or one git has never seen. Only the second is all-additions —
        # conflating them showed every unchanged file as brand new.
        diff = _untracked_as_diff(absolute, rel)
    if not (diff or "").strip():
        return f"{_BOLD}{rel}{_RESET}\n\n  {_GRAY}No uncommitted changes.{_RESET}"

    colored, dropped = colorize(diff)
    head = f"{_BOLD}{rel}{_RESET}"
    if _mine(root, rel, _changed_paths(client)):
        head += f" {_MARK}✎{_RESET}{_GRAY} written this session{_RESET}"
    out = [head, "", colored]
    if dropped:
        out.append("")
        out.append(f"  {_GRAY}… +{dropped} more lines — run: git diff HEAD -- {rel}{_RESET}")
    return "\n".join(out)


def report(client: Any, path: str = "") -> str:
    """``/diff [path]`` for a live client; never raises."""
    try:
        if path.strip():
            return file_report(client, path.strip())
        collected = collect(client)
        if collected is None:
            return _no_repo(os.getcwd())
        return render(*collected)
    except Exception as e:  # noqa: BLE001 — a report must not become the problem
        logger.error(f"/diff failed: {one_line(e)}", extra={"console": False})
        return f"Could not read the working tree: {one_line(e)}"


def _no_repo(cwd: str) -> str:
    """Why there is nothing to show, without implying something broke."""
    return (
        f"{_short(cwd)} is not inside a git repository, so there is nothing to "
        f"diff.\n  {_GRAY}/files lists what this session touched anyway.{_RESET}"
    )


def _short(absolute: str) -> str:
    """``~``-relative path, so the header fits on one line."""
    home = os.path.expanduser("~")
    if home and absolute.startswith(home + os.sep):
        return "~" + absolute[len(home):]
    return absolute


def _clip(text: str, width: int) -> str:
    """``text`` fitted to ``width``, keeping the tail (the filename identifies it)."""
    return text if len(text) <= width else "…" + text[-(width - 1):]
