"""Fast search tools using glob and ripgrep."""

import base64
import fnmatch
import json
import os
import re
import subprocess
import time

from mcp.server.fastmcp import FastMCP

from mnemoai.utils.logger import logger
from mnemoai.utils.path_utils import normalize_path

# ripgrep's --max-columns is a no-op in --json mode (it emits the full line
# regardless), so we cap long lines here to stop minified/base64 lines from
# blowing up the result.
MAX_LINE_CHARS = 500

# Noise dirs glob_search skips by default (opt out with include_ignored=True).
# ripgrep already honors .gitignore/.git/hidden; stdlib glob does not, so glob
# needs its own set — kept on stdlib so glob still works with no rg installed.
_DEFAULT_IGNORED_DIRS = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}

# Hard cap on matches buffered when sorting by mtime, so "collect all before
# sorting" on an enormous tree can't exhaust memory.
_GLOB_SCAN_CEILING = 100000

# Wall-clock bound on ONE glob_search, matching grep_search's subprocess timeout.
# The tool must fail with a usable partial result long before the CLIENT's
# LLM.MCP_CALL_TIMEOUT (300s) fires: that timeout kills the call without a retry
# and, until it does, occupies a slot the agent is waiting on. A caller pointing
# a `**` pattern at $HOME is the ordinary way to get here.
_GLOB_TIME_BUDGET_S = 30.0

_MAGIC = re.compile(r"[*?\[]")


class _ScanBudget:
    """Wall-clock bound for a directory walk; ``expired()`` is what stops it."""

    def __init__(self, seconds: float) -> None:
        self.deadline = time.monotonic() + seconds
        self.timed_out = False

    def expired(self) -> bool:
        """True once the budget is spent (latched, so it's cheap to re-ask)."""
        if not self.timed_out and time.monotonic() >= self.deadline:
            self.timed_out = True
        return self.timed_out


def _split_glob_pattern(root: str, pattern: str) -> tuple[str, list[str]]:
    """Split ``pattern`` into a concrete search root and its magic segments.

    Leading segments with no wildcard are folded into the root, so
    ``src/**/*.ts`` starts walking at ``src`` instead of filtering everything
    under ``root``. An absolute pattern anchors itself and ignores ``root``, the
    way ``os.path.join`` used to.
    """
    parts = [p for p in pattern.replace(os.sep, "/").split("/") if p not in ("", ".")]
    if os.path.isabs(pattern):
        root = os.path.splitdrive(pattern)[0] + os.sep
    while parts and not _MAGIC.search(parts[0]):
        root = os.path.join(root, parts.pop(0))
    return os.path.normpath(root) if root else root, parts


def _iter_glob_files(
    root: str, segments: list[str], include_ignored: bool, budget: _ScanBudget
):
    """Yield files under ``root`` matching ``segments``, one segment per path part.

    Replaces ``glob.iglob(recursive=True)``, which bounds its OUTPUT but not its
    WORK: the ignore list was applied to the matches it yielded, so a `**` walk
    still descended every ``node_modules``/``build``/``dist`` tree only to throw
    the results away, nothing checked the clock, and on Python < 3.13 ``**``
    follows directory symlinks — a link back to an ancestor made the walk
    effectively unbounded. Here the noise dirs are PRUNED before descending,
    directory symlinks are never followed, and the budget is checked as we go.

    Hidden names are skipped unless the matching segment itself starts with a dot,
    which is stdlib glob's rule (and why ``.git`` was never the slow part).
    """
    if not segments:
        # A pattern with no wildcard at all: the root IS the candidate.
        if os.path.isfile(root):
            yield root
        return

    stack = [(root, 0)]
    seen = set()
    while stack:
        if budget.expired():
            return
        state = stack.pop()
        if state in seen:  # two `**` segments can reach the same state
            continue
        seen.add(state)
        dirpath, index = state
        segment = segments[index]
        recursive = segment == "**"
        last = index == len(segments) - 1
        if recursive and not last:
            # `**` matches zero directories too, so the next segment also gets a
            # shot at THIS directory.
            stack.append((dirpath, index + 1))
        hidden_ok = segment.startswith(".")
        try:
            entries = list(os.scandir(dirpath))
        except OSError:  # unreadable/vanished dir — skip it, keep the walk alive
            continue
        for entry in entries:
            if budget.expired():
                return
            name = entry.name
            if name.startswith(".") and not hidden_ok:
                continue
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_dir:
                if not include_ignored and name in _DEFAULT_IGNORED_DIRS:
                    continue
                if recursive:
                    stack.append((entry.path, index))
                elif not last and fnmatch.fnmatch(name, segment):
                    stack.append((entry.path, index + 1))
                continue
            # A file only ever answers the LAST segment ("**" as the last segment
            # means "everything below here").
            if not last:
                continue
            if not recursive and not fnmatch.fnmatch(name, segment):
                continue
            try:
                if entry.is_file():  # follows a symlink to a real file
                    yield entry.path
            except OSError:
                continue


def register_search_tools(mcp: FastMCP) -> None:
    """Register fast search tools.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    def glob_search(
        pattern: str,
        path: str = None,
        max_results: int = 1000,
        sort_by_mtime: bool = True,
        include_ignored: bool = False,
    ) -> str:
        """Fast file pattern matching.

        Use this to find files by name patterns, NOT for content search.
        For large searches (entire home dir or system-wide), use execute_bash with 'find' instead.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "src/**/*.ts", "*.yaml")
            path: Directory to search (default: current directory)
            max_results: Maximum number of results to return (default: 1000, use 0 for unlimited)
            sort_by_mtime: Sort by modification time (default: True, disable for faster large searches)
            include_ignored: Include files inside common noise dirs (.git, node_modules, .venv, __pycache__, build, dist, ...); default False

        Returns:
            JSON string with matching file paths. The scan is bounded: it stops
            after 30 seconds and returns what it found with timed_out=True, so a
            pattern aimed at a huge tree gives a partial answer instead of hanging.
            Symlinked directories are not traversed (a link to an ancestor would
            make the walk endless); symlinks to files still match.

        Examples:
            glob_search(pattern="**/*.py", max_results=100)  # First 100 Python files
            glob_search(pattern="*.yaml", path="/home/user/configs")  # YAML files in specific dir
            glob_search(pattern="test_*.py", sort_by_mtime=False)  # Faster, unsorted results

        Performance tip: For system-wide searches, use execute_bash with:
            find /path -name "*.py" -type f | wc -l  # Just count
            find /path -name "*.py" -type f -print | head -1000  # First 1000
        """
        if path is None:
            path = os.getcwd()

        # Expand ~ and tolerate shell escaping/quoting in the path.
        path = normalize_path(path)

        if not os.path.exists(path):
            return json.dumps(
                {"error": True, "message": f"Path does not exist: {path}"}
            )

        try:
            search_root, segments = _split_glob_pattern(path, pattern)
            budget = _ScanBudget(_GLOB_TIME_BUDGET_S)

            def _mtime(p: str) -> float:
                # A file yielded by glob can vanish before we stat it; treat a
                # missing file as oldest rather than crashing the whole search.
                try:
                    return os.path.getmtime(p)
                except OSError:
                    return 0.0

            matches = []
            truncated = False
            scan_ceiling_hit = False

            found = _iter_glob_files(search_root, segments, include_ignored, budget)

            if sort_by_mtime:
                # Collect ALL matches (up to a hard ceiling), THEN sort, THEN
                # slice — so a capped result is the truly-newest N, not the first
                # N the walk happened to yield.
                for match in found:
                    matches.append(match)
                    if len(matches) >= _GLOB_SCAN_CEILING:
                        scan_ceiling_hit = True
                        logger.debug(
                            f"glob_search hit scan ceiling: {_GLOB_SCAN_CEILING}"
                        )
                        break
                matches.sort(key=_mtime, reverse=True)
                total = len(matches)
                if max_results > 0 and total > max_results:
                    matches = matches[:max_results]
                    truncated = True
            else:
                # Unsorted: stop lazily as soon as max_results is reached (fast).
                for match in found:
                    matches.append(match)
                    if max_results > 0 and len(matches) >= max_results:
                        truncated = True
                        break

            result = {
                "success": True,
                "matches": matches,
                "count": len(matches),
                "pattern": pattern,
                "search_path": path,
            }

            if truncated:
                result["truncated"] = True
                if sort_by_mtime and scan_ceiling_hit:
                    # Past the scan ceiling: `total` is only the buffered subset
                    # (not the true match count) and the newest-first ordering is
                    # over that subset, so don't claim either as global.
                    result["message"] = (
                        f"Showing {max_results} matches from the first "
                        f"{_GLOB_SCAN_CEILING} scanned (not globally newest — the "
                        "tree exceeds the scan ceiling). Use a more specific "
                        "pattern/path to narrow the search."
                    )
                elif sort_by_mtime:
                    result["message"] = (
                        f"Showing {max_results} of {total} matches (newest first). "
                        "Increase max_results or use max_results=0 for unlimited."
                    )
                else:
                    result["message"] = (
                        f"Results limited to {max_results} (unsorted). Use "
                        "sort_by_mtime=True for newest-first or max_results=0 for unlimited."
                    )
            if scan_ceiling_hit:
                result["scan_ceiling_hit"] = True
            if budget.timed_out:
                # Partial, and say so plainly: an unflagged short list reads as
                # "that's all there is". Overwrites any truncation message —
                # running out of time is the more important fact.
                result["truncated"] = True
                result["timed_out"] = True
                result["message"] = (
                    f"Scan stopped after {int(_GLOB_TIME_BUDGET_S)}s with "
                    f"{len(matches)} match(es) — the tree is too large to walk "
                    "fully. Narrow the pattern or point path at a subdirectory, "
                    "or use execute_bash with 'find' for a system-wide search."
                )
                logger.debug(
                    f"glob_search timed out after {_GLOB_TIME_BUDGET_S}s: "
                    f"{pattern} under {search_root}"
                )

            return json.dumps(result)

        except Exception as e:
            logger.error(f"Error in glob_search: {str(e)}", exc_info=True)
            return json.dumps({"error": True, "message": str(e)})

    @mcp.tool()
    def grep_search(
        pattern: str,
        path: str = None,
        file_pattern: str = None,
        case_insensitive: bool = False,
        output_mode: str = "files_with_matches",
        context_lines: int = 0,
        context_before: int = 0,
        context_after: int = 0,
        max_results: int = 100,
        offset: int = 0,
    ) -> str:
        """Fast content search using ripgrep.

        Use this to search FILE CONTENT, not filenames.
        This is 10-100x faster than traditional grep for large codebases.

        Args:
            pattern: Regex pattern to search for in file contents
            path: Directory to search (default: current directory)
            file_pattern: Filter files by glob (e.g., "*.py", "*.{ts,tsx}")
            case_insensitive: Case-insensitive search (default: False)
            output_mode: "files_with_matches" (default), "content", or "count"
            context_lines: Symmetric lines of context around each match, content mode (default: 0)
            context_before: Lines of context BEFORE each match (overrides context_lines; content mode)
            context_after: Lines of context AFTER each match (overrides context_lines; content mode)
            max_results: Max results to return, a TOTAL cap counting MATCHES only
                (context lines are extra); 0 = unlimited (default: 100)
            offset: Skip the first N results before applying max_results, for paging (default: 0)

        Returns:
            JSON string with search results

        Examples:
            grep_search(pattern="class Foo")  # Find class Foo definition
            grep_search(pattern="TODO|FIXME", file_pattern="*.py", case_insensitive=True)
            grep_search(pattern="import React", file_pattern="*.{ts,tsx}", output_mode="content")
        """
        if path is None:
            path = os.getcwd()

        # Expand ~ and tolerate shell escaping/quoting in the path.
        path = normalize_path(path)

        if not os.path.exists(path):
            return json.dumps(
                {"error": True, "message": f"Path does not exist: {path}"}
            )

        # Always run ripgrep in --json mode and derive every output mode from the
        # parsed match events. We deliberately do NOT pass --files-with-matches /
        # --count (they OVERRIDE --json and emit plain text this parser can't
        # read — which silently returned empty results) nor --max-count (a
        # PER-FILE cap, not the total the caller expects). The limit is applied
        # in Python after parsing, so max_results caps the true number of results
        # across all files in every mode. max_results <= 0 means unlimited.
        # --sort path forces a deterministic cross-file order (rg is parallel and
        # otherwise emits files in a nondeterministic order): the total cap and
        # offset paging are only meaningful over a STABLE order.
        cmd = ["rg", "--json", "--sort", "path"]

        if case_insensitive:
            cmd.append("-i")

        if file_pattern:
            cmd.append(f"--glob={file_pattern}")

        if output_mode == "content":
            # Asymmetric -A/-B (individual flags cover the symmetric case too);
            # context_before/after override the context_lines shorthand.
            before = context_before if context_before > 0 else context_lines
            after = context_after if context_after > 0 else context_lines
            if before > 0:
                cmd.append(f"-B{before}")
            if after > 0:
                cmd.append(f"-A{after}")

        # -e guards a pattern that begins with '-' from being read as a flag.
        cmd += ["-e", pattern, path]

        unlimited = max_results <= 0
        offset = max(0, offset)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # ripgrep gives an object its text as {"text": ...} for valid UTF-8,
            # or {"bytes": <base64>} for a line/path it couldn't decode (binary,
            # a stray byte). Prefer text; decode the bytes leniently otherwise so
            # a single non-UTF-8 match can't KeyError and abort the whole search.
            def _text(obj: dict) -> str:
                if not isinstance(obj, dict):
                    return ""
                if "text" in obj:
                    return obj["text"]
                b64 = obj.get("bytes")
                if b64:
                    try:
                        return base64.b64decode(b64).decode("utf-8", "replace")
                    except (ValueError, TypeError):
                        return ""
                return ""

            def _cap(text: str) -> str:
                # rg's --max-columns is a no-op under --json; cap here so a
                # minified/base64 line can't blow up the result.
                if len(text) > MAX_LINE_CHARS:
                    return text[:MAX_LINE_CHARS] + f"… [+{len(text) - MAX_LINE_CHARS} chars]"
                return text

            # Single pass over the events: an ordered row list (matches +, in
            # content mode, their -A/-B/-C context), the unique file order, and
            # per-file counts — enough to build any mode.
            all_matches = []
            file_order = []
            counts = {}
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Keep 'context' events too (rg emits them for -A/-B/-C) so the
                # model sees the surrounding lines in content mode; only real
                # matches count toward file_order/counts and the max_results cap.
                event = data.get("type")
                if event not in ("match", "context"):
                    continue
                ev_data = data["data"]
                file_path = _text(ev_data.get("path", {}))
                if not file_path:
                    continue  # can't identify the file — skip
                is_context = event == "context"
                all_matches.append(
                    {
                        "file": file_path,
                        "line": ev_data.get("line_number"),
                        "text": _cap(_text(ev_data.get("lines", {})).rstrip()),
                        "is_context": is_context,
                    }
                )
                if is_context:
                    continue
                if file_path not in counts:
                    counts[file_path] = 0
                    file_order.append(file_path)
                counts[file_path] += 1

            # ripgrep exits 0 (matches), 1 (no matches — a normal empty result),
            # or 2 (an error, e.g. an invalid regex OR a per-file permission
            # denial). Only treat exit >= 2 as FATAL when NO events were parsed:
            # a bad regex yields empty stdout + exit 2 (surface it), but an
            # unreadable file among readable ones still streams valid matches on
            # stdout with exit 2 (keep them, note the error non-fatally).
            if result.returncode >= 2 and not all_matches:
                msg = result.stderr.strip() or "ripgrep failed"
                return json.dumps(
                    {
                        "error": True,
                        "message": f"Search failed: {msg}",
                        "pattern": pattern,
                    }
                )
            partial_warning = (
                result.stderr.strip()
                if (result.returncode >= 2 and all_matches)
                else ""
            )

            if output_mode == "files_with_matches":
                files = (
                    file_order[offset:]
                    if unlimited
                    else file_order[offset : offset + max_results]
                )
                out = {
                    "success": True,
                    "files": files,
                    "count": len(files),
                    "pattern": pattern,
                    "search_path": path,
                }
                if not unlimited and len(file_order) > offset + max_results:
                    out["truncated"] = True
                    out["message"] = (
                        f"Showing {len(files)} of {len(file_order)} matching "
                        f"files (offset {offset}). Increase max_results/offset "
                        "or narrow the search."
                    )
                if partial_warning:
                    out["warnings"] = partial_warning
                return json.dumps(out)

            elif output_mode == "content":
                # Cap/offset count MATCHES only; context lines ride along with
                # THEIR match (window is [offset, offset+max_results)). Each
                # context row is owned by exactly ONE match — the nearest by line
                # number — so an off-window match's context can't leak in and the
                # same context line can't duplicate across pages (rg emits a
                # before-context row just before its match, an after-context row
                # just after; OR-ing both neighbours would double-attribute it).
                total_matches = sum(1 for r in all_matches if not r["is_context"])
                hi = None if unlimited else offset + max_results

                def _in_window(mi):
                    return mi is not None and mi >= offset and (hi is None or mi < hi)

                n = len(all_matches)
                # Ordinal of each match row (by appearance order).
                ordinal = {}
                c = -1
                for pos, r in enumerate(all_matches):
                    if not r["is_context"]:
                        c += 1
                        ordinal[pos] = c
                # Backward pass: for each position, the next match's ordinal+line.
                next_idx = [None] * n
                next_line = [None] * n
                ni = nl = None
                for pos in range(n - 1, -1, -1):
                    r = all_matches[pos]
                    if not r["is_context"]:
                        ni, nl = ordinal[pos], r["line"]
                    next_idx[pos], next_line[pos] = ni, nl
                # Forward pass: track the previous match; own each context row by
                # the nearer match (tie -> previous, both are correct).
                matches = []
                prev_idx = prev_line = None
                for pos, row in enumerate(all_matches):
                    if not row["is_context"]:
                        prev_idx, prev_line = ordinal[pos], row["line"]
                        if _in_window(prev_idx):
                            matches.append(row)
                        continue
                    rl = row["line"]
                    cand = []  # (line-distance, owning match ordinal)
                    if prev_idx is not None and prev_line is not None and rl is not None:
                        cand.append((abs(rl - prev_line), prev_idx))
                    if next_idx[pos] is not None and next_line[pos] is not None and rl is not None:
                        cand.append((abs(next_line[pos] - rl), next_idx[pos]))
                    if not cand:
                        continue  # no line info to attribute — drop the stray row
                    cand.sort(key=lambda t: t[0])
                    if _in_window(cand[0][1]):
                        matches.append(row)
                shown = sum(1 for r in matches if not r["is_context"])
                out = {
                    "success": True,
                    "matches": matches,
                    "count": shown,
                    "pattern": pattern,
                    "search_path": path,
                }
                if not unlimited and total_matches > offset + shown:
                    out["truncated"] = True
                    out["message"] = (
                        f"Showing {shown} of {total_matches} matches. "
                        "Increase max_results/offset or narrow the search."
                    )
                if partial_warning:
                    out["warnings"] = partial_warning
                return json.dumps(out)

            elif output_mode == "count":
                # One count line per file, so the total cap bounds the file set.
                capped = (
                    file_order[offset:]
                    if unlimited
                    else file_order[offset : offset + max_results]
                )
                capped_counts = {f: counts[f] for f in capped}
                out = {
                    "success": True,
                    "counts": capped_counts,
                    "total_matches": sum(capped_counts.values()),
                    "files_with_matches": len(capped_counts),
                    "pattern": pattern,
                    "search_path": path,
                }
                if not unlimited and len(file_order) > offset + max_results:
                    out["truncated"] = True
                    out["message"] = (
                        f"Showing counts for {len(capped)} of {len(file_order)} "
                        "matching files. Increase max_results/offset or narrow the search."
                    )
                if partial_warning:
                    out["warnings"] = partial_warning
                return json.dumps(out)

            else:
                return json.dumps(
                    {
                        "error": True,
                        "message": (
                            f"Invalid output_mode '{output_mode}'. Use "
                            "'files_with_matches', 'content', or 'count'."
                        ),
                    }
                )

        except subprocess.TimeoutExpired:
            return json.dumps(
                {
                    "error": True,
                    "message": "Search timed out after 30 seconds. Try narrowing your search with file_pattern.",
                }
            )
        except FileNotFoundError:
            return json.dumps(
                {
                    "error": True,
                    "message": "ripgrep (rg) not installed. Install with: brew install ripgrep (macOS) or apt install ripgrep (Linux)",
                }
            )
        except Exception as e:
            logger.error(f"Error in grep_search: {str(e)}", exc_info=True)
            return json.dumps({"error": True, "message": str(e)})
